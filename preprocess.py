#!/usr/bin/env python3
"""
视频预处理器 — Layer 1
功能：场景切换检测、关键帧去重、PPT 过滤、音频提取、OSS 上传、manifest 生成
依赖：全部为传统算法库，零 AI 模型、零 GPU
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import imagehash
import numpy as np
import yaml
from PIL import Image
from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Step 1: 音频提取 (FFmpeg)
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, output_dir: str, sample_rate: int = 16000) -> str:
    """提取音频为 WAV 16kHz 单声道，供 ASR 使用"""
    audio_path = os.path.join(output_dir, "audio.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        audio_path,
    ]
    log.info("提取音频: %s", video_path)
    subprocess.run(cmd, check=True, capture_output=True)
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    log.info("音频提取完成: %.1f MB", size_mb)
    return audio_path


# ---------------------------------------------------------------------------
# Step 2: 场景切换检测 (PySceneDetect · AdaptiveDetector)
# ---------------------------------------------------------------------------

def detect_scenes(video_path: str, threshold: float = 3.0,
                  min_scene_len: float = 1.0) -> list[dict]:
    """
    使用 AdaptiveDetector 检测场景切换点。
    原理：HSV 色彩空间帧间差异 + 滑动窗口自适应阈值。
    返回 [{start_time, end_time, start_frame, end_frame}, ...]
    """
    log.info("场景检测中 (阈值=%.1f)...", threshold)
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        AdaptiveDetector(adaptive_threshold=threshold,
                         min_scene_len=int(min_scene_len * video.frame_rate))
    )
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    scenes = []
    for start, end in scene_list:
        scenes.append({
            "start_time": start.get_seconds(),
            "end_time": end.get_seconds(),
            "start_frame": start.get_frames(),
            "end_frame": end.get_frames(),
        })
    log.info("检测到 %d 个场景", len(scenes))
    return scenes


# ---------------------------------------------------------------------------
# Step 3: 关键帧提取
# ---------------------------------------------------------------------------

def extract_keyframes(video_path: str, scenes: list[dict],
                      output_dir: str) -> list[dict]:
    """从每个场景的中间位置提取一帧作为关键帧"""
    frames_dir = os.path.join(output_dir, "keyframes")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    keyframes = []

    for i, scene in enumerate(scenes):
        mid_time = (scene["start_time"] + scene["end_time"]) / 2
        mid_frame = int(mid_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        if not ret:
            continue
        filename = f"keyframe_{i:04d}_{mid_time:.2f}s.jpg"
        filepath = os.path.join(frames_dir, filename)
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        keyframes.append({
            "index": i,
            "timestamp": mid_time,
            "scene_start": scene["start_time"],
            "scene_end": scene["end_time"],
            "filepath": filepath,
            "filename": filename,
        })

    cap.release()
    log.info("提取了 %d 个关键帧", len(keyframes))
    return keyframes


# ---------------------------------------------------------------------------
# Step 4: 感知哈希去重 (pHash)
# ---------------------------------------------------------------------------

def deduplicate_by_phash(keyframes: list[dict],
                         threshold: int = 5) -> list[dict]:
    """
    pHash 去重：汉明距离 < threshold 判定为相同画面。
    原理：图片缩小→灰度→DCT→取低频→二值化→64位哈希
    """
    if not keyframes:
        return keyframes

    unique = [keyframes[0]]
    prev_hash = imagehash.phash(Image.open(keyframes[0]["filepath"]))

    for kf in keyframes[1:]:
        curr_hash = imagehash.phash(Image.open(kf["filepath"]))
        if curr_hash - prev_hash >= threshold:
            unique.append(kf)
            prev_hash = curr_hash

    removed = len(keyframes) - len(unique)
    log.info("pHash 去重: %d → %d (移除 %d 张重复)", len(keyframes), len(unique), removed)
    return unique


# ---------------------------------------------------------------------------
# Step 5: PPT 类型过滤 (OpenCV 三特征融合)
# ---------------------------------------------------------------------------

def is_ppt_frame(filepath: str) -> bool:
    """
    三特征融合判断是否为 PPT 画面：
    1. Canny 边缘密度（PPT 文字产生大量边缘）
    2. 颜色标准差（PPT 背景色单一，std 低）
    3. Hough 水平线检测（文字基线）
    """
    img = cv2.imread(filepath)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / (h * w)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_std = np.mean(np.std(hsv[:, :, 0].astype(float)))

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=w * 0.3, maxLineGap=20)
    horizontal_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10 or angle > 170:
                horizontal_count += 1

    is_ppt = (edge_density > 0.02 and color_std < 40 and horizontal_count >= 3)
    return bool(is_ppt)


def filter_ppt_frames(keyframes: list[dict]) -> list[dict]:
    """标记每个关键帧是否为 PPT 类型"""
    for kf in keyframes:
        kf["is_ppt"] = is_ppt_frame(kf["filepath"])
    ppt_count = sum(1 for kf in keyframes if kf["is_ppt"])
    log.info("PPT 过滤: %d/%d 帧识别为 PPT", ppt_count, len(keyframes))
    return keyframes


# ---------------------------------------------------------------------------
# Step 6: OSS 上传
# ---------------------------------------------------------------------------

def upload_to_oss(config: dict, files: list[str],
                  prefix: str = "") -> dict[str, str]:
    """上传文件到阿里云 OSS，返回 {local_path: oss_url}"""
    try:
        import oss2
    except ImportError:
        log.warning("oss2 未安装，跳过 OSS 上传")
        return {}

    oss_cfg = config.get("oss", {})
    auth = oss2.Auth(oss_cfg["access_key_id"], oss_cfg["access_key_secret"])
    bucket = oss2.Bucket(auth, oss_cfg["endpoint"], oss_cfg["bucket"])
    oss_prefix = oss_cfg.get("prefix", "video-knowledge/")

    url_map = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        key = f"{oss_prefix}{prefix}{filename}"
        log.info("上传 OSS: %s → %s", filename, key)
        bucket.put_object_from_file(key, filepath)
        bucket_name = oss_cfg["bucket"]
        endpoint = oss_cfg["endpoint"]
        url = endpoint.replace("https://", f"https://{bucket_name}.") + f"/{key}"
        url_map[filepath] = url

    log.info("OSS 上传完成: %d 个文件", len(url_map))
    return url_map


# ---------------------------------------------------------------------------
# Step 7: 生成 manifest.json
# ---------------------------------------------------------------------------

def generate_manifest(video_path: str, audio_path: str,
                      keyframes: list[dict], oss_urls: dict,
                      output_dir: str) -> str:
    """生成 manifest.json 供 Dify Workflow 消费"""
    video_name = Path(video_path).stem

    manifest = {
        "version": "1.0",
        "video_name": video_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "audio": {
            "local_path": audio_path,
            "oss_url": oss_urls.get(audio_path, ""),
        },
        "keyframes": [],
    }

    for kf in keyframes:
        entry = {
            "index": kf["index"],
            "timestamp": kf["timestamp"],
            "scene_start": kf["scene_start"],
            "scene_end": kf["scene_end"],
            "is_ppt": kf.get("is_ppt", False),
            "filename": kf["filename"],
            "oss_url": oss_urls.get(kf["filepath"], ""),
        }
        manifest["keyframes"].append(entry)

    manifest["stats"] = {
        "total_scenes": len(keyframes),
        "ppt_frames": sum(1 for kf in keyframes if kf.get("is_ppt")),
        "non_ppt_frames": sum(1 for kf in keyframes if not kf.get("is_ppt")),
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log.info("Manifest 已生成: %s", manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(video_path: str, config_path: str = "config.yaml",
                 output_dir: str | None = None, skip_oss: bool = False):
    """完整预处理流水线"""
    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        log.error("视频文件不存在: %s", video_path)
        sys.exit(1)

    config = load_config(config_path)
    prep_cfg = config.get("preprocess", {})

    if output_dir is None:
        video_name = Path(video_path).stem
        output_dir = os.path.join("output", video_name)
    os.makedirs(output_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("开始预处理: %s", video_path)
    log.info("输出目录: %s", output_dir)
    log.info("=" * 60)

    t0 = time.time()

    audio_path = extract_audio(
        video_path, output_dir,
        sample_rate=prep_cfg.get("audio_sample_rate", 16000),
    )

    scenes = detect_scenes(
        video_path,
        threshold=prep_cfg.get("scene_threshold", 3.0),
        min_scene_len=prep_cfg.get("min_scene_duration", 1.0),
    )

    keyframes = extract_keyframes(video_path, scenes, output_dir)

    keyframes = deduplicate_by_phash(
        keyframes,
        threshold=prep_cfg.get("phash_threshold", 5),
    )

    keyframes = filter_ppt_frames(keyframes)

    oss_urls = {}
    if not skip_oss:
        all_files = [audio_path] + [kf["filepath"] for kf in keyframes]
        video_name = Path(video_path).stem
        oss_urls = upload_to_oss(config, all_files, prefix=f"{video_name}/")

    manifest_path = generate_manifest(
        video_path, audio_path, keyframes, oss_urls, output_dir,
    )

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("预处理完成! 耗时 %.1f 秒", elapsed)
    log.info("Manifest: %s", manifest_path)
    log.info("关键帧: %d 张 (PPT: %d, 非PPT: %d)",
             len(keyframes),
             sum(1 for kf in keyframes if kf.get("is_ppt")),
             sum(1 for kf in keyframes if not kf.get("is_ppt")))
    log.info("=" * 60)
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频预处理器 — Layer 1")
    parser.add_argument("video", help="视频文件路径")
    parser.add_argument("-c", "--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("-o", "--output", default=None, help="输出目录")
    parser.add_argument("--skip-oss", action="store_true", help="跳过 OSS 上传（本地调试用）")
    args = parser.parse_args()
    run_pipeline(args.video, args.config, args.output, args.skip_oss)
