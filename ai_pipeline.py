"""
AI Pipeline — 直接调用云端 API，替代 Dify Workflow
流程：ASR 转写 → 视觉分析 → 知识点提取 → 多格式输出
"""

import asyncio
import base64
import json
import logging
import os
from pathlib import Path

import httpx
import yaml

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────

def _load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# ── ASR: DashScope Paraformer ─────────────────────

async def transcribe_audio(audio_path: str, api_key: str) -> dict:
    """
    调用阿里云百炼 Paraformer-v2 进行语音转写。
    使用 DashScope 官方 SDK，自动处理文件上传。
    大 WAV 文件先压缩成 MP3 再上传。
    返回 {"text": "全文", "segments": [{"start": 0.0, "end": 1.5, "text": "..."}]}
    """
    log.info("ASR 转写: %s", audio_path)

    # Step 0: 如果是 WAV 且 > 50MB，先转 MP3
    upload_path = audio_path
    tmp_mp3 = None
    file_size = os.path.getsize(audio_path)
    if file_size > 50 * 1024 * 1024 and audio_path.lower().endswith(".wav"):
        tmp_mp3 = audio_path.rsplit(".", 1)[0] + "_asr.mp3"
        log.info("音频 %.0fMB 太大，转换为 MP3...", file_size / 1024 / 1024)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", audio_path,
            "-ac", "1", "-ar", "16000", "-b:a", "64k",
            tmp_mp3,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        if proc.returncode == 0 and os.path.exists(tmp_mp3):
            upload_path = tmp_mp3
            log.info("MP3 转换完成: %.1fMB", os.path.getsize(tmp_mp3) / 1024 / 1024)
        else:
            log.warning("MP3 转换失败，使用原始 WAV")
            tmp_mp3 = None

    try:
        import dashscope
        from dashscope.audio.asr import Transcription
        dashscope.api_key = api_key

        # Step 1: 用 SDK 上传文件到 DashScope 临时 OSS
        log.info("上传音频到 DashScope OSS: %s (%.1fMB)", upload_path, os.path.getsize(upload_path) / 1024 / 1024)
        oss_url = await asyncio.to_thread(dashscope.OssUtils.upload, upload_path)
        if not oss_url:
            raise RuntimeError(f"音频上传到 OSS 失败，返回空 URL")
        log.info("音频已上传: %s", oss_url[:80])

        # Step 2: 提交异步转写任务
        task_response = await asyncio.to_thread(
            Transcription.async_call,
            model="paraformer-v2",
            file_urls=[oss_url],
            language_hints=["zh", "en"],
        )

        task_id = ""
        if hasattr(task_response, "output") and task_response.output:
            task_id = task_response.output.get("task_id", "")
        if not task_id:
            raise RuntimeError(f"ASR 提交失败: {task_response}")
        log.info("ASR 任务已提交: %s", task_id)

        # Step 3: 用 SDK 的 wait 方法等待结果
        result = await asyncio.to_thread(Transcription.wait, task=task_id)
        if not hasattr(result, "output") or not result.output:
            raise RuntimeError(f"ASR 返回无效结果: {result}")

        task_status = result.output.get("task_status", "")
        if task_status != "SUCCEEDED":
            raise RuntimeError(f"ASR 失败: {json.dumps(result.output, ensure_ascii=False)}")

        # Step 4: 解析结果
        transcription_results = result.output.get("results", [])
        segments = []
        full_text_parts = []
        for item in transcription_results:
            url = item.get("transcription_url", "")
            if url:
                async with httpx.AsyncClient(timeout=30) as client:
                    tr_resp = await client.get(url)
                    tr_data = tr_resp.json()
                    for trans in tr_data.get("transcripts", []):
                        text = trans.get("text", "")
                        full_text_parts.append(text)
                        for sent in trans.get("sentences", []):
                            segments.append({
                                "start": sent.get("begin_time", 0) / 1000.0,
                                "end": sent.get("end_time", 0) / 1000.0,
                                "text": sent.get("text", ""),
                            })

        full_text = "\n".join(full_text_parts)
        log.info("ASR 完成: %d 段, %d 字", len(segments), len(full_text))
        return {"text": full_text, "segments": segments}

    finally:
        # 清理临时 MP3
        if tmp_mp3 and os.path.exists(tmp_mp3):
            os.remove(tmp_mp3)
            log.info("已清理临时 MP3: %s", tmp_mp3)


# ── Visual Analysis: Qwen-VL ─────────────────────

async def analyze_keyframes(keyframes: list[dict], api_key: str) -> list[dict]:
    """
    调用 Qwen-VL-Max 分析关键帧图片。
    先用 pHash 去重，再均匀采样最多 MAX_FRAMES 张，5 路并发。
    """
    MAX_FRAMES = 30
    CONCURRENCY = 5
    PHASH_THRESHOLD = 8  # pHash 汉明距离阈值，越小越严格

    # Step 1: pHash 去重 — 过滤掉相似帧
    try:
        import imagehash
        from PIL import Image
        deduped = []
        seen_hashes = []
        for kf in keyframes:
            fp = kf.get("filepath", "")
            if not fp or not os.path.exists(fp):
                continue
            try:
                h = imagehash.phash(Image.open(fp))
                is_dup = False
                for sh in seen_hashes:
                    if abs(h - sh) < PHASH_THRESHOLD:
                        is_dup = True
                        break
                if not is_dup:
                    deduped.append(kf)
                    seen_hashes.append(h)
            except Exception:
                deduped.append(kf)  # 无法计算 hash 的帧保留
        log.info("pHash 去重: %d → %d 张", len(keyframes), len(deduped))
    except ImportError:
        log.warning("imagehash 未安装，跳过 pHash 去重")
        deduped = [kf for kf in keyframes if kf.get("filepath") and os.path.exists(kf.get("filepath", ""))]

    # Step 2: 均匀采样
    if len(deduped) > MAX_FRAMES:
        step = len(deduped) / MAX_FRAMES
        sampled = [deduped[int(i * step)] for i in range(MAX_FRAMES)]
        log.info("视觉分析: 从 %d 张中采样 %d 张", len(deduped), len(sampled))
    else:
        sampled = deduped
        log.info("视觉分析: %d 张关键帧", len(sampled))

    sem = asyncio.Semaphore(CONCURRENCY)
    results = []

    async def analyze_one(kf: dict) -> dict | None:
        filepath = kf.get("filepath", "")
        if not filepath or not os.path.exists(filepath):
            return None

        with open(filepath, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        async with sem:
            async with httpx.AsyncClient(timeout=60) as client:
                try:
                    resp = await client.post(
                        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "qwen-vl-max",
                            "messages": [{
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                                    {"type": "text", "text": (
                                        "分析这张视频截图，用 JSON 格式回答：\n"
                                        '{"is_ppt": true/false, "visual_type": "ppt|code|chart|camera|other", '
                                        '"text_content": "图片中的文字内容", '
                                        '"description": "一句话描述画面内容"}\n'
                                        "只返回 JSON，不要其他内容。"
                                    )},
                                ],
                            }],
                            "max_tokens": 500,
                        },
                    )
                    resp.raise_for_status()
                    answer = resp.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    log.warning("  帧 %d 分析失败: %s", kf.get("index", 0), e)
                    return None

        try:
            parsed = json.loads(answer.strip().strip("```json").strip("```").strip())
        except json.JSONDecodeError:
            parsed = {"is_ppt": False, "visual_type": "other", "text_content": "", "description": answer[:200]}

        log.info("  帧 %d (%.1fs): %s", kf.get("index", 0), kf.get("timestamp", 0), parsed.get("visual_type", "?"))
        return {"index": kf.get("index", 0), "timestamp": kf.get("timestamp", 0), **parsed}

    tasks = [analyze_one(kf) for kf in sampled]
    raw = await asyncio.gather(*tasks)
    results = [r for r in raw if r is not None]

    log.info("视觉分析完成: %d 张", len(results))
    return results


# ── Knowledge Extraction: DeepSeek ────────────────

async def extract_knowledge(
    transcript: dict, visual_results: list[dict],
    video_name: str, deepseek_key: str, deepseek_url: str = "https://api.deepseek.com",
) -> dict:
    """
    调用 DeepSeek 融合音频+视觉信息，提取教学级知识点。
    返回 {"knowledge_points": [...], "summary": "...", "outline": [...]}
    """
    log.info("知识点提取: DeepSeek")

    # 构建视觉内容摘要（带帧索引，用于后续关联截图）
    visual_summary = []
    for vr in visual_results:
        line = f"[帧{vr.get('index',0)} {vr['timestamp']:.1f}s] {vr.get('visual_type','?')}"
        if vr.get("text_content"):
            line += f" | 文字: {vr['text_content'][:300]}"
        if vr.get("description"):
            line += f" | {vr['description'][:150]}"
        visual_summary.append(line)

    # 构建转写文本（带时间戳）
    transcript_lines = []
    for seg in transcript.get("segments", [])[:300]:
        transcript_lines.append(f"[{seg['start']:.1f}s-{seg['end']:.1f}s] {seg['text']}")

    has_transcript = len(transcript_lines) > 0
    has_visual = len(visual_summary) > 0

    prompt = f"""你是一位资深知识整理专家。你的任务是根据下面提供的**实际素材**，整理成一份可以直接用来学习的详细笔记。

## 重要约束（必须遵守）
1. **只能使用下面提供的素材内容**，严禁添加素材中没有的信息
2. **时间戳必须来自素材中的实际时间**，不要均匀切分或猜测
3. 如果音频转写为空，就只根据画面分析来整理，并在摘要中说明"本报告基于画面分析，缺少语音内容"
4. 如果某个知识点的时间范围不确定，用最近的关键帧时间戳
5. 宁可少写也不要编造内容

## 视频名称
{video_name}

## 音频转写（带时间戳）
{"（无音频转写数据）" if not has_transcript else chr(10).join(transcript_lines[:250])}
{"... (更多内容省略)" if len(transcript_lines) > 250 else ""}

## 画面分析（关键帧截图内容）
{"（无画面分析数据）" if not has_visual else chr(10).join(visual_summary)}

## 输出要求

输出一份**教学级别的详细知识笔记**。具体要求：

### 1. 每个知识点必须包含：
- **title**: 知识点标题
- **content**: 详细的教学内容（200-500 字），要求：
  - 基于素材中讲师实际说的话和展示的画面来组织
  - 如果涉及代码/命令，给出素材中出现的代码示例
  - 如果涉及步骤/流程，列出讲师实际演示的步骤
  - 包含讲师提到的注意事项、最佳实践
- **time_start_sec**: 该知识点在视频中的起始秒数（必须来自素材中的实际时间戳）
- **time_end_sec**: 该知识点在视频中的结束秒数（必须来自素材中的实际时间戳）
- **importance**: high/medium/low
- **related_frame_indices**: 相关的关键帧索引号列表（如 [3, 5, 8]），必须是画面分析中实际存在的帧号
- **key_takeaways**: 该知识点的 2-3 个核心要点（字符串列表）

### 2. 整体结构：
- **summary**: 整体摘要（100-200字）
- **outline**: 视频大纲，按时间顺序列出章节 [{{"title": "...", "time_start_sec": 0, "time_end_sec": 300}}]
- **knowledge_points**: 所有知识点（按时间顺序）

请用以下 JSON 格式输出：
```json
{{
  "summary": "...",
  "outline": [
    {{"title": "章节名", "time_start_sec": 0, "time_end_sec": 300}}
  ],
  "knowledge_points": [
    {{
      "title": "...",
      "content": "详细的教学内容...",
      "time_start_sec": 0,
      "time_end_sec": 150,
      "importance": "high",
      "related_frame_indices": [0, 3],
      "key_takeaways": ["要点1", "要点2"]
    }}
  ]
}}
```
只返回 JSON，不要其他内容。"""

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{deepseek_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {deepseek_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16384,
                "temperature": 0.3,
            },
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"]

    try:
        result = json.loads(answer.strip().strip("```json").strip("```").strip())
    except json.JSONDecodeError:
        result = {"summary": answer[:500], "knowledge_points": [], "outline": []}

    log.info("知识点提取完成: %d 个知识点", len(result.get("knowledge_points", [])))
    return result


# ── Multi-format Output ───────────────────────────

def _sec_to_ts(sec) -> str:
    """秒数转 H:MM:SS 格式"""
    try:
        s = int(float(sec))
    except (ValueError, TypeError):
        return "0:00"
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _sec_to_srt_ts(sec) -> str:
    """秒数转 SRT 时间戳 HH:MM:SS,000"""
    try:
        s = int(float(sec))
    except (ValueError, TypeError):
        return "00:00:00,000"
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},000"


def generate_markdown(video_name: str, knowledge: dict, visual_results: list[dict] = None,
                      keyframes: list[dict] = None, video_web_path: str = "") -> str:
    """生成带截图和视频时间戳的 Markdown 报告"""
    visual_results = visual_results or []
    keyframes = keyframes or []

    # 建立帧索引 → 文件路径的映射
    frame_map = {}
    for kf in keyframes:
        frame_map[kf.get("index", -1)] = kf

    lines = [f"# {video_name} — 知识笔记\n"]

    # 摘要
    if knowledge.get("summary"):
        lines.append(f"## 📋 摘要\n\n{knowledge['summary']}\n")

    # 大纲（带时间戳）
    outline = knowledge.get("outline", [])
    if outline:
        lines.append("## 📑 视频大纲\n")
        lines.append("| 章节 | 时间 |")
        lines.append("|------|------|")
        for ch in outline:
            ts = _sec_to_ts(ch.get("time_start_sec", 0))
            te = _sec_to_ts(ch.get("time_end_sec", 0))
            lines.append(f"| {ch['title']} | {ts} - {te} |")
        lines.append("")

    # 知识点
    lines.append("## 📚 知识点详解\n")
    for i, kp in enumerate(knowledge.get("knowledge_points", []), 1):
        imp = {"high": "🔴 重要", "medium": "🟡 一般", "low": "🟢 了解"}.get(kp.get("importance", ""), "")

        ts_start = _sec_to_ts(kp.get("time_start_sec", 0))
        ts_end = _sec_to_ts(kp.get("time_end_sec", 0))

        lines.append(f"### {i}. {kp['title']}\n")
        lines.append(f"⏱️ **{ts_start} - {ts_end}** &nbsp; {imp}\n")

        # 嵌入关联的关键帧截图
        related_frames = kp.get("related_frame_indices", [])
        if related_frames and keyframes:
            for fi in related_frames[:3]:  # 最多 3 张
                kf = frame_map.get(fi)
                if kf and kf.get("filename"):
                    from urllib.parse import quote
                    img_path = f"/output/{quote(video_name)}/keyframes/{quote(kf['filename'])}"
                    lines.append(f"![帧{fi} ({_sec_to_ts(kf.get('timestamp', 0))})]({img_path})\n")

        # 正文内容
        lines.append(f"{kp['content']}\n")

        # 核心要点
        takeaways = kp.get("key_takeaways", [])
        if takeaways:
            lines.append("**💡 核心要点：**\n")
            for t in takeaways:
                lines.append(f"- {t}")
            lines.append("")

        lines.append("---\n")

    return "\n".join(lines)


def generate_json_report(video_name: str, knowledge: dict) -> str:
    return json.dumps({"video_name": video_name, **knowledge}, ensure_ascii=False, indent=2)


def generate_srt(knowledge: dict) -> str:
    lines = []
    for i, kp in enumerate(knowledge.get("knowledge_points", []), 1):
        start = _sec_to_srt_ts(kp.get("time_start_sec", 0))
        end = _sec_to_srt_ts(kp.get("time_end_sec", 0))
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(kp["title"])
        takeaways = kp.get("key_takeaways", [])
        if takeaways:
            lines.append(" | ".join(takeaways[:3]))
        lines.append("")
    return "\n".join(lines)


# ── Main Pipeline ─────────────────────────────────

async def run_ai_pipeline(manifest_path: str, progress_cb=None) -> dict:
    """
    完整 AI 分析流水线。
    progress_cb(stage, progress_pct) 用于更新进度。
    返回 {"markdown", "json", "srt"}
    """
    config = _load_config()
    ds_key = config.get("dashscope", {}).get("api_key", "")
    dk_key = config.get("deepseek", {}).get("api_key", "")
    dk_url = config.get("deepseek", {}).get("base_url", "https://api.deepseek.com")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    video_name = manifest.get("video_name", "unknown")
    audio_path = manifest.get("audio", {}).get("local_path", "")
    keyframes = manifest.get("keyframes", [])

    # 补全 filepath（manifest 里可能只有 filename）
    manifest_dir = str(Path(manifest_path).parent)
    for kf in keyframes:
        if not kf.get("filepath"):
            kf["filepath"] = os.path.join(manifest_dir, "keyframes", kf.get("filename", ""))

    errors = []  # 收集每一步的错误

    # Step 1: ASR — 必须成功，否则报错
    transcript = {"text": "", "segments": []}
    if not ds_key or ds_key == "your-dashscope-api-key":
        errors.append("❌ ASR 失败: DashScope API Key 未配置")
    elif not audio_path or not os.path.exists(audio_path):
        errors.append(f"❌ ASR 失败: 音频文件不存在 ({audio_path})")
    else:
        if progress_cb:
            await progress_cb("ASR 语音转写", 50)
        try:
            transcript = await transcribe_audio(audio_path, ds_key)
            if not transcript.get("text", "").strip():
                errors.append("⚠️ ASR 返回空文本，可能是音频无语音内容或转写服务异常")
            else:
                log.info("ASR 成功: %d 字, %d 段", len(transcript["text"]), len(transcript.get("segments", [])))
        except Exception as e:
            err_detail = str(e) or repr(e)
            if hasattr(e, 'response'):
                try:
                    err_detail += f" | HTTP {e.response.status_code}: {e.response.text[:500]}"
                except Exception:
                    pass
            errors.append(f"❌ ASR 失败: {err_detail}")

    # Step 2: Visual Analysis — 失败记录但不阻断
    visual_results = []
    if not ds_key or ds_key == "your-dashscope-api-key":
        errors.append("❌ 视觉分析失败: DashScope API Key 未配置")
    elif not keyframes:
        errors.append("⚠️ 视觉分析跳过: 无关键帧")
    else:
        if progress_cb:
            await progress_cb("视觉分析 (Qwen-VL)", 65)
        try:
            visual_results = await analyze_keyframes(keyframes, ds_key)
            log.info("视觉分析成功: %d 张", len(visual_results))
        except Exception as e:
            err_detail = str(e) or repr(e)
            if hasattr(e, 'response'):
                try:
                    err_detail += f" | HTTP {e.response.status_code}: {e.response.text[:500]}"
                except Exception:
                    pass
            errors.append(f"❌ 视觉分析失败: {err_detail}")

    # Step 3: Knowledge Extraction — 必须有 ASR 文本才能提取
    knowledge = {"summary": "", "knowledge_points": [], "outline": []}
    if not dk_key or dk_key == "your-deepseek-api-key":
        errors.append("❌ 知识点提取失败: DeepSeek API Key 未配置")
    elif not transcript.get("text", "").strip():
        errors.append("❌ 知识点提取跳过: 没有 ASR 转写文本，无法提取知识点（请先修复 ASR 问题）")
    else:
        if progress_cb:
            await progress_cb("知识点提取 (DeepSeek)", 80)
        try:
            knowledge = await extract_knowledge(transcript, visual_results, video_name, dk_key, dk_url)
            if not knowledge.get("knowledge_points"):
                errors.append("⚠️ DeepSeek 未返回任何知识点")
            else:
                log.info("知识点提取成功: %d 个", len(knowledge["knowledge_points"]))
        except Exception as e:
            err_detail = str(e) or repr(e)
            if hasattr(e, 'response'):
                try:
                    err_detail += f" | HTTP {e.response.status_code}: {e.response.text[:500]}"
                except Exception:
                    pass
            errors.append(f"❌ 知识点提取失败: {err_detail}")

    # Step 4: 生成报告 — 如果有错误，在报告开头显示
    if progress_cb:
        await progress_cb("生成报告", 95)

    if errors:
        error_block = "## ⚠️ 处理过程中遇到以下问题\n\n"
        for err in errors:
            error_block += f"- {err}\n"
        error_block += "\n请检查以上问题后重新上传视频。\n\n---\n\n"
    else:
        error_block = ""

    md = error_block + generate_markdown(video_name, knowledge, visual_results, keyframes)
    rj = generate_json_report(video_name, {**knowledge, "errors": errors})
    srt = generate_srt(knowledge)

    return {"markdown": md, "json": rj, "srt": srt}
