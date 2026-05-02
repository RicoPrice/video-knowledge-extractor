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
    先上传音频到 DashScope 文件服务，再提交异步转写任务。
    返回 {"text": "全文", "segments": [{"start": 0.0, "end": 1.5, "text": "..."}]}
    """
    log.info("ASR 转写: %s", audio_path)

    # Step 0: 上传音频文件到 DashScope
    log.info("上传音频到 DashScope...")
    async with httpx.AsyncClient(timeout=300) as client:
        with open(audio_path, "rb") as f:
            resp = await client.post(
                "https://dashscope.aliyuncs.com/api/v1/uploads",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (os.path.basename(audio_path), f, "audio/wav")},
                data={"purpose": "file-extract"},
            )
        resp.raise_for_status()
        upload_data = resp.json()
        file_url = upload_data.get("data", {}).get("uploaded_url", "")
        if not file_url:
            # 备选：尝试从 id 构造
            file_id = upload_data.get("data", {}).get("file_id") or upload_data.get("id", "")
            file_url = f"dashscope://file-{file_id}" if file_id else ""
        if not file_url:
            raise RuntimeError(f"音频上传失败: {upload_data}")
        log.info("音频已上传: %s", file_url[:80])

    # Step 1: 提交转写任务
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
            },
            json={
                "model": "paraformer-v2",
                "input": {"file_urls": [file_url]},
                "parameters": {
                    "language_hints": ["zh", "en"],
                },
            },
        )
        resp.raise_for_status()
        task_data = resp.json()
        task_id = task_data.get("output", {}).get("task_id", "")
        if not task_id:
            raise RuntimeError(f"ASR 提交失败: {task_data}")
        log.info("ASR 任务已提交: %s", task_id)

    # Step 2: 轮询结果
    async with httpx.AsyncClient(timeout=30) as client:
        for _ in range(120):  # 最多等 10 分钟
            await asyncio.sleep(5)
            resp = await client.get(
                f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            result = resp.json()
            status = result.get("output", {}).get("task_status", "")
            if status == "SUCCEEDED":
                break
            elif status == "FAILED":
                raise RuntimeError(f"ASR 失败: {result}")
            log.info("ASR 进行中... (%s)", status)
        else:
            raise TimeoutError("ASR 超时")

    # Step 3: 解析结果
    transcription = result.get("output", {}).get("results", [])
    segments = []
    full_text_parts = []
    for item in transcription:
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


# ── Visual Analysis: Qwen-VL ─────────────────────

async def analyze_keyframes(keyframes: list[dict], api_key: str) -> list[dict]:
    """
    调用 Qwen-VL-Max 分析关键帧图片。
    最多分析 MAX_FRAMES 张（均匀采样），5 路并发。
    """
    MAX_FRAMES = 30
    CONCURRENCY = 5

    # 均匀采样
    if len(keyframes) > MAX_FRAMES:
        step = len(keyframes) / MAX_FRAMES
        sampled = [keyframes[int(i * step)] for i in range(MAX_FRAMES)]
        log.info("视觉分析: 从 %d 张中采样 %d 张", len(keyframes), len(sampled))
    else:
        sampled = keyframes
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
    调用 DeepSeek 融合音频+视觉信息，提取结构化知识点。
    返回 {"knowledge_points": [...], "summary": "..."}
    """
    log.info("知识点提取: DeepSeek")

    # 构建视觉内容摘要
    visual_summary = []
    for vr in visual_results:
        line = f"[{vr['timestamp']:.1f}s] {vr.get('visual_type','?')}"
        if vr.get("text_content"):
            line += f" | 文字: {vr['text_content'][:200]}"
        if vr.get("description"):
            line += f" | {vr['description'][:100]}"
        visual_summary.append(line)

    # 构建转写文本（带时间戳）
    transcript_lines = []
    for seg in transcript.get("segments", [])[:200]:  # 限制长度
        transcript_lines.append(f"[{seg['start']:.1f}s-{seg['end']:.1f}s] {seg['text']}")

    prompt = f"""你是一个专业的知识点提取专家。请根据以下视频录播的音频转写和画面分析，提取结构化的知识点。

## 视频名称
{video_name}

## 音频转写（带时间戳）
{chr(10).join(transcript_lines[:150])}
{"... (更多内容省略)" if len(transcript_lines) > 150 else ""}

## 画面分析（关键帧）
{chr(10).join(visual_summary)}

## 要求
请提取出所有重要知识点，每个知识点包含：
1. title: 简洁的标题
2. content: 详细说明（2-5句话）
3. time_range: 大致时间范围
4. importance: high/medium/low
5. related_visual: 相关的画面内容（如有）

同时生成一个整体摘要（summary），200字以内。

请用以下 JSON 格式输出：
```json
{{
  "summary": "整体摘要",
  "knowledge_points": [
    {{
      "title": "...",
      "content": "...",
      "time_range": "0:00-2:30",
      "importance": "high",
      "related_visual": "..."
    }}
  ]
}}
```
只返回 JSON，不要其他内容。"""

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{deepseek_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {deepseek_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": 0.3,
            },
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"]

    try:
        result = json.loads(answer.strip().strip("```json").strip("```").strip())
    except json.JSONDecodeError:
        result = {"summary": answer[:500], "knowledge_points": []}

    log.info("知识点提取完成: %d 个知识点", len(result.get("knowledge_points", [])))
    return result


# ── Multi-format Output ───────────────────────────

def generate_markdown(video_name: str, knowledge: dict) -> str:
    lines = [f"# {video_name} — 知识点报告\n"]
    if knowledge.get("summary"):
        lines.append(f"## 摘要\n\n{knowledge['summary']}\n")
    lines.append("## 知识点\n")
    for i, kp in enumerate(knowledge.get("knowledge_points", []), 1):
        imp = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(kp.get("importance", ""), "⚪")
        lines.append(f"### {i}. {kp['title']} {imp}\n")
        lines.append(f"**时间**: {kp.get('time_range', '未知')}\n")
        lines.append(f"{kp['content']}\n")
        if kp.get("related_visual"):
            lines.append(f"> 📷 画面: {kp['related_visual']}\n")
        lines.append("")
    return "\n".join(lines)


def generate_json_report(video_name: str, knowledge: dict) -> str:
    return json.dumps({"video_name": video_name, **knowledge}, ensure_ascii=False, indent=2)


def generate_srt(knowledge: dict) -> str:
    lines = []
    for i, kp in enumerate(knowledge.get("knowledge_points", []), 1):
        tr = kp.get("time_range", "0:00-0:00")
        parts = tr.split("-")
        start = _time_to_srt(parts[0].strip()) if parts else "00:00:00,000"
        end = _time_to_srt(parts[1].strip()) if len(parts) > 1 else "00:00:00,000"
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(f"{kp['title']}: {kp['content'][:100]}")
        lines.append("")
    return "\n".join(lines)


def _time_to_srt(t: str) -> str:
    """Convert '1:23' or '1:23:45' to SRT format '01:23:00,000'"""
    parts = t.split(":")
    try:
        if len(parts) == 2:
            m, s = int(parts[0]), int(parts[1])
            return f"00:{m:02d}:{s:02d},000"
        elif len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{h:02d}:{m:02d}:{s:02d},000"
    except ValueError:
        pass
    return "00:00:00,000"


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

    # Step 1: ASR
    transcript = {"text": "", "segments": []}
    if ds_key and ds_key != "your-dashscope-api-key" and audio_path and os.path.exists(audio_path):
        if progress_cb:
            await progress_cb("ASR 语音转写", 50)
        try:
            transcript = await transcribe_audio(audio_path, ds_key)
        except Exception as e:
            log.error("ASR 失败: %s", e)
    else:
        log.warning("跳过 ASR: DashScope API Key 未配置或音频不存在")

    # Step 2: Visual Analysis
    visual_results = []
    if ds_key and ds_key != "your-dashscope-api-key" and keyframes:
        if progress_cb:
            await progress_cb("视觉分析 (Qwen-VL)", 65)
        try:
            visual_results = await analyze_keyframes(keyframes, ds_key)
        except Exception as e:
            log.error("视觉分析失败: %s", e)
    else:
        log.warning("跳过视觉分析: DashScope API Key 未配置或无关键帧")

    # Step 3: Knowledge Extraction
    knowledge = {"summary": "", "knowledge_points": []}
    if dk_key and dk_key != "your-deepseek-api-key":
        if progress_cb:
            await progress_cb("知识点提取 (DeepSeek)", 80)
        try:
            knowledge = await extract_knowledge(transcript, visual_results, video_name, dk_key, dk_url)
        except Exception as e:
            log.error("知识点提取失败: %s", e)
    else:
        log.warning("跳过知识点提取: DeepSeek API Key 未配置")

    # Step 4: 如果没有 AI 结果，生成基础报告
    if not knowledge.get("knowledge_points"):
        knowledge = _fallback_report(manifest, transcript, visual_results)

    # Step 5: Multi-format output
    if progress_cb:
        await progress_cb("生成报告", 95)

    md = generate_markdown(video_name, knowledge)
    rj = generate_json_report(video_name, knowledge)
    srt = generate_srt(knowledge)

    return {"markdown": md, "json": rj, "srt": srt}


def _fallback_report(manifest: dict, transcript: dict, visual_results: list) -> dict:
    """API Key 未配置时的降级报告"""
    kps = []
    if transcript.get("segments"):
        kps.append({
            "title": "音频转写结果",
            "content": transcript["text"][:500] + ("..." if len(transcript["text"]) > 500 else ""),
            "time_range": "全程",
            "importance": "high",
            "related_visual": "",
        })
    for vr in visual_results[:10]:
        kps.append({
            "title": f"画面内容 ({vr['timestamp']:.1f}s)",
            "content": vr.get("description", "") or vr.get("text_content", ""),
            "time_range": f"{vr['timestamp']:.0f}s",
            "importance": "medium",
            "related_visual": vr.get("visual_type", ""),
        })
    stats = manifest.get("stats", {})
    summary = f"视频共检测到 {stats.get('total_scenes', 0)} 个场景，其中 PPT 帧 {stats.get('ppt_frames', 0)} 张。"
    if not kps:
        kps.append({
            "title": "预处理完成",
            "content": summary + " API Key 未配置，无法进行 AI 分析。请在 config.yaml 中填入 DashScope 和 DeepSeek 的 API Key。",
            "time_range": "全程",
            "importance": "high",
            "related_visual": "",
        })
    return {"summary": summary, "knowledge_points": kps}
