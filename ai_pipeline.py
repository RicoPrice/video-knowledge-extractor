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

    prompt = f"""你是一位资深技术讲师和知识整理专家。你的任务是将一段技术分享/课程录播的内容，整理成一份**可以直接用来学习的详细笔记**。

## 视频名称
{video_name}

## 音频转写（带时间戳）
{chr(10).join(transcript_lines[:250])}
{"... (更多内容省略)" if len(transcript_lines) > 250 else ""}

## 画面分析（关键帧截图内容）
{chr(10).join(visual_summary)}

## 输出要求

你需要输出一份**教学级别的详细知识笔记**，不是简单的摘要。具体要求：

### 1. 每个知识点必须包含：
- **title**: 知识点标题
- **content**: 详细的教学内容（至少 200-500 字），要求：
  - 像教科书一样解释概念的定义、原理、用途
  - 如果涉及代码/命令，给出完整的代码示例和解释
  - 如果涉及步骤/流程，列出详细的操作步骤
  - 如果涉及对比，用表格或列表说明区别
  - 包含讲师提到的注意事项、最佳实践、常见坑
- **time_start_sec**: 该知识点在视频中的起始秒数（数字）
- **time_end_sec**: 该知识点在视频中的结束秒数（数字）
- **importance**: high/medium/low
- **related_frame_indices**: 相关的关键帧索引号列表（如 [3, 5, 8]），对应画面分析中的帧号
- **key_takeaways**: 该知识点的 2-3 个核心要点（字符串列表）

### 2. 整体结构：
- **summary**: 整体摘要（100-200字），说明这个视频讲了什么主题、适合什么水平的学习者
- **outline**: 视频大纲，按时间顺序列出章节 [{{"title": "...", "time_start_sec": 0, "time_end_sec": 300}}]
- **knowledge_points**: 所有知识点（按时间顺序）

### 3. 内容深度：
- 不要只写"讲师介绍了XXX"这种摘要，要把XXX的具体内容写出来
- 如果讲师演示了代码，把代码和解释都写出来
- 如果讲师画了图/展示了架构，用文字详细描述架构的每个部分

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
                    img_path = f"/output/{video_name}/keyframes/{kf['filename']}"
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

    md = generate_markdown(video_name, knowledge, visual_results, keyframes)
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
            "time_start_sec": 0,
            "time_end_sec": 0,
            "importance": "high",
            "related_frame_indices": [],
            "key_takeaways": [],
        })
    for vr in visual_results[:10]:
        kps.append({
            "title": f"画面内容 ({vr['timestamp']:.1f}s)",
            "content": vr.get("description", "") or vr.get("text_content", ""),
            "time_start_sec": vr.get("timestamp", 0),
            "time_end_sec": vr.get("timestamp", 0),
            "importance": "medium",
            "related_frame_indices": [vr.get("index", 0)],
            "key_takeaways": [],
        })
    stats = manifest.get("stats", {})
    summary = f"视频共检测到 {stats.get('total_scenes', 0)} 个场景，其中 PPT 帧 {stats.get('ppt_frames', 0)} 张。"
    if not kps:
        kps.append({
            "title": "预处理完成",
            "content": summary + " API Key 未配置，无法进行 AI 分析。请在 config.yaml 中填入 DashScope 和 DeepSeek 的 API Key。",
            "time_start_sec": 0,
            "time_end_sec": 0,
            "importance": "high",
            "related_frame_indices": [],
            "key_takeaways": [],
        })
    return {"summary": summary, "knowledge_points": kps, "outline": []}
