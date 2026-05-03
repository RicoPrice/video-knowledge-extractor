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

HOTWORDS_PATH = os.path.join(os.path.dirname(__file__), "hotwords.yaml")


def _load_hotwords_config() -> dict:
    """加载 hotwords.yaml，不存在则返回空 dict。"""
    if os.path.exists(HOTWORDS_PATH):
        with open(HOTWORDS_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_vocabulary_id(vocab_id: str):
    """把 vocabulary_id 写回 hotwords.yaml。"""
    cfg = _load_hotwords_config()
    cfg["vocabulary_id"] = vocab_id
    with open(HOTWORDS_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)


async def _ensure_vocabulary(api_key: str) -> str:
    """
    确保热词表存在，返回 vocabulary_id。
    - 如果 hotwords.yaml 不存在或 hotwords 为空 → 返回 ""
    - 如果 vocabulary_id 已缓存 → 直接返回
    - 否则调 DashScope API 创建热词表 → 缓存并返回
    """
    hw_cfg = _load_hotwords_config()
    hotwords = hw_cfg.get("hotwords", {})
    if not hotwords:
        return ""

    vocab_id = hw_cfg.get("vocabulary_id", "")
    if vocab_id:
        log.info("使用已缓存的热词表: %s", vocab_id)
        return vocab_id

    # 创建热词表
    log.info("创建热词表: %d 个热词", len(hotwords))
    vocabulary = []
    for word, weight in hotwords.items():
        lang = "zh" if any('\u4e00' <= c <= '\u9fff' for c in str(word)) else "en"
        vocabulary.append({"text": str(word), "weight": int(weight), "lang": lang})

    import httpx
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/customization",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "speech-biasing",
                "input": {
                    "action": "create_vocabulary",
                    "target_model": "paraformer-v2",
                    "prefix": "vke",
                    "vocabulary": vocabulary,
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()

    vocab_id = data.get("output", {}).get("vocabulary_id", "")
    if not vocab_id:
        log.warning("热词表创建失败: %s", data)
        return ""

    log.info("热词表创建成功: %s", vocab_id)
    _save_vocabulary_id(vocab_id)
    return vocab_id

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

        # Step 1: 上传 MP3 到阿里云 OSS，生成签名 URL
        import oss2
        oss_cfg = _load_config().get("oss", {})
        oss_ak = oss_cfg.get("access_key_id", "")
        oss_sk = oss_cfg.get("access_key_secret", "")
        oss_endpoint = oss_cfg.get("endpoint", "")
        oss_bucket_name = oss_cfg.get("bucket", "")
        oss_prefix = oss_cfg.get("prefix", "video-knowledge/")

        if not all([oss_ak, oss_sk, oss_endpoint, oss_bucket_name]):
            raise RuntimeError("OSS 未配置，请在 config.yaml 中填写 oss.access_key_id/secret/endpoint/bucket")

        auth = oss2.Auth(oss_ak, oss_sk)
        bucket = oss2.Bucket(auth, oss_endpoint, oss_bucket_name)

        oss_key = f"{oss_prefix}{os.path.basename(upload_path)}"
        file_size_mb = os.path.getsize(upload_path) / 1024 / 1024
        log.info("上传音频到 OSS: %s (%.1fMB) -> %s", upload_path, file_size_mb, oss_key)

        await asyncio.to_thread(bucket.put_object_from_file, oss_key, upload_path)

        # 生成 1 小时有效的签名 URL
        signed_url = bucket.sign_url('GET', oss_key, 3600)
        log.info("OSS 签名 URL: %s", signed_url[:80])

        # Step 2: 获取热词表（如果配置了）
        vocab_id = await _ensure_vocabulary(api_key)
        if vocab_id:
            log.info("使用热词表: %s", vocab_id)

        # Step 3: 提交异步转写任务
        asr_kwargs = dict(
            model="paraformer-v2",
            file_urls=[signed_url],
            language_hints=["zh", "en"],
        )
        if vocab_id:
            asr_kwargs["vocabulary_id"] = vocab_id

        task_response = await asyncio.to_thread(
            Transcription.async_call,
            **asr_kwargs,
        )

        task_id = ""
        if hasattr(task_response, "output") and task_response.output:
            task_id = task_response.output.get("task_id", "")
        if not task_id:
            raise RuntimeError(f"ASR 提交失败: {task_response}")
        log.info("ASR 任务已提交: %s", task_id)

        # Step 3: 等待结果
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

        # 清理 OSS 临时文件
        try:
            await asyncio.to_thread(bucket.delete_object, oss_key)
            log.info("已清理 OSS 临时文件: %s", oss_key)
        except Exception:
            pass

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
    两轮采样策略：
      1. pHash 去重 → 均匀采样 60 张 → 快速分类（只问 visual_type）
      2. 按优先级选 30 张有信息量的帧 → 详细分析
    优先级：chart > ppt > code > other > camera
    """
    CLASSIFY_FRAMES = 60   # 第一轮快速分类的帧数
    DETAIL_FRAMES = 30     # 第二轮详细分析的帧数
    CONCURRENCY = 5
    PHASH_THRESHOLD = 8

    # ── Step 1: pHash 去重 ──
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
                deduped.append(kf)
        log.info("pHash 去重: %d → %d 张", len(keyframes), len(deduped))
    except ImportError:
        log.warning("imagehash 未安装，跳过 pHash 去重")
        deduped = [kf for kf in keyframes if kf.get("filepath") and os.path.exists(kf.get("filepath", ""))]

    # ── Step 2: 均匀采样候选帧 ──
    if len(deduped) > CLASSIFY_FRAMES:
        step = len(deduped) / CLASSIFY_FRAMES
        candidates = [deduped[int(i * step)] for i in range(CLASSIFY_FRAMES)]
    else:
        candidates = deduped
    log.info("第一轮候选: %d 张", len(candidates))

    # ── Step 3: 快速分类（只问 visual_type，省 token）──
    sem = asyncio.Semaphore(CONCURRENCY)

    async def classify_one(kf: dict) -> dict | None:
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
                                        "这张图片属于哪种类型？只回答一个词：\n"
                                        "ppt / chart / code / camera / transition / other\n"
                                        "（ppt=幻灯片/演示文稿, chart=图表/K线图/数据图, "
                                        "code=代码/终端, camera=人物出镜/摄像头, "
                                        "transition=软件切换画面/OBS过渡/部分遮挡, other=其他）\n"
                                        "注意：transition 仅指能看到操作系统桌面/任务栏、OBS控制面板、"
                                        "或同一画面递归缩小重复（套娃）的情况。股票软件的多面板界面属于 chart，不是 transition。"
                                    )},
                                ],
                            }],
                            "max_tokens": 20,
                        },
                    )
                    resp.raise_for_status()
                    vtype = resp.json()["choices"][0]["message"]["content"].strip().lower()
                    # 规范化
                    for valid in ("ppt", "chart", "code", "camera", "transition"):
                        if valid in vtype:
                            vtype = valid
                            break
                    else:
                        vtype = "other"
                except Exception as e:
                    log.warning("  帧 %d 分类失败: %s", kf.get("index", 0), e)
                    vtype = "other"
        return {**kf, "_visual_type": vtype}

    classify_tasks = [classify_one(kf) for kf in candidates]
    classified = [r for r in await asyncio.gather(*classify_tasks) if r is not None]
    log.info("快速分类完成: %d 张 (%s)",
             len(classified),
             ", ".join(f"{t}={sum(1 for c in classified if c['_visual_type']==t)}"
                       for t in ("chart", "ppt", "code", "camera", "other")))

    # ── Step 4: 按优先级选帧，保证时间分布均匀 ──
    PRIORITY = {"chart": 0, "ppt": 1, "code": 2, "other": 3, "camera": 99, "transition": 99}

    # 按时间排序
    classified.sort(key=lambda x: x.get("timestamp", 0))

    # 分成时间桶（每 10 分钟一桶），每桶内按优先级排序
    bucket_seconds = 600
    buckets: dict[int, list] = {}
    for c in classified:
        bucket_id = int(c.get("timestamp", 0) // bucket_seconds)
        buckets.setdefault(bucket_id, []).append(c)

    selected = []
    # 每桶按优先级选，camera 排最后
    for bid in sorted(buckets.keys()):
        bucket = sorted(buckets[bid], key=lambda x: PRIORITY.get(x["_visual_type"], 3))
        # 每桶最多选 ceil(DETAIL_FRAMES / len(buckets)) 张
        per_bucket = max(1, -(-DETAIL_FRAMES // max(len(buckets), 1)))
        selected.extend(bucket[:per_bucket])

    # 如果选多了，按优先级裁剪；如果选少了，从剩余帧补充
    if len(selected) > DETAIL_FRAMES:
        selected.sort(key=lambda x: (PRIORITY.get(x["_visual_type"], 3), x.get("timestamp", 0)))
        selected = selected[:DETAIL_FRAMES]
    elif len(selected) < DETAIL_FRAMES:
        selected_set = {c.get("index") for c in selected}
        remaining = [c for c in classified if c.get("index") not in selected_set]
        remaining.sort(key=lambda x: PRIORITY.get(x["_visual_type"], 3))
        for r in remaining:
            if len(selected) >= DETAIL_FRAMES:
                break
            selected.append(r)

    # 按时间排序
    selected.sort(key=lambda x: x.get("timestamp", 0))
    type_counts = {}
    for s in selected:
        t = s["_visual_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    log.info("第二轮选帧: %d 张 (%s)", len(selected),
             ", ".join(f"{t}={c}" for t, c in sorted(type_counts.items())))

    # ── Step 5: 对选中的帧做详细分析 ──
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
                                        '{"is_ppt": true/false, "visual_type": "ppt|code|chart|camera|transition|other", '
                                        '"text_content": "图片中的文字内容", '
                                        '"description": "一句话描述画面内容"}\n'
                                        "visual_type 说明：\n"
                                        "- ppt: 幻灯片/演示文稿\n"
                                        "- chart: 图表/K线图/数据图/股票软件界面\n"
                                        "- code: 代码/终端\n"
                                        "- camera: 人物出镜/摄像头画面（无教学信息）\n"
                                        "- transition: 直播软件界面(OBS等)/场景切换过渡/画面部分遮挡\n"
                                        "- other: 其他有信息量的画面\n"
                                        "注意：transition 仅指能看到操作系统桌面/任务栏、OBS控制面板、"
                                        "或同一画面递归缩小重复（套娃）的情况。股票软件的多面板界面属于 chart，不是 transition。\n"
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
            parsed = {"is_ppt": False, "visual_type": kf.get("_visual_type", "other"),
                       "text_content": "", "description": answer[:200]}
        log.info("  帧 %d (%.1fs): %s", kf.get("index", 0), kf.get("timestamp", 0), parsed.get("visual_type", "?"))
        return {"index": kf.get("index", 0), "timestamp": kf.get("timestamp", 0), **parsed}

    detail_tasks = [analyze_one(kf) for kf in selected]
    raw = await asyncio.gather(*detail_tasks)
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
    长视频自动分块摘要再汇总，确保覆盖全部内容。
    返回 {"knowledge_points": [...], "summary": "...", "outline": [...]}
    """
    log.info("知识点提取: DeepSeek")

    # 构建视觉内容摘要（带帧索引）
    visual_summary = []
    for vr in visual_results:
        line = f"[帧{vr.get('index',0)} {vr['timestamp']:.1f}s] {vr.get('visual_type','?')}"
        if vr.get("text_content"):
            line += f" | 文字: {vr['text_content'][:300]}"
        if vr.get("description"):
            line += f" | {vr['description'][:150]}"
        visual_summary.append(line)

    # 构建全部转写文本（不截断）
    all_segments = transcript.get("segments", [])
    log.info("ASR 总段数: %d", len(all_segments))

    has_transcript = len(all_segments) > 0
    has_visual = len(visual_summary) > 0

    # ── 分块策略：按 15 分钟分块 ──
    CHUNK_SECONDS = 900  # 15 分钟

    if has_transcript and len(all_segments) > 400:
        # 长视频：分块摘要 → 汇总
        log.info("长视频模式: 分块摘要再汇总")
        chunk_summaries = await _chunked_summarize(
            all_segments, visual_summary, video_name,
            deepseek_key, deepseek_url, CHUNK_SECONDS
        )
        # 用分块摘要做最终汇总
        result = await _final_synthesis(
            chunk_summaries, visual_summary, video_name,
            deepseek_key, deepseek_url
        )
    else:
        # 短视频：直接一次性处理
        transcript_lines = []
        for seg in all_segments:
            transcript_lines.append(f"[{seg['start']:.1f}s-{seg['end']:.1f}s] {seg['text']}")
        result = await _single_pass_extract(
            transcript_lines, visual_summary, video_name,
            has_transcript, has_visual, deepseek_key, deepseek_url
        )

    log.info("知识点提取完成: %d 个知识点", len(result.get("knowledge_points", [])))
    return result


async def _chunked_summarize(
    segments: list[dict], visual_summary: list[str], video_name: str,
    dk_key: str, dk_url: str, chunk_seconds: int,
) -> list[dict]:
    """把 ASR 文本按时间分块，每块单独提取要点。"""
    # 按时间分块
    chunks = []
    current_chunk = []
    chunk_start = 0
    chunk_idx = 0

    for seg in segments:
        t = seg.get("start", 0)
        if t >= chunk_start + chunk_seconds and current_chunk:
            chunks.append({
                "index": chunk_idx,
                "start_sec": chunk_start,
                "end_sec": t,
                "lines": current_chunk,
            })
            chunk_idx += 1
            chunk_start = t
            current_chunk = []
        current_chunk.append(f"[{seg['start']:.1f}s-{seg['end']:.1f}s] {seg['text']}")

    if current_chunk:
        end_t = segments[-1].get("end", chunk_start + chunk_seconds)
        chunks.append({
            "index": chunk_idx,
            "start_sec": chunk_start,
            "end_sec": end_t,
            "lines": current_chunk,
        })

    log.info("分块: %d 块 (每块 %d 分钟)", len(chunks), chunk_seconds // 60)

    # 找每块对应的视觉分析
    def get_visual_for_chunk(start, end):
        result = []
        for v in visual_summary:
            # 从 "[帧X 123.4s]" 提取时间
            try:
                ts = float(v.split("s]")[0].split()[-1])
                if start <= ts <= end:
                    result.append(v)
            except (ValueError, IndexError):
                pass
        return result

    # 并发处理每块（最多 3 路并发避免限流）
    CONCURRENCY = 3
    sem = asyncio.Semaphore(CONCURRENCY)
    chunk_results = []

    async def process_chunk(chunk):
        async with sem:
            chunk_visual = get_visual_for_chunk(chunk["start_sec"], chunk["end_sec"])
            start_m = int(chunk["start_sec"] // 60)
            end_m = int(chunk["end_sec"] // 60)
            transcript_text = "\n".join(chunk["lines"])
            visual_text = "\n".join(chunk_visual) if chunk_visual else "（该时段无画面分析）"

            prompt = f"""你是知识整理专家。以下是一段视频的第 {start_m}-{end_m} 分钟的内容。
请提取这段时间内的**所有知识要点**。

## 重要约束
1. 只能使用下面提供的素材，严禁添加素材中没有的信息
2. 时间戳必须来自素材中的实际时间
3. 如果这段时间是闲聊/寒暄/无实质内容，直接返回空列表。以下都算闲聊：
   - 主播打招呼、问候观众、等人进直播间
   - 聊天气、吃饭、日常琐事
   - 纯粹的弹幕互动（"谢谢关注"、"欢迎新朋友"）
   - 没有传递可学习的知识或技能
4. 宁可少写也不要编造

## 视频名称
{video_name}

## 音频转写（{start_m}-{end_m}分钟）
{transcript_text}

## 画面分析
{visual_text}

请用 JSON 格式输出：
```json
{{
  "segment_summary": "这段时间的一句话概述",
  "has_knowledge": true/false,
  "points": [
    {{
      "title": "知识点标题",
      "content": "详细内容（100-300字）",
      "time_start_sec": 0,
      "time_end_sec": 0,
      "importance": "high/medium/low",
      "related_frame_indices": [],
      "key_takeaways": ["要点1", "要点2"]
    }}
  ]
}}
```
只返回 JSON。"""

            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{dk_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {dk_key}",
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
                data = json.loads(answer.strip().strip("```json").strip("```").strip())
            except json.JSONDecodeError:
                data = {"segment_summary": answer[:200], "has_knowledge": False, "points": []}

            data["start_sec"] = chunk["start_sec"]
            data["end_sec"] = chunk["end_sec"]
            log.info("  块 %d/%d (%d-%d分钟): %d 个知识点",
                     chunk["index"] + 1, len(chunks), start_m, end_m,
                     len(data.get("points", [])))
            return data

    tasks = [process_chunk(c) for c in chunks]
    chunk_results = await asyncio.gather(*tasks)
    return sorted(chunk_results, key=lambda x: x.get("start_sec", 0))


async def _final_synthesis(
    chunk_summaries: list[dict], visual_summary: list[str],
    video_name: str, dk_key: str, dk_url: str,
) -> dict:
    """把分块摘要汇总成最终报告。分组合并避免 JSON 截断。"""
    # 构建分块摘要文本
    summary_lines = []
    all_points = []
    for cs in chunk_summaries:
        start_m = int(cs.get("start_sec", 0) // 60)
        end_m = int(cs.get("end_sec", 0) // 60)
        summary_lines.append(f"[{start_m}-{end_m}分钟] {cs.get('segment_summary', '无摘要')}")
        for pt in cs.get("points", []):
            all_points.append(pt)

    log.info("汇总: %d 块摘要, %d 个知识点", len(chunk_summaries), len(all_points))

    if len(all_points) <= 25:
        # 知识点不多，直接一轮合并
        result = await _merge_points(all_points, summary_lines, video_name, dk_key, dk_url)
    else:
        # 知识点太多，先按时间段分组合并
        GROUP_SIZE = 20
        groups = []
        for i in range(0, len(all_points), GROUP_SIZE):
            groups.append(all_points[i:i + GROUP_SIZE])

        log.info("分组合并: %d 个知识点 → %d 组", len(all_points), len(groups))

        merged_points = []
        for gi, group in enumerate(groups):
            merged = await _merge_points(group, summary_lines, video_name, dk_key, dk_url)
            merged_pts = merged.get("knowledge_points", group)
            merged_points.extend(merged_pts)
            log.info("  组 %d/%d: %d → %d 个知识点", gi + 1, len(groups),
                     len(group), len(merged_pts))

        # 单独生成 outline 和 summary（轻量请求，不传知识点全文）
        meta = await _generate_outline(merged_points, summary_lines, video_name, dk_key, dk_url)
        result = {
            "summary": meta.get("summary", ""),
            "outline": meta.get("outline", []),
            "knowledge_points": merged_points,
        }

    # 确保字段完整
    if not result.get("knowledge_points"):
        result["knowledge_points"] = all_points

    log.info("合并后: %d → %d 个知识点", len(all_points), len(result["knowledge_points"]))
    return result


async def _merge_points(
    points: list[dict], summary_lines: list[str],
    video_name: str, dk_key: str, dk_url: str,
) -> dict:
    """合并一组知识点（同主题跨块合并），生成摘要和大纲。"""
    points_text = json.dumps(points, ensure_ascii=False)

    prompt = f"""以下是从视频 "{video_name}" 中提取的知识点（{len(points)} 个）。
请完成两件事：

### 任务 1：合并知识点
- 如果多个知识点讲的是同一个主题（比如连续几个都在讲 MACD），合并成一个完整的知识点
- 合并时保留最详细的内容，时间范围取最早的 start 和最晚的 end
- 如果某个知识点是独立的，保持原样
- 不要添加素材中没有的信息

### 任务 2：生成摘要和大纲
- summary: 100-200 字的整体摘要
- outline: 按时间顺序的章节列表

## 分段摘要
{chr(10).join(summary_lines)}

## 知识点列表
{points_text}

用 JSON 格式输出：
```json
{{
  "summary": "整体摘要...",
  "outline": [{{"title": "章节名", "time_start_sec": 0, "time_end_sec": 300}}],
  "knowledge_points": [
    {{
      "title": "...",
      "content": "合并后的详细内容...",
      "time_start_sec": 0,
      "time_end_sec": 2700,
      "importance": "high",
      "related_frame_indices": [0, 3],
      "key_takeaways": ["要点1", "要点2"]
    }}
  ]
}}
```
只返回 JSON。"""

    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            f"{dk_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {dk_key}",
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
        log.error("合并结果 JSON 解析失败，使用原始知识点")
        result = {"summary": "", "outline": [], "knowledge_points": points}

    return result


async def _generate_outline(
    points: list[dict], summary_lines: list[str],
    video_name: str, dk_key: str, dk_url: str,
) -> dict:
    """单独生成 outline 和 summary（轻量请求，只传知识点标题和时间）。"""
    # 只传标题和时间，不传完整 content，避免超长
    point_briefs = []
    for pt in points:
        title = pt.get("title", "")
        t0 = pt.get("time_start_sec", 0)
        t1 = pt.get("time_end_sec", 0)
        point_briefs.append(f"[{t0:.0f}s-{t1:.0f}s] {title}")

    prompt = f"""根据以下视频的分段摘要和知识点列表，生成：
1. 一段 100-200 字的整体摘要
2. 视频大纲（按内容主题划分的章节列表，用于目录索引）

大纲要求：
- 按**内容主题**划分章节，不要按固定时间间隔（如每15分钟）均分
- 把相邻的、主题相近的知识点归入同一章节
- 一个章节可以跨越 5 分钟也可以跨越 40 分钟，取决于主题的实际时长
- 时间范围必须来自知识点的实际时间戳，不要凑整数
- 章节数量控制在 5-15 个

## 视频名称
{video_name}

## 分段摘要
{chr(10).join(summary_lines)}

## 知识点列表（{len(points)} 个）
{chr(10).join(point_briefs)}

用 JSON 格式输出：
```json
{{
  "summary": "整体摘要...",
  "outline": [{{"title": "章节名", "time_start_sec": 0, "time_end_sec": 300}}]
}}
```
只返回 JSON。"""

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{dk_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {dk_key}",
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
        log.error("大纲生成 JSON 解析失败")
        result = {"summary": "", "outline": []}

    log.info("大纲生成: %d 个章节", len(result.get("outline", [])))
    return result


async def _single_pass_extract(
    transcript_lines: list[str], visual_summary: list[str],
    video_name: str, has_transcript: bool, has_visual: bool,
    dk_key: str, dk_url: str,
) -> dict:
    """短视频：一次性提取知识点。"""
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
{"（无音频转写数据）" if not has_transcript else chr(10).join(transcript_lines)}

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
            f"{dk_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {dk_key}",
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

    # 记录已使用的帧索引，避免同一帧反复出现
    used_frame_indices = set()

    # 建立帧索引 → visual_type 的映射（只需建一次）
    vr_type_map = {}
    for vr in visual_results:
        vr_type_map[vr.get("index", -1)] = vr.get("visual_type", "other")
    SKIP_TYPES = {"camera", "transition"}

    for i, kp in enumerate(knowledge.get("knowledge_points", []), 1):
        imp = {"high": "🔴 重要", "medium": "🟡 一般", "low": "🟢 了解"}.get(kp.get("importance", ""), "")

        ts_start = _sec_to_ts(kp.get("time_start_sec", 0))
        ts_end = _sec_to_ts(kp.get("time_end_sec", 0))

        lines.append(f"### {i}. {kp['title']}\n")
        lines.append(f"⏱️ **{ts_start} - {ts_end}** &nbsp; {imp}\n")

        # 配图：用从原视频截取的帧
        screenshot_fn = kp.get("screenshot_filename", "")
        if screenshot_fn:
            from urllib.parse import quote
            img_path = f"/output/{quote(video_name)}/keyframes/{quote(screenshot_fn)}"
            lines.append(f"![{kp['title'][:30]} ({ts_start})]({img_path})\n")

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


def generate_raw_srt(transcript: dict) -> str:
    """把 ASR 原始 segments 转成 SRT 格式字幕文件。"""
    segments = transcript.get("segments", [])
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _sec_to_srt_ts(seg.get("start", 0))
        end = _sec_to_srt_ts(seg.get("end", 0))
        text = seg.get("text", "").strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


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


# ── 知识点配图：从原视频截帧 ─────────────────────

async def _extract_kp_screenshots(
    kp_list: list[dict], video_path: str, output_dir: str, ds_key: str,
):
    """
    为每个知识点从原视频截帧。
    策略：在知识点时间段内每隔 10 秒截一帧，用 Qwen-VL 快速分类，
    找到第一个 chart/ppt/code 帧就用，跳过 camera/transition。
    结果写入 kp["screenshot_path"]。
    """
    import subprocess

    kf_dir = os.path.join(output_dir, "keyframes")
    os.makedirs(kf_dir, exist_ok=True)

    SKIP_TYPES = {"camera", "transition"}
    sem = asyncio.Semaphore(3)  # 控制 Qwen-VL 并发

    async def classify_image(filepath: str) -> str:
        """快速分类一张图片，返回 visual_type。"""
        with open(filepath, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        async with sem:
            async with httpx.AsyncClient(timeout=30) as client:
                try:
                    resp = await client.post(
                        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {ds_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "qwen-vl-max",
                            "messages": [{
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                                    {"type": "text", "text": (
                                        "这张图片属于哪种类型？只回答一个词：\n"
                                        "ppt / chart / code / camera / transition / other\n"
                                        "（chart=图表/K线图/股票软件, ppt=幻灯片, code=代码, "
                                        "camera=人物出镜, transition=OBS/直播软件/场景切换/遮挡）\n"
                                        "注意：transition 仅指能看到操作系统桌面/任务栏、OBS控制面板、"
                                        "或同一画面递归缩小重复（套娃）的情况。股票软件的多面板界面属于 chart，不是 transition。"
                                    )},
                                ],
                            }],
                            "max_tokens": 20,
                        },
                    )
                    resp.raise_for_status()
                    vtype = resp.json()["choices"][0]["message"]["content"].strip().lower()
                    for valid in ("ppt", "chart", "code", "camera", "transition"):
                        if valid in vtype:
                            return valid
                    return "other"
                except Exception:
                    return "other"

    async def find_best_frame_for_kp(kp: dict, kp_idx: int):
        """在知识点时间段内截帧，找到有信息量的就停。"""
        t_start = kp.get("time_start_sec", 0)
        t_end = kp.get("time_end_sec", t_start + 60)
        mid = (t_start + t_end) / 2

        # 从中间开始，向两边扩展，每 10 秒试一次
        probe_times = [mid]
        for offset in range(10, int((t_end - t_start) / 2) + 20, 10):
            probe_times.append(mid + offset)
            probe_times.append(mid - offset)
        # 限制在时间段内，去重
        probe_times = [t for t in probe_times if t_start - 5 <= t <= t_end + 5]
        probe_times = list(dict.fromkeys(probe_times))[:8]  # 最多试 8 次

        for t in probe_times:
            fname = f"kp_{kp_idx:03d}_{t:.0f}s.jpg"
            fpath = os.path.join(kf_dir, fname)

            # ffmpeg 截帧
            try:
                proc = await asyncio.to_thread(
                    subprocess.run,
                    ["ffmpeg", "-y", "-ss", str(t), "-i", video_path,
                     "-frames:v", "1", "-q:v", "2", fpath],
                    capture_output=True, timeout=15,
                )
                if proc.returncode != 0 or not os.path.exists(fpath):
                    continue
            except Exception:
                continue

            # 分类
            vtype = await classify_image(fpath)
            if vtype not in SKIP_TYPES:
                kp["screenshot_path"] = fpath
                kp["screenshot_filename"] = fname
                log.info("  KP %d: %.0fs → %s (%s)", kp_idx, t, fname, vtype)
                return

            # 是 camera/transition，删掉继续试
            try:
                os.unlink(fpath)
            except Exception:
                pass

        log.info("  KP %d: 未找到有信息量的帧", kp_idx)

    # 并发处理所有知识点
    tasks = [find_best_frame_for_kp(kp, i) for i, kp in enumerate(kp_list)]
    await asyncio.gather(*tasks)

    found = sum(1 for kp in kp_list if kp.get("screenshot_path"))
    log.info("知识点配图: %d/%d 个找到有信息量的帧", found, len(kp_list))


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

    # Step 4: 为每个知识点从原视频截取最佳配图
    video_path = manifest.get("video_path", "")
    if not video_path or not os.path.exists(video_path):
        # manifest 里没有 video_path，从 uploads 目录推算
        import glob
        candidates = glob.glob(f"uploads/*/{video_name}.mp4")
        if candidates:
            video_path = candidates[0]
            log.info("从 uploads 找到视频: %s", video_path)
    kp_list = knowledge.get("knowledge_points", [])
    if kp_list and video_path and os.path.exists(video_path) and ds_key:
        if progress_cb:
            await progress_cb("截取知识点配图", 90)
        try:
            await _extract_kp_screenshots(kp_list, video_path, manifest_dir, ds_key)
        except Exception as e:
            log.warning("知识点配图截取失败: %s", e)

    # Step 5: 生成报告 — 如果有错误，在报告开头显示
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
    raw = generate_raw_srt(transcript)

    return {"markdown": md, "json": rj, "srt": srt, "raw_srt": raw}
