"""Web App — 视频知识点提取平台"""

import asyncio
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import database as db
import ai_pipeline

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="视频知识点提取平台")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# Background task registry
_running_tasks: dict[str, asyncio.Task] = {}


def load_config() -> dict:
    cfg_path = BASE_DIR / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


@app.on_event("startup")
async def startup():
    await db.init_db()
    (BASE_DIR / "static").mkdir(exist_ok=True)
    (BASE_DIR / "templates").mkdir(exist_ok=True)


# ── Pages ─────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/task/{task_id}", response_class=HTMLResponse)
async def task_detail_page(request: Request, task_id: str):
    task = await db.get_task(task_id)
    if not task:
        return HTMLResponse("任务不存在", status_code=404)
    return templates.TemplateResponse("task.html", {"request": request, "task": task})


# ── API ───────────────────────────────────────────

@app.get("/api/tasks")
async def api_list_tasks():
    tasks = await db.list_tasks()
    return tasks


@app.get("/api/tasks/{task_id}")
async def api_get_task(task_id: str):
    task = await db.get_task(task_id)
    if not task:
        return JSONResponse({"error": "not found"}, 404)
    return task


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse({"error": "no file"}, 400)

    task_id = uuid.uuid4().hex[:12]
    video_name = Path(file.filename).stem
    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    video_path = task_dir / file.filename

    # 边写入边计算 SHA-256
    sha = hashlib.sha256()
    with open(video_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
            sha.update(chunk)
    file_hash = sha.hexdigest()

    # 检查是否有相同文件的已有任务
    existing = await db.find_by_hash(file_hash)
    if existing:
        if existing["status"] == "completed":
            # 已完成：跳过上传，直接返回旧任务
            shutil.rmtree(task_dir)
            return JSONResponse({
                "task_id": existing["id"],
                "video_name": existing["video_name"],
                "duplicate": True,
                "status": existing["status"],
            })
        elif existing["status"] in ("failed", "cancelled"):
            # 失败/取消：删除旧任务，用旧文件路径重新处理
            old_id = existing["id"]
            old_video_path = existing.get("video_path", "")
            await db.delete_task(old_id)
            old_upload = UPLOAD_DIR / old_id
            if old_upload.exists():
                shutil.rmtree(old_upload)
            # 用新任务继续
        elif existing["status"] in ("pending", "processing"):
            # 正在处理中：跳过，返回当前任务
            shutil.rmtree(task_dir)
            return JSONResponse({
                "task_id": existing["id"],
                "video_name": existing["video_name"],
                "duplicate": True,
                "status": existing["status"],
            })

    await db.create_task(task_id, video_name, str(video_path), file_hash)
    bg = asyncio.create_task(run_pipeline(task_id, str(video_path)))
    _running_tasks[task_id] = bg
    return {"task_id": task_id, "video_name": video_name, "duplicate": False}


@app.delete("/api/tasks/{task_id}")
async def api_delete_task(task_id: str):
    task = await db.get_task(task_id)
    if not task:
        return JSONResponse({"error": "not found"}, 404)
    if task_id in _running_tasks:
        _running_tasks[task_id].cancel()
        del _running_tasks[task_id]
    upload_path = UPLOAD_DIR / task_id
    output_path = OUTPUT_DIR / (task.get("video_name") or task_id)
    if upload_path.exists():
        shutil.rmtree(upload_path)
    if output_path.exists():
        shutil.rmtree(output_path)
    await db.delete_task(task_id)
    return {"ok": True}


@app.get("/api/tasks/{task_id}/download/{fmt}")
async def api_download(task_id: str, fmt: str):
    task = await db.get_task(task_id)
    if not task:
        return JSONResponse({"error": "not found"}, 404)
    field_map = {"md": "report_markdown", "json": "report_json", "srt": "report_srt", "html": "report_html"}
    field = field_map.get(fmt)
    if not field or not task.get(field):
        return JSONResponse({"error": "report not available"}, 404)
    ext_map = {"md": ".md", "json": ".json", "srt": ".srt", "html": ".html"}
    content = task[field]
    filename = f"{task['video_name']}_report{ext_map[fmt]}"
    tmp_path = OUTPUT_DIR / filename
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    return FileResponse(tmp_path, filename=filename)


# ── Background pipeline ──────────────────────────

async def run_pipeline(task_id: str, video_path: str):
    """Run the full pipeline: preprocess → AI analysis."""
    try:
        await db.update_task(task_id, status="processing", stage="预处理", progress=5)

        task = await db.get_task(task_id)
        video_name = task["video_name"]
        out_dir = str(OUTPUT_DIR / video_name)

        await db.update_task(task_id, stage="提取音频", progress=10)
        proc = await asyncio.create_subprocess_exec(
            str(BASE_DIR / "venv" / "bin" / "python3"), "-u",
            str(BASE_DIR / "preprocess.py"),
            video_path,
            "-c", str(BASE_DIR / "config.yaml"),
            "-o", out_dir,
            "--skip-oss",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            limit=10 * 1024 * 1024,  # 10MB buffer for long progress lines
        )
        # 逐块读取输出（PySceneDetect 用 \r 刷新进度条，不产生 \n）
        output_lines = []
        while True:
            try:
                chunk = await asyncio.wait_for(proc.stdout.read(8192), timeout=300)
            except asyncio.TimeoutError:
                break
            if not chunk:
                break
            text = chunk.decode(errors="replace")
            for line in text.replace("\r", "\n").split("\n"):
                line = line.strip()
                if not line:
                    continue
                output_lines.append(line)
                ll = line.lower()
                if "audio" in ll or "ffmpeg" in ll:
                    await db.update_task(task_id, stage="提取音频", progress=15)
                elif "scene" in ll or "detect" in ll:
                    await db.update_task(task_id, stage="场景检测", progress=20)
                elif "hash" in ll or "phash" in ll or "dedup" in ll:
                    await db.update_task(task_id, stage="关键帧去重", progress=28)
                elif "ppt" in ll or "filter" in ll:
                    await db.update_task(task_id, stage="PPT 过滤", progress=32)
                elif "manifest" in ll or "done" in ll:
                    await db.update_task(task_id, stage="生成 Manifest", progress=38)
        await proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"预处理失败: {chr(10).join(output_lines[-10:])}")

        await db.update_task(task_id, stage="预处理完成", progress=40)

        manifest_path = os.path.join(out_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"manifest.json 未生成: {out_dir}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = f.read()
        await db.update_task(task_id, manifest_json=manifest)

        # AI 分析（直接调用 API，不经过 Dify）
        async def progress_cb(stage, pct):
            await db.update_task(task_id, stage=stage, progress=pct)

        await db.update_task(task_id, stage="AI 分析中", progress=45)
        result = await ai_pipeline.run_ai_pipeline(manifest_path, progress_cb=progress_cb)

        await db.update_task(
            task_id, status="completed", stage="完成", progress=100,
            report_markdown=result.get("markdown", ""),
            report_json=result.get("json", ""),
            report_srt=result.get("srt", ""),
            report_html="",
        )
    except asyncio.CancelledError:
        await db.update_task(task_id, status="cancelled", stage="已取消")
    except Exception as e:
        await db.update_task(task_id, status="failed", stage="失败", error=str(e)[:1000])
    finally:
        _running_tasks.pop(task_id, None)
