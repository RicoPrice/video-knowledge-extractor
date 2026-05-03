"""SQLite database layer for task history and report storage."""

import aiosqlite
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "app.db")


async def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                video_name TEXT NOT NULL,
                video_path TEXT,
                file_hash TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                progress INTEGER DEFAULT 0,
                stage TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                manifest_json TEXT,
                report_markdown TEXT,
                report_json TEXT,
                report_srt TEXT,
                report_html TEXT,
                raw_srt TEXT,
                error TEXT
            )
        """)
        # 兼容旧表：如果 raw_srt 列不存在则添加
        try:
            await db.execute("ALTER TABLE tasks ADD COLUMN raw_srt TEXT")
        except Exception:
            pass  # 列已存在
        await db.commit()


async def create_task(task_id: str, video_name: str, video_path: str, file_hash: str = "") -> dict:
    now = datetime.now().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO tasks (id, video_name, video_path, file_hash, status, created_at, updated_at) VALUES (?, ?, ?, ?, 'pending', ?, ?)",
            (task_id, video_name, video_path, file_hash, now, now),
        )
        await db.commit()
    return {"id": task_id, "video_name": video_name, "status": "pending", "created_at": now}


async def update_task(task_id: str, **kwargs):
    kwargs["updated_at"] = datetime.now().isoformat()
    fields = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [task_id]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(f"UPDATE tasks SET {fields} WHERE id = ?", values)
        await db.commit()


async def get_task(task_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def list_tasks(limit: int = 50, offset: int = 0) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, video_name, status, progress, stage, created_at, updated_at, error FROM tasks ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cur:
            return [dict(row) async for row in cur]


async def delete_task(task_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        await db.commit()


async def find_by_hash(file_hash: str) -> dict | None:
    """查找相同文件哈希的已有任务"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM tasks WHERE file_hash = ? ORDER BY created_at DESC LIMIT 1",
            (file_hash,),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None
