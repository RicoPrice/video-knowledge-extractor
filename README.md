# 视频知识点提取平台 Video Knowledge Extractor

上传长视频后，自动完成音频转写、关键帧分析和知识点抽取，输出结构化报告（Markdown / JSON / SRT）。

## 当前架构

```text
Web App (FastAPI)
  └─ app.py
     ├─ 上传视频（SHA-256 去重）
     ├─ 调用 preprocess.py（子进程）
     └─ 调用 ai_pipeline.py（异步）

preprocess.py
  ├─ FFmpeg 提取音频
  ├─ PySceneDetect 场景检测
  ├─ pHash 去重 + PPT 启发式过滤
  └─ 输出 manifest.json

ai_pipeline.py
  ├─ DashScope Paraformer（ASR）
  ├─ Qwen-VL-Max（关键帧视觉分析）
  └─ DeepSeek（知识点汇总与结构化）
```

## 运行环境

- Linux（推荐 Ubuntu 22.04+）
- Python 3.10+
- FFmpeg
- DashScope API Key（ASR + Qwen-VL）
- DeepSeek API Key
- 阿里云 OSS（当前 ASR 路径依赖 OSS 上传音频）

## 快速开始

### 1) 安装依赖

```bash
git clone https://github.com/RicoPrice/video-knowledge-extractor.git
cd video-knowledge-extractor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

安装 FFmpeg：

```bash
sudo apt update
sudo apt install -y ffmpeg
```

### 2) 配置

```bash
cp config.example.yaml config.yaml
```

至少需要填写：

| 字段 | 说明 | 必填 |
|------|------|------|
| `dashscope.api_key` | 阿里云百炼 API Key | 是 |
| `deepseek.api_key` | DeepSeek API Key | 是 |
| `deepseek.base_url` | DeepSeek API 地址 | 否（默认即可） |
| `oss.access_key_id` | OSS Access Key ID | 是 |
| `oss.access_key_secret` | OSS Access Key Secret | 是 |
| `oss.endpoint` | OSS Endpoint | 是 |
| `oss.bucket` | OSS Bucket 名称 | 是 |
| `oss.prefix` | OSS 对象前缀 | 否 |

> 注意：`config.example.yaml` 中有 `dify` 字段，当前主流程未使用 Dify。

### 3) 启动 Web 服务

```bash
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 7860
```

浏览器访问：`http://<服务器IP>:7860`

## 主要目录

| 路径 | 用途 |
|------|------|
| `app.py` | FastAPI 主入口，任务编排 |
| `preprocess.py` | 视频预处理 |
| `ai_pipeline.py` | AI 分析流水线 |
| `database.py` | SQLite 数据层 |
| `templates/` | 前端页面模板 |
| `static/` | 静态资源 |
| `uploads/` | 上传视频（按任务ID分目录） |
| `output/` | 预处理与报告输出 |
| `data/app.db` | 任务数据库 |

## 处理流程

1. 上传视频（Web）并计算 SHA-256 去重。
2. 预处理：提取音频、场景检测、关键帧去重、PPT 帧标记。
3. 生成 `manifest.json`。
4. AI 分析：ASR -> 视觉分析 -> 知识点提取。
5. 写回数据库并提供报告预览/下载。

## 前端能力

### 首页列表
- 上传区支持拖拽或点击，显示实时上传进度
- SHA-256 去重：相同视频不会重复处理
- 分类筛选：顶部 chip 栏按分类过滤（全部 / 未分类 / 自定义），计数实时更新
- 新建分类直接在页面内完成，之后上传的视频自动归入该分类
- 每张任务卡分类徽章点击即可修改，名称含 emoji / 引号也不会破坏
- 标题点击直接进入报告页，旁边铅笔按钮内联编辑（回车保存 / Esc 取消）
- 排序下拉：最新在前 / 最旧在前 / 名称 A→Z / Z→A / 按状态分组，选择持久化到 localStorage
- 名称模糊搜索
- 实时进度轮询：有任务进行中时 1.5s 轮询，空闲时 5s；页面切回前台立即刷新
- 失败任务用红色左边条和醒目"↻ 重试"按钮高亮；已完成可低调"↻ 重新生成"；已取消可"↻ 重新处理"
- 后端重启残留的僵尸任务会被启动时自动标为"后端维护，请重试"

### 报告详情页
- Markdown 预览 / 原文 / JSON / Manifest 多标签切换
- 视频播放，时间戳点击跳转
- 图片灯箱预览
- 滚动触发视频小窗（PiP）
- 下载 `Markdown / JSON / SRT（总结） / SRT（原始字幕）`
- 浏览器打印导出 PDF
- 顶部工具栏内的"✎ 修改标题"支持内联重命名
- 进行中任务显示带脉冲动画的进度条，完成后本页自动刷新
- 失败态显示错误详情，一键"↻ 重试处理"

## API 概览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/upload` | 上传视频，可选表单字段 `category` |
| GET | `/api/tasks` | 任务列表（带 no-cache 响应头） |
| GET | `/api/tasks/{id}` | 单个任务详情 |
| PATCH | `/api/tasks/{id}` | 修改 `category` 或 `video_name` |
| POST | `/api/tasks/{id}/retry` | 用原视频重新跑一次 pipeline |
| DELETE | `/api/tasks/{id}` | 删除任务及其上传 / 输出文件 |
| GET | `/api/categories` | 分类列表及计数 |
| GET | `/api/tasks/{id}/download/{kind}` | 下载报告（kind: `md` / `json` / `srt` / `raw_srt`） |

## 部署指南

生产部署（systemd、自启动、日志、排错）见：`DEPLOYMENT.md`

## 已知限制

- 当前 ASR 流程依赖 OSS 临时上传音频。
- 默认单进程运行，长任务会占用较多处理时间。
- 视觉分析基于采样帧，极长视频可能遗漏部分画面。
