# 视频知识点提取平台 Video Knowledge Extractor

超长知识分享直播录播的自动化知识点提取系统。上传视频，自动完成音频转写、画面分析、知识点提取，生成结构化笔记报告。

## 架构

```
┌─────────────────────────────────────────────────┐
│  Web App (FastAPI :7860)                        │
│  上传视频 → 任务管理 → 报告预览/下载            │
└──────────┬──────────────────────────────────────┘
           │
     ┌─────▼─────┐
     │ preprocess │  FFmpeg 提取音频
     │    .py     │  PySceneDetect 场景检测
     │            │  pHash 去重 + PPT 过滤
     └─────┬─────┘
           │ manifest.json
     ┌─────▼──────────────────────────────────────┐
     │ ai_pipeline.py                              │
     │  ① DashScope Paraformer → ASR 语音转写      │
     │  ② Qwen-VL-Max → 关键帧视觉分析            │
     │  ③ DeepSeek → 知识点提取 + 结构化输出       │
     └─────┬──────────────────────────────────────┘
           │
     ┌─────▼─────┐
     │ 多格式输出 │  Markdown / JSON / SRT
     └───────────┘
```

## 快速开始

### 环境要求

- Ubuntu 22.04+ / DGX Spark
- Python 3.10+
- FFmpeg
- 阿里云百炼 API Key（DashScope）
- DeepSeek API Key

### 安装

```bash
git clone https://github.com/RicoPrice/video-knowledge-extractor.git
cd video-knowledge-extractor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 配置

```bash
cp config.example.yaml config.yaml
```

编辑 `config.yaml`，填入：

| 字段 | 说明 | 必填 |
|------|------|------|
| `dashscope.api_key` | 阿里云百炼 API Key | ✅ |
| `deepseek.api_key` | DeepSeek API Key | ✅ |
| `deepseek.base_url` | DeepSeek API 地址 | 默认 `https://api.deepseek.com` |
| `oss.*` | 阿里云 OSS 配置 | ❌ 可选 |

### 启动

```bash
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 7860
```

浏览器打开 `http://<IP>:7860`，拖拽视频文件上传即可。

## 文件说明

| 文件 | 说明 |
|------|------|
| `app.py` | FastAPI Web 应用主入口 |
| `preprocess.py` | 视频预处理（音频提取、场景检测、关键帧去重） |
| `ai_pipeline.py` | AI 分析流水线（ASR + 视觉分析 + 知识点提取） |
| `database.py` | SQLite 数据层（任务历史、报告存储） |
| `config.example.yaml` | 配置模板 |
| `run.sh` | 命令行一键处理脚本 |
| `templates/` | Jinja2 前端页面 |

## 处理流程

1. **上传** — 拖拽视频到 Web 页面，SHA-256 去重检测
2. **预处理** — FFmpeg 提取音频 → PySceneDetect 场景检测 → pHash 去重 → PPT 帧过滤
3. **ASR 转写** — 音频上传至 OSS → DashScope Paraformer-v2 语音转文字（带时间戳）
4. **视觉分析** — Qwen-VL-Max 两轮采样：60 张快速分类 → 按优先级选 30 张详细分析（chart > ppt > code > other，过滤 camera/transition）
5. **知识点提取** — DeepSeek 分块摘要（15 分钟/块，3 路并发）→ 分组合并同主题 → 生成大纲
6. **知识点配图** — 从原视频按知识点时间段截帧 → Qwen-VL 分类过滤 camera/OBS 过渡帧 → 只保留有信息量的截图
7. **报告生成** — Markdown（带截图 + 目录大纲）/ JSON / SRT 多格式输出

## 技术栈

- **后端**: FastAPI + asyncio + aiosqlite
- **预处理**: FFmpeg + PySceneDetect + OpenCV + imagehash
- **ASR**: 阿里云百炼 Paraformer-v2
- **视觉**: Qwen-VL-Max (DashScope)
- **知识提取**: DeepSeek Chat
- **前端**: Tailwind CSS + marked.js
- **存储**: SQLite + 本地文件系统

## 已知限制

- ASR 需要上传音频到 DashScope 云端，大文件上传耗时较长
- 视觉分析采样 30 帧，超长视频可能遗漏部分画面
- 单 uvicorn worker，不支持多任务并行处理
