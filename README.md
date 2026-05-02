# 视频知识点提取平台

超长知识分享直播录播的自动化知识点提取系统。上传视频，自动完成音频转写、画面分析、知识点提取，生成结构化报告。

## 架构

```
浏览器 (局域网)
  │  上传视频
  ▼
Web App (FastAPI :7860)
  │
  ├─ Layer 1: 预处理 (preprocess.py)
  │   ├─ FFmpeg 提取音频
  │   ├─ PySceneDetect 场景检测
  │   ├─ pHash 关键帧去重
  │   └─ PPT/非PPT 分类
  │
  └─ Layer 2: AI 分析 (ai_pipeline.py)
      ├─ DashScope Paraformer-v2  → ASR 语音转文字
      ├─ DashScope Qwen-VL-Max   → 关键帧视觉分析
      └─ DeepSeek Chat            → 知识点提取 + 报告生成
```

## 快速开始

### 环境要求

- Ubuntu 22.04+ / DGX Spark
- Python 3.10+
- FFmpeg
- 阿里云百炼 (DashScope) API Key
- DeepSeek API Key

### 安装

```bash
git clone https://github.com/RicoPrice/video-knowledge-extractor.git
cd video-knowledge-extractor

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp config.example.yaml config.yaml
# 编辑 config.yaml，填入 API Key
```

### 启动

```bash
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 7860
```

浏览器访问 `http://<机器IP>:7860`

### 使用

1. 拖拽视频文件到上传区域
2. 等待处理（进度条实时更新）
3. 完成后点击「查看报告」预览
4. 支持下载 Markdown / JSON / SRT 格式

## 配置说明

复制 `config.example.yaml` 为 `config.yaml`，填入：

| 字段 | 说明 | 必填 |
|------|------|------|
| `dashscope.api_key` | 阿里云百炼 API Key（ASR + Qwen-VL） | 是 |
| `deepseek.api_key` | DeepSeek API Key（知识点提取） | 是 |
| `deepseek.base_url` | DeepSeek API 地址，默认 `https://api.deepseek.com` | 否 |
| `oss.*` | 阿里云 OSS（当前未使用，可跳过） | 否 |

## 项目结构

```
├── app.py                 # FastAPI Web 应用
├── ai_pipeline.py         # AI 分析流水线（ASR + 视觉 + 知识提取）
├── preprocess.py          # 视频预处理（音频提取 + 场景检测 + 去重）
├── database.py            # SQLite 任务存储
├── config.example.yaml    # 配置模板
├── requirements.txt       # Python 依赖
├── templates/
│   ├── index.html         # 首页（上传 + 任务列表）
│   └── task.html          # 报告预览页
└── static/                # 静态资源
```

## 功能特性

- **拖拽上传**：支持 MP4/MKV/AVI/MOV
- **重复检测**：SHA-256 哈希去重，相同视频不重复处理
- **实时进度**：预处理和 AI 分析阶段进度条实时更新
- **多格式输出**：Markdown（带截图）、JSON、SRT
- **报告预览**：Markdown 渲染 + 关键帧截图 + 点击放大
- **历史记录**：所有任务持久化存储，支持查看和删除

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端 | FastAPI + uvicorn |
| 前端 | Tailwind CSS + vanilla JS |
| 存储 | SQLite (aiosqlite) |
| 预处理 | FFmpeg + PySceneDetect + OpenCV |
| ASR | DashScope Paraformer-v2 |
| 视觉分析 | DashScope Qwen-VL-Max |
| 知识提取 | DeepSeek Chat |

## 已知限制（v0.1 原型）

- ASR 需要上传音频到 DashScope 云端，大文件上传较慢
- 视觉分析最多采样 30 张关键帧（均匀采样）
- 无用户认证，适合局域网内部使用
- 单进程处理，不支持多任务并行

## License

MIT
