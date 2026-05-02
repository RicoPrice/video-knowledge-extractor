# 视频知识点提取平台

从知识分享类直播录播中，自动提取结构化知识点。

## 架构

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Web 前端    │────▶│  FastAPI 后端  │────▶│  预处理器     │
│  上传/预览    │◀────│  任务管理      │     │  FFmpeg       │
│  历史记录     │     │  SQLite       │     │  PySceneDetect│
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                     ┌──────▼───────┐
                     │  AI Pipeline  │
                     │  DashScope    │
                     │  (ASR+VL)     │
                     │  DeepSeek     │
                     │  (知识提取)    │
                     └──────────────┘
```

**处理流程：**

1. 用户通过 Web 页面上传视频
2. 预处理器提取音频 + 场景检测 + 关键帧去重 + PPT 过滤
3. AI Pipeline：
   - Paraformer-v2 语音转文字（带时间戳）
   - Qwen-VL-Max 分析关键帧图片（OCR / PPT 识别）
   - DeepSeek 融合音视觉信息，提取结构化知识点
4. 生成 Markdown / JSON / SRT 多格式报告
5. 在线预览和下载

## 快速开始

### 环境要求

- Ubuntu 22.04+ / DGX Spark
- Python 3.12+
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
# 编辑 config.yaml，填入：
#   dashscope.api_key  — 阿里云百炼 API Key
#   deepseek.api_key   — DeepSeek API Key
#   oss.*              — （可选）阿里云 OSS 配置
```

### 启动

```bash
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 7860
```

浏览器访问 `http://<IP>:7860`，拖拽视频文件上传即可。

## 项目结构

```
├── app.py                 # FastAPI Web 应用
├── ai_pipeline.py         # AI 分析流水线（ASR + VL + 知识提取）
├── preprocess.py          # 视频预处理（音频提取、场景检测、去重）
├── database.py            # SQLite 数据层
├── config.example.yaml    # 配置模板
├── requirements.txt       # Python 依赖
├── run.sh                 # CLI 一键运行脚本
├── templates/
│   ├── index.html         # 首页（上传 + 任务列表）
│   └── task.html          # 报告预览页
├── static/                # 静态资源
└── workflow.yml           # Dify Workflow DSL（备用）
```

## 技术栈

| 组件 | 技术 |
|------|------|
| Web 后端 | FastAPI + Uvicorn |
| 前端 | HTML + Tailwind CSS + 原生 JS |
| 数据库 | SQLite (aiosqlite) |
| 视频处理 | FFmpeg + PySceneDetect + OpenCV |
| 图片去重 | pHash (imagehash) |
| ASR | 阿里云百炼 Paraformer-v2 |
| 视觉分析 | 阿里云百炼 Qwen-VL-Max |
| 知识提取 | DeepSeek-V3 |

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 首页 |
| `/task/{id}` | GET | 报告预览页 |
| `/api/upload` | POST | 上传视频（multipart/form-data） |
| `/api/tasks` | GET | 任务列表 |
| `/api/tasks/{id}` | GET | 任务详情 |
| `/api/tasks/{id}` | DELETE | 删除任务 |
| `/api/tasks/{id}/download/{fmt}` | GET | 下载报告（md/json/srt/html） |

## 已知限制

- 视觉分析最多采样 30 张关键帧（均匀采样），超长视频可能遗漏部分画面
- ASR 需要先上传音频到 DashScope 文件服务，大文件上传较慢
- 单次处理为串行流水线，不支持多视频并行处理
- 前端为轻量实现，无用户认证

## License

MIT
