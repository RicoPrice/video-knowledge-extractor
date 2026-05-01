# video-knowledge-extractor

超长知识分享直播录播 · 自动化知识点提取方案（全 API · 本地预处理版）

## 架构

三层分离设计：

- **Layer 1 — 本地预处理器**：Python 脚本，使用传统图像处理算法（零 AI 模型、零 GPU），完成视频场景切换检测、关键帧去重、PPT 过滤、音频提取，结果上传至 OSS
- **Layer 2 — Dify Workflow**：编排 ASR 转写、多模态视觉分析、知识点融合提取、多格式输出
- **Layer 3 — 云端 AI API**：阿里云百炼（Paraformer ASR + Qwen-VL）、DeepSeek 文本推理

## 快速开始

### 1. 安装依赖

```bash
sudo apt install -y ffmpeg
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml，填入你的 API Key 和 OSS 配置
```

### 3. 运行预处理

```bash
./run.sh /path/to/video.mp4
```

### 4. Dify Workflow

预处理完成后，将生成的 `manifest.json` 提交给 Dify Workflow 进行 AI 分析。

## 输出产物

- 学习笔记 Markdown
- 结构化 JSON
- SRT 字幕轨
- 可交互 HTML 回看页面

## 部署环境

- Ubuntu 24.04 (ARM64 / x86_64)
- Python 3.8+
- FFmpeg
- Docker + Docker Compose（用于 Dify）
