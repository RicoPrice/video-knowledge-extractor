#!/usr/bin/env bash
# 一键启动预处理脚本（Linux 版）
# 用法: ./run.sh /path/to/video.mp4 [--skip-oss]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

if [ $# -lt 1 ]; then
    echo "用法: $0 <视频文件路径> [--skip-oss]"
    echo "示例: $0 /home/rico/videos/lecture.mp4"
    echo "      $0 /home/rico/videos/lecture.mp4 --skip-oss"
    exit 1
fi

if ! command -v ffmpeg &>/dev/null; then
    echo "错误: 未安装 ffmpeg，请运行: sudo apt install -y ffmpeg"
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "创建 Python 虚拟环境..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    source "$VENV_DIR/bin/activate"
fi

if [ ! -f "$SCRIPT_DIR/config.yaml" ]; then
    echo "警告: config.yaml 不存在，已从模板复制"
    echo "请编辑 config.yaml 填入你的 API Key 和 OSS 配置"
    cp "$SCRIPT_DIR/config.example.yaml" "$SCRIPT_DIR/config.yaml"
fi

cd "$SCRIPT_DIR"
python3 preprocess.py "$@"
