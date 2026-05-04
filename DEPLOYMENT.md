# 部署指南（Linux / systemd）

本文档基于当前仓库实现编写，适用于把项目部署为长期运行的服务。

## 1. 前置条件

- Linux 服务器（推荐 Ubuntu 22.04+）
- 已安装 Python 3.10+
- 已安装 FFmpeg
- 可以访问 DashScope、DeepSeek、阿里云 OSS

安装系统依赖：

```bash
sudo apt update
sudo apt install -y python3 python3-venv ffmpeg
```

## 2. 获取代码并初始化环境

```bash
git clone https://github.com/RicoPrice/video-knowledge-extractor.git
cd video-knowledge-extractor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. 配置密钥与参数

复制配置模板：

```bash
cp config.example.yaml config.yaml
```

编辑 `config.yaml`，至少确认以下字段：

- `dashscope.api_key`
- `deepseek.api_key`
- `deepseek.base_url`（可使用默认）
- `oss.access_key_id`
- `oss.access_key_secret`
- `oss.endpoint`
- `oss.bucket`
- `oss.prefix`（可选）

说明：

- 当前 `ai_pipeline.py` 中 ASR 会把音频先上传到 OSS，再提交 Paraformer。
- 若 OSS 未配置完整，任务会在 ASR 阶段失败。

## 4. 本地验证启动

先手工运行验证配置是否正确：

```bash
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 7860
```

浏览器打开 `http://<服务器IP>:7860`，上传一个短视频验证流程。

## 5. 配置 systemd 服务

仓库已提供示例：`vke.service`。

将该服务文件复制到系统目录并根据实际用户名/路径调整：

```bash
sudo cp vke.service /etc/systemd/system/vke.service
sudo systemctl daemon-reload
sudo systemctl enable --now vke.service
```

检查状态：

```bash
sudo systemctl status vke.service
```

查看日志：

```bash
journalctl -u vke.service -f
```

## 6. 目录与权限建议

项目运行期间会写入以下目录：

- `uploads/`：用户上传视频
- `output/`：预处理中间产物与临时下载文件
- `data/`：SQLite 数据库（`data/app.db`）
- `hotwords.yaml`：首次创建热词表后会回写 `vocabulary_id`

请确保服务用户对仓库目录具备读写权限。

## 7. 网络与安全建议

- 生产环境建议通过 Nginx/Caddy 反向代理并开启 HTTPS。
- 限制 `7860` 端口访问来源，或只监听内网。
- `config.yaml` 含密钥，不要提交到版本库。
- 建议定期备份 `data/app.db` 与业务输出目录。

## 8. 常见问题排查

1) 页面可访问但任务失败  
- 查看任务错误信息（前端任务卡片）  
- 再查看 `journalctl -u vke.service -f` 获取后端异常栈

2) 提示 FFmpeg 相关错误  
- 确认 `ffmpeg` 在系统 PATH 中：`which ffmpeg`

3) ASR 阶段失败  
- 核查 `config.yaml` 中 `oss.*` 配置是否完整
- 核查 DashScope API Key 是否有效

4) 启动失败（找不到 Python 或 uvicorn）  
- 核查 `vke.service` 中 `ExecStart` 与 `WorkingDirectory` 是否为实际路径
- 确认已在该目录创建并安装 `venv`

## 9. 升级流程建议

```bash
cd /path/to/video-knowledge-extractor
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart vke.service
sudo systemctl status vke.service
```

升级后建议上传一个短视频做冒烟验证。
