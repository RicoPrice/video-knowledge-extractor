# Dify Workflow 搭建指南

## 视频知识点提取 Workflow 节点结构

按文章第三节设计，Workflow 包含以下节点：

### 1. Start 节点
**输入变量：**
- `manifest_json` (paragraph, required) — 预处理器生成的 manifest.json 内容

### 2. Code 节点 — 解析 Manifest
**Python 代码：**
```python
import json
def main(manifest_json: str) -> dict:
    m = json.loads(manifest_json)
    return {
        "audio_url": m["audio"]["oss_url"],
        "keyframe_urls": json.dumps([kf["oss_url"] for kf in m.get("keyframes", [])]),
        "keyframe_info": json.dumps(m.get("keyframes", []), ensure_ascii=False),
        "video_name": m.get("video_name", "unknown"),
    }
```

### 3. HTTP Request 节点 — ASR 转写
**配置：**
- Method: POST
- URL: `https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription`
- Headers:
  - `Authorization: Bearer {{阿里云百炼 API Key}}`
  - `Content-Type: application/json`
- Body:
```json
{
  "model": "paraformer-v2",
  "input": {
    "audio_url": "{{parse_manifest.audio_url}}"
  },
  "parameters": {
    "format": "pcm",
    "sample_rate": 16000,
    "enable_words": true
  }
}
```

### 4. Code 节点 — 语义分段
**功能：** 根据 ASR 时间戳和语义变化切分段落

### 5. LLM 节点 — 视觉分析 (Qwen-VL-Max)
**模型：** Tongyi / qwen-vl-max
**Prompt：**
```
分析这张关键帧图片：
1. 判断是否为 PPT 画面
2. 如果是 PPT，提取所有文字内容（OCR）
3. 如果是代码截图，识别代码内容
4. 如果是图表，描述图表含义

图片 URL: {{keyframe_url}}
时间戳: {{timestamp}}

输出 JSON 格式：
{
  "is_ppt": true/false,
  "text_content": "...",
  "visual_type": "ppt|code|chart|other",
  "description": "..."
}
```

### 6. Code 节点 — 时间戳对齐
**功能：** 将视觉关键帧与 ASR 文本按时间戳对齐

### 7. Code 节点 — 规则去噪
**功能：** 移除口头禅、重复内容、无关片段

### 8. LLM 节点 — 知识点提取 (DeepSeek-V3)
**模型：** DeepSeek / deepseek-chat
**Prompt：**
```
你是一个知识点提取专家。根据以下信息提取结构化知识点：

**音频转写文本：**
{{asr_transcript}}

**视觉内容（PPT/代码/图表）：**
{{visual_content}}

**时间戳对齐信息：**
{{aligned_segments}}

请提取出：
1. 核心知识点（标题 + 详细说明）
2. 每个知识点的时间范围（开始-结束）
3. 相关的 PPT 内容或代码片段
4. 重要程度（高/中/低）

输出 JSON 数组格式：
[
  {
    "title": "知识点标题",
    "content": "详细说明",
    "time_start": 120.5,
    "time_end": 185.3,
    "ppt_content": "相关 PPT 文字",
    "code_snippet": "相关代码",
    "importance": "high"
  },
  ...
]
```

### 9. Code 节点 — 去重聚合
**功能：** 合并相似知识点，移除重复

### 10. LLM 节点 — 最终润色 (DeepSeek-V3)
**功能：** 优化知识点表述，补充细节

### 11. Code 节点 — 多格式输出
**功能：** 生成 Markdown、JSON、SRT、HTML 四种格式

### 12. End 节点
**输出变量：**
- `markdown_output`
- `json_output`
- `srt_output`
- `html_output`

---

## 快速开始

### 方式 1：手动搭建（推荐学习）
1. 在 Dify Studio 点击 "Create from Blank" → 选择 "Workflow"
2. 按上述节点结构逐个添加
3. 连接节点，配置变量传递

### 方式 2：导入 DSL（快速）
1. 等待完整 DSL 文件生成
2. 在 Dify Studio 点击 "Import DSL file"
3. 上传 `workflow.yml`

### 方式 3：先测试 Layer 1（最实用）
1. 先运行预处理器测试：
   ```bash
   cd /home/rico/video-knowledge-extractor
   ./run.sh /path/to/test-video.mp4 --skip-oss
   ```
2. 检查生成的 `output/*/manifest.json`
3. 确认预处理器工作正常后，再搭建 Workflow

---

## 注意事项

- **ASR 调用**：Paraformer 不是 LLM，需要用 HTTP Request 节点调用 DashScope API
- **Qwen-VL 调用**：在 Dify 中配置 Tongyi 模型供应商后，可以直接用 LLM 节点选择 qwen-vl-max
- **批量处理**：关键帧分析需要循环处理，可以用 Iteration 节点或在 Code 节点里批量调用
- **成本控制**：Qwen-VL 和 DeepSeek API 都按 token 计费，建议先用小视频测试

---

## 下一步

建议先测试 Layer 1 预处理器，确保能正常提取音频和关键帧。Workflow 可以逐步完善。
