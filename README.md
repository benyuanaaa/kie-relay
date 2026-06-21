# kie.ai Relay — OpenAI 兼容 API 中转站

将 kie.ai 的 task-based API 包装成 OpenAI 兼容格式，支持直接用 OpenAI SDK 调用。

## 架构

```
用户客户端 (OpenAI SDK) → kie-relay (端口 5001) → kie.ai API (task-based)
                                              → 轮询等待结果
                                              → 返回 OpenAI 格式
```

## 快速开始

### 1. 安装依赖

```bash
cd kie-relay
pip install -r requirements.txt
```

### 2. 配置

编辑 `.env` 文件：

```env
# 从 kie.ai 后台获取: https://kie.ai/api-key
KIE_API_KEY=你的kie-api-key

# 可选：设置中转站自己的 API Key（用户需要带这个 Key 才能调用）
RELAY_API_KEY=my-secret-key
```

### 3. 启动

```bash
python run.py
```

服务默认运行在 `http://localhost:5001`。

## 支持的接口

### `GET /health`
健康检查，返回余额信息。

### `GET /v1/models`
列出可用模型（OpenAI 兼容）。

### `POST /v1/images/generations`
图片生成（完整 OpenAI 兼容）。

```bash
curl http://localhost:5001/v1/images/generations \
  -H "Authorization: Bearer 你的RELAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "z-image",
    "prompt": "一只白色3D卡通熊戴着金色船长帽",
    "n": 1,
    "size": "1024x1024"
  }'
```

### `POST /v1/chat/completions`
聊天补全（将用户消息作为 prompt 生成图片）。

## 模型映射

| 用户使用的名称 | kie.ai 实际模型 |
|:---|:---|
| `z-image` | z-image |
| `google/imagen-4` | google/imagen4 |
| `google/imagen-4-fast` | google/imagen4-fast |
| `ideogram-v3` | ideogram/v3-text-to-image |
| `bytedance/seedream` | bytedance/seedream |
| `grok-imagine` | grok-imagine/text-to-image |
| `black-forest-labs/flux-pro` | black-forest-labs/flux-pro |
| `black-forest-labs/flux-dev` | black-forest-labs/flux-dev |
| `hailuo/text-to-video` | hailuo/02-text-to-video-pro |
| `kling/v2.1-standard` | kling/v2-1-standard |

也支持透传原始的 kie.ai 模型 ID（包含 `/` 的会自动通过）。

## 搭配 One API 使用

```bash
# 启动 One API (端口 3000)
docker run -d --name one-api --restart always \
  -p 3000:3000 -e TZ=Asia/Shanghai \
  -v /home/oneapi/data:/data \
  justsong/one-api
```

然后在 One API 后台添加渠道：
- **类型**: OpenAI（兼容）
- **名称**: kie-ai
- **API Key**: 你的 RELAY_API_KEY（或留空）
- **Base URL**: `http://你的服务器IP:5001`
- **模型**: 添加上面列出的模型名
