# 语音聊天机器人 — 云边端双模 + 声纹识别

基于云端大模型（Ollama + CosyVoice2）和本地小模型（Ollama `qwen2:1.5b`）的双模语音助手。云端优先，网络不佳时自动切换到本地。集成声纹识别，支持多说话人识别。

## 核心功能

| 功能 | 说明 |
|------|------|
| 唤醒词检测 | `openwakeword`（纯 Python，跨平台），预训练模型 "alexa" |
| 语音识别 (ASR) | 云端 Vosk 服务，TCP 流式传输 |
| 大模型对话 (LLM) | 云端优先（Ollama）→ 超时自动切本地（Ollama `qwen2:1.5b`） |
| 语音合成 (TTS) | 云端优先（CosyVoice2 流式）→ 本地兜底（Microsoft edge-tts） |
| 声纹识别 | SpeechBrain ECAPA-TDNN，多说话人注册与识别 |
| 降噪 + VAD | `noisereduce` 降噪 + WebRTC VAD 语音活动检测 |

## 架构

```
[麦克风] → 唤醒词检测 → VAD → 降噪 → ASR(云端TCP) → LLM路由 → TTS → [扬声器]
                                                    ├─ 云端 Ollama + CosyVoice2（优先）
                                                    └─ 本地 Ollama + edge-tts（兜底）

[声纹识别] → SpeechBrain 嵌入提取 → 余弦相似度匹配 → 说话人 ID（并行，非阻塞）
```

## 项目结构

```
dual_mode/
├── config.yaml                       # 所有配置文件
├── orchestrator.py                   # 主编排器（程序入口）
├── utils/
│   └── config_loader.py             # YAML 配置解析器
├── audio/
│   ├── audio_capture.py             # 音频捕获（PyAudio + VAD + 降噪 + ASR 流式）
│   └── wake_word.py                 # 唤醒词检测（openwakeword）
├── llm/
│   ├── llm_router.py                # 云边双模路由器（含云端 TTS 客户端）
│   └── local_ollama_client.py       # 本地 Ollama HTTP 客户端
├── tts/
│   └── local_tts_client.py          # 本地 TTS 客户端（edge-tts）
├── voiceprint/
│   ├── speechbrain_engine.py        # SpeechBrain ECAPA-TDNN 声纹提取
│   └── speaker_manager.py           # 声纹注册、识别、存储
├── prompts/
│   └── local_fallback.txt           # 本地 LLM 系统提示词
└── enrolled_speakers/               # 声纹模板存储（.npy 文件）
```

## 环境要求

- **操作系统**：Windows 10/11、Linux（树莓派等）
- **Python**：3.10+

### 依赖安装

```bash
# 核心依赖
pip install pyaudio webrtcvad-wheels noisereduce openwakeword
pip install speechbrain edge-tts pydub pyyaml requests numpy

# 本地 LLM（Ollama）
# 从 https://ollama.com 下载安装，然后：
ollama pull qwen2:1.5b
```

### 云端服务

确保以下服务可访问（在 `config.yaml` 中配置）：

| 服务 | 端口 | 说明 |
|------|------|------|
| ASR | TCP 12306 | Vosk 流式语音识别 |
| TTS | TCP 11451 | Ollama LLM + CosyVoice2 TTS |
| Ollama | HTTP 11434 | LLM API（云端/本地） |

## 配置

编辑 `dual_mode/config.yaml`：

```yaml
servers:
  cloud:
    asr:
      host: "100.125.0.54"     # 云端 ASR 地址
      port: 12306
    tts:
      host: "100.125.0.54"     # 云端 TTS 地址
      port: 11451
  local:
    ollama:
      base_url: "http://localhost:11434"  # 本地 Ollama

llm:
  cloud:
    tts_timeout_seconds: 10.0   # 云端超时秒数，超时切本地
  local:
    model: "qwen2:1.5b"        # 本地模型名

voiceprint:
  enabled: true
  identification_threshold: 0.55  # 识别阈值（0-1，越高越严格）

wake_word:
  enabled: true
  model: "alexa"               # 唤醒词模型
  threshold: 0.5               # 灵敏度
```

## 使用方式

### 1. 声纹管理

```bash
# 列出已注册的说话人
python dual_mode/orchestrator.py --list-speakers

# 注册新说话人（需要麦克风，录制 3 次语音）
python dual_mode/orchestrator.py --enroll 张三
```

### 2. 启动语音对话

```bash
python dual_mode/orchestrator.py
```

对话流程：

1. **待机** — 等待唤醒词 "Alexa"
2. **聆听** — 检测到唤醒词后开始录音，自动检测语音结束
3. **播报** — 云端 LLM+TTS 回复（超时自动切换本地）

说出 "退出"、"再见"、"exit"、"quit" 结束程序。

### 3. 使用自定义配置

```bash
python dual_mode/orchestrator.py --config /path/to/custom.yaml
```

## 代码运行流程

```
main()
  ├── 解析命令行参数（--config / --enroll / --list-speakers）
  ├── ConfigLoader 加载 config.yaml
  │
  ├── [--list-speakers] → SpeakerManager.list_enrolled() → 打印列表
  │
  ├── [--enroll NAME] → enroll_speaker()
  │     ├── 循环 3 次：录制音频 → SpeechBrain 提取嵌入 → 收集
  │     └── SpeakerManager.enroll() 平均嵌入 → 保存 .npy
  │
  └── [默认] → start_conversation()
        ├── 初始化 AudioCapture + WakeWordDetector + LLMRouter
        ├── 连接云端 ASR 服务器
        │
        └── 对话循环（状态机）:
              │
              ├── [IDLE] 等待唤醒词
              │     ├── openwakeword 检测 "alexa"
              │     └── 检测到 → 播放 ding.wav → 进入 LISTENING
              │
              ├── [LISTENING] 聆听用户语音
              │     ├── 启用声纹音频累积
              │     ├── listen_and_stream()
              │     │     ├── 从 PyAudio 队列读取音频
              │     │     ├── noisereduce 降噪
              │     │     ├── WebRTC VAD 检测语音起止
              │     │     ├── 语音数据通过 TCP 发送到 ASR
              │     │     └── yield ASR 识别结果（partial/text）
              │     ├── 获取累积音频 → _identify_async()
              │     │     └── SpeechBrain 嵌入 → 余弦相似度匹配
              │     └── 得到最终文本 → 进入 SPEAKING
              │
              └── [SPEAKING] LLM + TTS 播报
                    ├── LLMRouter.get_response_and_play()
                    │     ├── 尝试云端：线程中调用 CloudTTSClient
                    │     │     ├── TCP 发送文本到云端 TTS 服务器
                    │     │     └── 接收 TEXT/AUDO/EOT → PyAudio 播放
                    │     └── 超时/失败：切换本地
                    │           ├── LocalOllamaClient → 文本回复
                    │           └── LocalTTSClient → edge-tts → PyAudio 播放
                    └── 返回 LISTENING（或退出）
```

## 云端 TTS 协议

```
客户端 → 服务器: "{文本}|\n"
服务器 → 客户端: 流式返回
  [4字节 'TEXT'][4字节长度][UTF-8 文本]
  [4字节 'AUDO'][4字节长度][Float32 PCM 音频]
  [4字节 'EOT '][4字节 0x00000000]
```

## 声纹识别流程

```
注册:
  麦克风 → 录制 N 次语音 → SpeechBrain 提取 192 维嵌入 → 平均 → 保存 .npy

识别:
  麦克风 → 用户说话 → 累积音频 → SpeechBrain 提取嵌入
  → 与所有已注册模板计算余弦相似度 → 最高分 ≥ 阈值 → 识别成功
```

## 常见问题

**Q: 无法导入 openwakeword 模型？**  
A: 首次使用会自动下载模型（~10MB），确保网络通畅。

**Q: SpeechBrain 下载失败？**  
A: 首次加载会从 HuggingFace 下载模型（~20MB），需要可访问 huggingface.co。

**Q: PyAudio 无法打开麦克风？**  
A: Windows 下检查麦克风权限设置；Linux 下检查 ALSA/PulseAudio 配置。

**Q: 唤醒词检测不到？**  
A: 在 `config.yaml` 中降低 `wake_word.threshold`（如 0.3），或更换模型为 "hey_jarvis"。

## License

MIT
