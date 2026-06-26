

# 智能语音助手 - 基于Qwen系列模型

这是一个基于 **Qwen3-ASR** 语音识别模型和 **Qwen1.5-0.5B-Chat** 对话模型的本地语音助手。通过按住键盘的 `Ctrl` 键说话，松开后自动进行语音识别并调用大模型生成回答，实现端到端的语音对话交互。

## ✨ 功能特点

- **实时语音录入**：通过 `sounddevice` 捕获麦克风音频，支持按住 `Ctrl` 键触发录音
- **本地语音识别**：使用 `Qwen3-ASR-0.6B` 模型将语音转换为中文文本
- **本地对话生成**：基于 `Qwen1.5-0.5B-Chat` 模型生成智能回复
- **智能灯光控制**：当用户提到“开灯”/“关灯”或相关暗示时，模型会直接输出简洁指令（无需额外意图识别模块）
- **完全离线运行**：所有模型均在本地加载，无需联网，保护隐私

## 🛠️ 技术栈

| 组件 | 技术 |
| :--- | :--- |
| 语音识别 | Qwen3-ASR-0.6B（ModelScope） |
| 对话模型 | Qwen1.5-0.5B-Chat（Transformers） |
| 音频采集 | sounddevice + numpy |
| 键盘监听 | keyboard |
| 深度学习框架 | PyTorch |

## 📦 安装与配置

### 环境要求

- Python 3.8+
- 支持 CUDA 的 GPU（可选，CPU 也可运行但速度较慢）
- 麦克风设备

### 安装依赖

```bash
pip install numpy sounddevice keyboard torch modelscope transformers
```

### 下载模型文件

本项目需要两个预训练模型，请确保你有足够的磁盘空间（约 8-10 GB）：

1. **Qwen3-ASR-0.6B**：语音识别模型
2. **Qwen1.5-0.5B-Chat**：对话生成模型

你可以通过 ModelScope 或 HuggingFace 下载，并将模型路径配置在代码中的以下变量：

```python
ASR_MODEL_PATH = r"D:\modelscope\hub\models\Qwen\Qwen3-ASR-0___6B"
CHAT_MODEL_PATH = r"D:\modelscope\hub\models\Qwen\Qwen1___5-0___5B-Chat"
```

> ⚠️ **注意**：请根据你的实际下载路径修改上述路径。

## 🚀 使用指南

### 启动程序

在项目根目录下执行：

```bash
python voice_assistant.py
```

### 交互方式

| 操作 | 功能 |
| :--- | :--- |
| **按住 `Ctrl` 键** | 开始录音（最长 10 秒） |
| **松开 `Ctrl` 键** | 自动结束录音并开始识别 |
| **按 `ESC` 键** | 退出程序 |

### 示例对话

```
按住【Ctrl】说话 → 松开识别 → AI回答
按【ESC】退出程序

=======录音中=======
录音时长为 3.25 秒
=======处理中=======
=====请耐心等待=====
识别中...
今天天气怎么样？
AI回答：今天天气晴朗，气温22度，适合出门活动。
```

### 特殊指令

当你说出以下内容时，模型会直接输出简短指令，而不展开对话：

- “开灯” / “好暗” / “看不清” → 输出：`开灯`
- “关灯” / “太刺眼” / “好亮” → 输出：`关灯`

## ⚙️ 配置调整

### 修改运行设备

默认使用 CPU 运行，如果你的机器有 NVIDIA GPU，可以将代码中的：

```python
device = "cpu"
```

改为：

```python
device = "cuda"
```

### 调整录音时长上限

在 `while keyboard.is_pressed('Ctrl'):` 循环中，将 `10` 改为你想要的秒数：

```python
if time.time() - start_time > 10:  # 修改这里的数字
    stop_Sign = True
    break
```

### 修改系统提示词（System Prompt）

如果你想改变模型的回答风格或特殊指令，可以修改 `chat_with_local_model` 函数中的 `system` 内容：

```python
{"role": "system", "content": "请用中文简洁回答。如果用户..."}
```

## 📁 项目结构

```
.
├── voice_assistant.py          # 主程序文件
├── README.md                   # 项目说明文档
└── requirements.txt            # 依赖清单（可选）
```

---

## 🚀 一键安装命令

在项目根目录下执行：

```bash
pip install -r requirements.txt
```

---

## 💡 按需安装建议

如果你的网络环境或硬件配置有特殊需求，可以参考以下分组安装：

### 最小化安装（仅核心依赖）

```bash
pip install torch torchaudio transformers modelscope sounddevice numpy keyboard
```

### 完整安装（含推荐扩展）

```bash
pip install -r requirements.txt
```

### 如果使用 GPU（CUDA 版本）

PyTorch 的 GPU 版本需要单独指定安装源，建议去 [PyTorch官网](https://pytorch.org/get-started/locally/) 根据你的 CUDA 版本生成安装命令，通常是：

```bash
# 示例：CUDA 11.8 版本
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

然后再安装其他依赖：

```bash
pip install transformers modelscope sounddevice numpy keyboard
```

### 如果使用 CPU 运行（推荐）

直接使用上面的完整安装命令即可，PyTorch 会自动安装 CPU 版本。

---

## ⚠️ 注意事项

| 问题 | 解决方法 |
| :--- | :--- |
| **sounddevice 安装失败** | Windows 用户需要安装 [PortAudio](https://www.portaudio.com/)，或者使用 `pip install pyaudio` 替代（但需要修改代码中的音频采集部分） |
| **torch 下载太慢** | 使用国内镜像源：`pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| **modelscope 依赖冲突** | 如果遇到版本冲突，可以尝试 `pip install modelscope --upgrade` |
| **keyboard 在 Linux/macOS 需要权限** | Linux 用户可能需要 `sudo` 运行，macOS 用户需要在系统设置中授予终端辅助功能权限 |

---

## ❓ 常见问题

### Q1: 提示 `sounddevice.PortAudioError` 或找不到音频设备？

请检查麦克风是否正常连接，或尝试指定音频设备索引：

```python
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32',
    callback=audio_callback,
    device=1  # 指定设备索引，可在Python中运行 sd.query_devices() 查看
)
```

### Q2: 模型加载时间过长或内存不足？

- 确保你的电脑有至少 8GB 可用内存（CPU 运行）或 6GB 显存（GPU 运行）
- 可以尝试使用量化版模型（如果官方提供）来降低资源占用

### Q3: 按 `Ctrl` 键无反应或录音不触发？

- 请以**管理员身份**运行终端（Windows）或授予终端**麦克风权限**（macOS/Linux）
- 检查是否有其他程序占用了 `Ctrl` 快捷键
- 代码中监听的按键是 `'ctrl'`（左 Ctrl 和右 Ctrl 均可），如果需要区分左右键，可改用 `'left ctrl'` 或 `'right ctrl'`

## 📄 许可证

本项目仅供学习和研究使用，请遵守 Qwen 系列模型的开源协议。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进本项目。

## 📧 联系方式

如有问题，请通过 GitHub Issues 联系。



