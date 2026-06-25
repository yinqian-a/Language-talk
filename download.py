from qwen_asr import Qwen3ASRModel

# 直接使用官方模型名称，不需要本地路径
ASR_MODEL_PATH = "Qwen/Qwen3-ASR-0.6B"

print("正在下载并加载语音识别模型...")
asr_model = Qwen3ASRModel.from_pretrained(
    ASR_MODEL_PATH,
    trust_remote_code=True
)
print("模型加载成功！")