import numpy as np
import sounddevice as sd
import keyboard
import time
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from qwen_asr import Qwen3ASRModel

# 语音识别模型本地路径
ASR_MODEL_PATH = r"D:\modelscope\hub\models\Qwen\Qwen3-ASR-0___6B"
# 对话大模型本地路径
CHAT_MODEL_PATH = r"D:\modelscope\hub\models\Qwen\Qwen1___5-0___5B-Chat"

# 音频配置
AUDIO_BUFFER = []
SAMPLE_RATE = 16000  # 语音模型要求 16k 采样率
device = "cpu"  # 运行设备（有GPU改为cuda）

print("正在加载 语音识别模型 Qwen3-ASR-0.6B...")
asr_model = Qwen3ASRModel.from_pretrained(ASR_MODEL_PATH)
print("正在加载 对话模型 Qwen1.5-0.5B-Chat...")
chat_model = AutoModelForCausalLM.from_pretrained(
    CHAT_MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_PATH)
print("所有模型加载完成！")

def audio_callback(in_data, frame_count, time_info, status_flags):
    # if status_flags:print(in_data)
    # # in_data声道获取的数据
    # # [:,0]“:”表示所有采样点；0表示第0个声道(左声道)
    # # AUDIO_BUFFER.append(...) 将复制的音频数据添加到缓冲区列表中
    # AUDIO_BUFFER.append(in_data[:,0].copy())

    # 取左声道数据，存入缓冲区
    AUDIO_BUFFER.append(in_data[:, 0].copy())
    # 打印异常状态（调试用）
    if status_flags:
        print(f"音频状态警告: {status_flags}")

#你是一个乐于助人的智能助手，请用中文简洁回答。如果用户说的话暗示或明示你打开灯光或用户提到看不清或很黑，那就只用回复开灯两个字即可
def chat_with_local_model(tokenizer, model, user_text):
    # 构造对话格式
    messages = [
        {"role": "system", "content": "请用中文简洁回答。如果用户说的话暗示或明示你打开灯光或用户提到看不清或很黑，那就只用回复开灯两个字即可，如果用户说的话暗示或明示你关闭灯光或用户提到很刺眼或很亮，那就只用回复关灯两个字即可"},
        {"role": "user", "content": user_text}
    ]
    # 应用模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # 转换为模型输入
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    # 生成回答
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    # 截取新生成的内容
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    # 解码输出
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# ===================== 主程序 =====================
if __name__ == '__main__':
    # 启动音频流
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        callback=audio_callback
    )
    stream.start()
    # 提示信息
    print("按住【Ctrl】说话 → 松开识别 → AI回答")
    print("按【ESC】退出程序")

    while True:
        # ESC 退出程序
        if keyboard.is_pressed('esc'):
            stream.stop()
            stream.close()
            print("\n程序已退出！")
            break

        # 按住 Ctrl 开始录音
        if keyboard.is_pressed('ctrl'):
            print('=======录音中=======')
            AUDIO_BUFFER.clear()  # 清空上一次音频
            start_time = time.time()
            stop_Sign = False

            # 按住期间持续录音
            while keyboard.is_pressed('Ctrl'):
                time.sleep(0.03)  # 因为my_stream一直在start状态，所以回调函数(声音数据获取)会每隔0.03秒执行
                if time.time() - start_time > 10:
                    stop_Sign = True
                    break

            if len(AUDIO_BUFFER) == 0:
                print("未检测到声音，请重新说话！")
                continue

            if stop_Sign:
                print('录入已达到10秒，停止语音输入！！！')

            # audio_np_array = np.array(AUDIO_BUFFER, dtype=np.float32)
            audio_np_array = np.concatenate(AUDIO_BUFFER, axis=0)  # axis操作唯独
            print(f'录音时长为{len(audio_np_array) / SAMPLE_RATE:.2f}秒')
            print('=======处理中=======')
            print('=====请耐心等待=====')
            # ===================== 语音转文字 =====================
            print("识别中...")
            # torch.set_grad_enabled(False)
            # transcribe表示变换(此处音转文)
            result = asr_model.transcribe(audio=(audio_np_array, SAMPLE_RATE),language='Chinese')  # (需要处理的元组，采样频率，转换的语种)
            # torch.set_grad_enabled(True)
            if len(result) > 0:
                text = result[0].text

            print(text)
            ai_response = chat_with_local_model(tokenizer, chat_model, text)
            print(f"AI回答：{ai_response}")