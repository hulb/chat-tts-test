import numpy as np
import pyaudio
import wave
import ChatTTS

# 初始化 ChatTTS
chat = ChatTTS.Chat()
chat.load_models()

# 文本列表
texts = ["给你一瓶魔法药水，带我一起去月球旅行。怎么样[laugh],我说的[laugh]还不错吧?",]

# 生成音频数据
wavs = chat.infer(texts, use_decoder=True)

# 保存音频数据到 WAV 文件
output_file = "output.wav"
sample_rate = 24000

# 将生成的音频数据转换为 16-bit PCM 格式
audio_data = (np.array(wavs[0]) * 32767).astype(np.int16)

# 使用 wave 模块保存音频文件
with wave.open(output_file, 'w') as wav_file:
    wav_file.setnchannels(1)  # 单声道
    wav_file.setsampwidth(2)  # 16-bit PCM
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(audio_data.tobytes())

# 播放音频文件
def play_audio(file_path):
    # 打开音频文件
    wf = wave.open(file_path, 'rb')

    # 初始化 PyAudio
    p = pyaudio.PyAudio()

    # 打开音频流
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # 读取并播放音频数据
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    # 关闭音频流
    stream.stop_stream()
    stream.close()
    p.terminate()

# 播放生成的音频文件
play_audio(output_file)
