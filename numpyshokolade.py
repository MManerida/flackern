from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt


# 1. 以当前脚本所在文件夹为起点
project_dir = Path(__file__).resolve().parent

import random

# 想观察哪个单词，就修改这里
WORD = "bird"

# 找到这个单词的文件夹
word_dir = (
    project_dir
    / "test_data"
    / "SpeechCommands"
    / "speech_commands_v0.02"
    / WORD
)

# 收集这个文件夹里的 WAV 录音
audio_files = sorted(
    path for path in word_dir.glob("*.wav")
    if path.is_file()
)

# 没找到录音时，给出明确提示
if not audio_files:
    raise FileNotFoundError(
        f"这个文件夹中没有找到 WAV 录音：{word_dir}"
    )

# 随机选择一条录音的路径
audio_path = random.choice(audio_files)

print("本次选择的单词：", WORD)
print("准备读取文件：", audio_path)


# 2. 直接用 SoundFile 读取 WAV，绕过 TorchCodec
# always_2d=True：即使只有一个声道，也返回二维数组
# 返回形状：[采样点数, 声道数]
audio_array, sample_rate = sf.read(
    str(audio_path),
    dtype="float32",
    always_2d=True,
)

if audio_array.shape[0] == 0:
    raise ValueError("音频文件没有采样数据。")


# 3. NumPy 数组 → PyTorch Tensor
# 转置后形状：[声道数, 采样点数]
waveform = torch.from_numpy(audio_array.T.copy())

# 如果有多个声道，取平均，合并成单声道
waveform = waveform.mean(dim=0, keepdim=True)

# Speech Commands 的文件夹名就是单词标签
label = audio_path.parent.name

print("单词：", label)
print("采样率：", sample_rate)
print("波形形状：", tuple(waveform.shape))


# 4. 波形 → 梅尔功率频谱 stft把声音分成许多短片段，分析每一段的频率组成，再按时间排列，代码里的参数在控制这些步骤
# sample_rate	告诉工具声音的采样率，用于建立正确的频率刻度
# n_fft=1024	每次傅里叶变换使用的点数
# win_length=1024	每个分析窗口包含的采样点数
# hop_length=512	相邻窗口向前移动的采样点数，因此窗口之间有重叠
# n_mels=64	将频谱汇总成 64 个梅尔频带
# power=2.0	对傅里叶变换结果的幅值取平方，得到功率
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    win_length=1024,
    hop_length=512,
    n_mels=64,
    power=2.0,
)

mel = mel_transform(waveform)

# 功率 → 分贝刻度，方便观察强弱差异
db_transform = torchaudio.transforms.AmplitudeToDB(
    stype="power",
    top_db=80,
)
mel_db = db_transform(mel)

print("梅尔频谱形状：", tuple(mel_db.shape))
import torch.nn as nn

# 六个候选类别
WORDS = ["yes", "no", "up", "down", "go", "stop"]

# 创建一个小型 CNN
model = nn.Sequential(
    # 在频谱上寻找局部模式
    nn.Conv2d(
        in_channels=1,
        out_channels=16,
        kernel_size=3,
        padding=1,
    ),
    nn.ReLU(),
    nn.MaxPool2d(2),

    # 在前一层的特征上继续提取模式
    nn.Conv2d(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        padding=1,
    ),
    nn.ReLU(),

    # 汇总特征，同时保留粗略的频率、时间位置
    nn.AdaptiveAvgPool2d((4, 4)),
    nn.Flatten(),

    # 为每个单词输出一个得分
    nn.Linear(32 * 4 * 4, len(WORDS)),
)

# 增加批量维度
inputs = mel_db.unsqueeze(0)

# 对这个样本做简单标准化
inputs = (inputs - inputs.mean()) / (
    inputs.std(unbiased=False) + 1e-6
)

# 这里只测试数据能否通过网络，不进行训练
model.eval()
with torch.no_grad():
    scores = model(inputs)

print("CNN 输入形状：", tuple(inputs.shape))
print("CNN 输出形状：", tuple(scores.shape))
print("各类别的原始得分：", scores)


# 5. 画图
# 去掉单声道维度，得到 [梅尔频带数, 时间帧数]
image = mel_db.squeeze(0).cpu().numpy()

plt.figure(figsize=(8, 4))
plt.imshow(
    image,
    origin="lower",
    aspect="auto",
    cmap="viridis",
)
plt.title(f"Mel Spectrogram: {label}")
plt.xlabel("Time frame")
plt.ylabel("Mel band")
plt.colorbar(label="Power (dB)")
plt.tight_layout()

# 保存图片
output_path = project_dir / "speech_mel.png"
plt.savefig(output_path, dpi=150)
print("频谱图已保存到：", output_path)

plt.show()
