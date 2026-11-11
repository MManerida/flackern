from pathlib import Path
from datetime import datetime
import random

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader


# ==================================================
# 1. 配置：可修改
# ==================================================
WORDS = ["yes", "no", "up", "down", "go", "stop"]

SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 400
HOP_LENGTH = 160

BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 0.001
SEED = 42

# 每个类别最多使用多少条；设为 None 表示使用全部
TRAIN_PER_WORD = 300
VAL_PER_WORD = 100
TEST_PER_WORD = 100

# 先用 CPU 跑通；之后可改为 True 尝试 Mac GPU
USE_MPS = False

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = (
    PROJECT_DIR
    / "test_data"
    / "SpeechCommands"
    / "speech_commands_v0.02"
)


# ==================================================
# 2. 按官方清单划分数据
# ==================================================
def read_split_list(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少官方划分清单：{path}\n"
            "请检查 DATA_DIR 是否指向完整数据集。"
            "本程序不会自动重新下载数据。"
        )

    return {
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def build_splits():
    val_names = read_split_list(DATA_DIR / "validation_list.txt")
    test_names = read_split_list(DATA_DIR / "testing_list.txt")

    if val_names & test_names:
        raise ValueError("验证清单和测试清单存在重复文件，请检查数据集。")

    splits = {"train": [], "val": [], "test": []}
    limits = {
        "train": TRAIN_PER_WORD,
        "val": VAL_PER_WORD,
        "test": TEST_PER_WORD,
    }
    rng = random.Random(SEED)#为了减少数据量，打乱内部每组，每个单词类别

    for label_id, word in enumerate(WORDS):
        files = sorted((DATA_DIR / word).glob("*.wav"))
        if not files:
            raise FileNotFoundError(f"没有找到录音：{DATA_DIR / word}")

        word_splits = {"train": [], "val": [], "test": []}

        for path in files:
            relative_name = path.relative_to(DATA_DIR).as_posix()

            if relative_name in test_names:
                split = "test"
            elif relative_name in val_names:
                split = "val"
            else:
                split = "train"

            word_splits[split].append((path, label_id))

        for split in splits:
            samples = word_splits[split]
            rng.shuffle(samples)#在各自集合内部选择本次使用的部分数据

            limit = limits[split]
            if limit is not None:
                samples = samples[:limit]

            if not samples:
                raise ValueError(f"{word} 的 {split} 集没有样本。")

            splits[split].extend(samples)

        print(
            f"{word:>4}："
            f"官方训练 {len(word_splits['train'])}，"
            f"验证 {len(word_splits['val'])}，"
            f"测试 {len(word_splits['test'])}"
        )

    # 额外检查：本次选中的三个集合不能共享说话人
    def speaker_ids(samples):
        return {
            path.stem.split("_nohash_")[0]
            for path, _ in samples
        }

    speakers = {
        name: speaker_ids(samples)
        for name, samples in splits.items()
    }

    for left, right in [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]:
        if speakers[left] & speakers[right]:
            raise ValueError(
                f"{left} 和 {right} 存在重复说话人，请检查划分。"
            )

    print("\n本次实际使用的录音数：")
    for name, samples in splits.items():
        print(f"  {name}: {len(samples)}")

    return splits


# ==================================================
# 3. SoundFile 读取音频 → 梅尔频谱
# ==================================================
class VoiceDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

        self.to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=80,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        # 不调用 torchaudio.load，因此不走 TorchCodec
        audio_array, sr = sf.read(
            str(path),
            dtype="float32",#用32位浮点数表示读出的音频数值
            always_2d=True,
        )

        if audio_array.shape[0] == 0:
            raise ValueError(f"录音为空：{path}")

        # [采样点数, 声道数] → [声道数, 采样点数] 这里调整数据的排列方式
        waveform = torch.from_numpy(audio_array.T.copy())#改变数据容器类型，得到的tensor也会保留对应浮点数类型
        # 多声道合并成单声道
        waveform = waveform.mean(dim=0, keepdim=True)

        # 若采样率不同，重采样到统一采样率
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, sr, SAMPLE_RATE
            )

        # 统一为一秒：短的补零，长的截断（波形）
        waveform = waveform[:, :SAMPLE_RATE]
        missing = SAMPLE_RATE - waveform.shape[1]
        if missing > 0:
            waveform = F.pad(waveform, (0, missing))

        mel_db = self.to_db(self.to_mel(waveform))

        # 每条音频独立标准化，不使用其他集合的数据统计量
        mel_db = (
            (mel_db - mel_db.mean())
            / mel_db.std(unbiased=False).clamp_min(1e-6)
        )

        # 返回：频谱 Tensor、正确的类别编号
        return mel_db, label


# ==================================================
# 4. CNN：频谱 → 六个类别的得分
# ==================================================
class VoiceCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
#第一部分：特征提取，池化层，两次最大池化，一次自适应平均池化
        self.features = nn.Sequential(#特征提取模块
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),#池化

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
#第二部分：分类
        self.classifier = nn.Sequential(
            nn.Flatten(),#把tensor按维度展平，把多维特征图转换成一维向量
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),#表示训练时，每个元素有20%的概率被置为0.处理本次向前计算中的中间特征值，减少模型过度依赖
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ==================================================
# 5. 评估：不计算梯度、不更新参数
# ==================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()#把模型切换到评估模式，关掉dropout的随机屏蔽行为
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)#及上，让频谱，标签与模型处于兼容设备上，方便模型计算和损失比较

        scores = model(features)#计算得分
        loss = criterion(scores, labels)#对照答案，计算损失

        count = labels.size(0)#统计当前批次有多少条录音，读取实际数量
        total_loss += loss.item() * count#累计损失
        correct += (scores.argmax(dim=1) == labels).sum().item()#累计预测正确的数量
        total += count

    return total_loss / total, correct / total


# ==================================================
# 6. 主流程：训练 → 验证选模型 → 最后测试
# ==================================================
def main():
    random.seed(SEED)
    np.random.seed(SEED)#numpy 随机数生成器的种子
    torch.manual_seed(SEED)#从数据集抽取样本，初始化网络参数，打乱训练样本顺序，dropout随机丢弃部分特征

    device = torch.device(
        "mps"
        if USE_MPS and torch.backends.mps.is_available()
        else "cpu"
    )
    print("使用设备：", device)
    print("数据目录：", DATA_DIR)

    splits = build_splits()#划分数据，创建数据集对象

    train_set = VoiceDataset(splits["train"])
    val_set = VoiceDataset(splits["val"])
    test_set = VoiceDataset(splits["test"])#当程序要第某个样本时，怎样读取录音，转成频谱，并返回正确标签

    # 每次取出一批样本，把它们组合成批量tensor，训练时打乱录音顺序，以下是3个加载器
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,#控制额外启动多少个工作进程，帮忙加载和处理数据，=0的意思是不启动额外的数据加载进程，由主进程完成
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,#验证集，测试集使用
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # 开始训练前，先验证一条音频是否能正常读取
    sample, label = train_set[0]
    print("\n第一条训练样本：", WORDS[label])
    print("单条频谱形状：", tuple(sample.shape))
    # 创建模型，损失函数和优化器
    model = VoiceCNN(len(WORDS)).to(device)#从频谱计算六个类别得分
    criterion = nn.CrossEntropyLoss()#比较得分与正确标签，计算损失
    optimizer = torch.optim.Adam(#优化器选择 Adam算法
        model.parameters(),#把模型的可训练参数交给它管理，包括卷积层和全连接层的权重，偏置
        lr=LEARNING_RATE,#设置学习率，影响更新步幅
    )

    # 每次运行单独保存，避免覆盖上一次实验
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = PROJECT_DIR / "checkpoints" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_voice_cnn.pt"

    # 保存本次使用的文件清单，方便复查
    for name, samples in splits.items():
        text = "\n".join(
            path.relative_to(DATA_DIR).as_posix()
            for path, _ in samples
        )
        (output_dir / f"{name}_files.txt").write_text(
            text, encoding="utf-8"
        )

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):#外层循环：训练多轮
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for step, (features, labels) in enumerate(train_loader, 1):#内层循环：逐批训练
            features = features.to(device)
            labels = labels.to(device)
            #训练核心
            optimizer.zero_grad()#清除上一批梯度
            scores = model(features)#scores不是单词，是一条录音对应的六个数
            loss = criterion(scores, labels)

            loss.backward()#反向传播，计算并积累梯度
            optimizer.step()#前面被创建成了Adam，这里执行的是Adam的更新规则；如果前面创建的是SGD，这里更新的就是SGD

            count = labels.size(0)
            train_loss_sum += loss.item() * count
            train_correct += (
                scores.argmax(dim=1) == labels#最大值索引
            ).sum().item()#predicted_ids=scores.argmax(dim=1),is_correct=predicted_ids==labels
            train_total += count#统计这批录音有多少条预测正确

            if step % 20 == 0 or step == len(train_loader):
                print(
                    f"第 {epoch}/{EPOCHS} 轮，"
                    f"批次 {step}/{len(train_loader)}"
                )
        #每轮结束用验证集评估
        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"训练 Loss={train_loss_sum / train_total:.4f}，"
            f"训练准确率={train_correct / train_total:.1%} | "
            f"验证 Loss={val_loss:.4f}，"
            f"验证准确率={val_accuracy:.1%}"
        )

        # 只根据验证集选择模型，不看测试集成绩
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                    "words": WORDS,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "sample_rate": SAMPLE_RATE,
                    "n_mels": N_MELS,
                    "n_fft": N_FFT,
                    "hop_length": HOP_LENGTH,
                },
                checkpoint_path,
            )
            print("已保存验证集表现最好的模型。\n")

    # 加载选出的最佳模型，再进行最终测试
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    test_loss, test_accuracy = evaluate(
        model, test_loader, criterion, device
    )

    print("\n训练结束")
    print("选中的模型来自第", checkpoint["epoch"], "轮")
    print(f"测试 Loss：{test_loss:.4f}")
    print(f"测试准确率：{test_accuracy:.1%}")
    print("模型保存位置：", checkpoint_path)


if __name__ == "__main__":
    main()
