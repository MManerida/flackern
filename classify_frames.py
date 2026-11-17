from pathlib import Path#处理文件路径
import csv#读写csv清单

import torch
from PIL import Image#打开图片文件
from torchvision.models import resnet18, ResNet18_Weights#现成的模型和权重
from torchvision import transforms#图像预处理工具


TOP_K = 3#每张图片保留置信度最高的前3个类别

# 预训练模型自带的图像预处理标准：
# 短边缩放到 256、中心裁剪 224x224、转张量、按 ImageNet 的均值方差归一化
prepare_image = transforms.Compose([
    transforms.Resize(256),#短边缩放到256像素
    transforms.CenterCrop(224),#从中心裁剪224X224（模型要求的输入尺寸）
    transforms.ToTensor(),#图片到张量的转换，像素值缩放到0～1
    transforms.Normalize(#用imageNet的统计量做标准化
        mean=[0.485, 0.456, 0.406],#这里是resnet18当年用imagenet统计量训练，我们必须用同一组mean/std
        std=[0.229, 0.224, 0.225],
    ),
])


def main():
    frames_csv = Path("video_frames/frames.csv")#上一个抽帧脚本产出的清单
    output_csv = Path("video_frames/frames_predictions.csv")#加上了分类结果的清单

    with frames_csv.open(encoding="utf-8-sig") as file:#输出加上了分类结果的清单
        rows = list(csv.DictReader(file))

    print("待分类帧数：", len(rows))

    # 加载预训练权重；首次运行会自动下载模型文件
    weights = ResNet18_Weights.DEFAULT
    categories = weights.meta["categories"]#1000个类别名

    model = resnet18(weights=weights)#按照resnet18的结构创造模型，把下载好的权重装进去，切换到推理模式
    model.eval()

    results = []

    for row in rows:#逐张打开图片，统一转成rgb三通道
        image = Image.open(row["图片路径"]).convert("RGB")
        features = prepare_image(image).unsqueeze(0)#features特征，unsqueeze增加维度

        with torch.inference_mode():
            scores = model(features)
            probabilities = torch.softmax(scores, dim=1)[0]

        top = torch.topk(probabilities, TOP_K)

        record = {
            "时间_秒": row["目标时间_秒"],
            "图片路径": row["图片路径"],
        }

        for rank in range(TOP_K):
            class_id = int(top.indices[rank].item())
            probability = float(top.values[rank].item())

            record[f"类别{rank + 1}"] = categories[class_id]
            record[f"比例{rank + 1}"] = f"{probability:.1%}"

        results.append(record)#把编号译回类别名字

        print(
            f"第 {row['目标时间_秒']} 秒："
            f"{record['类别1']}（{record['比例1']}）"
        )

    field_names = list(results[0].keys())

    with output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(results)

    print("完整结果：", output_csv)


if __name__ == "__main__":
    main()