# flackern 抓取视频-分析视频-创造视频
音频：

1.numpyschokolade:读取wav录音得到numpy，进行傅立叶转换，cnn向前传播测试（Forward Pass Test），验证数据能不能穿透网络结构。 

2.trian_voice_cnn.py: 这段代码，我们自行设置了六个分类，soundfile读取音频得到numpy数组和采样率，转成tensor（1维，点，1s标准格式转换成16000Hz，【16000】），再转成梅尔频谱（音频到图像的转换，得到，时间帧frame+频率Hz，这2维，【80，100】），（batch+通道数+高+宽，变成4维张量【16，1，64，101】进入cnn，cnn处理特定数据如图片要求4维张量，图像领域是一个转换器）在cnn里面进行特征提取再分类（这里是吃算力的地方，只设置了六类，以单词形式），选择最高分的类别编号，再查出对应的单词名称，输出分类结果。

视频：

1.classify_frames.py:这是单张图片如何进入预训练分类模型，并把结果写成表格的代码。按frames.csv里的路径打开图片，转换成RGB三通道；prepare_image依次进行缩放、中心裁剪、转成 Tensor 和标准化，得到形状为 [3, 224, 224] 的数据。.unsqueeze(0) 增加批次维度，变成 [1, 3, 224, 224]，表示这次给模型输入1张图片。程序加载别人已训练好的ResNet18_Weights.DEFAULT权重来预测，类别名称取自 weights.meta["categories"]，、对应ImageNet的1000个类别；最后选出得分最高的 前3类写入CSV。全是预制菜。

# ps:
0.https://www.kaggle.com/notebooks；https://colab.research.google.com/ ？《云端炼丹》

1.numpy: numpy是读取音频与pytorch的中间格式，soundfile默认读取出来numpy，torch_from_numpy()转成tensor

2.cnn:里面进行：卷积-relu（激活）-池化，重复堆叠，展平，最后分类；池化层（降维），最大池化：区域内最大值，最强信号，cnn中间层提取特征；平均池化：区域内平均值，cnn最后基层（分类汇总），尺寸对其；embedding词嵌入，一个词用什么特征表示比较好。

3.nn.Linear（）：全连接层，把卷积层输出的多维向量展平成一维，是线性函数；nn.Relu()：激活函数，是非线性部分（热传导/扩散模型相关概念）。

4.权重：如当写了nn.Conv2d() or nn.Linear()时，pytorch会在后台自动生成一个巨大的数字矩阵，这个矩阵里的每一个数字就是一个权重。在AI训练没开始时，权重是随机的，乱的；此时输入一张yes图，模型猜错，反向传播（梯度在这里，是向量，用链式法则）根据误差（loss）计算出哪一个权重错误，优化器（optimizer）进行调整。-optimizer：新的权重 = 旧的权重 - (学习率 × 梯度)。
算子（operator）定义了计算的动作或规则，如矩阵乘法，卷积，全连接。它规定了数据该怎么流动，怎么运算，是抽象的。
权重（weight）是具体存储在内存中的数值。

5.函数本身不规定是测试还是验证，取决于传进来哪个loader：evaluate(model, val_loader, criterion, device)# 验证；evaluate(model, test_loader, criterion, device)# 测试

6.候选类别：模型从所有候选选项中，选出最可能，是，的单词，并给出高分。

7.Adam：优化器，具体公式在pytorch库内部实现。它会利用，本次计算的梯度；梯度的历史移动平均；梯度平方的历史移动平均；学习率及其他配置；来结算参数更新量，并进行偏差修正。

8.混淆矩阵：把“正确答案是什么”和“实际判成什么”交叉统计，让我们既看到错了多少，也看清哪些类别最容易被混淆，从而有针对性地复核、改进和评测。
<img width="1794" height="746" alt="1795067331522" src="https://github.com/user-attachments/assets/904f5e21-2eca-4027-b3f8-28b6596a7823" />
<img width="2752" height="1536" alt="混淆矩阵" src="https://github.com/user-attachments/assets/41882c77-1fbb-462a-bba3-82e43a085cc2" />

