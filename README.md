# flackern 抓取音频-分析音频-创造音频
1.numpyschokolade:读取wav录音得到numpy，进行傅立叶转换，cnn向前传播测试（Forward Pass Test），验证数据能不能穿透网络结构。 

2.trian_voice_cnn.py: 这段代码，我们自行设置了六个分类，soundfile读取音频得到numpy数组和采样率，转成tensor，再转成梅尔频谱（音频到图像的转换），在cnn里面进行特征提取再分类（这里是吃算力的地方，只设置了六类，以单词形式），选择最高分的类别编号，再查出对应的单词名称，输出分类结果。

Predict_voice.py:

# ps:？《云端炼丹》
0.https://www.kaggle.com/notebooks；https://colab.research.google.com/

1.numpy: numpy是读取音频与pytorch的中间格式，soundfile默认读取出来numpy，torch_from_numpy()转成tensor

2.cnn:池化层（降维），最大池化：区域内最大值，最强信号，cnn中间层提取特征；平均池化：区域内平均值，cnn最后基层（分类汇总），尺寸对其

3.权重：如当写了nn.Conv2d() or nn.Linear()时，pytorch会在后台自动生成一个巨大的数字矩阵，这个矩阵里的每一个数字就是一个权重。在AI训练没开始时，权重是随机的，乱的；此时输入一张yes图，模型猜错，反向传播（梯度在这里，是向量，用链式法则）根据误差（loss）计算出哪一个权重错误，优化器（optimizer）进行调整。-optimizer：新的权重 = 旧的权重 - (学习率 × 梯度)

4.函数本身不规定是测试还是验证，取决于传进来哪个loader：evaluate(model, val_loader, criterion, device)# 验证；evaluate(model, test_loader, criterion, device)# 测试

5.候选类别：模型从所有候选选项中，选出最可能，是，的单词，并给出高分。
