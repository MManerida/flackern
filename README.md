# flackern 抓取音频-分析音频-创造音频
numpyschokolade:用numpy读取wav录音，进行傅立叶转换，cnn向前传播测试（Forward Pass Test），验证数据能不能穿透网络结构。 

trian_voice_cnn.py: 这段代码，我们自行设置了六个分类，soundfile读取音频得到numpy数组和采样率，转成tensor，再转成梅尔频谱（音频到图像的转换），在cnn里面进行特征提取再分类（这里是吃算力的地方，只设置了六类，），选择最高分的类别编号，再查出对应的单词名称，输出分类结果。

Predict_voice.py:

# ps:
numpy:
cnn:
