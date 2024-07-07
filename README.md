# Palm-Recognition

> 四春 毕设



## 数据集

- <a href="http://biometrics.idealtest.org/#/">CASIA</a>
- <a href="https://cslinzhang.github.io/ContactlessPalm/">Tongji</a>



## 文件结构

```shell
.
├── dataset		# 存放数据集原始图片、ROI 提取结果、特征提取结果
├── feature		# 特征提取相关代码
├── log			# 模型运行日志文件
├── model		# 模型代码
├── result		# 模型结果
├── roi			# ROI 提取相关代码 
├── ui			# 结果可视化
└── util		# 工具
```



## 运行方式

- 下载数据集
- 运行 `roi/roi.py` 进行 ROI 提取
- 分别运行 `feature` 文件夹中的 Python 文件，进行不同特征的提取
- 运行 `model` 文件夹中的任意 Python 文件，进行掌纹识别
- 掌纹识别准确率可在 `log` 文件夹中查看，不同方法准确率与运行时间的关系可在 `util/time.ipynb` 中查看



## 参考

**论文：**

- <a href="https://zhuanlan.zhihu.com/p/488323541">知乎：新手看计算机领域论文前需要了解的知识及其注意事项</a>

**ROI 提取：**

- <a href="https://github.com/yyaddaden/PROIE">GitHub：PROIE</a>
- <a href="https://github.com/faceluhao/PalmprintROIExtraction">GitHub：PalmprintROIExtraction</a>
- <a href="https://blog.csdn.net/jiangpeng59/article/details/109542852">CSDN：python-cv2: 求直线和轮廓的交点</a>
- <a href="https://blog.csdn.net/weixin_40522801/article/details/106454622">CSDN：Opencv：图像旋转，cv2.getRotationMatrix2D 和 cv2.warpAffine 函数</a>
- <a href="https://www.jianshu.com/p/90572b07e48f">简书：OpenCV Python 实现旋转矩形的裁剪</a>

**数据集划分：**

- <a href="https://zhuanlan.zhihu.com/p/48976706">知乎：训练集、验证集和测试集</a>
- <a href="https://zhuanlan.zhihu.com/p/114391603">知乎：【机器学习】训练集，验证集，测试集；验证和交叉验证</a>
- <a href="https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d">Medium：One Shot Learning with Siamese Networks using Keras</a>

**LBP 特征：**

- <a href="https://github.com/1044197988/Python-Image-feature-extraction">GitHub：Python-Image-feature-extraction</a>
- <a href="https://github.com/Kasra1377/lbp-face-recognition/tree/master">GitHub：lbp-face-recognition</a>
- <a href="https://blog.csdn.net/wzyaiwl/article/details/107614819">CSDN：Python 实现 LBP 算法</a>
- <a href="https://senitco.github.io/2017/06/12/image-feature-lbp/">个人博客：图像特征提取之 LBP 特征</a>

**HOG 特征：**

- <a href="https://blog.csdn.net/hujingshuang/article/details/47337707">CSDN：【特征检测】HOG 特征算法</a>
- <a href="https://blog.csdn.net/qq_42312574/article/details/132061458">CSDN：【计算机视觉】关于图像处理的一些基本操作</a>

**ResNet 特征：**

- <a href="https://github.com/josharnoldjosh/Resnet-Extract-Image-Feature-Pytorch-Python/tree/master">GitHub：Resnet-Extract-Image-Feature-Pytorch-Python</a>
- <a href="https://zhuanlan.zhihu.com/p/376291615">知乎：ImageNet 预训练</a>
- <a href="https://zhuanlan.zhihu.com/p/577262080">知乎：PyTorch 搭建预训练AlexNet、DenseNet、ResNet、VGG 实现猫狗图片分类</a>
- <a href="https://www.kaggle.com/code/hirotaka0122/feature-extraction-by-resnet/notebook">Kaggle：Feature Extraction by ResNet</a>

**多视图：**

- <a href="https://blog.csdn.net/weixin_36378508/article/details/114898102">CSDN：求导方式</a>
- <a href="https://zhuanlan.zhihu.com/p/508625294">知乎：Python 计算余弦相似性（cosine similarity）方法汇总</a>
- <a href="https://zhuanlan.zhihu.com/p/159244903">知乎：点积相似度、余弦相似度、欧几里得相似度</a>
- <a href="https://levy96.github.io/articles/python-import.html">个人博客：ImportError: No module named *** 问题？——理解绝对导入和相对导入</a>

**分类：**

- <a href="https://blog.csdn.net/Dream_angel_Z/article/details/49406573">CSDN：Scikit-learn Preprocessing 预处理</a>
- <a href="https://blog.csdn.net/qq_39856931/article/details/106342911">CSDN：【机器学习】两种方法实现 KNN 算法：纯 Python 实现 + 调用 Sklearn 库实现（使用 Iris 数据集）</a>
- <a href="https://blog.csdn.net/Dr_maker/article/details/121985749">CSDN：ROC 曲线及 EER 介绍</a>
- <a href="https://blog.csdn.net/u011501388/article/details/78242856">CSDN：模式识别分类器评价指标之 CMC 曲线</a>
- <a href="https://zhuanlan.zhihu.com/p/573964757">知乎：ROC 曲线</a>
- <a href="https://juejin.cn/post/7088561002450518023">稀土掘金：分类评价指标：TP、TN、FP、FN、Recall，以及人脸识别评价指标 TAR、FAR、FRR</a>

**其它：**

- <a href="https://github.com/ruofei7/Palmprint_Recognition">GitHub：Palmprint_Recognition</a>
- <a href="https://github.com/auduongtansang/PalmprintRecognition/tree/master">GitHub：PalmprintRecognition</a>
- <a href="https://github.com/AdrianUng/palmprint-feature-extraction-techniques">GitHub：palmprint-feature-extraction-techniques</a>
- <a href="https://github.com/szaboa/PalmPrintAuthentication">GitHub：PalmPrintAuthentication</a>
- <a href="https://github.com/goodrahstar/Palmprint_Recognition">GitHub：Palmprint_Recognition</a>
- <a href="https://github.com/1119231393/03">GitHub：03 年掌纹识别实现</a>

