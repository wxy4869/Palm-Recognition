import numpy as np
import pandas as pd

import cv2
import skimage.feature


GRIDX = 4
GRIDY = 4
H = int(128 / GRIDX)
W = int(128 / GRIDY)
P = 8
R = 1
HISTSIZE = P * (P - 1) + 2 + 1


# 加载特征
def load_lbp(dataset):
    return np.load('dataset/%s/feature/lbp.npy' % dataset, allow_pickle=True)
    

# 计算 LBPH
def cal_lbph(file):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    lbp_image = skimage.feature.local_binary_pattern(image, P, R, method='uniform')  # 获得 LBP 图像
    lbp_image = lbp_image.astype(np.uint8)
    lbph = []
    for i in range(GRIDX):  # 对图像进行分割
        for j in range(GRIDY):
            tmp_image = lbp_image[i * H : (i + 1) * H , j * W : (j + 1) * W]
            hist = cv2.calcHist([tmp_image], [0], None, [HISTSIZE], [0, HISTSIZE])  # 计算直方图
            hist = cv2.normalize(hist, hist)
            lbph.append(hist)
    lbph = np.array(lbph).squeeze()
    lbph = lbph.flatten()
    return lbph


# 处理 CASIA 数据集
def CASIA():
    path = 'dataset/CASIA/roi/'
    df = pd.read_csv('dataset/CASIA/data_split.csv')
    feature = []
    for file in df['file_name']:
        feature.append(cal_lbph(path + file))
    np.save('dataset/CASIA/feature/lbp.npy', np.array(feature))


# 处理 tongji 数据集
def tongji():
    path = 'dataset/tongji/roi/'
    df = pd.read_csv('dataset/tongji/data_split.csv')
    feature = []
    for file in df['file_name']:
        feature.append(cal_lbph(path + file))
    np.save('dataset/tongji/feature/lbp.npy', np.array(feature))


def main():
    CASIA()
    tongji()
    

if __name__ == '__main__':
    main()
