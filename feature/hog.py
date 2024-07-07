import numpy as np
import pandas as pd

import cv2
import skimage.feature


# 加载特征
def load_hog(dataset):
    return np.load('dataset/%s/feature/hog.npy' % dataset, allow_pickle=True)
    

# 计算 HOG
def cal_hog(file):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    image = cv2.equalizeHist(image)
    hog = skimage.feature.hog(image, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2), feature_vector=True)
    return hog


# 处理 CASIA 数据集
def CASIA():
    path = 'dataset/CASIA/roi/'
    df = pd.read_csv('dataset/CASIA/data_split.csv')
    feature = []
    for file in df['file_name']:
        feature.append(cal_hog(path + file))
    np.save('dataset/CASIA/feature/hog.npy', np.array(feature))


# 处理 tongji 数据集
def tongji():
    path = 'dataset/tongji/roi/'
    df = pd.read_csv('dataset/tongji/data_split.csv')
    feature = []
    for file in df['file_name']:
        feature.append(cal_hog(path + file))
    np.save('dataset/tongji/feature/hog.npy', np.array(feature))


def main():
    CASIA()
    tongji()
    

if __name__ == '__main__':
    main()
