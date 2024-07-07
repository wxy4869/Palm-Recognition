import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from util import param


N_COMPONENTS = 100


# 加载特征
def load_lda(dataset):
    return np.load('dataset/%s/feature/lda.npy' % dataset, allow_pickle=True)


# 计算特征
def cal_lda(df, data, i):
    x_train = data[df[(df['is_train'] == True) & (df['k_fold'] != i)].index]
    y_train = df[(df['is_train'] == True) & (df['k_fold'] != i)]['class_id']
    lda = LDA(n_components=N_COMPONENTS)
    lda.fit(x_train, y_train)
    return lda.transform(data)


# 处理 CASIA 数据集
def CASIA():
    path = 'dataset/CASIA/roi/'
    df = pd.read_csv('dataset/CASIA/data_split.csv')
    data = []
    for file in df['file_name']:
        image = cv2.imread((path + file), cv2.IMREAD_GRAYSCALE)
        image = image.flatten()
        data.append(image)
    data = np.array(data)
    feature = []
    for i in range(param.K_FOLD):
        print(i)
        feature.append(cal_lda(df, data, i))
    np.save('dataset/CASIA/feature/lda.npy', np.array(feature))
    print('CASIA done')


# 处理 tongji 数据集
def tongji():
    path = 'dataset/tongji/roi/'
    df = pd.read_csv('dataset/tongji/data_split.csv')
    data = []
    for file in df['file_name']:
        image = cv2.imread((path + file), cv2.IMREAD_GRAYSCALE)
        image = image.flatten()
        data.append(image)
    data = np.array(data)
    feature = []
    for i in range(param.K_FOLD):
        print(i)
        feature.append(cal_lda(df, data, i))
    np.save('dataset/tongji/feature/lda.npy', np.array(feature))
    print('tongji done')


def main():
    CASIA()
    tongji()
    

if __name__ == '__main__':
    main()
