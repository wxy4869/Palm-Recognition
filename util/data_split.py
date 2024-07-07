import os
import random
import numpy as np
import pandas as pd

from itertools import combinations
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

import param


random.seed(42)
np.random.seed(42)


# 产生 size 组 (sample_1, sample_2, label) 数据对用于多识图算法, 正例和反例各占一半
def pos_neg(size, dataset, test_fold, data_type='train'):
    """
    size: 正例反例数据对的总数量
    dataset: 数据集名称, CASIA 或 tongji
    test_fold: 在 k-fold 中, 训练集是哪一组
    type: train 或 test
    """

   # 用于产生数据对的原始数据
    df = pd.read_csv('dataset/%s/data_split.csv' % dataset)
    if data_type == 'train':
        data = df[(df['is_train'] == True) & (df['k_fold'] != test_fold)]
    elif data_type == 'val':
        data = df[(df['is_train'] == True) & (df['k_fold'] == test_fold)]
    elif data_type == 'test':
        data = df[(df['is_train'] == False)]
    n_classes = data['class_id'].unique()  # 掌纹类别标签
    n_examples = data['class_id'].value_counts(sort=False)  # 每种掌纹类别中的样本数量
    data = data.reset_index()
    data['index'] = data.index
    start_pos = data.groupby('class_id').first().reset_index().set_index('class_id')['index']  # 每种掌纹类别的开始位置

    # 从每个类别所有样本的排列组合中选取 size // 2 个, 构成正例
    pos_data = []
    for i in n_classes:  # 每个类别所有样本的排列组合
        tmp = list(combinations(range(n_examples[i]), 2))  
        tmp = [(x + start_pos[i], y + start_pos[i]) for x, y in tmp]
        pos_data.extend(tmp)
    pos_data = random.sample(pos_data, size // 2)

    # 从所有掌纹类别标签的组合中选取 size // 2 个, 构成反例
    n_classes_combination = list(combinations(n_classes, 2))  # 所有掌纹类别标签的组合
    neg_data = random.sample(n_classes_combination, size // 2)

    # 产生 size 个数据对
    out = pd.DataFrame(columns=['sample_1', 'sample_2', 'label'])
    for i in range(size):
        if i < size // 2:  # 前一半是正例
            idx_1, idx_2 = pos_data[i]
            sample_1 = data.loc[idx_1, 'file_name']
            sample_2 = data.loc[idx_2, 'file_name']
            label = 1
        else:  # 后一半是反例
            category_1, category_2 = neg_data[i - size // 2]
            idx_1 = np.random.randint(0, n_examples[category_1])
            idx_2 = np.random.randint(0, n_examples[category_2])
            sample_1 = data.loc[start_pos[category_1] + idx_1, 'file_name']
            sample_2 = data.loc[start_pos[category_2] + idx_2, 'file_name']
            label = -1
        out.loc[i] = [sample_1, sample_2, label]
    # out.to_csv('tmp.csv', index=False)
    return out


# 处理 CASIA 数据集
def train_test_CASIA():
    # 获取文件列表
    class_id = []
    file_name = []
    
    path = 'dataset/CASIA/roi/'
    dir = os.listdir(path)
    dir.sort()
    for file in dir:
        if file.split('.')[-1] != 'jpg':  # 排除 .DS_Store
            continue
        hand_type = file.split('_')[2]
        if hand_type == 'l':
            class_id.append(int(file.split('_')[0]) * 2 - 2)  # 0001_m_l 的类别为 0
        else:
            class_id.append(int(file.split('_')[0]) * 2 - 1)  # 0001_m_r 的类别为 1
        file_name.append(file)
    df = pd.DataFrame.from_dict({'class_id': class_id, 'file_name': file_name})
    df = df.sort_values(['class_id', 'file_name'])

    # 划分出训练集
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for i, (train_index, test_index) in enumerate(sss.split(df, class_id)):
        df.loc[train_index, 'is_train'] = True
        df.loc[test_index, 'is_train'] = False

    # 划分出 k-fold 验证集
    data = df[df['is_train'] == True].index
    label = df[df['is_train'] == True]['class_id']
    kf = StratifiedKFold(n_splits=param.K_FOLD, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kf.split(data, label)):
        df.loc[data[test_index], 'k_fold'] = i
    df.to_csv('dataset/CASIA/data_split.csv', index=False)


# 处理 tongji 数据集
def train_test_tongji():
    # 获取文件列表
    class_id = []
    file_name = []
    
    path = 'dataset/tongji/roi/session1/'
    dir = os.listdir(path)
    dir.sort()
    for file in dir:
        if file.split('.')[-1] != 'bmp':  # 排除 .DS_Store
            continue
        class_id.append((int(file[1 : 5]) - 1) // 10)
        file_name.append('session1/' + file)
    
    path = 'dataset/tongji/roi/session2/'
    dir = os.listdir(path)
    dir.sort()
    for file in dir:
        if file.split('.')[-1] != 'bmp':  # 排除 .DS_Store
            continue
        class_id.append((int(file[1 : 5]) - 1) // 10)
        file_name.append('session2/' + file)
    df = pd.DataFrame.from_dict({'class_id': class_id, 'file_name': file_name})
    df = df.sort_values(['class_id', 'file_name'])

    # 划分出训练集
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for i, (train_index, test_index) in enumerate(sss.split(df, class_id)):
        df.loc[train_index, 'is_train'] = True
        df.loc[test_index, 'is_train'] = False

    # 划分出 k-fold 验证集
    data = df[df['is_train'] == True].index
    label = df[df['is_train'] == True]['class_id']
    kf = StratifiedKFold(n_splits=param.K_FOLD, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kf.split(data, label)):
        df.loc[data[test_index], 'k_fold'] = i
    df.to_csv('dataset/tongji/data_split.csv', index=False)


def main():
    train_test_CASIA()
    train_test_tongji()


if __name__ == '__main__':
    main()
    # pos_neg(2000, 'CASIA', 0, data_type='train')
    