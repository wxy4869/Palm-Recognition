import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import datetime
import numpy as np
import pandas as pd

import torch
from itertools import combinations
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, classification_report

from feature.lbp import load_lbp
from feature.lda import load_lda
from feature.hog import load_hog
from feature.res import load_res
from util.data_split import pos_neg
from util.logger import Logger
from util import param


torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True


DATASET = 'CASIA'  # 采用的数据集
FEATURES = ['lbp', 'hog', 'res']  # 多视图采用的特征
SIZE_TRAIN = 3000  # 训练集正例反例数据对的总数量
SIZE_VAL = 500  # 验证集正例反例数据对的总数量
SIZE_TEST = 500  # 测试集正例反例数据对的总数量

Q = 100  # 多视图时, 将不同视图的特征维度统一为 Q; 单视图时, lbp 取 944, hog 取 324, res 取 1000 (划掉)
T = 50  # 迭代次数
K = 3  # 视图数
E = 1e-6  # 收敛误差
MU = 1  # 学习率
T_P = 0.8  # 阈值
T_N = 0.1  # 阈值
LAMBDA_1 = 0.01  # 正则化项系数
LAMBDA_2 = 1  # 正则化项系数

DEVICE = ('mps' if torch.backends.mps.is_available() else 'cpu')
if K == 1:
    LOG = Logger('log/reg/%s_k=%d_%s_size=%d_q=%d_t=%d_e=%f_mu=%f_tp=%f_tn=%f_lambda1=%f_lambda2=%f_knn=%d.log' % (DATASET, K, FEATURES[0], SIZE_TRAIN, Q, T, E, MU, T_P, T_N, LAMBDA_1, LAMBDA_2, param.N_NEIGHBORS), level='info')
else:
    LOG = Logger('log/reg/%s_k=%d_multi_size=%d_q=%d_t=%d_e=%f_mu=%f_tp=%f_tn=%f_lambda1=%f_lambda2=%f_knn=%d.log' % (DATASET, K, SIZE_TRAIN, Q, T, E, MU, T_P, T_N, LAMBDA_1, LAMBDA_2, param.N_NEIGHBORS), level='info')


# 标准化
def feature_scaler(df, feature, i):
    re_feature = []
    scaler_list = []
    for k in range(K):
        data = feature[k][df[(df['is_train'] == True) & (df['k_fold'] != i)].index]
        scaler = preprocessing.StandardScaler()
        scaler.fit(data)
        re_feature.append(scaler.transform(feature[k]))
        scaler_list.append(scaler)
    return re_feature, scaler_list


# 降维
def reduce_dim(df, feature, i):
    re_feature = []
    pca_list = []
    for k in range(K):
        data = feature[k][df[(df['is_train'] == True) & (df['k_fold'] != i)].index]
        pca = PCA(n_components=Q)
        pca.fit(data)
        re_feature.append(pca.transform(feature[k]))
        pca_list.append(pca)
    return re_feature, pca_list


# 获得用于多视图相似度学习的训练数据
def get_pos_neg_data(df, feature, df_pos_neg):
    """
    x1: 数组, 长度为 '视图数量', 数组中每个元素是 torch.Tensor, 维度为 (样本数量, 特征维度)
    x2: 数组, 长度为 '视图数量', 数组中每个元素是 torch.Tensor, 维度为 (样本数量, 特征维度)
    label: ndarray, 维度为(样本数量, 1)
    """

    df['index'] = df.index
    df = df.set_index('file_name')
    x1 = [feature[k][df.loc[df_pos_neg['sample_1']]['index']] for k in range(K)]
    x2 = [feature[k][df.loc[df_pos_neg['sample_2']]['index']] for k in range(K)]
    label = df_pos_neg[['label']].values.flatten()
    return x1, x2, label


# 将每个样本的特征映射到公共空间 x -> h
def mapping(x, W):
    """
    h: torch.Tensor, 维度为 (视图数量, 样本数量, 特征维度)
    """
    
    h = [(x[k] @ W[k].T) for k in range(K)]
    h = torch.stack(h, dim=0)
    return h
    

# 计算所有视图所有正例反例数据对的相似度
def cal_similarity(x1, x2, W):
    """
    similarity: torch.Tensor, 维度为 (样本数量, 视图数量)
    """

    h1 = mapping(x1, W)
    h2 = mapping(x2, W)
    similarity = [torch.diag(h1[:, i, :] @ h2[:, i, :].T) / (torch.norm(h1[:, i, :], dim=1) * torch.norm(h2[:, i, :], dim=1)) for i in range(x1[0].shape[0])]
    similarity = torch.stack(similarity, dim=0)
    return similarity


# 计算待优化函数 J
def cal_J(similarity, label, W, W0):
    J = 0
    similarity_multi = torch.sum(similarity, 1) / K
    for i in range(SIZE_TRAIN):
        if label[i] == 1:
            J += max(T_P - similarity_multi[i], 0) / (SIZE_TRAIN / 2) * 10
        else:
            J += max(similarity_multi[i] - T_N, 0) / (SIZE_TRAIN / 2) * 10
        if K != 1:
            tmp = list(combinations(range(K), 2))
            reg = 0
            for k, l in tmp:
                reg += (similarity[i, k] - similarity[i, l]).item() ** 2
            J += LAMBDA_2 * reg / SIZE_TRAIN        
    for k in range(K):
        J += LAMBDA_1 * (torch.norm(W[k] - W0) ** 2)
    return J


# 多视图相似度学习, 训练
def multi(x1, x2, label):
    # 初始化映射矩阵 W
    W = [torch.eye(Q, Q, device=DEVICE) + torch.rand(Q, Q, device=DEVICE) / 10 for k in range(K)]
    W = [w.requires_grad_(True) for w in W]
    # W = [torch.eye(Q, Q, device=DEVICE, requires_grad=True) for k in range(K)]
    W0 = torch.eye(Q, Q, device=DEVICE)

    # 计算相似度和待优化函数 J
    similarity = cal_similarity(x1, x2, W)
    # torch.save(similarity, 'result/similarity/reg_before.pt')
    J = cal_J(similarity, label, W, W0)

    # 优化
    for t in range(T):
        # 更新映射矩阵 W
        J.backward()
        for k in range(K):
            W[k] = W[k] - MU * W[k].grad
            W[k].retain_grad()

        # 计算相似度和待优化函数 J
        similarity = cal_similarity(x1, x2, W)
        J_new = cal_J(similarity, label, W, W0)  
        LOG.logger.info('iter = %s, J = %s, diff = %s' % (t, J_new.item(), abs(J - J_new).item()))
        if abs(J - J_new) < E:
            # torch.save(similarity, 'result/similarity/reg_after.pt')
            # quit()
            return W
        J = J_new
        
    # torch.save(similarity, 'result/similarity/reg_after.pt')
    # quit()
    
    return W


# KNN 自定义距离度量矩阵
def my_metric(h1, h2):
    h1 = h1.reshape(K, Q)
    h2 = h2.reshape(K, Q)
    return 1 - np.sum(np.diag(h1 @ h2.T) / (np.linalg.norm(h1, axis=1) * np.linalg.norm(h2, axis=1))) / K
    

# 掌纹分类, 验证
def val_identity(df, feature, i, W):
    with torch.no_grad():
        # 获得用于验证的数据
        data = mapping(feature, W).cpu().numpy()
        x_train = data[:, df[(df['is_train'] == True) & (df['k_fold'] != i)].index].transpose(1, 0, 2).reshape(-1, K * Q)
        y_train = df[(df['is_train'] == True) & (df['k_fold'] != i)]['class_id']
        x_test = data[:, df[(df['is_train'] == True) & (df['k_fold'] == i)].index].transpose(1, 0, 2).reshape(-1, K * Q)
        y_test = df[(df['is_train'] == True) & (df['k_fold'] == i)]['class_id']
        
        # 验证, KNN
        knn = KNeighborsClassifier(n_neighbors=param.N_NEIGHBORS, metric=my_metric)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        acc = accuracy_score(y_pred, y_test)
        LOG.logger.info("val identity acc: %s" % acc)
        return acc, knn


# 掌纹匹配, 验证
def val_verify(df, feature, i, W, best_threshold):
    with torch.no_grad():
        # 获得用于验证的数据
        val_pos_neg = pos_neg(SIZE_VAL, DATASET, i, 'val')
        x1, x2, label = get_pos_neg_data(df, feature, val_pos_neg)

        # 验证
        similarity = cal_similarity(x1, x2, W)
        similarity = (torch.sum(similarity, 1) / K).cpu()
        y_pred = np.where(similarity > best_threshold, 1, -1)
        acc = accuracy_score(y_pred, label)
        LOG.logger.info("val verify acc: %s" % acc)
        return acc


# 掌纹分类, 测试
def test_identity(df, i, scaler, pca, W, knn):
    with torch.no_grad():
        # 读取特征并进行预处理
        feature = []
        if 'lbp' in FEATURES:
            feature.append(load_lbp(DATASET))
        if 'lda' in FEATURES:
            feature.append(load_lda(DATASET)[i])
        if 'hog' in FEATURES:
            feature.append(load_hog(DATASET))
        if 'res' in FEATURES:
            feature.append(load_res(DATASET))
        feature = [scaler[k].transform(feature[k]) for k in range(K)] if scaler is not None else feature  # 标准化
        feature = [pca[k].transform(feature[k]) for k in range(K)] if pca is not None else feature  # 降维
        feature = [(torch.from_numpy(f).float()).to(DEVICE) for f in feature]
        data = mapping(feature, W).cpu().numpy()

        # 获得用于测试的数据
        x_test = data[:, df[df['is_train'] == False].index].transpose(1, 0, 2).reshape(-1, K * Q)
        y_test = df[df['is_train'] == False]['class_id']

        # 测试, KNN
        y_pred = knn.predict(x_test)
        acc = accuracy_score(y_pred, y_test)
        LOG.logger.info("test identity acc: %s" % acc)


# 掌纹匹配, 测试
def test_verify(df, i, scaler, pca, W, best_model):
    with torch.no_grad():
        # 读取特征并进行预处理
        feature = []
        if 'lbp' in FEATURES:
            feature.append(load_lbp(DATASET))
        if 'lda' in FEATURES:
            feature.append(load_lda(DATASET)[i])
        if 'hog' in FEATURES:
            feature.append(load_hog(DATASET))
        if 'res' in FEATURES:
            feature.append(load_res(DATASET))
        feature = [scaler[k].transform(feature[k]) for k in range(K)] if scaler is not None else feature  # 标准化
        feature = [pca[k].transform(feature[k]) for k in range(K)] if pca is not None else feature  # 降维
        feature = [(torch.from_numpy(f).float()).to(DEVICE) for f in feature]

        # 获得用于测试的数据
        test_pos_neg = pos_neg(SIZE_TEST, DATASET, i, 'test')
        x1, x2, label = get_pos_neg_data(df, feature, test_pos_neg)

        # 测试
        similarity = cal_similarity(x1, x2, W)
        similarity = (torch.sum(similarity, 1) / K).cpu()
        fpr, tpr, thresholds = roc_curve(label, similarity, pos_label=1)
        if K == 1:
            np.save('result/reg/%s_k=%d_%s.npy' % (DATASET, K, FEATURES[0]), {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
        else:
            np.save('result/reg/%s_k=%d_multi.npy' % (DATASET, K), {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
        y_pred = np.where(similarity > best_model, 1, -1)
        acc = accuracy_score(y_pred, label)
        LOG.logger.info("test verify acc: %s" % acc)
        LOG.logger.info("\n%s" % classification_report(label, y_pred))


def main():
    LOG.logger.info('%s, k=%d, size=%d, q=%d, t=%d, e=%f, mu=%f, tp=%f, tn=%f, lambda1=%f, lambda2=%f, kfold=%d, knn=%d' % (DATASET, K, SIZE_TRAIN, Q, T, E, MU, T_P, T_N, LAMBDA_1, LAMBDA_2, param.K_FOLD, param.N_NEIGHBORS))
    LOG.logger.info('%s' % FEATURES)
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    LOG.logger.info('program start time = %s' % time)

    # 读取数据列表
    df = pd.read_csv('dataset/%s/data_split.csv' % DATASET)

    # k-fold 交叉验证
    acc_identity_list = []
    best_acc_identity = 0
    acc_verify_list = []
    best_acc_verify = 0
    for i in range(param.K_FOLD):
        print(i)
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        LOG.logger.info('train start time = %s' % time)

        # 读取特征并进行预处理
        feature = []
        if 'lbp' in FEATURES:
            feature.append(load_lbp(DATASET))
        if 'lda' in FEATURES:
            feature.append(load_lda(DATASET)[i])
        if 'hog' in FEATURES:
            feature.append(load_hog(DATASET))
        if 'res' in FEATURES:
            feature.append(load_res(DATASET))
        if K != 0:  # 若设置为 K != 1, 则是对单视图不降维 (中期报告的逻辑)
            feature, scaler = feature_scaler(df, feature, i)  # 标准化
            feature, pca = reduce_dim(df, feature, i)  # 降维
        else:
            scaler, pca = None, None
        feature = [(torch.from_numpy(f).float()).to(DEVICE) for f in feature]
        print('preprocess done')

        # 训练
        train_pos_neg = pos_neg(SIZE_TRAIN, DATASET, i, 'train')  # 获得正例反例的数据集划分
        x1, x2, label = get_pos_neg_data(df, feature, train_pos_neg)  # 获得用于多视图相似度学习的训练数据
        W = multi(x1, x2, label)  # 多视图相似度学习, 掌纹分类, 训练
        with torch.no_grad():  # 掌纹匹配, 训练
            similarity = cal_similarity(x1, x2, W)
            similarity = (torch.sum(similarity, 1) / K).cpu()
            fpr, tpr, thresholds = roc_curve(label, similarity, pos_label=1)
            best_threshold_index = np.argmax(tpr - fpr)
            best_threshold = thresholds[best_threshold_index]
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        LOG.logger.info('train done time = %s' % time)

        # 验证
        acc_identity, knn_identity = val_identity(df, feature, i, W)  # 掌纹识别, 验证
        acc_identity_list.append(acc_identity)
        if acc_identity > best_acc_identity:
            best_acc_identity = acc_identity
            best_i_identity, best_scaler_identity, best_pca_identity, best_W_identity, best_knn_identity = i, scaler, pca, W, knn_identity
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        LOG.logger.info('val identity done time = %s' % time)

        acc_verify = val_verify(df, feature, i, W, best_threshold)  # 掌纹匹配, 验证
        acc_verify_list.append(acc_verify)
        if acc_verify > best_acc_verify:
            best_acc_verify = acc_verify
            best_i_verify, best_scaler_verify, best_pca_verify, best_W_verify, best_model_verify = i, scaler, pca, W, best_threshold
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        LOG.logger.info('val verify done time = %s' % time)
    
    acc_identity_list = np.array(acc_identity_list)
    acc_verify_list = np.array(acc_verify_list)
    LOG.logger.info("val acc identity mean: %s, val acc identity std: %s" % (acc_identity_list.mean(), acc_identity_list.std() ** 2))
    LOG.logger.info("val acc verify mean: %s, val acc verify std: %s" % (acc_verify_list.mean(), acc_verify_list.std() ** 2))

    # 测试
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    LOG.logger.info('test start time = %s' % time)
    
    test_identity(df, best_i_identity, best_scaler_identity, best_pca_identity, best_W_identity, best_knn_identity)
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    LOG.logger.info('test identity done time = %s' % time)
    
    test_verify(df, best_i_verify, best_scaler_verify, best_pca_verify, best_W_verify, best_model_verify)
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    LOG.logger.info('test verify done time = %s' % time)


if __name__ == '__main__':
    main()
