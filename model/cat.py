import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pickle
import datetime
import numpy as np
import pandas as pd

import torch
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
SIZE_TRAIN = 3000  # 正例反例数据对的总数量
SIZE_VAL = 500  # 验证集正例反例数据对的总数量
SIZE_TEST = 500  # 测试集正例反例数据对的总数量

Q = 100  # 多视图时, 将不同视图的特征维度统一为 Q; 单视图时, lbp 取 944, hog 取 324, res 取 1000 (划掉)
T = 100  # 迭代次数
K = 3  # 视图数
E = 1e-2  # 收敛误差
MU = 1  # 学习率
T_P = 0.8  # 阈值
T_N = 0.1  # 阈值
LAMBDA = 0.01  # 正则化项系数

DEVICE = ('mps' if torch.backends.mps.is_available() else 'cpu')
if K == 1:
    LOG = Logger('log/cat/%s_k=%d_%s_size=%d_q=%d_t=%d_e=%f_mu=%f_tp=%f_tn=%f_lambda=%f_knn=%d.log' % (DATASET, K, FEATURES[0], SIZE_TRAIN, Q, T, E, MU, T_P, T_N, LAMBDA, param.N_NEIGHBORS), level='info')
else:
    LOG = Logger('log/cat/%s_k=%d_multi_size=%d_q=%d_t=%d_e=%f_mu=%f_tp=%f_tn=%f_lambda=%f_knn=%d.log' % (DATASET, K, SIZE_TRAIN, Q, T, E, MU, T_P, T_N, LAMBDA, param.N_NEIGHBORS), level='info')


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
def mapping(x, Ws, Wc):
    """
    h: torch.Tensor, 维度为 (样本数量, 特征维度, 1)
    """

    hs = [x[k] @ Ws[k].T for k in range(K)]
    hc = [x[k] @ Wc[k].T for k in range(K)]
    hs = torch.cat(hs, dim=1)
    hc = torch.stack(hc, dim=1)
    hc = torch.sum(hc, dim=1) / K
    h = torch.cat([hs, hc], dim=1).reshape(-1, (K + 1) * Q, 1)
    return h


# 计算所有正例反例数据对的相似度
def cal_similarity(x1, x2, Ws, Wc):
    """
    similarity: 数组, 长度为 '样本数量'
    """

    h1 = mapping(x1, Ws, Wc)
    h2 = mapping(x2, Ws, Wc)
    similarity = [h1[i].T @ h2[i] / (torch.norm(h1[i]) * torch.norm(h2[i])) for i in range(len(h1))]
    return similarity
    

# 计算待优化函数 J
def cal_J(similarity, label, Ws, Wc):
    J = 0
    for i in range(SIZE_TRAIN):
        if label[i] == 1:
            J += max(T_P - similarity[i], 0) / (SIZE_TRAIN / 2)
        else:
            J += max(similarity[i] - T_N, 0) / (SIZE_TRAIN / 2)
    for k in range(K):
        J += LAMBDA * (torch.norm(Ws[k]) ** 2 + torch.norm(Wc[k]) ** 2)
    return J


# 多视图相似度学习, 训练
def multi(x1, x2, label):
    # 初始化映射矩阵 W
    Ws = [torch.eye(Q, Q, device=DEVICE, requires_grad=True) for k in range(K)]
    Wc = [torch.eye(Q, Q, device=DEVICE, requires_grad=True) for k in range(K)]

    # 计算相似度和待优化函数 J
    similarity = cal_similarity(x1, x2, Ws, Wc)
    # torch.save(similarity, 'result/similarity/cat_tongji_before.pt')
    J = cal_J(similarity, label, Ws, Wc)

    # 优化
    for t in range(T):
        # 更新映射矩阵 W
        J.backward()
        for k in range(K):
            Ws[k] = Ws[k] - MU * Ws[k].grad
            Wc[k] = Wc[k] - MU * Wc[k].grad
            Ws[k].retain_grad()
            Wc[k].retain_grad()

        # 计算相似度和待优化函数 J
        similarity = cal_similarity(x1, x2, Ws, Wc)
        J_new = cal_J(similarity, label, Ws, Wc)
        LOG.logger.info('iter = %s, J = %s, diff = %s' % (t, J_new.item(), abs(J - J_new).item()))
        if abs(J - J_new) < E:
            # torch.save(similarity, 'result/similarity/cat_tongji_after.pt')
            # quit()
            return Ws, Wc
        J = J_new

    # torch.save(similarity, 'result/similarity/cat_tongji_after.pt')
    # quit()
    
    return Ws, Wc


# 掌纹分类, 验证
def val_identity(df, feature, i, Ws, Wc):
    with torch.no_grad():
        # 获得用于验证的数据
        data = mapping(feature, Ws, Wc)
        data = data.cpu().squeeze()
        x_train = data[df[(df['is_train'] == True) & (df['k_fold'] != i)].index]
        y_train = df[(df['is_train'] == True) & (df['k_fold'] != i)]['class_id']
        x_test = data[df[(df['is_train'] == True) & (df['k_fold'] == i)].index]
        y_test = df[(df['is_train'] == True) & (df['k_fold'] == i)]['class_id']
        
        # 验证, KNN
        knn = KNeighborsClassifier(n_neighbors=param.N_NEIGHBORS, metric='cosine')
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        acc = accuracy_score(y_pred, y_test)
        LOG.logger.info("val identity acc: %s" % acc)
        return acc, knn


# 掌纹匹配, 验证
def val_verify(df, feature, i, Ws, Wc, best_threshold):
    with torch.no_grad():
        # 获得用于验证的数据
        val_pos_neg = pos_neg(SIZE_VAL, DATASET, i, 'val')
        x1, x2, label = get_pos_neg_data(df, feature, val_pos_neg)

        # 验证
        similarity = torch.tensor(cal_similarity(x1, x2, Ws, Wc))
        y_pred = np.where(similarity > best_threshold, 1, -1)
        acc = accuracy_score(y_pred, label)
        LOG.logger.info("val verify acc: %s" % acc)
        return acc


# 掌纹分类, 测试
def test_identity(df, i, scaler, pca, Ws, Wc, knn):
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
        data = mapping(feature, Ws, Wc)
        data = data.cpu().squeeze()

        # 获得用于测试的数据
        x_test = data[df[df['is_train'] == False].index]
        y_test = df[df['is_train'] == False]['class_id']

        # 测试, KNN
        y_pred = knn.predict(x_test)
        acc = accuracy_score(y_pred, y_test)
        LOG.logger.info("test identity acc: %s" % acc)


# 掌纹匹配, 测试
def test_verify(df, i, scaler, pca, Ws, Wc, best_model):
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
        similarity = torch.tensor(cal_similarity(x1, x2, Ws, Wc))
        fpr, tpr, thresholds = roc_curve(label, similarity, pos_label=1)
        if K == 1:
            np.save('result/cat/%s_k=%d_%s.npy' % (DATASET, K, FEATURES[0]), {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
        else:
            np.save('result/cat/%s_k=%d_multi.npy' % (DATASET, K), {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
        y_pred = np.where(similarity > best_model, 1, -1)
        acc = accuracy_score(y_pred, label)
        LOG.logger.info("test verify acc: %s" % acc)
        LOG.logger.info("\n%s" % classification_report(label, y_pred))


def main():
    LOG.logger.info('%s, k=%d, size=%d, q=%d, t=%d, e=%f, mu=%f, tp=%f, tn=%f, lambda=%f, kfold=%d, knn=%d' % (DATASET, K, SIZE_TRAIN, Q, T, E, MU, T_P, T_N, LAMBDA, param.K_FOLD, param.N_NEIGHBORS))
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
        df_pos_neg = pos_neg(SIZE_TRAIN, DATASET, i, 'train')  # 获得正例反例的数据集划分
        x1, x2, label = get_pos_neg_data(df, feature, df_pos_neg)  # 获得用于多视图相似度学习的训练数据
        Ws, Wc = multi(x1, x2, label)  # 多视图相似度学习, 掌纹分类, 训练
        with torch.no_grad():  # 掌纹匹配, 训练
            similarity = torch.tensor(cal_similarity(x1, x2, Ws, Wc))
            fpr, tpr, thresholds = roc_curve(label, similarity, pos_label=1)
            best_threshold_index = np.argmax(tpr - fpr)
            best_threshold = thresholds[best_threshold_index]
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        LOG.logger.info('train done time = %s' % time)

        # 验证
        acc_identity, knn_identity = val_identity(df, feature, i, Ws, Wc)  # 掌纹匹配, 验证
        acc_identity_list.append(acc_identity)
        if acc_identity > best_acc_identity:
            best_acc_identity = acc_identity
            best_i_identity, best_scaler_identity, best_pca_identity, best_Ws_identity, best_Wc_identity, best_knn_identity = i, scaler, pca, Ws, Wc, knn_identity
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        LOG.logger.info('val identity done time = %s' % time)

        acc_verify = val_verify(df, feature, i, Ws, Wc, best_threshold)  # 掌纹匹配, 验证
        acc_verify_list.append(acc_verify)
        if acc_verify > best_acc_verify:
            best_acc_verify = acc_verify
            best_i_verify, best_scaler_verify, best_pca_verify, best_Ws_verify, best_Wc_verify, best_model_verify = i, scaler, pca, Ws, Wc, best_threshold
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        LOG.logger.info('val verify done time = %s' % time)
        
    acc_identity_list = np.array(acc_identity_list)
    acc_verify_list = np.array(acc_verify_list)
    LOG.logger.info("val acc identity mean: %s, val acc identity std: %s" % (acc_identity_list.mean(), acc_identity_list.std() ** 2))
    LOG.logger.info("val acc verify mean: %s, val acc verify std: %s" % (acc_verify_list.mean(), acc_verify_list.std() ** 2))

    # 测试
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    LOG.logger.info('test start time = %s' % time)
    
    test_identity(df, best_i_identity, best_scaler_identity, best_pca_identity, best_Ws_identity, best_Wc_identity, best_knn_identity)
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    LOG.logger.info('test identity done time = %s' % time)
    
    test_verify(df, best_i_verify, best_scaler_verify, best_pca_verify, best_Ws_verify, best_Wc_verify, best_model_verify)
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    LOG.logger.info('test verify done time = %s' % time)

    '''
    # 保存模型
    file_name = 'result/cat/demo%s_k=%d_multi_identity.pkl' % (DATASET, K)
    with open(file_name, "wb") as f:
        pickle.dump({'best_i_identity': best_i_identity, 
                    'best_scaler_identity': best_scaler_identity, 
                    'best_pca_identity': best_pca_identity, 
                    'best_Ws_identity': best_Ws_identity, 
                    'best_Wc_identity': best_Wc_identity, 
                    'best_knn_identity': best_knn_identity
                    }, f)
    file_name = 'result/cat/demo%s_k=%d_multi_verify.pkl' % (DATASET, K)
    with open(file_name, "wb") as f:
        pickle.dump({'best_i_verify': best_i_verify, 
                    'best_scaler_verify': best_scaler_verify, 
                    'best_pca_verify': best_pca_verify, 
                    'best_Ws_verify': best_Ws_verify, 
                    'best_Wc_verify': best_Wc_verify, 
                    'best_model_verify': best_model_verify
                    }, f)
    '''


if __name__ == '__main__':
    main()
