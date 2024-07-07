import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pickle
import pandas as pd

import torch

from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QTabWidget, QListWidget, QLabel, QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from feature.lbp import load_lbp
from feature.lda import load_lda
from feature.hog import load_hog
from feature.res import load_res
from util.data_split import pos_neg
from model.cat import get_pos_neg_data, mapping, cal_similarity


# 与 model 保持一致
DATASET = 'CASIA'
FEATURES = ['lbp', 'hog', 'res']
SIZE_TEST = 500
K = 3
DEVICE = ('mps' if torch.backends.mps.is_available() else 'cpu')


class ImageViewer(QMainWindow):
    # 初始化
    def __init__(self):
        super().__init__()
        self.initData()
        self.initUI()


    def initData(self):
        self.df = pd.read_csv('dataset/%s/data_split.csv' % DATASET)
        
        self.identity_model = pickle.load(open('result/cat/demo%s_k=%d_multi_identity.pkl' % (DATASET, K), "rb"))
        self.verify_model = pickle.load(open('result/cat/demo%s_k=%d_multi_verify.pkl' % (DATASET, K), "rb"))

        feature = []
        if 'lbp' in FEATURES:
            feature.append(load_lbp(DATASET))
        if 'lda' in FEATURES:
            feature.append(load_lda(DATASET)[self.identity_model['best_i_identity']])
        if 'hog' in FEATURES:
            feature.append(load_hog(DATASET))
        if 'res' in FEATURES:
            feature.append(load_res(DATASET))
        feature = [self.identity_model['best_scaler_identity'][k].transform(feature[k]) for k in range(K)] if self.identity_model['best_scaler_identity'] is not None else feature  # 标准化
        feature = [self.identity_model['best_pca_identity'][k].transform(feature[k]) for k in range(K)] if self.identity_model['best_pca_identity'] is not None else feature  # 降维
        feature = [(torch.from_numpy(f).float()).to(DEVICE) for f in feature]
        self.feature = feature

        self.identity_data = self.df[self.df['is_train'] == False]
        self.identity_data = self.identity_data.sample(frac=1, random_state=42).reset_index(drop=True)
        self.verify_data = pos_neg(SIZE_TEST, DATASET, None, 'test')
        self.verify_data = self.verify_data.sample(frac=1, random_state=42).reset_index(drop=True)


    def initUI(self):
        # 创建主窗口
        self.setWindowTitle('掌纹识别')
        self.setGeometry(100, 100, 800, 600)

        # 创建布局
        central_widget = QWidget()
        main_layout = QGridLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 创建选项卡控件
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self.tab_changed)

        # 掌纹分类选项卡
        tab_identity = QWidget()
        tab_identity_layout = QGridLayout()
        tab_identity.setLayout(tab_identity_layout)

        self.identity_image_list = QListWidget()
        self.identity_image_list.setMaximumWidth(300)
        self.identity_image_list.currentItemChanged.connect(self.display_identity_images)
        for _, row in self.identity_data.iterrows():
            self.identity_image_list.addItem(row['file_name'])
        self.identity_image_label = QLabel()
        self.identity_image_label.setAlignment(Qt.AlignCenter)
        self.identity_button = QPushButton('掌纹分类')
        self.identity_button.clicked.connect(self.identity)
        self.identity_output_label = QLabel()
        
        tab_identity_layout.addWidget(self.identity_image_list, 0, 0, 3, 1)
        tab_identity_layout.addWidget(self.identity_image_label, 0, 1, 1, 2)
        tab_identity_layout.addWidget(self.identity_button, 1, 1, 1, 2)
        tab_identity_layout.addWidget(self.identity_output_label, 2, 1, 1, 2)

        self.identity_image_list.setCurrentRow(0)

        # 掌纹验证选项卡
        tab_verify = QWidget()
        tab_verify_layout = QGridLayout()
        tab_verify.setLayout(tab_verify_layout)

        self.verify_image_list = QListWidget()
        self.verify_image_list.setMaximumWidth(300)
        self.verify_image_list.currentItemChanged.connect(self.display_verify_images)
        for _, row in self.verify_data.iterrows():
            self.verify_image_list.addItem('%s\t%s' % (row['sample_1'], row['sample_2']))
        self.verify_image_label1 = QLabel()
        self.verify_image_label1.setAlignment(Qt.AlignCenter)
        self.verify_image_label2 = QLabel()
        self.verify_image_label2.setAlignment(Qt.AlignCenter)
        self.verify_button = QPushButton('掌纹验证')
        self.verify_button.clicked.connect(self.verify)
        self.verify_output_label = QLabel()
        
        tab_verify_layout.addWidget(self.verify_image_list, 0, 0, 3, 1)
        tab_verify_layout.addWidget(self.verify_image_label1, 0, 1)
        tab_verify_layout.addWidget(self.verify_image_label2, 0, 2)
        tab_verify_layout.addWidget(self.verify_button, 1, 1, 1, 2)
        tab_verify_layout.addWidget(self.verify_output_label, 2, 1, 1, 2)

        self.verify_image_list.setCurrentRow(0)

        # 将两个选项卡添加到选项卡控件中, 将选项卡控件添加到整体布局中
        self.tab_widget.addTab(tab_identity, '掌纹分类')
        self.tab_widget.addTab(tab_verify, '掌纹验证')
        main_layout.addWidget(self.tab_widget)


    # 切换选项卡
    def tab_changed(self, index):
        if index == 0:
            feature = []
            if 'lbp' in FEATURES:
                feature.append(load_lbp(DATASET))
            if 'lda' in FEATURES:
                feature.append(load_lda(DATASET)[self.identity_model['best_i_identity']])
            if 'hog' in FEATURES:
                feature.append(load_hog(DATASET))
            if 'res' in FEATURES:
                feature.append(load_res(DATASET))
            feature = [self.identity_model['best_scaler_identity'][k].transform(feature[k]) for k in range(K)] if self.identity_model['best_scaler_identity'] is not None else feature  # 标准化
            feature = [self.identity_model['best_pca_identity'][k].transform(feature[k]) for k in range(K)] if self.identity_model['best_pca_identity'] is not None else feature  # 降维
            feature = [(torch.from_numpy(f).float()).to(DEVICE) for f in feature]
            self.feature = feature
        elif index == 1:
            feature = []
            if 'lbp' in FEATURES:
                feature.append(load_lbp(DATASET))
            if 'lda' in FEATURES:
                feature.append(load_lda(DATASET)[self.verify_model['best_i_verify']])
            if 'hog' in FEATURES:
                feature.append(load_hog(DATASET))
            if 'res' in FEATURES:
                feature.append(load_res(DATASET))
            feature = [self.verify_model['best_scaler_verify'][k].transform(feature[k]) for k in range(K)] if self.verify_model['best_scaler_verify'] is not None else feature  # 标准化
            feature = [self.verify_model['best_pca_verify'][k].transform(feature[k]) for k in range(K)] if self.verify_model['best_pca_verify'] is not None else feature  # 降维
            feature = [(torch.from_numpy(f).float()).to(DEVICE) for f in feature]
            self.feature = feature


    # 展示图片
    def display_identity_images(self, current):
        # 展示图片
        image_names = current.text()
        self.identity_image_label.setPixmap(QPixmap('dataset/%s/roi/%s' % (DATASET, image_names)))

        # 保存当前图片的信息用于掌纹验证, 并清空预测结果
        self.current_data_identity = image_names
        self.identity_output_label.setText('')


    def display_verify_images(self, current):
        # 展示图片
        image_names = current.text().split('\t')
        self.verify_image_label1.setPixmap(QPixmap('dataset/%s/roi/%s' % (DATASET, image_names[0])))
        self.verify_image_label2.setPixmap(QPixmap('dataset/%s/roi/%s' % (DATASET, image_names[1])))

        # 保存当前图片的信息用于掌纹验证, 并清空预测结果
        self.current_data_verify = self.verify_data[(self.verify_data['sample_1'] == image_names[0]) & 
                                                    (self.verify_data['sample_2'] == image_names[1])]
        self.verify_output_label.setText('')


    # 掌纹识别
    def identity(self):
        # 掌纹分类
        index = self.df[self.df['file_name'] == self.current_data_identity].index
        label = self.df.loc[index, 'class_id'].values[0]
        
        x = [self.feature[k][index] for k in range(K)]
        x = mapping(x, self.identity_model['best_Ws_identity'], self.identity_model['best_Wc_identity'])
        x = x.cpu().squeeze().detach().reshape(1, -1)
        y_pred = self.identity_model['best_knn_identity'].predict(x)[0]

        # 更新预测结果
        res_str1 = '该掌纹属于类别 %s, ' % y_pred
        res_str2 = '预测正确' if y_pred == label else '预测错误, 实际类别为 %s' % label
        self.identity_output_label.setText(str(res_str1 + res_str2))


    def verify(self):
        # 掌纹验证
        x1, x2, label = get_pos_neg_data(self.df, self.feature, self.current_data_verify)
        similarity = torch.tensor(cal_similarity(x1, x2, self.verify_model['best_Ws_verify'], self.verify_model['best_Wc_verify']))
        y_pred = 1 if similarity > self.verify_model['best_model_verify'] else -1
        
        # 更新预测结果
        res_str1 = '两张掌纹图片属于同一类, ' if y_pred == 1 else '两张掌纹图片属于不同类, '
        res_str2 = '预测正确' if y_pred == label else '预测错误'
        self.verify_output_label.setText(str(res_str1 + res_str2))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
