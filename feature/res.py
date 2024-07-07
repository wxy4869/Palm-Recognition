import numpy as np
import pandas as pd

import torch
from PIL import Image
from torchvision import models
from torch.autograd import Variable
from torchvision.transforms import transforms


torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
model.eval()

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


# 加载特征
def load_res(dataset):
    return np.load('dataset/%s/feature/res.npy' % dataset, allow_pickle=True)


# 计算特征
def cal_res_feature(file):
    image = Image.open(file)
    rgb_image = Image.new('RGB', image.size)
    rgb_image.paste(image)
    image = rgb_image.convert('RGB')
    image = Variable(normalize(to_tensor(scaler(image))).unsqueeze(0))
    feature = model(image).detach().numpy().squeeze()
    return feature


# 处理 CASIA 数据集
def CASIA():
    path = 'dataset/CASIA/roi/'
    df = pd.read_csv('dataset/CASIA/data_split.csv')
    feature = []
    for file in df['file_name']:
        feature.append(cal_res_feature(path + file))
    np.save('dataset/CASIA/feature/res.npy', np.array(feature))


# 处理 tongji 数据集
def tongji():
    path = 'dataset/tongji/roi/'
    df = pd.read_csv('dataset/tongji/data_split.csv')
    feature = []
    for file in df['file_name']:
        feature.append(cal_res_feature(path + file))
    np.save('dataset/tongji/feature/res.npy', np.array(feature))


def main():
    CASIA()
    tongji()
    

if __name__ == '__main__':
    main()
