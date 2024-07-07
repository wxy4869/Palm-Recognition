import os
import math
import numpy as np
import matplotlib.pyplot as plt

import cv2
from skimage.draw import line


C1 = 0.60
C2 = 0.55
C3 = 0.15
N = 128


# 计算欧几里德距离
def dist(pointA, pointB):
    return math.sqrt((pointA['x'] - pointB['x']) ** 2 + (pointA['y'] - pointB['y']) ** 2)


# 求解在经过点 point 斜率为 x_slope 的直线上距离 pointL 的长度为 distance 的两个点的坐标
def find_point_on_line(point, slope, distance):
    intercept_x = distance / math.sqrt(1 + slope ** 2)
    intercept_y = distance / math.sqrt(1 + slope ** 2) * slope
    pointA = {'x': point['x'] - intercept_x, 'y': point['y'] - intercept_y}
    pointB = {'x': point['x'] + intercept_x, 'y': point['y'] + intercept_y}
    return pointA, pointB


# 获取手指间隙的坐标
def gap_point(image):
    # 得到二值图像
    blur_image = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary_image = cv2.threshold(blur_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 计算质心
    mu = cv2.moments(binary_image)
    pointC = {'x': mu['m10'] // mu['m00'], 'y': mu['m01'] // mu['m00']}

    # 求轮廓, 并将轮廓重新排序, 使横纵坐标和最小的点排在最前面
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 得到的 contours 是 tuple 类型
    areas = [cv2.contourArea(contour) for contour in contours]
    contours = contours[areas.index(max(areas))].reshape(-1, 2)  # 取最大的轮廓, 维度是 (n, 1, 2), reshape 维度为 (n, 2), n 代表轮廓中点的个数
    index = np.argmin(contours.sum(-1))
    contours = np.concatenate([contours[index:, :], contours[:index, :]])
    
    # 找到轮廓中距质心的距离变化的极值点
    distance = np.sqrt(np.square(contours - [pointC['x'], pointC['y']]).sum(-1))
    f = np.fft.rfft(distance)  # 傅立叶变换, 使距离变平滑
    f = np.concatenate([f[:15], 0 * f[15:]])  # 只要变化最剧烈的部分
    distance = np.fft.irfft(f)  # 傅立叶逆变换
    derivative = np.diff(distance)  # 一阶导
    sign_change = np.diff(np.sign(derivative)) / 2  # 二阶导, 即 (1 1 1 0 -1 -1 -1 0 1 1) 的导数
    points = contours[np.where(sign_change > 0)[0]]

    # 找到手指间隙
    # '''
    points = points[np.argsort(points[:, 1])][:4]
    points = points[np.argsort(points[:, 0])]
    distances = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            distances[i, j] = math.sqrt(np.square(points[i] - points[j]).sum(-1))
    distanceL = distances[0, 1] + distances[1, 2] + distances[0, 2]
    distanceR = distances[1, 2] + distances[2, 3] + distances[1, 3]
    if distanceL < distanceR:
        pointG1 = {'x': points[0][0], 'y': points[0][1]}
        pointG2 = {'x': points[2][0], 'y': points[2][1]}
    else:
        pointG1 = {'x': points[1][0], 'y': points[1][1]}
        pointG2 = {'x': points[3][0], 'y': points[3][1]}
    # '''
    ''' debug
    points = points[np.argsort(points[:, 1])][:5]
    points = points[np.argsort(points[:, 0])]
    pointG1 = {'x': points[0][0], 'y': points[0][1]}
    pointG2 = {'x': points[2][0], 'y': points[2][1]}
    '''
    return contours, pointG1, pointG2


# 建立坐标系: 求解以 pointG1, pointG2 为 x 轴, 其垂直平分线为 y 轴的坐标系
def coordinate_system(pointG1, pointG2):
    origin = {'x': (pointG1['x'] + pointG2['x']) / 2, 'y': (pointG1['y'] + pointG2['y']) / 2}
    x_slope = (pointG1['y'] - pointG2['y']) / (pointG1['x'] - pointG2['x'])
    x_slope = 0.000001 if abs(x_slope) < 0.000001 else x_slope
    y_slope = -1 / x_slope
    pointT1, pointT2 = find_point_on_line(origin, y_slope, dist(pointG1, pointG2) * C1)
    pointL = pointT1 if pointT1['y'] > origin['y'] else pointT2
    return origin, x_slope, y_slope, pointL


# 获取手掌边缘的坐标: 求解经过点 pointL 斜率为 x_slope 的直线与手掌轮廓 contours 的交点
def edge_point(contours, pointL, x_slope, hand_type):
    x, _, w, _ = cv2.boundingRect(contours)  # 返回值是左上角的 x, y 坐标、矩形的宽和高
    line_start = {'x': x, 'y': round(x_slope * (x - pointL['x']) + pointL['y'])}
    line_end = {'x': x + w, 'y': round(x_slope * (x + w - pointL['x']) + pointL['y'])}
    points = []
    last = -1
    cnt = 0
    for point in zip(*line(line_start['x'], line_start['y'], line_end['x'], line_end['y'])):
        point = tuple([int(round(point[0])), int(round(point[1]))])
        now = cv2.pointPolygonTest(contours, point, False)
        if now == 0 or last * now < 0:
            if cnt == 0:
                points.append(point) 
                cnt += 1
            elif cnt > 0 and abs(points[cnt - 1][0] - point[0]) > 10:
                points.append(point) 
                cnt += 1
        last = now
    if hand_type == 'l':
        pointE1 = {'x': points[-2][0], 'y': points[-2][1]}
        pointE2 = {'x': points[-1][0], 'y': points[-1][1]}
    else:
        pointE1 = {'x': points[0][0], 'y': points[0][1]}
        pointE2 = {'x': points[1][0], 'y': points[1][1]}
    return pointE1, pointE2 


# 获取 ROI 顶点的坐标: 求解相对于 pointL, 处于特定位置的 ROI 的四个顶点的坐标
def roi_vertex(pointL, pointE1, pointE2, x_slope, y_slope):
    side = dist(pointE1, pointE2) * C2
    _, pointT1 = find_point_on_line(pointL, y_slope, dist(pointE1, pointE2) * C3)
    pointT1, pointT2 = find_point_on_line(pointL, y_slope, dist(pointE1, pointE2) * C3)
    pointS1 = pointT1 if pointT1['y'] < pointL['y'] else pointT2
    pointT1, pointT2 = find_point_on_line(pointL, y_slope, side - dist(pointE1, pointE2) * C3)
    pointS2 = pointT1 if pointT1['y'] > pointL['y'] else pointT2
    pointV1, pointV2 = find_point_on_line(pointS1, x_slope, side / 2)
    pointV3, pointV4 = find_point_on_line(pointS2, x_slope, side / 2)
    return pointV1, pointV2, pointV3, pointV4


def roi(path):
    # 读取图像
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    hand_type = path.split('/')[-1].split('_')[2]
    
    # 获取手指间隙的坐标、建立坐标系、获取手掌边缘的坐标、获取 ROI 顶点的坐标
    contours, pointG1, pointG2 = gap_point(image)
    _, x_slope, y_slope, pointL = coordinate_system(pointG1, pointG2)
    pointE1, pointE2 = edge_point(contours, pointL, x_slope, hand_type)
    pointV1, pointV2, pointV3, pointV4 = roi_vertex(pointL, pointE1, pointE2, x_slope, y_slope)
    
    # 保存图像
    points = np.array([(pointV1['x'], pointV1['y']), (pointV2['x'], pointV2['y']), (pointV3['x'], pointV3['y']), (pointV4['x'], pointV4['y'])], dtype=np.float32)
    rotated_box = cv2.minAreaRect(points)  # 求最小外接矩形, 即 ROI
    center, size, angle = rotated_box[0], rotated_box[1], rotated_box[2]  # 返回中心点, 宽和高, 旋转角度; 旋转角度是 x 轴逆时针旋转第一次碰到举行的边时的角度
    center, size = tuple(map(int, center)), tuple(map(int, size))
    angle = math.degrees(math.atan(x_slope))  # 重新计算角度
    height, width = image.shape[0], image.shape[1]
    mu = cv2.getRotationMatrix2D(center, angle, 1)  # 将原图绕 ROI 中心点旋转
    rotate_image = cv2.warpAffine(image, mu, (width, height))
    crop_image = cv2.getRectSubPix(rotate_image, size, center)  # 裁剪 ROI
    resize_image = cv2.resize(crop_image, (N, N))  # 压缩大小
    cv2.imwrite('dataset/CASIA/roi/%s' % (path.split('/')[-1]), resize_image)


def main():
    for i in range(1, 313):
        path = 'dataset/CASIA/original/%04d/' % (i)
        for file in os.listdir(path):
            if file.split('.')[-1] != 'jpg':  # 排除 .DS_Store
                continue
            roi(path + file)


def debug():
    # path_list = ['0076/0076_f_l_05.jpg', '0076/0076_f_l_06.jpg', '0076/0076_f_l_07.jpg', '0117/0117_f_l_04.jpg', '0123/0123_f_l_03.jpg', '0287/0287_m_l_09.jpg', '0290/0290_m_l_01.jpg']  # 选最高五个点, 取第 2 个和第 4 个
    path_list = ['0306/0306_f_r_04.jpg', '0308/0308_f_r_04.jpg', '0308/0308_f_r_10.jpg']  # 选最高五个点, 取第 0 个和第 2 个
    for i in path_list:
        roi('dataset/CASIA/original/%s' % i)
    '''
    path = 'dataset/CASIA/original/0132/'  # 0121 0127 0132
    for file in os.listdir(path):
        if file.split('.')[-1] != 'jpg':  # 排除 .DS_Store
            continue
        roi(path + file)
    '''


if __name__ == '__main__':
    main()
    # debug()
