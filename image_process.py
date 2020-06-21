# 本文件的主要作用是对图像进行处理
import pandas as pd
import numpy as np
import re
import os
import cv2
DIR_PATH = '../比赛/global-wheat-detection'
DIR_TRAIN = f'{DIR_PATH}/train'
FILE_PATH = f'{DIR_PATH}/train.csv'

# 把训练集的train_df扩大
def expand_train_df(ori_df, train_ids):
    train_df = ori_df[ori_df['image_id'].isin(train_ids)]

    # 把图片旋转90度之后的数据加入进去
    train_rotate90 = rotate_90(train_df)
    train_rotate180 = rotate_180(train_df)
    train_rotate270 = rotate_270(train_df)

    merge_df = pd.concat([train_df,train_rotate90,train_rotate180,train_rotate270])
    # 把四个文件合并在一起
    return merge_df


def rotate_270(dataFrame):
    dataFrame_2 = dataFrame.copy()

    image_ids = dataFrame_2['image_id'].unique()
    # 如果统一旋转90度,宽度和高度互换

    # 先找到框的中心位置
    x_center = dataFrame_2['x'] + dataFrame_2['w'] / 2
    y_center = dataFrame_2['y'] + dataFrame_2['h'] / 2

    # 接下来就是找到变化后中心的位置
    x_center_2 = y_center
    y_center_2 = 1024 - x_center

    # 然后宽高互换
    w1 = dataFrame_2['w']
    h1 = dataFrame_2['h']
    temp = dataFrame_2['w'].copy()
    dataFrame_2['w'] = dataFrame_2['h']
    dataFrame_2['h'] = temp

    w2 = dataFrame_2['w']
    h2 = dataFrame_2['h']

    dataFrame_2['x'] = x_center_2 - dataFrame_2['w'] / 2
    dataFrame_2['y'] = y_center_2 - dataFrame_2['h'] / 2

    dataFrame_2['image_id'] = dataFrame_2['image_id'] + '_rotate270'

    return dataFrame_2

    # 显示标注
    # i = 0
    # for image in image_ids:
    #     if image == ".DS_Store":
    #         continue
    #     i = i + 1
    #     img_path = DIR_PATH + '/train' + '/' + image + '.jpg'
    #     print(str(i) + ' : ' + img_path)
    #     img = cv2.imread(img_path)
    #
    #     rotate_img = cv2.rotate(img, 2)
    #
    #     records = dataFrame_2[dataFrame_2['image_id'] == image]
    #     boxes = records[['x', 'y', 'w', 'h']].values
    #     boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    #     boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    #
    #     for box in boxes:
    #         cv2.rectangle(rotate_img,
    #                       (box[0], box[1]),
    #                       (box[2], box[3]),
    #                       (0, 0, 255), 3)

        # cv2.imwrite(image+'.jpg',rotate_img)


def rotate_180(dataFrame):
    # 先复制一个一样的
    dataFrame_2 = dataFrame.copy()

    image_ids = dataFrame_2['image_id'].unique()
    # 如果统一旋转90度,宽度和高度互换

    # 先找到框的中心位置
    x_center = dataFrame_2['x'] + dataFrame_2['w'] / 2
    y_center = dataFrame_2['y'] + dataFrame_2['h'] / 2

    # 接下来就是找到变化后中心的位置
    x_center_2 = 1024 - x_center
    y_center_2 = 1024 - y_center

    dataFrame_2['x'] = x_center_2 - dataFrame_2['w'] / 2
    dataFrame_2['y'] = y_center_2 - dataFrame_2['h'] / 2
    # 显示标注
    # image_list = os.listdir(DIR_TRAIN)
    # i = 0
    # for image in image_ids:
    #     if image == ".DS_Store":
    #         continue
    #     i = i + 1
    #     img_path = DIR_PATH + '/train' + '/' + image + '.jpg'
    #     print(str(i) + ' : ' + img_path)
    #     img = cv2.imread(img_path)
    #
    #     rotate_img = cv2.rotate(img, 1)
    #
    #     records = dataFrame[dataFrame['image_id'] == image]
    #     boxes = records[['x', 'y', 'w', 'h']].values
    #     boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    #     boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    #
    #     for box in boxes:
    #         cv2.rectangle(rotate_img,
    #                       (box[0], box[1]),
    #                       (box[2], box[3]),
    #                       (0, 0, 255), 3)
    #
    #     cv2.imwrite(image+'.jpg',rotate_img)

    dataFrame_2['image_id'] = dataFrame_2['image_id'] + '_rotate180'

    return dataFrame_2

def rotate_90(dataFrame):
    # 先复制一个一样的
    dataFrame_2 = dataFrame.copy()

    image_ids = dataFrame_2['image_id'].unique()
    # 如果统一旋转90度,宽度和高度互换

    # 先找到框的中心位置
    x_center = dataFrame_2['x'] + dataFrame_2['w'] / 2
    y_center = dataFrame_2['y'] + dataFrame_2['h'] / 2

    # 旋转90度需要变化中心的位置

    x_center_2 = 1024 - y_center
    y_center_2 = x_center

    # 然后宽高互换
    w1 = dataFrame_2['w']
    h1 = dataFrame_2['h']
    temp = dataFrame_2['w'].copy()
    dataFrame_2['w'] = dataFrame_2['h']
    dataFrame_2['h'] = temp

    w2 = dataFrame_2['w']
    h2 = dataFrame_2['h']

    # 旋转之后中心位置改变，根据中心位置和宽高找到左上角位置
    dataFrame_2['x'] = x_center_2 - dataFrame_2['w'] / 2
    dataFrame_2['y'] = y_center_2 - dataFrame_2['h'] / 2

    # image_list = os.listdir(DIR_TRAIN)
    # i = 0
    # for image in image_ids:
    #     if image == ".DS_Store":
    #         continue
    #     i = i + 1
    #     img_path = DIR_PATH + '/train' + '/' + image + '.jpg'
    #     print(str(i) + ' : ' + img_path)
    #     img = cv2.imread(img_path)
    #
    #     rotate_img = cv2.rotate(img, 0)
    #
    #     records = dataFrame[dataFrame['image_id'] == image]
    #     boxes = records[['x', 'y', 'w', 'h']].values
    #     boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    #     boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    #
    #     for box in boxes:
    #         cv2.rectangle(rotate_img,
    #                       (box[0], box[1]),
    #                       (box[2], box[3]),
    #                       (0, 0, 255), 3)
    #
    #     cv2.imwrite(image+'.jpg',rotate_img)
    # 最后需要把列名换一换
    # for image in image_ids:
    #     dataFrame[dataFrame['image_id'] == image] = image + '_rotate90'
    dataFrame_2['image_id'] = dataFrame_2['image_id'] + '_rotate90'

    return dataFrame_2


def read_and_change_name(FILE_PATH):
    dataFrame = pd.read_csv(FILE_PATH)
    image_ids = dataFrame['image_id'].unique()

    for image in image_ids:
        if 'E+' in image or len(image) == 8:
            print(image)

    f_read = open('wrong_name.txt','r')
    contents = f_read.readlines()
    for line in contents:
        if line.startswith('#'):
            continue
        # 先去除前后的空格
        line = line.strip()
        part_1, part_2 = line.split(' ')
        # print(str(part_1) + ' : ' + str(part_2))
        records = dataFrame[dataFrame['image_id'] == str(part_1)]
        # 找到索引位置
        index = records.index.array.to_numpy()
        for i in index:
            # 根绝索引位置进行修改
            dataFrame.iat[i,0] = str(part_2)


    # 如果没有什么意外，到这里就已经全部转换好了

    return dataFrame

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r


if __name__ == '__main__':
        
    dataFrame = read_and_change_name(FILE_PATH)
    dataFrame['x'] = -1
    dataFrame['y'] = -1
    dataFrame['w'] = -1
    dataFrame['h'] = -1

    dataFrame[['x', 'y', 'w', 'h']] = np.stack(dataFrame['bbox'].apply(lambda x: expand_bbox(x)))
    dataFrame.drop(columns=['bbox'], inplace=True)
    dataFrame['x'] = dataFrame['x'].astype(np.float32)
    dataFrame['y'] = dataFrame['y'].astype(np.float32)
    dataFrame['w'] = dataFrame['w'].astype(np.float32)
    dataFrame['h'] = dataFrame['h'].astype(np.float32)
    image_ids = dataFrame['image_id'].unique()
    train_ids = image_ids[:-600]


    expand_train_df(dataFrame, train_ids)