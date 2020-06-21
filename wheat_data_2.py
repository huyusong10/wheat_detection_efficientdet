import pandas as pd
import numpy as np
import cv2
import os
import re

from torch.utils.data import Dataset
from torchvision import transforms

import torch
import torchvision

import image_process

input_sizes = [512, 640, 768, 896, 1024]
input_dir = '/home/huys/wheat_detection'
# input_dir = '../比赛/global-wheat-detection'


class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]
        # 这里需要判断图片是原来的图片还是旋转之后的图片
        if 'rotate90' in image_id:
            part1,part2 = image_id.split('_')
            img = cv2.imread(f'{self.image_dir}/{part1}.jpg', cv2.IMREAD_COLOR)
            image = cv2.rotate(img,0)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
        elif 'rotate180' in image_id:
            part1,part2 = image_id.split('_')
            img = cv2.imread(f'{self.image_dir}/{part1}.jpg', cv2.IMREAD_COLOR)
            image = cv2.rotate(img, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
        elif 'rotate270' in image_id:
            part1, part2 = image_id.split('_')
            img = cv2.imread(f'{self.image_dir}/{part1}.jpg', cv2.IMREAD_COLOR)
            image = cv2.rotate(img, 2)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
        else:
            img = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0

        boxes = np.zeros((len(records), 5))
        boxes[:, 0:4] = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        sample = {'img': image, 'annot': boxes}
        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        return self.image_ids.shape[0]


def get_train_transform(compound_coef=0):
    return transforms.Compose([
        Normalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        Augmenter(),
        Resizer(input_sizes[compound_coef])
    ])


def get_valid_transform(compound_coef=0):
    return transforms.Compose([
        Normalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        Augmenter(),
        Resizer(input_sizes[compound_coef])
    ])


def collate_fn(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


def get_data_set(compound_coef=0, DIR_INPUT=input_dir):
    ori_df = pd.read_csv(f'{DIR_INPUT}/train.csv')
    # ori_df = image_process.read_and_change_name(f'{DIR_INPUT}/train.csv')
    DIR_TRAIN = f'{DIR_INPUT}/train'

    ori_df['x'] = -1
    ori_df['y'] = -1
    ori_df['w'] = -1
    ori_df['h'] = -1

    ori_df[['x', 'y', 'w', 'h']] = np.stack(ori_df['bbox'].apply(lambda x: expand_bbox(x)))
    ori_df.drop(columns=['bbox'], inplace=True)
    ori_df['x'] = ori_df['x'].astype(np.float)
    ori_df['y'] = ori_df['y'].astype(np.float)
    ori_df['w'] = ori_df['w'].astype(np.float)
    ori_df['h'] = ori_df['h'].astype(np.float)
    # 去除10个比较大的框
    # 添加面积选项
    ori_df['area'] = ori_df['w'] * ori_df['h']
    # 接下来要去除比较大的面积框

    # 先获取面积这一列
    area_list = ori_df['area'].values.tolist()
    area_list_copy = area_list.copy()
    area_list_copy.sort(reverse=True)
    # 把其中几个大的框去掉
    for i in range(10):
        index = area_list.index(area_list_copy[i])
        ori_df.drop([index], inplace=True)


    # 划分训练集和测试集
    image_ids = ori_df['image_id'].unique()
    train_ids = image_ids[:-600]
    valid_ids = image_ids[-600:]

    train_df = image_process.expand_train_df(ori_df, train_ids)
    # train_df = ori_df[ori_df['image_id'].isin(train_ids)]
    # print(train_df.shape)
    valid_df = ori_df[ori_df['image_id'].isin(valid_ids)]
    # print(valid_df.shape)

    train_dataset = WheatDataset(train_df, DIR_TRAIN, get_train_transform(compound_coef))
    valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_valid_transform(compound_coef))

    return train_dataset, valid_dataset
