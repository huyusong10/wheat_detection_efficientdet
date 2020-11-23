import pandas as pd
import numpy as np
import cv2
import random
import os
import re

from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from sklearn.model_selection import StratifiedKFold
from distortion import augment
from utils import check_file

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

def get_train_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        # ToTensorV2()
    ], A.BboxParams(format='pascal_voc', label_fields=['ids']))

def get_valid_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        # ToTensorV2()
    ], A.BboxParams(format='pascal_voc', label_fields=['ids']))

class WheatDataset(Dataset):

    def __init__(self, dataframe, img_dir, transforms=None, noise_level=0):
        super().__init__()

        self.img_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.img_dir = img_dir
        self.transforms = transforms
        self.noise_level = noise_level
        self.epoch = -1

    def __getitem__(self, index: int):

        img_id = self.img_ids[index]
        records = self.df[self.df['image_id'] == img_id]

        img = self._imread(img_id)
        boxes = self._boxread(records)

        if self.transforms:
            # try:
            #     transformed = self.transforms(image=img, bboxes=boxes, ids=np.zeros(boxes.shape[0]))
            # except:
            #     print(img_id)
            transformed = self.transforms(image=img, bboxes=boxes, ids=np.zeros(boxes.shape[0]))
        else:
            raise RuntimeError
        
        noise_level = self._get_noise()

        img, boxes = augment(transformed['image'], np.array(transformed['bboxes']), noise_level=noise_level)

        sample = {'img': img, 'bboxes': boxes, 'scale': 1, 'noise_level': noise_level}

        return sample

    def __len__(self) -> int:
        return self.img_ids.shape[0]

    def _imread(self, img_id):

        img = cv2.imread(os.path.join(self.img_dir, img_id+'.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _boxread(self, records):

        boxes = np.zeros((len(records), 4))
        boxes[:, 0:4] = records[['x', 'y', 'w', 'h']].values

        # transform to xyxy
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        return boxes

    def _get_noise(self):

        if self.epoch%4 == 0:
            return self.noise_level
        else:
            return 0

    def set_epoch(self, epoch):
        self.epoch = epoch



def collate_fn(data):
    imgs = [s['img'] for s in data]
    bboxes = [s['bboxes'] for s in data]
    scales = [s['scale'] for s in data]
    noise_level = [s['noise_level'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in bboxes)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(bboxes), max_num_annots, 5)) * -1

        for idx, annot in enumerate(bboxes):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], 0:4] = torch.from_numpy(annot)
                annot_padded[idx, :annot.shape[0], 4] = 0
    else:
        annot_padded = torch.ones((len(bboxes), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'imgs': imgs, 'bboxes': annot_padded, 'scales': scales, 'noise_level': noise_level}

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

def get_xywh(train_df):
    bboxs = np.stack(train_df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        train_df[column] = bboxs[:, i]
    train_df.drop(columns=['bbox'], inplace=True)
    return train_df

def filter_bbox(train_df):
    train_df['area'] = train_df['w'] * train_df['h']

    area_list = train_df['area'].values.tolist()
    area_list_copy = area_list.copy()
    area_list_copy.sort(reverse=True)

    for i in range(10):
        index = area_list.index(area_list_copy[i])
        train_df.drop([index], inplace=True)

    train_df.index = range(len(train_df))
    train_df.drop(columns=['area'], inplace=True)
    return train_df

def split_data(df, n=5):
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=47)

    df_folds = df[['image_id']].copy()
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'source'] = df[['image_id', 'source']].groupby('image_id').min()['source']
    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['source'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    )
    df_folds.loc[:, 'fold'] = 0
    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

    return df_folds

def get_loader(params):
    # n0_df = pd.read_csv('{}/train.csv'.format(params.input_dir))
    n0_df = pd.read_csv('/home/huys/wheat_detection/wheat_detection_efficientdet/distortion/noiselevel1.csv')
    image_dir = '{}/train'.format(params.input_dir)

    n0_df = get_xywh(n0_df)
    n0_df = filter_bbox(n0_df)

    image_ids = n0_df['image_id'].unique()
    count = len(image_ids)
    train_ids = image_ids[:-int(count * 0.2)]
    # train_ids = image_ids[:64]
    random.shuffle(train_ids)
    valid_ids = image_ids[-int(count * 0.2):]

    # split_df = split_data(n0_df, n=5)
    # valid_ids = list(split_df[split_df['fold'] == fold_number].index)
    # train_ids = list(split_df[split_df['fold'] != fold_number].index)

    train_df = n0_df[n0_df['image_id'].isin(train_ids)]
    valid_df = n0_df[n0_df['image_id'].isin(valid_ids)]

    train_set = WheatDataset(train_df, image_dir, get_train_transform(params.img_size), noise_level=params.noise_level)
    val_set = WheatDataset(valid_df, image_dir, get_valid_transform(params.img_size))

    train_loader = DataLoader(
        train_set,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, train_set


def noise_loader(params):
    
    n0_df_dir = check_file('noiselevel0.csv')
    n0_df = pd.read_csv(n0_df_dir)

    # nx_df_dir = check_file(f'noiselevel{params.noise_level}.csv')
    nx_df_dir = check_file(f'noiselevel1.csv')
    nx_df = pd.read_csv(nx_df_dir)

    get_xywh(n0_df)
    get_xywh(nx_df)

    image_dir = '{}/train'.format(params.input_dir)

    image_ids = n0_df['image_id'].unique()
    count = len(image_ids)
    train_ids = image_ids[:-int(count * 0.2)]
    random.shuffle(train_ids)
    valid_ids = image_ids[-int(count * 0.2):]

    train_df = nx_df[nx_df['image_id'].isin(train_ids)]
    valid_df = n0_df[n0_df['image_id'].isin(valid_ids)]

    train_set = WheatDataset(train_df, image_dir, get_train_transform(params.img_size), noise_level=params.noise_augment)
    val_set = WheatDataset(valid_df, image_dir, get_valid_transform(params.img_size))

    train_loader = myLoader(
        train_set,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, train_set


class myLoader(DataLoader):
    
    def __iter__(self, epoch):
        self.dataset.set_epoch(epoch)
        return super(myLoader, self).__iter__()

