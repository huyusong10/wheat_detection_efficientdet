import numpy as np
import copy
from .utils import calculate_iou, bbox_join, random_walk, clipboxes

p1 = [0, 0.3, 0.6, 1]       # merge
p2 = [0, 0.02, 0.035, 0.05]     # del bbox
p3 = [0, 0.15, 0.25, 0.35]       # deviate
p4 = [0, 0.025, 0.05, 0.1]      #  gen
p5 = [0, 0.15, 0.35, 0.5]       # scaling

def merge_bbox(bboxes, chart, p=0):

    if p == 0:
        return bboxes
 
    new_boxes = []
    flags = [False] * bboxes.shape[0]
    for i in range(bboxes.shape[0]):
        for j in range(bboxes.shape[0]-i-1):   
            if chart[i, j+i+1]>0.2:
                if np.random.rand() >= p:
                    continue
                new_boxes.append(bbox_join(bboxes[i], bboxes[j+i+1]))
                flags[i] = True
                flags[j+i+1] = True

        if not flags[i]:
            new_boxes.append(bboxes[i].tolist())

    return np.array(new_boxes)
        

def del_bbox(bboxes, p=0):

    if p == 0:
        return bboxes

    new_boxes = []
    for box in bboxes:
        if np.random.rand() < p:
            continue
        new_boxes.append(box)
    
    if len(new_boxes) == 0:
        return bboxes
    else:
        return np.array(new_boxes)


def deviate_bbox(bboxes, img_size, intensity=0.1, p=0):

    if p == 0:
        return bboxes

    new_boxes = []
    flag = np.zeros(bboxes.shape[0])
    for i, box in enumerate(bboxes):
        if np.random.rand() < p:
            new_boxes.append(random_walk(box, img_size, intensity))
            flag[i] = 1
        else:
            new_boxes.append(box)
    new_boxes = clipboxes(np.array(new_boxes), img_size)

    return new_boxes, flag


def random_scaling(bboxes, flag, intensity=0.1, p=0):

    if p == 0:
        return bboxes

    new_boxes = []

    for i, box in enumerate(bboxes):
        if np.random.rand() < p and flag[i] == 0:
            scale = (np.random.rand() * 3 - 2) * intensity
            scale *= 0.5 if scale < 0 else scale
            w = np.floor((box[2] - box[0]) * scale)
            h = np.floor((box[3] - box[1]) * scale)
            new_boxes.append(np.array([box[0]+w, box[1]+h, box[2]-w, box[3]-h]))
        else:
            new_boxes.append(box)
     
    return np.array(new_boxes)


def random_gen(bboxes, img_size, p=0):

    if p == 0:
        return bboxes

    new_boxes = []

    mean_w = np.mean(bboxes[..., 2] - bboxes[..., 0])
    mean_h = np.mean(bboxes[..., 3] - bboxes[..., 1])

    for box in bboxes:
        if np.random.rand() < p:
            x1 = np.floor(np.random.rand() * img_size)
            y1 = np.floor(np.random.rand() * img_size)
            w = (np.random.rand() * 2) * mean_w
            h = (np.random.rand() * 2) * mean_h
            new_boxes.append(np.array([x1, y1, w+x1, h+y1]))
        new_boxes.append(box)

    new_boxes = clipboxes(np.array(new_boxes), img_size)

    return new_boxes

def random_occlude(img, bboxes, p=0):

    if p == 0:
        return img, bboxes
    
    new_img = copy.deepcopy(img)
    l = bboxes.shape[0]

    for box in bboxes:
        if np.random.rand() < p:
            pass


def augment(img, bboxes, noise_level=0):

    if noise_level == 0:
        return img, bboxes
    
    img_size = img.shape[1]

    # shape = bboxes.shape

    # chart = np.ones((bboxes.shape[0], bboxes.shape[0])) * -1

    # for i in range(bboxes.shape[0]):
    #     for j in range(bboxes.shape[0]-i-1):
    #         iou = calculate_iou(bboxes[i], bboxes[j+i+1])
    #         chart[i, j+i+1] = iou
    # bboxes = merge_bbox(bboxes, chart, p=p1[noise_level])

    bboxes = del_bbox(bboxes, p=p2[noise_level])
    bboxes, flag = deviate_bbox(bboxes, img_size, intensity=0.1, p=p3[noise_level])
    bboxes = random_scaling(bboxes, flag, intensity=0.2, p=p5[noise_level])
    # bboxes = random_gen(bboxes, img_size, p=p4[noise_level])
    bboxes = np.round(bboxes)
    # print('orignal:', shape, bboxes.shape[0] - shape[0])
    
    bboxes = bboxes[(bboxes[:, 2] != bboxes[:, 0]) & (bboxes[:, 3] != bboxes[:, 1])]
    bboxes = clipboxes(bboxes, img_size)

    return img, bboxes


