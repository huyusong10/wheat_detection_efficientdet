import numpy as np
from .utils import calculate_iou, bbox_join, random_walk, clipboxes

# np.random.seed(9001)

p1 = [0, 0.3, 0.6, 1]
p2 = [0, 0.05, 0.1, 0.15]
p3 = [0, 0.05, 0.1, 0.15]
p4 = [0, 0.025, 0.05, 0.1]

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


def deviate_bbox(bboxes, img_size, intensity=0.2, p=0):

    if p == 0:
        return bboxes

    new_boxes = []
    for box in bboxes:
        if np.random.rand() < p:
            new_boxes.append(random_walk(box, img_size, intensity))
        else:
            new_boxes.append(box)
    new_boxes = clipboxes(np.array(new_boxes), img_size)

    return new_boxes


def random_gen(bboxes, img_size, p=0):

    if p == 0:
        return bboxes

    new_boxes = []

    mean_w = np.mean(bboxes[..., 2] - bboxes[..., 0])
    mean_h = np.mean(bboxes[..., 3] - bboxes[..., 1])

    for box in bboxes:
        if np.random.rand() < p:
            x1 = np.floor(np.random.rand() * img_size*0.9)
            y1 = np.floor(np.random.rand() * img_size*0.9)
            w = (np.random.rand() * 2) * mean_w
            h = (np.random.rand() * 2) * mean_h
            new_boxes.append(np.array([x1, y1, w+x1, h+y1]))
        new_boxes.append(box)

    new_boxes = clipboxes(np.array(new_boxes), img_size)

    return new_boxes


def distort(bboxes, img_size, noise_level=0):

    if noise_level == 0:
        return bboxes

    chart = np.ones((bboxes.shape[0], bboxes.shape[0])) * -1

    for i in range(bboxes.shape[0]):
        for j in range(bboxes.shape[0]-i-1):
            iou = calculate_iou(bboxes[i], bboxes[j+i+1])
            chart[i, j+i+1] = iou

    np.random.seed(10010)

    bboxes = merge_bbox(bboxes, chart, p=p1[noise_level])
    bboxes = del_bbox(bboxes, p=p2[noise_level])
    bboxes = deviate_bbox(bboxes, img_size, intensity=0.2, p=p3[noise_level])
    bboxes = random_gen(bboxes, img_size, p=p4[noise_level])
    bboxes = np.floor(bboxes)
    bboxes = bboxes[(bboxes[:, 2] != bboxes[:, 0]) & (bboxes[:, 3] != bboxes[:, 1])]

    return bboxes

