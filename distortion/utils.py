import numpy as np
import copy
import sys

def calculate_iou(b1, b2, form='pascal_voc'):

    if form == 'coco':
        b1 = xywh2xyxy(b1)
        b2 = xywh2xyxy(b2)

    dx = min(b1[2], b2[2]) - max(b1[0], b2[0]) + 1
    if dx < 0:
        return 0.0
    dy = min(b1[3], b2[3]) - max(b1[1], b2[1]) + 1

    if dy < 0:
        return 0.0
    overlap_area = dx * dy
    union_area = ((b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1) +
                  (b2[2] - b2[0] + 1) * (b2[3] - b2[1] + 1) - overlap_area)

    return overlap_area / union_area

def xywh2xyxy(box):

    trans = copy.deepcopy(box)
    trans[2] = trans[0] + trans[2]
    trans[3] = trans[1] + trans[3]

    return trans

def bbox_join(b1, b2, form='pascal_voc'):

    if form == 'coco':
        b1 = xywh2xyxy(b1)
        b2 = xywh2xyxy(b2)

        x1 = min(b1[0], b2[0])
        y1 = min(b1[1], b2[1])
        x2 = max(b1[2], b2[2])
        y2 = max(b1[3], b2[3])

        return np.array([x1, y1, x2-x1, y2-y1])

    elif form == 'pascal_voc':
        
        x1 = min(b1[0], b2[0])
        y1 = min(b1[1], b2[1])
        x2 = max(b1[2], b2[2])
        y2 = max(b1[3], b2[3])

        return np.array([x1, y1, x2, y2])

def random_walk(bbox, img_size, intensity):

    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    step = (w + h) / 2

    walker = (np.random.rand() * 2 - 1) * intensity *step

    return bbox+walker

def clipboxes(bboxes, img_size):

    new_boxes = copy.deepcopy(bboxes)

    try:
        new_boxes[..., 0] = np.clip(bboxes[..., 0], a_min=0, a_max=img_size-1)
        new_boxes[..., 1] = np.clip(bboxes[..., 1], a_min=0, a_max=img_size-1)
        new_boxes[..., 2] = np.clip(bboxes[..., 2], a_min=0, a_max=img_size-1)
        new_boxes[..., 3] = np.clip(bboxes[..., 3], a_min=0, a_max=img_size-1)
    except:
        print(new_boxes)
        sys.exit()

    return new_boxes

     


