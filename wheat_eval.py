import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

from backbone import EfficientDetBackbone
import numpy as np
import numba
from tqdm import tqdm

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import postprocess
from utils.eval_utils import calculate_image_precision, calculate_precision
from wheat_data import get_data_set, collate_fn

torch.cuda.set_device(0)
use_cuda = True
compound_coef = 4
pth_path = '/home/huys/wheat_detection/result/model_d4_1e-3_1200_0.5/savedByLoss-d4_3_2700.pth'
# pth_path = '/home/huys/wheat_detection/result/model_test/final_stage.pth'

threshold = 0.5
iou_threshold = 0.25
obj_list = ['wheat spike']
batch_size = 4

val_params = {'batch_size': batch_size,
            'shuffle': False,
            'drop_last': True,
            'collate_fn': collate_fn,
            'num_workers': 4}

def eval_data(dataset, dataset_params, model, threshold, iou_threshold):
    val_generator = DataLoader(dataset, **dataset_params)
    eval_result = []
    for i, data in enumerate(tqdm(val_generator)):
        if use_cuda:
            imgs = torch.stack([img.cuda() for img in data['img']], 0)
        else:
            imgs = torch.stack([img for img in data['img']], 0)
        batch_gts = data['annot'].int()

        with torch.no_grad():
            features, regression, classification, anchors = model(imgs)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            # use NMS
            out = postprocess(imgs,
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                threshold, iou_threshold)
            # use WBF
            # out = postprocess(imgs,
            #                     anchors, regression, classification,
            #                     regressBoxes, clipBoxes,
            #                     use_WBF=True, WBF_thr=0.5, WBF_iou_thr=0.55, input_size=512)

        batch_result = []
        for i in range(batch_size):
            preds = out[i]['rois'].astype(int)
            if preds.size == 0:
                batch_result.append(0)
                continue
            gts = batch_gts[i]
            gts = gts[gts[::,4] > -1].numpy()
            image_precision = calculate_image_precision(gts,
                                                        preds,
                                                        thresholds=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75),)
            batch_result.append(image_precision)

        mean_precision = np.mean(batch_result)
        # print(mean_precision)
        eval_result.append(mean_precision)

    valid_precision = np.mean(eval_result)
    # print('last:', valid_precision)
    return valid_precision

if __name__ == '__main__':

    _, val_set = get_data_set(compound_coef)

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                                scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    if use_cuda:
        model.load_state_dict(torch.load(pth_path, map_location=torch.device('cuda')))
    else:
        model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()
    if use_cuda:
        model.cuda()

    result = eval_data(val_set, val_params, model, threshold, iou_threshold)
    print(result)



