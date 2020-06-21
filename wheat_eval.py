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
compound_coef = 0
pth_path = '/home/huys/wheat_detection/result/model_without_precision_2/savedByLoss-d0_6_1211.pth'

threshold = 0.5
iou_threshold = 0.2
obj_list = ['wheat spike']
eval_thresholds = (0.5,)
batch_size = 16

val_params = {'batch_size': batch_size,
            'shuffle': False,
            'drop_last': True,
            'collate_fn': collate_fn,
            'num_workers': 4}

_, val_set = get_data_set(compound_coef)
val_generator = DataLoader(val_set, **val_params)

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

eval_result = []
for data in tqdm(val_generator):
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
        # out = postprocess(imgs,
        #                     anchors, regression, classification,
        #                     regressBoxes, clipBoxes,
        #                     threshold, iou_threshold)
        # use WBF
        out = postprocess(imgs,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            use_WBF=True, WBF_thr=0.5, WBF_skip_thr=0.0, input_size=512)

    batch_result = []
    for i in range(batch_size):
        preds = out[i]['rois'].astype(int)
        gts = batch_gts[i]
        gts = gts[gts[::,4] > -1].numpy()
        image_precision = calculate_image_precision(preds,
                                                    gts,
                                                    thresholds=eval_thresholds,
                                                    form='pascal_voc')
        batch_result.append(image_precision)

    mean_precision = np.mean(batch_result)
    print(mean_precision)
    eval_result.append(mean_precision)

valid_precision = np.mean(eval_result)
print('last:', valid_precision)


