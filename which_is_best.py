import torch
import numpy as np
from torch.utils.data import DataLoader

from backbone import EfficientDetBackbone
from tqdm import tqdm
import pickle
from tensorboardX import SummaryWriter

from wheat_eval import eval_data
from wheat_data import get_data_set, collate_fn

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import postprocess_on_iou_threshold
from utils.eval_utils import calculate_image_precision, calculate_precision


torch.cuda.set_device(3)
use_cuda = False
compound_coef = 4
pth_path = '/home/huys/wheat_detection/result/model_d4_1e-3_ReduceLROnPlateau/savedByPrecision-d4_23_16128.pth'

obj_list = ['wheat spike']
eval_thresholds = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75)
batch_size = 32

val_params = {'batch_size': batch_size,
            'shuffle': False,
            'drop_last': True,
            'collate_fn': collate_fn,
            'num_workers': 32}

if __name__ == '__main__':

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

    gts_ls = []
    anchors_ls = []
    classification_ls = []


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
            transformed_anchors = regressBoxes(anchors, regression)
            transformed_anchors = clipBoxes(transformed_anchors, imgs)
            gts_ls.append(batch_gts)
            anchors_ls.append(transformed_anchors)
            classification_ls.append(classification)


    
    writer = SummaryWriter('params_auto_adjust')

    best_precision = 0
    best_thresh = 0
    best_iou_thresh = 0

    all_result = []

    max_i = 60
    max_j = 40

    for i in tqdm(range(30, max_i+1)):
        threshold = i/100
        thresh_result = []
        for j in tqdm(range(15, max_j+1)):
            iou_threshold = j/100
            step_result = []
            for iters in range(len(val_generator)):
          
                gts_on_batch = gts_ls[iters]
                anchors = anchors_ls[iters]
                classification = classification_ls[iters]
                scores = torch.max(classification, dim=2, keepdim=True)[0]
                scores_over_thresh = (scores > threshold)[:, :, 0]
                out = postprocess_on_iou_threshold(batch_size, classification, anchors, scores, scores_over_thresh, iou_threshold)
                for x in range(len(out)):
                    preds = out[x]['rois'].astype(int)
                    if preds.size == 0:
                        step_result.append(0)
                        continue
                    gts = gts_on_batch[x]
                    gts = gts[gts[::,4] > -1].numpy()

                    image_precision = calculate_image_precision(gts,
                                                                preds)
                    step_result.append(image_precision)

            precision_over_step = np.mean(step_result)
            thresh_result.append(precision_over_step)
            if j == max_j:
                all_result.append(thresh_result)

            writer.add_scalar('precision_over_step', precision_over_step, i*31+j)
            if precision_over_step > best_precision:
                best_precision = precision_over_step
                best_thresh = threshold
                best_iou_thresh = iou_threshold
                print(f'最佳精确度更新为{best_precision}，此时threshold为{best_thresh}，iou_threshold为{best_iou_thresh}')
        
        precision_over_thresh = np.mean(thresh_result)
        writer.add_scalar('precision_over_thresh', precision_over_thresh, i)
    print(f'此次测试最佳精确度为{best_precision}，此时threshold为{best_thresh}，iou_threshold为{best_iou_thresh}')

    precision_over_iou_thresh = np.mean(all_result, axis=0)
    # with open('tmp.pkl', 'wb') as f:
    #     pickle.dump(all_result, f)

    for i in range(len(precision_over_iou_thresh)):
        writer.add_scalar('precision_over_iou_thresh', precision_over_iou_thresh[i], i+15)
    writer.close()


    