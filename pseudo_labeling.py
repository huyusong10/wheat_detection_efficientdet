import os
import torch
import pandas as pd
import numpy as np
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess


def predict_test(imgs_dir, model):
    testdf_psuedo = []
    for img_dir in os.listdir(imgs_dir):
        if not img_dir.endswith('.jpg'):
            continue
        full_dir = os.path.join(imgs_dir, img_dir)
        ori_imgs, framed_imgs, framed_metas = preprocess(full_dir, max_size=input_size)
        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
        x = x.to(torch.float32).permute(0, 3, 1, 2)
        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            out = postprocess(x, anchors,
                              regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)
        out = invert_affine(framed_metas, out)

        for i in range(len(ori_imgs)):
            if len(out[i]['rois']) == 0:
                result = {
                    'image_id': img_dir.split('.')[0],
                    'PredictionString': ''
                }
                testdf_psuedo.append(result)
                continue
            boxes = out[i]['rois'].astype(np.int)
            scores = out[i]['scores'].astype(np.float16)

            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            for j in zip(scores, boxes):
                if j[0] < score_threshold:
                    continue
                result = {
                    'image_id': img_dir.split('.')[0],
                    'width': 1024,
                    'height': 1024,
                    'bbox': "[{0:.1f} {1:.1f} {2:.1f} {3:.1f}]".format(j[1][0], j[1][1], j[1][2], j[1][3]),
                    'source': 'BULLBULL'
                }
                testdf_psuedo.append(result)

        test_df_pseudo = pd.DataFrame(testdf_psuedo,
                                      columns=['image_id', 'width', 'height', 'bbox', 'source'])
        return test_df_pseudo


def generate_train(test_df_pseudo):
    train_df = pd.read_csv('../input/global-wheat-detection/train.csv')
    frames = [train_df, test_df_pseudo]
    train_df = pd.concat(frames)

    return train_df


if __name__ == '__mian__':
    FILE_PREMODEL = r'../input/wheatdetection/wheat_detection/pth/final_stage.pth'
    DIR_TEST = r'../input/wheatdetection/wheat_detection/test'
    compound_coef = 0

    threshold = 0.3
    score_threshold = threshold
    iou_threshold = 0.2
    input_size = 512
    use_cuda = True

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=1,
                                 ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                                 scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    if use_cuda:
        model.load_state_dict(torch.load(FILE_PREMODEL, map_location=torch.device('cuda')))
    else:
        model.load_state_dict(torch.load(FILE_PREMODEL, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()

    test_df_pseudo = predict_test(DIR_TEST, model)
    final_train_csv = generate_train(test_df_pseudo)
