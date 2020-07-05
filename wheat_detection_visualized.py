import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import matplotlib.pyplot as plt
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

compound_coef = 4
img_path = '/home/huys/wheat_detection/test/2fd875eaa.jpg'
pth_path = '/home/huys/wheat_detection/result/model_d4_3e-4_1200_0.7/savedByLoss-d4_2_2025.pth'

threshold = 0.5
iou_threshold = 0.25
input_size = 1024


obj_list = ['wheat spike']
ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
x = x.to(torch.float32).permute(0, 3, 1, 2)

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                             scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])


model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
model.requires_grad_(False)
model.eval()

with torch.no_grad():
    features, regression, classification, anchors = model(x)

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      use_WBF=False, WBF_thr=threshold, WBF_iou_thr=iou_threshold, input_size=1024)

out = invert_affine(framed_metas, out)


plt.figure(figsize=(8,8))
plt.axis('off')
for i in range(len(ori_imgs)):
    if len(out[i]['rois']) == 0:
        continue

    for j in range(len(out[i]['rois'])):
        (x1, y1, x2, y2) = out[i]['rois'][j].astype(np.int)
        cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
        obj = obj_list[out[i]['class_ids'][j]]
        score = float(out[i]['scores'][j])

        cv2.putText(ori_imgs[i], '{}, {:.3f}'.format(obj, score),
                    (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1)

        plt.imshow(ori_imgs[i][:,:,(2,1,0)])

plt.show()