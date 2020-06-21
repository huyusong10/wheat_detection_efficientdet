import torch._namedtensor_internals

from backbone import EfficientDetBackbone
from tqdm import tqdm
from tensorboardX import SummaryWriter

from wheat_eval import eval_data
from wheat_data import get_data_set, collate_fn
from wheat_data import get_data_set, collate_fn

torch.cuda.set_device(3)
use_cuda = False
compound_coef = 0
pth_path = '/home/huys/wheat_detection/result/model_without_precision_2/savedByLoss-d0_5_1260.pth'

obj_list = ['wheat spike']
eval_thresholds = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75)
batch_size = 16

val_params = {'batch_size': batch_size,
            'shuffle': False,
            'drop_last': True,
            'collate_fn': collate_fn,
            'num_workers': 16}

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

    writer = SummaryWriter('params_auto_adjust')

    best_precision = 0.0
    best_thresh = 0
    best_iou_thresh = 0
    for i in tqdm(range(30, 61, 1)):
        for j in range(15, 36, 1):
            result = eval_data(val_set, val_params, model, i/100, j/100)
            writer.add_scalars('precision_over_iou_thresh', result, i*31+j)
            if result > best_precision:
                best_precision = result
                best_thresh = i/100
                best_iou_thresh = j/100
                print(f'最佳精确度更新为{best_precision},此时threshold为{i},iou_threshold为{j}')
        writer.add_scalars('best_precision_over_thresh', best_precision, i)
    writer.close()
    