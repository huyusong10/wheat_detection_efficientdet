import datetime
import os

import torch
import numpy as np
import yaml
from torch import nn
from backbone import EfficientDetBackbone
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, LambdaLR

from efficientdet.loss import FocalLoss
from utils.utils import check_file, select_device, init_seeds
# from efficientdet.utils import BBoxTransform, ClipBoxes
# from utils.eval_utils import calculate_image_precision, calculate_precision
# from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights
# from utils.sync_batchnorm import patch_replication_callback

from wheat_data import get_loader, noise_loader
from runner import Runner

class Params:
    def __init__(self, file=r'./params.yml'):
        self.params = yaml.safe_load(open(file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

    def get_dict(self):
        return self.params


def get_model(params):

    model = EfficientDetBackbone(num_classes=len(params.obj_list),
                                 compound_coef=params.compound_coef,
                                 ratios=eval(params.anchors_ratios),
                                 scales=eval(params.anchors_scales))

    assert params.weights.endswith('.pth'), 'weights file no correct!'
    if not params.resume:
        params.weights = check_file(params.weights)
        ckpt = None
        try:
            model.load_state_dict(torch.load(params.weights), strict=False)
        except RuntimeError as e:
            print('redundant pretrained weights was deprecated.')
    else:
        params.resume = check_file(params.resume)
        ckpt = torch.load(params.resume, strict=False)
        model.load_state_dict(ckpt['model'], strict=False)
    return model, ckpt

def get_scheduler(optim, sche_type, step_size, epochs):
    if sche_type == "exp":
        return StepLR(optim, step_size, 0.9)
    elif sche_type == "auto":
        return LambdaLR(optim, lambda x: (((1 + np.cos(x * np.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1)
    else:
        return None


if __name__ == '__main__':
    params_file = 'params.yml'
    params_file = check_file(params_file)
    params = Params(params_file)
    init_seeds(10086)

    params.save_dir = os.path.join(os.getcwd(), params.save_dir)
    os.makedirs(params.save_dir, exist_ok=True)
    device = select_device(params.device, batch_size=params.batch_size)

    train_loader, val_loader, _ = noise_loader(params)

    model, ckpt = get_model(params)
    model = nn.DataParallel(model).to(device)
    loss = FocalLoss()

    optim = {
        "adamw" : lambda : torch.optim.AdamW(model.parameters(), lr=params.lr, betas=eval(params.beta), weight_decay=params.weight_decay),
        "SGD": lambda : torch.optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum, nesterov=True, weight_decay=params.weight_decay)
    }[params.optim]()

    scheduler = get_scheduler(optim, params.scheduler, 1, params.epoch)

    writer = SummaryWriter(params.save_dir +f'/{datetime.datetime.now().strftime("%Y%m%d:%H%M")}'+f'-{params.project_name}') 
    params.save_dir = writer.logdir
    with open(os.path.join(params.save_dir, 'params.yml'), 'w') as f:  # save parameters
        yaml.dump(params.get_dict(), f, sort_keys=False)
    
    runner = Runner(params, model, optim, device, loss, writer, scheduler, ckpt)
    try:
        runner.train(train_loader, val_loader)
    except KeyboardInterrupt:
        print('interrupt by user')
    finally:
        writer.close()
        torch.cuda.empty_cache()
        print('training complete')

