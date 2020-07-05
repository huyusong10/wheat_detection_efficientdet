import datetime
import os, sys
import argparse
import traceback

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from backbone import EfficientDetBackbone
from tensorboardX import SummaryWriter
import numpy as np
from tqdm.autonotebook import tqdm

from efficientdet.loss import FocalLoss
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.eval_utils import calculate_image_precision, calculate_precision
from utils.utils import postprocess
from utils.utils import replace_w_sync_bn, CustomDataParallel, CustomPrecisionParallel, get_last_weights, init_weights
from utils.sync_batchnorm import patch_replication_callback

from wheat_data import get_data_set, collate_fn

class Params:
    def __init__(self, file=r'./params.yml'):
        self.params = yaml.safe_load(open(file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


# def get_args():
#     parser = argparse.ArgumentParser('wheat detection used efficientdet')
#     parser.add_argument('-p', '--project', type=str, default='wheat_detection')

#     parser.add_argument('-c', '--compound_coef', type=int, default=2)
#     parser.add_argument('-n', '--num_workers', type=int, default=4, help='num_workers of dataloader')
#     parser.add_argument('--batch_size', type=int, default=4, help='The number of images per batch among all devices')
#     parser.add_argument('--head_only', type=boolean_string, default=False,
#                         help='whether finetunes only the regressor and the classifier, '
#                               'useful in early stage convergence or small/easy dataset')
#     parser.add_argument('--lr', type=float, default=1e-4)
#     parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
#                                                                    'suggest using \'admaw\' until the'
#                                                                    ' very final stage then switch to \'sgd\'')
#     parser.add_argument('--num_epochs', type=int, default=20)
#     parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
#     parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
#     parser.add_argument('--es_min_delta', type=float, default=0.0,
#                         help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
#     parser.add_argument('--es_patience', type=int, default=0,
#                         help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')

#     parser.add_argument('--log_path', type=str, default='/home/huys/wheat_detection/result/logs/tensorboard/')
#     parser.add_argument('-w', '--load_weights', type=str, default='/home/huys/wheat_detection/efficientdet-d2.pth',
#                         help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
#     parser.add_argument('--saved_path', type=str, default='/home/huys/wheat_detection/result/model/')
#     parser.add_argument('--debug', type=boolean_string, default=False, help='whether visualize the predicted boxes of training, '
#                                                                   'the output images will be in test/')

#     args = parser.parse_args()
#     return args
# def boolean_string(s):
#     if s not in {'False', 'True'}:
#         raise ValueError('Not a valid boolean string')
#     return s == 'True'


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification,
                                                regression,
                                                anchors,
                                                annotations,
                                                imgs=imgs,
                                                obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression,
                                                anchors, annotations)
        return cls_loss, reg_loss


def train(params):

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if params.num_gpus ==1:
        torch.cuda.set_device(params.cuda_id)
    torch.manual_seed(42)

    os.makedirs(params.log_path, exist_ok=True)
    os.makedirs(params.saved_path, exist_ok=True)

    training_params = {
        'batch_size': params.batch_size,
        'shuffle': True,
        'drop_last': True,
        'collate_fn': collate_fn,
        'num_workers': params.num_workers
    }

    val_params = {
        'batch_size': params.batch_size,
        'shuffle': False,
        'drop_last': True,
        'collate_fn': collate_fn,
        'num_workers': params.num_workers
    }

    training_set, val_set = get_data_set(params.compound_coef)

    training_generator = DataLoader(training_set, **training_params)
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(params.obj_list),
                                 compound_coef=params.compound_coef,
                                 ratios=eval(params.anchors_ratios),
                                 scales=eval(params.anchors_scales))

    # load last weights
    if params.load_weights is not None:
        if params.load_weights.endswith('.pth'):
            weights_path = params.load_weights
        else:
            weights_path = get_last_weights(params.saved_path)
        try:
            last_step = int(
                os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '警告：看到这条信息不要慌张，这可能是因为pretrained的模型的类型数量和你训练的不同，其余的weight已经加载完毕。'
            )

        print(
            f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}'
        )
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if params.head_only:

        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    if params.num_gpus > 1 and params.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False


    writer = SummaryWriter(
        params.log_path +
        f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'+f'_d{params.compound_coef}_{params.lr}/')

    model_with_loss = ModelWithLoss(model, debug=params.debug)

    if params.num_gpus > 0:
        model_with_loss = model_with_loss.cuda()
        if params.num_gpus > 1:
            model_with_loss = CustomDataParallel(model_with_loss, params.num_gpus)
            model = CustomPrecisionParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model_with_loss)

    if params.optim == 'adamw':
        optimizer = torch.optim.AdamW(model_with_loss.parameters(), params.lr)
    elif params.optim == 'adamax':
        optimizer = torch.optim.Adamax(model_with_loss.parameters(), params.lr)
    elif params.optim == 'SGD':
        optimizer = torch.optim.SGD(model_with_loss.parameters(),
                                    params.lr,
                                    momentum=0.9,
                                    nesterov=True)
    else:
        print(f'无{params.optim}优化器')
        raise Exception

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.1,
                                                           patience=1,
                                                           min_lr=1e-6)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, 3e-6)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5000, gamma=0.1)

    epoch = 0
    best_loss = 1e5
    best_precision = 0
    best_epoch = 0
    step = max(0, last_step)
    num_iter_per_epoch = len(training_generator)
    model_with_loss.train()
    use_precision = params.train_with_precision
    precision = 0.0

    try:
        for epoch in range(params.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iters, data in enumerate(progress_bar):
                if iters < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']

                    # 脏数据
                    # if np.any(np.isnan(imgs.numpy().astype('float32'))):
                    #     continue

                    if params.num_gpus == 1:
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    cls_loss, reg_loss = model_with_loss(imgs,
                                               annot,
                                               obj_list=params.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        print('loss等于0或者无限')
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_with_loss.parameters(), 0.1)
                    optimizer.step()
                    # scheduler.step()
                    # scheduler.step(loss)

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        '第{}步。 轮次: {}/{}。 迭代次数: {}/{}. 分类loss: {:.5f}. 回归loss: {:.5f}. 总loss: {:.5f}'
                        .format(step, epoch, params.num_epochs, iters + 1,
                                num_iter_per_epoch, cls_loss.item(),
                                reg_loss.item(), loss.item()))
                    writer.add_scalars('总loss', {'训练集': loss}, step)
                    writer.add_scalars('回归loss', {'训练集': reg_loss}, step)
                    writer.add_scalars('分类loss', {'训练集': cls_loss}, step)

                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('学习率', current_lr, step)

                    step += 1

                    if step % params.save_interval == 0 and step > 0:
                        save_checkpoint(
                            model_with_loss,
                            f'savedByCheckpoint-d{params.compound_coef}_{epoch}_{step}.pth'
                        )
                        print(f'检查点，保存模型savedByCheckpoint-d{params.compound_coef}_{epoch}_{step}.pth')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    sys.exit()

            scheduler.step(np.mean(epoch_loss))

            if epoch % params.val_interval == 0:
                loss_regression_ls = []
                loss_classification_ls = []
                precision_ls = []
                model_with_loss.eval()

                # calculate valid set loss and precision(use mAP)
                for iters, data in enumerate(val_generator):

                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']
                        if use_precision:
                            batch_gts = data['annot'].int()

                        if params.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model_with_loss(
                            imgs, annot, obj_list=params.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()
                        loss = cls_loss + reg_loss

                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        if use_precision:
                            features, regression, classification, anchors = model(imgs)
                            regressBoxes = BBoxTransform()
                            clipBoxes = ClipBoxes()

                            out = postprocess(imgs,
                                            anchors, regression, classification,
                                            regressBoxes, clipBoxes,
                                            params.threshold, params.iou_threshold)
                            batch_precision = []
                            for i in range(params.batch_size):
                                preds = out[i]['rois'].astype(int)
                                if preds.size == 0:
                                    batch_precision.append(0)
                                    continue
                                gts = batch_gts[i]
                                gts = gts[gts[::,4] > -1].numpy()
                                image_precision = calculate_image_precision(gts,
                                                                            preds,
                                                                            thresholds=eval(params.eval_thresholds),)
                                batch_precision.append(image_precision)
                            mean_precision = np.mean(batch_precision)
                            precision_ls.append(mean_precision)


                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                        

                if not loss_classification_ls or not loss_regression_ls:
                    continue
                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                if use_precision:
                    precision = np.mean(precision_ls)
                loss = cls_loss + reg_loss

                print(
                    '测试集结果：轮次: {}/{}. 分类loss: {:1.5f}. 回归loss: {:1.5f}. 总loss: {:1.5f}. 精确度: {:1.5f}'
                    .format(epoch, params.num_epochs, cls_loss, reg_loss,
                            loss, precision))
                writer.add_scalars('总loss', {'测试集': loss}, step)
                writer.add_scalars('回归loss', {'测试集': reg_loss}, step)
                writer.add_scalars('分类loss', {'测试集': cls_loss}, step)
                if use_precision:
                    writer.add_scalars('精确度', {'测试集': precision}, step)

                loss_save = False
                if loss + params.es_min_delta < best_loss:
                    best_loss = loss
                    print(f'最佳总loss更新为{best_loss}，保存模型savedByLoss-d{params.compound_coef}_{epoch}_{step}.pth')
                    best_epoch = epoch

                    save_checkpoint(
                        model_with_loss,
                        f'savedByLoss-d{params.compound_coef}_{epoch}_{step}.pth'
                    )
                    loss_save = True

                if precision > best_precision and use_precision:
                    best_precision = precision
                    print(f'最佳精确度更新为{best_precision}，若本次没有通过loss保存模型，则保存为savedByPrecision-d{params.compound_coef}_{epoch}_{step}.pth')
                    if not loss_save:
                        save_checkpoint(
                            model_with_loss,
                            f'savedByPrecision-d{params.compound_coef}_{epoch}_{step}.pth'
                        )

                model_with_loss.train()

                # Early stopping
                if epoch - best_epoch > params.es_patience > 0:
                    print(
                        '[Info] 停止训练 {}. 最低能达到的loss是 {}'
                        .format(epoch, best_loss))
                    break
    except KeyboardInterrupt:
        save_checkpoint(
            model_with_loss, f'saveByInterrupt-d{params.compound_coef}_{epoch}_{step}.pth')
        writer.close()
    finally:
        print('本次训练信息总结：\n最佳loss为{:.5f}，最佳轮次为{}，最佳精确度为{:.5f}'.format(best_loss,best_epoch,best_precision))
    writer.close()


def save_checkpoint(model, name):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(),
                   os.path.join(params.saved_path, name))
    else:
        torch.save(model.model.state_dict(),
                   os.path.join(params.saved_path, name))


if __name__ == '__main__':
    params = Params(r'./params.yml')
    train(params)