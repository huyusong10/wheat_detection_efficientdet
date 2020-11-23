import os
import copy
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import postprocess
from utils.eval_utils import calculate_image_precision


class Runner():

    def __init__(self, params, model, optim, torch_device, loss, writer, scheduler, ckpt=None):
        self.params = params
        self.save_dir = params.save_dir
        self.result = os.path.join(self.save_dir, 'results.txt')
        self.writer = writer
        self.device = torch_device
        self.model = model

        self.ema = copy.deepcopy(model.module).cpu()
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.ema_decay = params.ema_decay

        self.criterion = loss
        self.optim = optim
        self.scheduler = scheduler

        self.start_epoch = 0
        self.best_metric = 0.5
        self.best_valid_loss = np.inf
        self.conf_thresh = params.conf_thresh

        if ckpt:
            self.load(ckpt)

        
    def save(self, epoch, filename):
        torch.save({"compound_coef": self.params.compound_coef,
                    "start_epoch": epoch,
                    "model": self.model.module.state_dict(),
                    "ema": self.ema.state_dict(),  # after training, either 'network' or 'ema' could be use
                    "optimizer": self.optim.state_dict(),
                    "best_metric": self.best_metric,
                    "best_valid_loss": self.best_valid_loss
                    }, self.save_dir + "/%s.pth" % (filename))
        print("Model saved %d epoch" % (epoch))


    def load(self, ckpt):
        self.ema.load_state_dict(ckpt['ema'])
        self.optim.load_state_dict(ckpt['optimizer'])
        self.start_epoch = ckpt['start_epoch']
        self.best_metric = ckpt["best_metric"]
        self.best_valid_loss = ckpt["best_valid_loss"]
        print("Model Loaded. Resuming training from epoch {}".format(self.start_epoch))


    def update_ema(self):
        with torch.no_grad():
            named_param = dict(self.net.module.named_parameters())
            for k, v in self.ema.named_parameters():
                param = named_param[k].detach().cpu()
                v.copy_(self.ema_decay * v + (1 - self.ema_decay) * param)


    def train(self, train_loader, val_loader=None):
        ni = len(train_loader)
        step = 0
        for epoch in range(self.start_epoch, self.params.epoch):
            self.model.train()
            cls_loss_ls = []
            reg_loss_ls = []

            pbar = tqdm(enumerate(train_loader.__iter__(epoch)), total=ni)
            for i, data in pbar:
                imgs = data['imgs'].to(self.device, non_blocking=True)
                labels = data['bboxes'].to(self.device, non_blocking=True)

                self.optim.zero_grad()
                _, regression, classification, anchors = self.model(imgs)
                cls_loss, reg_loss = self.criterion(classification, regression, anchors, labels)

                loss = cls_loss + reg_loss
                cls_loss = cls_loss.mean().item()
                reg_loss = reg_loss.mean().item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optim.step()

                cls_loss_ls.append(cls_loss)
                reg_loss_ls.append(reg_loss)

                self.writer.add_scalars('cls_loss', {'train set': cls_loss}, step)
                self.writer.add_scalars('reg_loss', {'train set': reg_loss}, step)

                pbar.set_description('Step{}. Epoch: {}/{}. Iteration: {}/{}. Cls_loss: {:.5f}. Reg_loss: {:.5f}.'
                        .format(step, epoch, self.params.epoch-1, i,
                                ni-1, cls_loss, reg_loss))
                step += 1

            self.scheduler.step()
            if val_loader is not None:
                cls_loss, reg_loss, precision, recall, f1, mAP = self.valid(epoch, val_loader, step)
            else:
                raise RuntimeError('val_loader not existed!')

            self.writer.add_scalar('lr', self.optim.state_dict()['param_groups'][0]['lr'], epoch)

            with open(self.result, 'a') as f:
                f.write('Epoch: {}/{}  Train cls_loss: {:.5f}  Train reg_loss: {:.5f}  Val cls_loss: {: .5f}  Val reg_loss: {: .5f}  Presicion: {: .5f}  Recall: {: .5f}  F1: {: .5f}  mAP: {: .5f}'
                        .format(epoch, self.params.epoch-1, np.mean(cls_loss_ls), np.mean(reg_loss_ls), cls_loss, reg_loss, precision, recall, f1, mAP) + '\n')
        # torch.cuda.empty_cache()
        return self.best_metric

        
    def valid(self, epoch, val_loader, step):
        print('validating...')

        cls_loss_ls = []
        reg_loss_ls = []
        precision_ls = []
        recall_ls = []
        f1_ls = []
        mAP_ls = []

        self.model.eval()
        with torch.no_grad():
            for data in tqdm(val_loader):

                imgs = data['imgs'].to(self.device, non_blocking=True)
                labels_cpu = data['bboxes'].int()
                labels = data['bboxes'].to(self.device, non_blocking=True)

                _, regression, classification, anchors = self.model(imgs)
                cls_loss, reg_loss = self.criterion(classification, regression, anchors, labels)

                cls_loss_ls.append(cls_loss.mean().item())
                reg_loss_ls.append(reg_loss.mean().item())

                out = postprocess(imgs, anchors[0], regression, classification, BBoxTransform(), ClipBoxes(), self.params.threshold, self.params.iou_threshold)

                for i in range(len(out)):
                    preds = out[i]['rois'].astype(int)

                    gts = labels_cpu[i]
                    gts = gts[gts[..., 4] > -1][..., 0:4].numpy()

                    precision, recall, f1, mAP = calculate_image_precision(gts, preds, thresholds=eval(self.params.eval_thresholds))
                    
                    precision_ls.append(precision)
                    recall_ls.append(recall)
                    f1_ls.append(f1)
                    mAP_ls.append(mAP)

            cls_loss = np.mean(cls_loss_ls)
            reg_loss = np.mean(reg_loss_ls)
            precision = np.mean(precision_ls)
            recall = np.mean(recall_ls)
            f1 = np.mean(f1_ls)
            mAP = np.mean(mAP_ls)

            self.writer.add_scalars('cls_loss', {'val set': cls_loss}, step)
            self.writer.add_scalars('reg_loss', {'val set': reg_loss}, step)
            self.writer.add_scalar('Precision', precision, epoch)
            self.writer.add_scalar('Recall', recall, epoch)
            self.writer.add_scalar('F1', f1, epoch)
            self.writer.add_scalar('mAP{}'.format(self.params.eval_thresholds), mAP, epoch)

            print('valid complete!')
            print('cls_loss: {:.5f}, reg_loss: {:.5f}, precision: {:.5f}, recall: {:.5f}, f1: {:.5f}, mAP: {:.5f}'.format(cls_loss, reg_loss, precision, recall, f1, mAP))

            if mAP > self.best_metric:
                print('saving model...')
                self.best_metric = mAP
                self.save(epoch, "ckpt_%d_%.4f" % (epoch, mAP))

            return cls_loss, reg_loss, precision, recall, f1, mAP
                    
    
