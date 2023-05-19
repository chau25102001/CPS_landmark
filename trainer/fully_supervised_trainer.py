import math
import os

import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn import DataParallel
from datasets.aflw import get_train_loader, get_test_loader
from models.network import SingleNetwork
from models.hgnet import HGNet
import torch.optim as optim
from losses.losses import *
from tqdm import tqdm
from utils.utils import AverageMeter, NME, Accuracy

import wandb


class FullySupervisedTrainer:
    def __init__(self, config):
        self.train_loader = get_train_loader(unsupervised=False)
        self.test_loader = get_test_loader()
        self.config = config
        if self.config.loss == 'awing':
            self.criterion = AdaptiveWingLoss(alpha=self.config.alpha,
                                              omega=self.config.omega,
                                              epsilon=self.config.epsilon,
                                              theta=self.config.theta,
                                              use_target_weight=self.config.use_target_weight,
                                              loss_weight=self.config.loss_weight,
                                              use_weighted_mask=self.config.use_weighted_mask)
        elif self.config.loss == 'mse':
            self.criterion = MSELoss()

        self.criterion_class = BCELoss()
        if self.config.model == 'hgnet':
            self.model = HGNet(num_classes=self.config.num_classes)
        else:
            self.model = SingleNetwork(num_classes=self.config.num_classes,
                                       model_size=self.config.model_size,
                                       model=self.config.model)

        self.model = DataParallel(self.model).to(self.config.device)
        self.base_lr = self.config.lr
        self.optimizer = optim.AdamW(params=self.model.parameters(),
                                     lr=self.config.lr,
                                     # alpha=self.config.momentum,
                                     weight_decay=self.config.weight_decay
                                     )
        # self.optimizer = optim.RMSprop(params=self.model.parameters(),
        #                                lr=self.config.lr,
        #                                alpha=0.99,
        #                                weight_decay=self.config.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                 T_max=self.config.joint_epoch * len(self.train_loader),
                                                                 eta_min=1e-6)
        self.current_epoch = 0
        self.evaluator = NME(h=self.config.img_height, w=self.config.img_width)
        self.evaluator_class = Accuracy(threshold=0.5)
        wandb.init(project='Fully supervised AFLW-neck', entity='chaunm', resume=False, config=config,
                   name=self.config.name)

    def _eval_epoch(self, epoch):
        self.model.eval()
        pbar = tqdm(self.test_loader, total=len(self.test_loader), desc=f'Eval epoch {epoch + 1}')
        loss_meter = AverageMeter()
        loss_class_meter = AverageMeter()

        nme_meter = AverageMeter()
        acc_meter = AverageMeter()

        for batch in pbar:
            image = batch['image'].to(self.config.device)
            heatmap = batch['heatmap'].to(self.config.device)
            mask = batch['mask_heatmap'].squeeze(1).unsqueeze(2).to(self.config.device)  # bsize x (k+1) x 1
            class_label = batch['mask_heatmap'].squeeze(1)[:, :-1].to(self.config.device)  # bsize x k
            landmark = batch['landmark'].to(self.config.device)
            with torch.no_grad():
                heatmap_pred, class_pred = self.model(image)

                loss = self.criterion(torch.sigmoid(heatmap_pred), heatmap, mask)
                nme = self.evaluator(torch.sigmoid(heatmap_pred), landmark, mask=mask[:, :-1, :].squeeze(-1))

                loss_meter.update(loss.item())
                nme_meter.update(nme.item())
                if self.config.classification:
                    loss_class = self.criterion_class(class_pred, class_label)
                    acc = self.evaluator_class(class_pred, class_label)

                    loss_class_meter.update(loss_class.item())
                    acc_meter.update(acc.item())
            pbar.set_postfix({
                "loss": loss_meter.average(),
                "loss class": loss_class_meter.average(),
                "nme": nme_meter.average(),
                "acc": acc_meter.average()
            })
        result = {
            "test/loss": loss_meter.average(),
            "test/loss class": loss_class_meter.average(),
            "test/nme": nme_meter.average(),
            "test/acc": acc_meter.average()
        }
        for k, v in result.items():
            wandb.log({k: v})
        return result

    def _train_epoch(self, epoch):
        self.model.train()
        len_loader = len(self.train_loader)
        pbar = tqdm(enumerate(range(len_loader)), total=len_loader,
                    desc=f'Train epoch {epoch + 1}/{self.config.joint_epoch}')
        data_loader = iter(self.train_loader)
        self.current_epoch = epoch
        loss_meter = AverageMeter()
        loss_class_meter = AverageMeter()

        nme_meter = AverageMeter()
        acc_meter = AverageMeter()

        for i, _ in pbar:
            self.optimizer.zero_grad()
            batch = next(data_loader)
            image = batch['image'].to(self.config.device)
            heatmap = batch['heatmap'].to(self.config.device)
            mask = batch['mask_heatmap'].squeeze(1).unsqueeze(2).to(self.config.device)  # bsize x (k+1) x 1
            class_label = batch['mask_heatmap'].squeeze(1)[:, :-1].to(self.config.device)  # bsize x k
            landmark = batch['landmark'].to(self.config.device)

            heatmap_pred, class_pred = self.model(image)
            loss = self.criterion(torch.sigmoid(heatmap_pred), heatmap, mask)
            nme = self.evaluator(torch.sigmoid(heatmap_pred), landmark, mask=mask[:, :-1, :].squeeze(-1))

            loss_meter.update(loss.item())
            nme_meter.update(nme.item())
            if self.config.classification:
                loss_class = self.criterion_class(class_pred, class_label)
                acc = self.evaluator_class(class_pred, class_label)

                loss_class_meter.update(loss_class.item())
                acc_meter.update(acc.item())
                loss += loss_class

            loss.backward()
            clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            wandb.log({"lr": lr})
            pbar.set_postfix({
                "loss": loss_meter.average(),
                "loss class": loss_class_meter.average(),
                "nme": nme_meter.average(),
                "acc": acc_meter.average()
            })
        result = {
            "train/loss": loss_meter.average(),
            "train/loss class": loss_class_meter.average(),
            "train/nme": nme_meter.average(),
            "train/acc": acc_meter.average()
        }
        for k, v in result.items():
            wandb.log({k: v})

    def save_checkpoint(self, dir='checkpoint_last.pt'):
        checkpoint_path = os.path.join(self.config.snapshot_dir, dir)
        checkpoint = {'state_dict': self.model.module.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'lr_scheduler': self.lr_scheduler.state_dict(),
                      'epoch': self.current_epoch,
                      }
        torch.save(checkpoint, checkpoint_path)
        print("----> save checkpoint")

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.config.snapshot_dir, 'checkpoint_last.pt')
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.module.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_l'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_1'])
        self.current_epoch = checkpoint['epoch']
        print("----> load checkpoint")

    def train(self, resume=False):
        best_score = 1e9
        if resume:
            self.load_checkpoint()

        for epoch in range(self.current_epoch, self.config.joint_epoch):
            self._train_epoch(epoch)
            result = self._eval_epoch(epoch)
            score = 0
            for k, v in result.items():
                if 'nme' in k:
                    score += v
            if score < best_score:
                best_score = score
                self.save_checkpoint('checkpoint_best.pt')
            self.save_checkpoint('checkpoint_last.pt')
