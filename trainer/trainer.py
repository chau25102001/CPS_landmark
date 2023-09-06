import os

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn import DataParallel
from datasets.aflw import get_train_loader, get_test_loader
from models.network import Network
import torch.optim as optim
from losses.losses import *
from tqdm import tqdm
from utils.utils import AverageMeter, NME, heatmap2coordinate
import wandb


class Trainer:
    def __init__(self, config):
        self.labeled_train_loader = get_train_loader(config, unsupervised=False)
        self.unlabeled_train_loader = get_train_loader(config, unsupervised=True)
        self.len_loader = max(len(self.labeled_train_loader), len(self.unlabeled_train_loader))
        self.test_loader = get_test_loader(config)
        self.config = config
        if self.config.loss == 'awing':
            self.criterion = AdaptiveWingLoss(alpha=self.config.alpha,
                                              omega=self.config.omega,
                                              epsilon=self.config.epsilon,
                                              theta=self.config.theta,
                                              use_target_weight=self.config.use_target_weight,
                                              loss_weight=self.config.loss_weight,
                                              use_weighted_mask=self.config.use_weighted_mask)
            self.criterion_cps = AdaptiveWingLoss(alpha=self.config.alpha,
                                                  omega=self.config.omega,
                                                  epsilon=self.config.epsilon,
                                                  theta=self.config.theta,
                                                  use_target_weight=self.config.use_target_weight,
                                                  loss_weight=self.config.loss_weight)
        elif self.config.loss == 'mse':
            self.criterion = MSELoss()
            self.criterion_cps = MSELoss()

        if self.config.use_aux_loss:
            self.criterion_aux = PeakLoss(H=self.config.img_height, W=self.config.img_width)
        self.model = Network(num_classes=self.config.num_classes,
                             model_size=self.config.model_size,
                             model=self.config.model)
        self.model = DataParallel(self.model).to(self.config.device)
        self.base_lr = self.config.lr

        self.optimizer_l = optim.AdamW(params=self.model.module.branch1.parameters(),
                                       lr=self.config.lr,
                                       weight_decay=self.config.weight_decay
                                       )
        self.optimizer_r = optim.AdamW(params=self.model.module.branch2.parameters(),
                                       lr=self.config.lr,
                                       weight_decay=self.config.weight_decay)
        self.lr_scheduler_1 = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer_l,
                                                                   T_max=self.config.labeled_epoch * self.len_loader,
                                                                   eta_min=1e-5)
        self.lr_scheduler_2 = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer_l,
                                                                   T_max=self.config.joint_epoch * self.len_loader,
                                                                   eta_min=1e-5)
        self.current_epoch = 0
        self.evaluator = NME(h=self.config.img_height, w=self.config.img_width)
        wandb.init(project='Cross Pseudo Label AFLW', entity='chaunm', resume=False, config=config,
                   name=self.config.name)

    def _eval_epoch(self, epoch):
        self.model.eval()
        pbar = tqdm(self.test_loader, total=len(self.test_loader), desc=f"Eval epoch {epoch}")
        loss_meter_1 = AverageMeter()
        loss_meter_2 = AverageMeter()

        nme_meter_1 = AverageMeter()
        nme_meter_2 = AverageMeter()

        for batch in pbar:
            image = batch['image'].to(self.config.device)
            heatmap = batch['heatmap'].to(self.config.device)
            landmark = batch['landmark'].to(self.config.device)
            mask = batch['mask_heatmap'].squeeze(1).unsqueeze(2).to(self.config.device)
            with torch.no_grad():
                pred1 = self.model(image, step=1)
                pred2 = self.model(image, step=2)
                loss_sup_l = self.criterion(torch.sigmoid(pred1), heatmap, mask=None)
                loss_sup_r = self.criterion(torch.sigmoid(pred2), heatmap, mask=None)
                loss_meter_1.update(loss_sup_l.item())
                loss_meter_2.update(loss_sup_r.item())
                nme1 = self.evaluator(torch.sigmoid(pred1), landmark, mask[:, :-1, :].squeeze(-1))
                nme2 = self.evaluator(torch.sigmoid(pred2), landmark, mask[:, :-1, :].squeeze(-1))
                # nme1 = self.evaluator(torch.sigmoid(pred1), heatmap, None)
                # nme2 = self.evaluator(torch.sigmoid(pred2), heatmap, None)
                nme_meter_1.update(nme1.item())
                nme_meter_2.update(nme2.item())
                pbar.set_postfix({'loss': [round(loss_meter_1.average(), 2), round(loss_meter_2.average(), 2)],
                                  'nme': [round(nme_meter_1.average(), 2), round(nme_meter_2.average(), 2)]})
        result = {'test/supervision loss 1': loss_meter_1.average(),
                  'test/supervision loss 2': loss_meter_2.average(),
                  'test/nme 1': nme_meter_1.average(),
                  'test/nme 2': nme_meter_2.average()}
        for k, v in result.items():
            wandb.log({k: v})
        return result

    def _train_supervise_epoch(self, epoch):  # train with labeled data only
        self.model.train()
        # pbar = tqdm(self.labeled_train_loader, total=len(self.labeled_train_loader),
        #             desc=f'Supervision only epoch {epoch}/{self.config.labeled_epoch}')

        pbar = tqdm(range(self.len_loader), total=self.len_loader,
                    desc=f'Supervision only epoch {epoch}/{self.config.labeled_epoch}')

        loss_meter_1 = AverageMeter()
        loss_meter_2 = AverageMeter()

        nme_meter_1 = AverageMeter()
        nme_meter_2 = AverageMeter()

        aux_loss_meter_1 = AverageMeter()
        aux_loss_meter_2 = AverageMeter()
        self.current_epoch = epoch
        supervised_dataloader = iter(self.labeled_train_loader)
        for _ in pbar:
            batch = next(supervised_dataloader)
            self.optimizer_l.zero_grad()
            self.optimizer_r.zero_grad()
            image = batch['image'].to(self.config.device)
            heatmap = batch['heatmap'].to(self.config.device)
            mask = batch['mask_heatmap'].squeeze(1).unsqueeze(2).to(self.config.device)

            pred1 = self.model(image, step=1)
            pred2 = self.model(image, step=2)
            loss_sup_l = self.criterion(torch.sigmoid(pred1), heatmap, mask=None)
            loss_sup_r = self.criterion(torch.sigmoid(pred2), heatmap, mask=None)
            loss = 0
            if self.config.use_aux_loss:
                aux_loss_l = self.criterion_aux(pred1, heatmap, mask=mask.squeeze(-1))
                aux_loss_r = self.criterion_aux(pred2, heatmap, mask=mask.squeeze(-1))
                loss = loss + (aux_loss_l + aux_loss_r)
                aux_loss_meter_1.update(aux_loss_l.item())
                aux_loss_meter_2.update(aux_loss_r.item())
            loss = loss + loss_sup_l + loss_sup_r
            loss.backward()
            clip_grad_norm_(self.model.module.branch1.parameters(), 5)
            clip_grad_norm_(self.model.module.branch2.parameters(), 5)

            loss_meter_1.update(loss_sup_l.item())
            loss_meter_2.update(loss_sup_r.item())
            self.optimizer_l.step()
            self.optimizer_r.step()
            self.lr_scheduler_1.step()
            lr = self.lr_scheduler_1.get_last_lr()[0]
            # print(lr)
            wandb.log({'lr': lr})
            for i in range(len(self.optimizer_l.param_groups)):
                self.optimizer_l.param_groups[i]['lr'] = lr
            for i in range(len(self.optimizer_r.param_groups)):
                self.optimizer_r.param_groups[i]['lr'] = lr

            nme1 = self.evaluator(torch.sigmoid(pred1), heatmap, mask[:, :-1, :].squeeze(-1))
            nme2 = self.evaluator(torch.sigmoid(pred2), heatmap, mask[:, :-1, :].squeeze(-1))
            # nme1 = self.evaluator(torch.sigmoid(pred1), heatmap, None)
            # nme2 = self.evaluator(torch.sigmoid(pred2), heatmap, None)
            nme_meter_1.update(nme1.item())
            nme_meter_2.update(nme2.item())
            pbar.set_postfix({'loss': [round(loss_meter_1.average(), 2), round(loss_meter_2.average(), 2)],
                              'nme': [round(nme_meter_1.average(), 2), round(nme_meter_2.average(), 2)],
                              'aux loss': [round(aux_loss_meter_1.average(), 2), round(aux_loss_meter_2.average(), 2)]})
        wandb.log({'train/warm-up supervision loss 1': loss_meter_1.average()})
        wandb.log({'train/warm-up supervision loss 2': loss_meter_2.average()})
        wandb.log({'train/warm-up nme 1': nme_meter_1.average()})
        wandb.log({'train/warm-up nme 2': nme_meter_2.average()})

    def _train_joint_epoch(self, epoch):  # train with both labeled and unlabeled data
        self.model.train()
        pbar = tqdm(range(self.len_loader), total=self.len_loader,
                    desc=f'Training jointly epoch {epoch}/{self.config.joint_epoch}')
        supervised_dataloader = iter(self.labeled_train_loader)
        unsupervised_dataloader = iter(self.unlabeled_train_loader)
        self.current_epoch = epoch

        loss_sup_l_meter = AverageMeter()
        loss_sup_r_meter = AverageMeter()
        loss_sup_cps_meter = AverageMeter()
        loss_unsup_cps_meter = AverageMeter()
        nme_meter_1 = AverageMeter()
        nme_meter_2 = AverageMeter()

        for id in pbar:
            self.optimizer_l.zero_grad()
            self.optimizer_r.zero_grad()
            labeled_batch = next(supervised_dataloader)
            unlabled_batch = next(unsupervised_dataloader)

            labeled_image = labeled_batch['image'].to(self.config.device)
            heatmap = labeled_batch['heatmap'].to(self.config.device)
            unlabeled_image = unlabled_batch['image'].to(self.config.device)
            mask = labeled_batch['mask_heatmap'].squeeze(1).unsqueeze(2).to(self.config.device)

            with torch.no_grad():  # generate cross pseudo label
                self.model.eval()

                unsup_pseudo_logits_1 = self.model(unlabeled_image, step=1).detach()  # model 1
                unsup_pseudo_logits_2 = self.model(unlabeled_image, step=2).detach()  # model 2

                sup_pseudo_logits_1 = self.model(labeled_image, step=1).detach()
                sup_pseudo_logits_2 = self.model(labeled_image, step=2).detach()
            sup_pseudo_label_1 = torch.sigmoid(sup_pseudo_logits_1)
            sup_pseudo_label_2 = torch.sigmoid(sup_pseudo_logits_2)
            unsup_pseudo_label_1 = torch.sigmoid(unsup_pseudo_logits_1)  # pseudo heatmap 1
            unsup_pseudo_label_2 = torch.sigmoid(unsup_pseudo_logits_2)  # pseudo heatmap 1
            self.model.train()
            logits_unsup_1 = self.model(unlabeled_image, step=1)
            logits_unsup_2 = self.model(unlabeled_image, step=2)
            logits_sup_1 = self.model(labeled_image, step=1)
            logits_sup_2 = self.model(labeled_image, step=2)
            # cross pseudo label supervision
            unsup_cps_loss = self.criterion_cps(torch.sigmoid(logits_unsup_1), unsup_pseudo_label_2, mask=None) \
                             + self.criterion_cps(torch.sigmoid(logits_unsup_2), unsup_pseudo_label_1, mask=None)
            sup_cps_loss = self.criterion_cps(torch.sigmoid(logits_sup_1), sup_pseudo_label_2, mask=None) \
                           + self.criterion_cps(torch.sigmoid(logits_sup_2), sup_pseudo_label_1, mask=None)
            loss_sup_cps_meter.update(sup_cps_loss.item())
            loss_unsup_cps_meter.update(unsup_cps_loss.item())

            cps_loss = 1 / 2 * unsup_cps_loss + 1 / 2 * sup_cps_loss

            sup_loss_l = self.criterion(torch.sigmoid(logits_sup_1), heatmap, mask=None)
            sup_loss_r = self.criterion(torch.sigmoid(logits_sup_2), heatmap, mask=None)
            loss_sup_l_meter.update(sup_loss_l.item())
            loss_sup_r_meter.update(sup_loss_r.item())

            loss_total = cps_loss + sup_loss_r + sup_loss_l
            loss_total.backward()
            clip_grad_norm_(self.model.module.branch1.parameters(), 5)
            clip_grad_norm_(self.model.module.branch2.parameters(), 5)

            self.optimizer_r.step()
            self.optimizer_l.step()
            self.lr_scheduler_2.step()
            lr = self.lr_scheduler_2.get_last_lr()[0]
            wandb.log({'lr': lr})
            for i in range(len(self.optimizer_l.param_groups)):
                self.optimizer_l.param_groups[i]['lr'] = lr
            for i in range(len(self.optimizer_r.param_groups)):
                self.optimizer_r.param_groups[i]['lr'] = lr
            nme1 = self.evaluator(torch.sigmoid(logits_sup_1), heatmap, mask[:, :-1, :].squeeze(-1))
            nme2 = self.evaluator(torch.sigmoid(logits_sup_2), heatmap, mask[:, :-1, :].squeeze(-1))
            # nme1 = self.evaluator(torch.sigmoid(logits_sup_1), heatmap, None)
            # nme2 = self.evaluator(torch.sigmoid(logits_sup_2), heatmap, None)
            nme_meter_1.update(nme1.item())
            nme_meter_2.update(nme2.item())

            pbar.set_postfix(
                {'supervision loss': [round(loss_sup_l_meter.average(), 2), round(loss_sup_r_meter.average(), 2)],
                 'nme': [round(nme_meter_1.average(), 2), round(nme_meter_2.average(), 2)]})
        result = {"train/loss_sup_l": loss_sup_l_meter.average(),
                  "train/loss_sup_r": loss_sup_r_meter.average(),
                  "train/loss_sup_cps": loss_sup_cps_meter.average(),
                  "train/loss_unsup_cps": loss_unsup_cps_meter.average(),
                  "train/nme 1": nme_meter_1.average(),
                  "train/nme 2": nme_meter_2.average()}
        for k, v in result.items():
            wandb.log({k: v})

    def save_checkpoint(self, dir='checkpoint_last.pt'):
        checkpoint_path = os.path.join(self.config.snapshot_dir, dir)
        checkpoint = {'state_dict': self.model.module.state_dict(),
                      'optimizer_l': self.optimizer_l.state_dict(),
                      'optimizer_r': self.optimizer_r.state_dict(),
                      'lr_scheduler_1': self.lr_scheduler_1.state_dict(),
                      'lr_scheduler_2': self.lr_scheduler_2.state_dict(),
                      'epoch': self.current_epoch,
                      }
        torch.save(checkpoint, checkpoint_path)
        print("----> save checkpoint")

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.config.snapshot_dir, 'checkpoint_last.pt')
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.module.load_state_dict(checkpoint['state_dict'])
        self.optimizer_l.load_state_dict(checkpoint['optimizer_l'])
        self.optimizer_r.load_state_dict(checkpoint['optimizer_r'])
        self.lr_scheduler_1.load_state_dict(checkpoint['lr_scheduler_1'])
        self.lr_scheduler_2.load_state_dict(checkpoint['lr_scheduler_2'])
        self.current_epoch = checkpoint['epoch']
        print("----> load checkpoint")

    def train(self, resume=False):
        best_score = 1e9
        if resume:
            self.load_checkpoint()
        for epoch in range(self.current_epoch, self.config.labeled_epoch + self.config.joint_epoch):
            if epoch < self.config.labeled_epoch:
                self._train_supervise_epoch(epoch)
            else:
                if epoch == self.config.labeled_epoch:
                    lr = self.config.lr  # restart learning rate
                    wandb.log({'lr': lr})
                    for i in range(len(self.optimizer_l.param_groups)):
                        self.optimizer_l.param_groups[i]['lr'] = lr
                    for i in range(len(self.optimizer_r.param_groups)):
                        self.optimizer_r.param_groups[i]['lr'] = lr
                self._train_joint_epoch(epoch)
            result = self._eval_epoch(epoch)
            score = 0
            for k, v in result.items():
                if 'nme' in k:
                    score += v
            if score < best_score:
                best_score = score
                self.save_checkpoint('checkpoint_best.pt')
            self.save_checkpoint('checkpoint_last.pt')
