import os

import termcolor
import torch
import torch.optim as optim
import wandb
from torch.nn import DataParallel
from tqdm import tqdm

from torch.nn.utils import clip_grad_norm_

from datasets.aflw import get_train_loader, get_test_loader
from losses.losses import *
from models.mean_teacher_network_vit import MeanTeacher_CPS_ViT
from utils.utils import AverageMeter, NME, adaptive_local_filter, get_theshold, local_filter, scale_function


def ema_decay_scheduler(start_ema_decay, end_ema_decay, max_step, step):
    if step > max_step:
        return end_ema_decay
    else:
        return start_ema_decay + (end_ema_decay - start_ema_decay) / max_step * step


class EMAVitTrainer:
    def __init__(self, config):
        self.labeled_train_loader = get_train_loader(config,
                                                     unsupervised=False)  # __getitem__ return image and annotations
        self.unlabeled_train_loader = get_train_loader(config, unsupervised=True)  # __getitem__ return image only
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

        self.model = MeanTeacher_CPS_ViT(config)
        self.model = DataParallel(self.model).to(config.device)
        self.base_lr = config.lr
        self.optimizer_l = optim.AdamW(params=self.model.module.branch1.parameters(),
                                       lr=self.config.lr,
                                       weight_decay=self.config.weight_decay
                                       )

        self.optimizer_r = optim.AdamW(params=self.model.module.branch2.parameters(),
                                       lr=self.config.lr,
                                       weight_decay=self.config.weight_decay)
        if config.lr_scheduler == 'cosine':
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer_l,
                                                                     T_max=self.config.total_epoch * self.len_loader,
                                                                     eta_min=1e-6)
        elif config.lr_scheduler == 'linear_warm_up':
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer_l,
                                                            lr_lambda=lambda step: ((step + 1) / (
                                                                config.warm_up_step)) if step < config.warm_up_step
                                                            else 1 - ((step + 1 - config.warm_up_step) / (
                                                                    config.total_epoch * self.len_loader - config.warm_up_step)))

        self.current_epoch = 0
        self.current_step = 0
        self.evaluator = NME(h=config.img_height, w=config.img_width)
        wandb.init(project="CPS EMA MAE for AFLW", entity='chaunm', resume=False,
                   name=self.config.name, config=config)

        self.ema_decay = self.config.start_ema_decay
        self.threshold_ema_decay = 0.99
        if self.config.adaptive_filter:
            self.adaptive_threshold_sup_1 = torch.ones((1, self.config.num_classes, 1),
                                                       device=self.config.device) * 0.5
            self.adaptive_threshold_sup_2 = torch.ones((1, self.config.num_classes, 1),
                                                       device=self.config.device) * 0.5
            self.adaptive_threshold_unsup_1 = torch.ones((1, self.config.num_classes, 1),
                                                         device=self.config.device) * 0.5
            self.adaptive_threshold_unsup_2 = torch.ones((1, self.config.num_classes, 1),
                                                         device=self.config.device) * 0.5

            print(termcolor.colored(
                f"ADAPTIVE THRESHOLD, CPS weight: {config.cps_loss_weight}, reduction: {config.threshold_reduction}, apply scale function: {config.apply_function}, clip norm: {config.clip_norm}",
                "blue"))
        self.count_channel_meters_sup_1 = [AverageMeter() for _ in range(self.config.num_classes)]
        self.count_channel_meters_sup_2 = [AverageMeter() for _ in range(self.config.num_classes)]
        self.count_channel_meters_unsup_1 = [AverageMeter() for _ in range(self.config.num_classes)]
        self.count_channel_meters_unsup_2 = [AverageMeter() for _ in range(self.config.num_classes)]
        self.warm_up_freeze_epoch = self.config.total_epoch // 4
        self.best_score = 1e9

    def _eval_epoch(self, epoch):
        self.model.eval()
        pbar = tqdm(self.test_loader, total=len(self.test_loader), desc=f'Eval epoch {epoch + 1}')
        loss_meter_student_1 = AverageMeter()
        loss_meter_student_2 = AverageMeter()
        loss_meter_teacher_1 = AverageMeter()
        loss_meter_teacher_2 = AverageMeter()

        nme_meter_student_1 = AverageMeter()
        nme_meter_student_2 = AverageMeter()
        nme_meter_teacher_1 = AverageMeter()
        nme_meter_teacher_2 = AverageMeter()
        for batch in pbar:
            image = batch['image'].to(self.config.device)
            heatmap = batch['heatmap'].to(self.config.device)
            mask = batch['mask_heatmap'].squeeze(1).unsqueeze(2).to(self.config.device)  # bsize x (k+1) x 1
            landmark = batch['landmark'].to(self.config.device)
            with torch.no_grad():
                s_pred_1, t_pred_1 = self.model(image, step=1)
                s_pred_2, t_pred_2 = self.model(image, step=2)

                # loss for model 1
                loss_sup_student_1 = self.criterion(torch.sigmoid(s_pred_1), heatmap, mask)
                loss_sup_teacher_1 = self.criterion(torch.sigmoid(t_pred_1), heatmap, mask)
                loss_meter_student_1.update(loss_sup_student_1.item())
                loss_meter_teacher_1.update(loss_sup_teacher_1.item())

                nme_student_1 = self.evaluator(torch.sigmoid(s_pred_1), landmark, mask=mask[:, :-1, :].squeeze(-1))
                nme_teacher_1 = self.evaluator(torch.sigmoid(t_pred_1), landmark, mask=mask[:, :-1, :].squeeze(-1))
                nme_meter_student_1.update(nme_student_1.item())
                nme_meter_teacher_1.update(nme_teacher_1.item())

                # loss for model 2
                loss_sup_student_2 = self.criterion(torch.sigmoid(s_pred_2), heatmap, mask)
                loss_sup_teacher_2 = self.criterion(torch.sigmoid(t_pred_2), heatmap, mask)
                loss_meter_student_2.update(loss_sup_student_2.item())
                loss_meter_teacher_2.update(loss_sup_teacher_2.item())

                nme_student_2 = self.evaluator(torch.sigmoid(s_pred_2), landmark, mask=mask[:, :-1, :].squeeze(-1))
                nme_teacher_2 = self.evaluator(torch.sigmoid(t_pred_2), landmark, mask=mask[:, :-1, :].squeeze(-1))
                nme_meter_student_2.update(nme_student_2.item())
                nme_meter_teacher_2.update(nme_teacher_2.item())

                pbar.set_postfix({
                    'loss': [round(loss_meter_student_1.average(), 2),
                             round(loss_meter_teacher_1.average(), 2),
                             round(loss_meter_student_2.average(), 2),
                             round(loss_meter_teacher_2.average(), 2)],
                    'nme': [round(nme_meter_student_1.average(), 2),
                            round(nme_meter_teacher_1.average(), 2),
                            round(nme_meter_student_2.average(), 2),
                            round(nme_meter_teacher_2.average(), 2)]
                })
        result = {'test/supervision loss student 1': loss_meter_student_1.average(),
                  'test/supervision loss student 2': loss_meter_student_2.average(),
                  'test/nme student 1': nme_meter_student_1.average(),
                  'test/nme student 2': nme_meter_student_2.average(),

                  'test/supervision loss teacher 1': loss_meter_teacher_1.average(),
                  'test/supervision loss teacher 2': loss_meter_teacher_2.average(),
                  'test/nme teacher 1': nme_meter_teacher_1.average(),
                  'test/nme teacher 2': nme_meter_teacher_2.average(),
                  }
        for k, v in result.items():
            wandb.log({k: v})
        return result

    def _train_epoch(self, epoch):
        if epoch == self.warm_up_freeze_epoch:
            print("unfreezing")
            for p in self.model.parameters():
                p.requires_grad = True
        self.model.train()
        pbar = tqdm(enumerate(range(self.len_loader)), total=self.len_loader,
                    desc=f'Training joinly epoch {epoch + 1}/{self.config.total_epoch}')
        supervised_dataloader = iter(self.labeled_train_loader)
        unsupervised_dataloader = iter(self.unlabeled_train_loader)
        self.current_epoch = epoch

        loss_meter_sup_student_1 = AverageMeter()
        loss_meter_sup_student_2 = AverageMeter()
        loss_meter_cps_sup = AverageMeter()
        loss_meter_cps_unsup = AverageMeter()

        nme_meter_student_1 = AverageMeter()
        nme_meter_student_2 = AverageMeter()
        nme_meter_teacher_1 = AverageMeter()
        nme_meter_teacher_2 = AverageMeter()

        for i, _ in pbar:
            self.current_step += 1
            self.optimizer_l.zero_grad()
            self.optimizer_r.zero_grad()

            labeled_batch = next(supervised_dataloader)
            unlabeled_batch = next(unsupervised_dataloader)

            unlabeled_image = unlabeled_batch['image'].to(self.config.device)
            labeled_image = labeled_batch['image'].to(self.config.device)
            heatmap = labeled_batch['heatmap'].to(self.config.device)
            mask = labeled_batch['mask_heatmap'].squeeze(1).unsqueeze(2).to(self.config.device)  # bsize x (k+1) x1
            landmark = labeled_batch['landmark'].to(self.config.device)

            logits_sup_student_1, logits_sup_teacher_1 = self.model(labeled_image, step=1)
            logits_sup_student_2, logits_sup_teacher_2 = self.model(labeled_image, step=2)

            sup_loss_student_1 = self.criterion(torch.sigmoid(logits_sup_student_1),
                                                heatmap, mask=mask)
            sup_loss_student_2 = self.criterion(torch.sigmoid(logits_sup_student_2),
                                                heatmap, mask=mask)

            loss_meter_sup_student_1.update(sup_loss_student_1.item())
            loss_meter_sup_student_2.update(sup_loss_student_2.item())

            cps_loss = 0
            with torch.no_grad():  # generate pseudo labels from teacher predictions
                self.model.eval()
                unlabeled_output_1 = self.model(unlabeled_image, step=1)
                unlabeled_output_2 = self.model(unlabeled_image, step=2)
                unsup_pseudo_logits_teacher_1 = unlabeled_output_1[1].detach()  # teacher 1 prediction
                unsup_pseudo_logits_teacher_2 = unlabeled_output_2[1].detach()  # teacher 2 prediction

                labeled_output_1 = self.model(labeled_image, step=1)
                labeled_output_2 = self.model(labeled_image, step=2)
                sup_pseudo_logits_teacher_1 = labeled_output_1[1].detach()  # teacher 1 prediction
                sup_pseudo_logits_teacher_2 = labeled_output_2[1].detach()  # teacher 2 prediction

                # heatmap pseudo label
                unsup_pseudo_label_teacher_1 = torch.sigmoid(unsup_pseudo_logits_teacher_1)
                unsup_pseudo_label_teacher_2 = torch.sigmoid(unsup_pseudo_logits_teacher_2)

                sup_pseudo_label_teacher_1 = torch.sigmoid(sup_pseudo_logits_teacher_1)
                sup_pseudo_label_teacher_2 = torch.sigmoid(sup_pseudo_logits_teacher_2)

                # filtering
                if self.config.adaptive_filter:
                    sup_max_teacher_1 = get_theshold(sup_pseudo_label_teacher_1, self.config.threshold_reduction)
                    sup_max_teacher_2 = get_theshold(sup_pseudo_label_teacher_2, self.config.threshold_reduction)
                    unsup_max_teacher_1 = get_theshold(unsup_pseudo_label_teacher_1,
                                                       self.config.threshold_reduction)
                    unsup_max_teacher_2 = get_theshold(unsup_pseudo_label_teacher_2,
                                                       self.config.threshold_reduction)
                    if self.config.adaptive_batched:
                        for b in range(self.config.train_batch_size):
                            self.adaptive_threshold_sup_1 = self.threshold_ema_decay * self.adaptive_threshold_sup_1 + (
                                    1 - self.threshold_ema_decay) * sup_max_teacher_1[b:b + 1, :, :]
                            self.adaptive_threshold_sup_2 = self.threshold_ema_decay * self.adaptive_threshold_sup_2 + (
                                    1 - self.threshold_ema_decay) * sup_max_teacher_2[b:b + 1, :, :]
                            self.adaptive_threshold_unsup_1 = self.threshold_ema_decay * self.adaptive_threshold_unsup_1 + (
                                    1 - self.threshold_ema_decay) * unsup_max_teacher_1[b:b + 1, :, :]
                            self.adaptive_threshold_unsup_2 = self.threshold_ema_decay * self.adaptive_threshold_unsup_2 + (
                                    1 - self.threshold_ema_decay) * unsup_max_teacher_2[b:b + 1, :, :]
                    else:
                        self.adaptive_threshold_sup_1 = self.threshold_ema_decay * self.adaptive_threshold_sup_1 + (
                                1 - self.threshold_ema_decay) * sup_max_teacher_1
                        self.adaptive_threshold_sup_2 = self.threshold_ema_decay * self.adaptive_threshold_sup_2 + (
                                1 - self.threshold_ema_decay) * sup_max_teacher_2
                        self.adaptive_threshold_unsup_1 = self.threshold_ema_decay * self.adaptive_threshold_unsup_1 + (
                                1 - self.threshold_ema_decay) * unsup_max_teacher_1
                        self.adaptive_threshold_unsup_2 = self.threshold_ema_decay * self.adaptive_threshold_unsup_2 + (
                                1 - self.threshold_ema_decay) * unsup_max_teacher_2
                    sup_mask_teacher_1 = adaptive_local_filter(sup_pseudo_label_teacher_1,
                                                               self.adaptive_threshold_sup_1,
                                                               apply_function=scale_function if self.config.apply_function else None)  # bx (k+1) x 1
                    sup_mask_teacher_2 = adaptive_local_filter(sup_pseudo_label_teacher_2,
                                                               self.adaptive_threshold_sup_2,
                                                               apply_function=scale_function if self.config.apply_function else None)
                    unsup_mask_teacher_1 = adaptive_local_filter(unsup_pseudo_label_teacher_1,
                                                                 self.adaptive_threshold_unsup_1,
                                                                 apply_function=scale_function if self.config.apply_function else None)
                    unsup_mask_teacher_2 = adaptive_local_filter(unsup_pseudo_label_teacher_2,
                                                                 self.adaptive_threshold_unsup_2,
                                                                 apply_function=scale_function if self.config.apply_function else None)
                else:
                    threshold_ema_decay = 0
                    unsup_mask_teacher_1, _ = local_filter(unsup_pseudo_label_teacher_1, self.config.threshold)
                    unsup_mask_teacher_2, _ = local_filter(unsup_pseudo_label_teacher_2, self.config.threshold)

                    sup_mask_teacher_1, _ = local_filter(sup_pseudo_label_teacher_1, self.config.threshold)
                    sup_mask_teacher_2, _ = local_filter(sup_pseudo_label_teacher_2, self.config.threshold)
                count_point_sup_1 = torch.mean(sup_mask_teacher_1, dim=0).squeeze(
                    -1).cpu().numpy().tolist()  # (k+1)
                count_point_sup_2 = torch.mean(sup_mask_teacher_2, dim=0).squeeze(
                    -1).cpu().numpy().tolist()  # (k+1)
                count_point_unsup_1 = torch.mean(unsup_mask_teacher_1, dim=0).squeeze(
                    -1).cpu().numpy().tolist()  # (k+1)
                count_point_unsup_2 = torch.mean(unsup_mask_teacher_2, dim=0).squeeze(
                    -1).cpu().numpy().tolist()  # (k+1)
                for i, _ in enumerate(count_point_sup_1):  # log number of pseudo labels used in this batch
                    self.count_channel_meters_sup_1[i].update(count_point_sup_1[i])
                    self.count_channel_meters_sup_2[i].update(count_point_sup_2[i])
                    self.count_channel_meters_unsup_1[i].update(count_point_unsup_1[i])
                    self.count_channel_meters_unsup_2[i].update(count_point_unsup_2[i])
                    wandb.log({f"count_point/sup1/{i}": self.count_channel_meters_sup_1[i].average()})
                    wandb.log({f"count_point/sup2/{i}": self.count_channel_meters_sup_2[i].average()})
                    wandb.log({f"count_point/unsup1/{i}": self.count_channel_meters_unsup_1[i].average()})
                    wandb.log({f"count_point/unsup2/{i}": self.count_channel_meters_unsup_2[i].average()})
            self.model.train()
            # cross pseudo supervision + ema + local filter
            logits_unsup_student_1, _ = self.model(unlabeled_image, step=1)
            logits_unsup_student_2, _ = self.model(unlabeled_image, step=2)

            unsup_cps_loss = self.criterion_cps(torch.sigmoid(logits_unsup_student_1),
                                                unsup_pseudo_label_teacher_2,
                                                mask=unsup_mask_teacher_2) \
                             + self.criterion_cps(torch.sigmoid(logits_unsup_student_2),
                                                  unsup_pseudo_label_teacher_1,
                                                  mask=unsup_mask_teacher_1)
            sup_cps_loss = self.criterion_cps(torch.sigmoid(logits_sup_student_1),
                                              sup_pseudo_label_teacher_2,
                                              mask=sup_mask_teacher_2) \
                           + self.criterion_cps(torch.sigmoid(logits_sup_student_2),
                                                sup_pseudo_label_teacher_1,
                                                mask=sup_mask_teacher_1)
            loss_meter_cps_sup.update(sup_cps_loss.item())
            loss_meter_cps_unsup.update(unsup_cps_loss.item())

            cps_loss = sup_cps_loss + unsup_cps_loss
            loss_total = sup_loss_student_1 + sup_loss_student_2 + cps_loss * self.config.cps_loss_weight
            loss_total.backward()
            if self.config.clip_norm is not None:  # clip grad norm
                clip_grad_norm_(self.model.module.branch1.parameters(), self.config.clip_norm)
                clip_grad_norm_(self.model.module.branch2.parameters(), self.config.clip_norm)

            self.optimizer_l.step()
            self.optimizer_r.step()
            self.model.module._update_ema_variables(self.ema_decay)  # update weights for 2 teachers
            self.ema_decay = ema_decay_scheduler(self.config.start_ema_decay,
                                                 self.config.end_ema_decay,
                                                 max_step=int(self.config.ema_linear_epoch * self.len_loader),
                                                 step=self.current_step)
            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            wandb.log({'lr': lr})
            for j in range(len(self.optimizer_l.param_groups)):
                self.optimizer_l.param_groups[j]['lr'] = lr
            for j in range(len(self.optimizer_r.param_groups)):
                self.optimizer_r.param_groups[j]['lr'] = lr

            nme_student1 = self.evaluator(torch.sigmoid(logits_sup_student_1), landmark, mask[:, :-1, :].squeeze(-1))
            nme_student2 = self.evaluator(torch.sigmoid(logits_sup_student_2), landmark, mask[:, :-1, :].squeeze(-1))
            nme_teacher1 = self.evaluator(torch.sigmoid(logits_sup_teacher_1), landmark, mask[:, :-1, :].squeeze(-1))
            nme_teacher2 = self.evaluator(torch.sigmoid(logits_sup_teacher_2), landmark, mask[:, :-1, :].squeeze(-1))

            nme_meter_student_1.update(nme_student1.item())
            nme_meter_student_2.update(nme_student2.item())
            nme_meter_teacher_1.update(nme_teacher1.item())
            nme_meter_teacher_2.update(nme_teacher2.item())
            pbar.set_postfix({
                'sup loss': [round(loss_meter_sup_student_1.average(), 2),
                             round(loss_meter_sup_student_2.average(), 2),
                             ],
                'cps loss': [round(loss_meter_cps_sup.average(), 2),
                             round(loss_meter_cps_unsup.average(), 2),
                             ],
                'nme': [round(nme_meter_student_1.average(), 2),
                        round(nme_meter_teacher_1.average(), 2),
                        round(nme_meter_student_2.average(), 2),
                        round(nme_meter_teacher_2.average(), 2)],
                'current_steps': self.current_step,
                'ema': [round(self.ema_decay, 2), round(self.threshold_ema_decay, 2)]
            })
        result = {
            "train/loss_sup_student_1": loss_meter_sup_student_1.average(),
            "train/loss_sup_student_2": loss_meter_sup_student_2.average(),
            "train/loss_sup_cps": loss_meter_cps_sup.average(),
            "train/loss_unsup_cps": loss_meter_cps_unsup.average(),
            "train/nme_student_1": nme_meter_student_1.average(),
            "train/nme_student_2": nme_meter_student_2.average(),
            "train/nme_teacher_1": nme_meter_teacher_1.average(),
            "train/nme_teacher_2": nme_meter_teacher_2.average(),

        }
        for k, v in result.items():
            wandb.log({k: v})

    def save_checkpoint(self, dir='checkpoint_last.pt', input=None):
        checkpoint_path = os.path.join(self.config.snapshot_dir, dir)
        checkpoint = {'state_dict': self.model.module.state_dict(),
                      'optimizer_l': self.optimizer_l.state_dict(),
                      'optimizer_r': self.optimizer_r.state_dict(),
                      'lr_scheduler': self.lr_scheduler.state_dict(),
                      'epoch': self.current_epoch,
                      'input': input
                      }
        torch.save(checkpoint, checkpoint_path)
        print("----> save checkpoint")

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.config.snapshot_dir, 'checkpoint_last.pt')
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.module.load_state_dict(checkpoint['state_dict'])
        self.optimizer_l.load_state_dict(checkpoint['optimizer_l'])
        self.optimizer_r.load_state_dict(checkpoint['optimizer_r'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_1'])
        self.current_epoch = checkpoint['epoch']
        print("----> load checkpoint")

    def train(self, resume=False):
        if resume:
            self.load_checkpoint()
        for epoch in range(self.current_epoch, self.config.total_epoch):
            self._train_epoch(epoch)
            result = self._eval_epoch(epoch)
            score = 0
            for k, v in result.items():
                if 'nme' in k:
                    score += v

            if score < self.best_score:
                self.best_score = score
                self.save_checkpoint('checkpoint_best.pt')
