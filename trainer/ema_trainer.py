import os

import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn import DataParallel
from datasets.aflw import get_train_loader, get_test_loader
from models.network import Network
from models.mean_teacher_network import MeanTeacher_CPS
import torch.optim as optim
from losses.losses import *
from tqdm import tqdm
from utils.utils import AverageMeter, NME, freeze_bn, unfreeze_bn, local_filter
import wandb


class EMATrainer:
    def __init__(self, config):
        self.labeled_train_loader = get_train_loader(unsupervised=False)
        self.unlabeled_train_loader = get_train_loader(unsupervised=True)
        self.len_loader = max(len(self.labeled_train_loader), len(self.unlabeled_train_loader))
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
            self.criterion_cps = AdaptiveWingLoss(alpha=self.config.alpha,
                                                  omega=self.config.omega,
                                                  epsilon=self.config.epsilon,
                                                  theta=self.config.theta,
                                                  use_target_weight=self.config.use_target_weight,
                                                  loss_weight=self.config.loss_weight)
            # self.criterion_cps = MSELoss()

        elif self.config.loss == 'mse':
            self.criterion = MSELoss()
            self.criterion_cps = MSELoss()

        if self.config.use_aux_loss:
            self.criterion_aux = PeakLoss(H=self.config.img_height, W=self.config.img_width)
        self.model = MeanTeacher_CPS(num_classes=self.config.num_classes,
                                     model_size=self.config.model_size,
                                     model=self.config.model)
        self.model = DataParallel(self.model).to(self.config.device)
        self.base_lr = self.config.lr

        self.optimizer_l = optim.AdamW(params=self.model.module.branch1.parameters(),
                                       lr=self.config.lr,
                                       # alpha=self.config.momentum,
                                       weight_decay=self.config.weight_decay
                                       )
        self.optimizer_r = optim.AdamW(params=self.model.module.branch2.parameters(),
                                       lr=self.config.lr,
                                       # alpha=self.config.momentum,
                                       weight_decay=self.config.weight_decay)
        self.lr_scheduler_1 = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer_l,
                                                                   T_max=self.config.labeled_epoch * self.len_loader,
                                                                   eta_min=1e-5)
        self.lr_scheduler_2 = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer_l,
                                                                   T_max=self.config.joint_epoch * self.len_loader,
                                                                   eta_min=1e-5)
        self.current_epoch = 0
        self.evaluator = NME(h=self.config.img_height, w=self.config.img_width)
        wandb.init(project='Cross Pseudo Label Mean Teacher AFLW', entity='chaunm', resume=False, config=config,
                   name=self.config.name)

    def _eval_epoch(self, epoch):
        self.model.eval()
        pbar = tqdm(self.test_loader, total=len(self.test_loader), desc=f'Eval epoch {epoch}')
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
            mask = batch['mask_heatmap'].squeeze(1).unsqueeze(2).to(self.config.device)
            landmark = batch['landmark'].to(self.config.device)
            with torch.no_grad():
                s_pred_1, t_pred_1 = self.model(image, step=1)
                s_pred_2, t_pred_2 = self.model(image, step=2)

                # loss for model 1
                loss_sup_student_1 = self.criterion(torch.sigmoid(s_pred_1), heatmap)
                loss_sup_teacher_1 = self.criterion(torch.sigmoid(t_pred_1), heatmap)
                loss_meter_student_1.update(loss_sup_student_1.item())
                loss_meter_teacher_1.update(loss_sup_teacher_1.item())

                nme_student_1 = self.evaluator(torch.sigmoid(s_pred_1), landmark, mask=mask[:, :-1, :].squeeze(-1))
                nme_teacher_1 = self.evaluator(torch.sigmoid(t_pred_1), landmark, mask=mask[:, :-1, :].squeeze(-1))
                nme_meter_student_1.update(nme_student_1.item())
                nme_meter_teacher_1.update(nme_teacher_1.item())

                # loss for model 2
                loss_sup_student_2 = self.criterion(torch.sigmoid(s_pred_2), heatmap)
                loss_sup_teacher_2 = self.criterion(torch.sigmoid(t_pred_2), heatmap)
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
                  'test/nme teacher 2': nme_meter_teacher_2.average()
                  }
        for k, v in result.items():
            wandb.log({k: v})
        return result

    def _train_supervise_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(range(self.len_loader), total=self.len_loader,
                    desc=f"Supervision only epoch {epoch}/{self.config.labeled_epoch}")
        loss_meter_student_1 = AverageMeter()
        loss_meter_student_2 = AverageMeter()
        loss_meter_teacher_1 = AverageMeter()
        loss_meter_teacher_2 = AverageMeter()

        nme_meter_student_1 = AverageMeter()
        nme_meter_student_2 = AverageMeter()
        nme_meter_teacher_1 = AverageMeter()
        nme_meter_teacher_2 = AverageMeter()

        self.current_epoch = epoch
        supervised_dataloader = iter(self.labeled_train_loader)
        for _ in pbar:
            batch = next(supervised_dataloader)
            self.optimizer_l.zero_grad()
            self.optimizer_r.zero_grad()
            image = batch['image'].to(self.config.device)
            heatmap = batch['heatmap'].to(self.config.device)
            mask = batch['mask_heatmap'].squeeze(1).unsqueeze(2).to(self.config.device)

            s_pred_1, t_pred_1 = self.model(image, step=1)
            s_pred_2, t_pred_2 = self.model(image, step=2)

            # loss for model 1
            loss_sup_student_1 = self.criterion(torch.sigmoid(s_pred_1), heatmap)
            loss_sup_teacher_1 = self.criterion(torch.sigmoid(t_pred_1), heatmap)
            loss_meter_student_1.update(loss_sup_student_1.item())
            loss_meter_teacher_1.update(loss_sup_teacher_1.item())

            # loss for model 2
            loss_sup_student_2 = self.criterion(torch.sigmoid(s_pred_2), heatmap, mask=mask)
            loss_sup_teacher_2 = self.criterion(torch.sigmoid(t_pred_2), heatmap, mask=mask)
            loss_meter_student_2.update(loss_sup_student_2.item())
            loss_meter_teacher_2.update(loss_sup_teacher_2.item())

            loss = loss_sup_student_1 + loss_sup_student_2
            loss.backward()
            clip_grad_norm_(self.model.module.branch1.parameters(), 5)
            clip_grad_norm_(self.model.module.branch2.parameters(), 5)

            # gradient update
            self.optimizer_l.step()
            self.optimizer_r.step()
            self.model.module._update_ema_variables(self.config.ema_decay)  # update weights for 2 teachers
            self.lr_scheduler_1.step()
            lr = self.lr_scheduler_1.get_last_lr()[0]
            wandb.log({'lr': lr})
            for i in range(len(self.optimizer_l.param_groups)):
                self.optimizer_l.param_groups[i]['lr'] = lr
            for i in range(len(self.optimizer_r.param_groups)):
                self.optimizer_r.param_groups[i]['lr'] = lr

            nme_student_1 = self.evaluator(torch.sigmoid(s_pred_1), heatmap, mask=mask[:, :-1, :].squeeze(-1))
            nme_teacher_1 = self.evaluator(torch.sigmoid(t_pred_1), heatmap, mask=mask[:, :-1, :].squeeze(-1))
            nme_meter_student_1.update(nme_student_1.item())
            nme_meter_teacher_1.update(nme_teacher_1.item())

            nme_student_2 = self.evaluator(torch.sigmoid(s_pred_2), heatmap, mask=mask[:, :-1, :].squeeze(-1))
            nme_teacher_2 = self.evaluator(torch.sigmoid(t_pred_2), heatmap, mask=mask[:, :-1, :].squeeze(-1))
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
        result = {'train/supervision loss student 1': loss_meter_student_1.average(),
                  'train/supervision loss student 2': loss_meter_student_2.average(),
                  'train/nme student 1': nme_meter_student_1.average(),
                  'train/nme student 2': nme_meter_student_2.average(),
                  'train/supervision loss teacher 1': loss_meter_teacher_1.average(),
                  'train/supervision loss teacher 2': loss_meter_teacher_2.average(),
                  'train/nme teacher 1': nme_meter_teacher_1.average(),
                  'train/nme teacher 2': nme_meter_teacher_2.average()
                  }
        for k, v in result.items():
            wandb.log({k: v})

    def _train_joint_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(range(self.len_loader), total=self.len_loader,
                    desc=f'Training joinly epoch {epoch}/{self.config.joint_epoch}')
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

        for _ in pbar:
            self.optimizer_l.zero_grad()
            self.optimizer_r.zero_grad()
            labeled_batch = next(supervised_dataloader)
            unlabeled_batch = next(unsupervised_dataloader)

            labeled_image = labeled_batch['image'].to(self.config.device)
            heatmap = labeled_batch['heatmap'].to(self.config.device)
            unlabeled_image = unlabeled_batch['image'].to(self.config.device)
            mask = labeled_batch['mask_heatmap'].squeeze(1).unsqueeze(2).to(self.config.device)
            landmark = labeled_batch['landmark'].to(self.config.device)
            with torch.no_grad():  # generate pseudo labels
                self.model.eval()
                unsup_pseudo_logits_teacher_1 = self.model(unlabeled_image, step=1)[1].detach()
                unsup_pseudo_logits_teacher_2 = self.model(unlabeled_image, step=2)[1].detach()

                sup_pseudo_logits_teacher_1 = self.model(labeled_image, step=1)[1].detach()
                sup_pseudo_logits_teacher_2 = self.model(labeled_image, step=2)[1].detach()

                unsup_pseudo_label_teacher_1 = torch.sigmoid(unsup_pseudo_logits_teacher_1)
                unsup_pseudo_label_teacher_2 = torch.sigmoid(unsup_pseudo_logits_teacher_2)

                sup_pseudo_label_teacher_1 = torch.sigmoid(sup_pseudo_logits_teacher_1)
                sup_pseudo_label_teacher_2 = torch.sigmoid(sup_pseudo_logits_teacher_2)

                unsup_mask_teacher_1 = local_filter(unsup_pseudo_label_teacher_1, self.config.threshold)
                unsup_mask_teacher_2 = local_filter(unsup_pseudo_label_teacher_2, self.config.threshold)

                sup_mask_teacher_1 = local_filter(sup_pseudo_label_teacher_1, self.config.threshold)
                sup_mask_teacher_2 = local_filter(sup_pseudo_label_teacher_2, self.config.threshold)

            self.model.train()

            logits_sup_student_1, logits_sup_teacher_1 = self.model(labeled_image, step=1)
            logits_sup_student_2, logits_sup_teacher_2 = self.model(labeled_image, step=2)
            sup_loss_student_1 = self.criterion(torch.sigmoid(logits_sup_student_1),
                                                heatmap, mask=None)
            sup_loss_student_2 = self.criterion(torch.sigmoid(logits_sup_student_2),
                                                heatmap, mask=None)
            loss_meter_sup_student_1.update(sup_loss_student_1.item())
            loss_meter_sup_student_2.update(sup_loss_student_2.item())
            loss_total = sup_loss_student_1 + sup_loss_student_2
            if epoch >= self.config.warm_up_epoch:
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

                loss_total += cps_loss * self.config.cps_loss_weight
            loss_total.backward()
            clip_grad_norm_(self.model.module.branch1.parameters(), 5)
            clip_grad_norm_(self.model.module.branch2.parameters(), 5)

            self.optimizer_l.step()
            self.optimizer_r.step()
            self.model.module._update_ema_variables(self.config.ema_decay)  # update weights for 2 teachers
            self.lr_scheduler_2.step()
            lr = self.lr_scheduler_2.get_last_lr()[0]
            wandb.log({'lr': lr})
            for i in range(len(self.optimizer_l.param_groups)):
                self.optimizer_l.param_groups[i]['lr'] = lr
            for i in range(len(self.optimizer_r.param_groups)):
                self.optimizer_r.param_groups[i]['lr'] = lr
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
                        round(nme_meter_teacher_2.average(), 2)]
            })

        result = {
            "train/loss_sup_student_1": loss_meter_sup_student_1.average(),
            "train/loss_sup_student_2": loss_meter_sup_student_2.average(),
            "train/loss_sup_cps": loss_meter_cps_sup.average(),
            "train/loss_unsup_cps": loss_meter_cps_unsup.average(),
            "train/nme_student_1": nme_meter_student_1.average(),
            "train/nme_student_2": nme_meter_student_2.average(),
            "train/nme_teacher_1": nme_meter_teacher_1.average(),
            "train/nme_teacher_2": nme_meter_teacher_2.average()

        }
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
