import os
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn import DataParallel
# from datasets.aflw import get_train_loader, get_test_loader
import datasets.aflw as aflw
import MAE.datasets.mae_neck as neck
from MAE.models.MAE import *
import torch.optim as optim
from tqdm import tqdm
import math
from utils.utils import AverageMeter

import wandb


class MAETrainer:
    def __init__(self, config, neck_data=False):
        if neck_data:
            self.train_loader = neck.get_train_loader(config, unsupervised=True)  # only load image
            self.test_loader = neck.get_test_loader(config)
        else:
            self.train_loader = aflw.get_train_loader(config, unsupervised=True)  # only load image
            self.test_loader = aflw.get_test_loader(config)
        self.config = config
        self.model = get_mae_model(encoder_embedding_dim=config.encoder_embedding_dim,
                                   encoder_layers=config.encoder_layers,
                                   n_heads_encoder_layer=config.n_heads_encoder_layer,
                                   decoder_embedding_dim=config.decoder_embedding_dim,
                                   decoder_layers=config.decoder_layers,
                                   n_heads_decoder_layer=config.n_heads_decoder_layer,
                                   patch_size=config.patch_size,
                                   num_patches=config.img_height // config.patch_size
                                   )
        self.model = DataParallel(self.model).to(self.config.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr,
                                     betas=(0.9, 0.95), weight_decay=config.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                              lr_lambda=lambda epoch:
                                                              (epoch + 1) / (
                                                                      config.warm_up_epoch + 1) if epoch < config.warm_up_epoch else (
                                                                  math.cos(
                                                                      (epoch + 1) / config.total_epoch * math.pi / 2)),
                                                              )
        self.current_epoch = 0
        self.best_score = 1e9
        wandb.init(project='MAE AFLW', entity='chaunm', resume=False, config=config,
                   name=self.config.name)

    def _train_epoch(self, epoch):
        self.model.train()
        len_loader = len(self.train_loader)
        pbar = tqdm(enumerate(range(len_loader)), total=len_loader,
                    desc=f'Train epoch {epoch + 1}/{self.config.total_epoch}')
        data_loader = iter(self.train_loader)
        self.current_epoch = epoch
        loss_meter = AverageMeter()
        for i, _ in pbar:
            data = next(data_loader)
            image = data['image'].to(self.config.device)
            image_patches = patchify(image, patch_size=self.config.patch_size)
            self.optimizer.zero_grad()
            predicted_patches, mask = self.model(image)
            loss = torch.sum(torch.mean(torch.square(image_patches - predicted_patches), dim=-1) * mask) / mask.sum()
            loss.backward()
            self.optimizer.step()
            loss_meter.update(loss.item())
            pbar.set_postfix({"loss": loss_meter.average()})
            wandb.log({"lr": self.lr_scheduler.get_last_lr()[0]})
        self.lr_scheduler.step()
        wandb.log({"train/loss": loss_meter.average()})

    def _val_epoch(self, epoch):
        self.model.eval()
        pbar = tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc=f"Eval epoch {epoch + 1}")
        loss_meter = AverageMeter()
        mean = torch.tensor(self.test_loader.dataset.mean).view(3, 1, 1)
        std = torch.tensor(self.test_loader.dataset.std).view(3, 1, 1)
        with torch.no_grad():
            for i, data in pbar:
                image = data['image'].to(self.config.device)
                image_patches = patchify(image, patch_size=self.config.patch_size)
                predicted_patches, mask = self.model(image)
                loss = torch.sum(
                    torch.mean(torch.square(image_patches - predicted_patches), dim=-1) * mask) / mask.sum()
                loss_meter.update(loss.item())

                if i == 0:  # save image in the first batch
                    # print(mask.unsqueeze(-1).tile(dims=(1, 1,
                    #                                     (self.config.img_height // self.config.patch_size) * (
                    #                                             self.config.img_height // self.config.patch_size) * 3)).shape)
                    mask = unpatchify(
                        mask.unsqueeze(-1).tile(dims=(1, 1, self.config.patch_size * self.config.patch_size * 3)),
                        patch_size=self.config.patch_size)
                    predicted_val_image = unpatchify(predicted_patches, patch_size=self.config.patch_size)
                    predicted_val_image = predicted_val_image * mask + image * (1 - mask)
                    img = torch.cat([image * (1 - mask), predicted_val_image, image], dim=0)  # 3 x B, C, H, W
                    img = einops.rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)  # 3 x H' x W'
                    img = img.cpu()
                    img = img * std + mean
                    images = wandb.Image(img)
                    wandb.log({"mae_image": images}, step=epoch)
                pbar.set_postfix({"loss": loss_meter.average()})

        wandb.log({"test/loss": loss_meter.average()})
        return loss_meter.average()

    def save_checkpoint(self, dir='checkpoint_last.pt'):
        checkpoint_path = os.path.join(self.config.snapshot_dir, dir)
        checkpoint = {'state_dict': self.model.module.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'lr_scheduler': self.lr_scheduler.state_dict(),
                      'epoch': self.current_epoch,
                      'score': self.best_score
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
        self.best_score = checkpoint['best_score']
        print("----> load checkpoint")

    def train(self, resume=False):
        if resume:
            self.load_checkpoint()

        for epoch in range(self.current_epoch, self.config.total_epoch):
            self._train_epoch(epoch)
            result = self._val_epoch(epoch)

            if result < self.best_score:
                self.best_score = result
                self.save_checkpoint('checkpoint_best.pt')
            # self.save_checkpoint('checkpoint_last.pt')
