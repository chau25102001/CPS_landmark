import shutil
import numpy as np
import termcolor
import sys

sys.path.append("..")
import datasets.aflw as aflw
import MAE.datasets.mae_neck as neck
from MAE.models.MAE import *
from tqdm import tqdm
import torch
from utils.utils import AverageMeter
import MAE.config.config as aflw_config
import MAE.config.config_neck as neck_config
import MAE.config.config_side as side_config
from utils.utils import seed_everything
from argparse import ArgumentParser
import os
import cv2

parser = ArgumentParser(description='MAE for AFLW')
parser.add_argument("--neck", action='store_true', default=False, help='use neck dataset or normal aflw')
parser.add_argument("--side", action="store_true", default=False, help='use side dataset or normal aflw')
args = parser.parse_args()

if __name__ == "__main__":
    if args.neck:
        get_cfg_func = neck_config.get_config
    elif args.side:
        get_cfg_func = side_config.get_config
    else:
        get_cfg_func = aflw_config.get_config
    cfg = get_cfg_func(train=False)
    seed_everything(cfg.seed)
    if args.neck:
        test_loader = neck.get_test_loader(cfg)
    else:
        test_loader = aflw.get_test_loader(cfg)

    model = get_mae_model(encoder_embedding_dim=cfg.encoder_embedding_dim,
                          encoder_layers=cfg.encoder_layers,
                          n_heads_encoder_layer=cfg.n_heads_encoder_layer,
                          decoder_embedding_dim=cfg.decoder_embedding_dim,
                          decoder_layers=cfg.decoder_layers,
                          n_heads_decoder_layer=cfg.n_heads_decoder_layer,
                          patch_size=cfg.patch_size,
                          num_patches=cfg.img_height // cfg.patch_size)
    if args.side:
        checkpoint_path = "/home/s/chaunm/CPS_landmarks/MAE/log_mae/snapshot/vit-mae_8/checkpoint_best.pt"
        save_dir = "mae_test_images_side"
    elif args.neck:
        checkpoint_path = "/home/s/chaunm/CPS_landmarks/MAE/log_mae/snapshot/vit-mae_6/checkpoint_best.pt"
        save_dir = "mae_test_images_neck"
    else:
        checkpoint_path = "/home/s/chaunm/CPS_landmarks/MAE/log_mae/snapshot/vit-mae_2/checkpoint_best.pt"
        save_dir = "mae_test_images_aflw"

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(cfg.device)
    model.eval()
    loss_meter = AverageMeter()
    mean = torch.tensor(test_loader.dataset.mean).view(3, 1, 1)
    std = torch.tensor(test_loader.dataset.std).view(3, 1, 1)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    with torch.no_grad():
        for i, data in pbar:
            image = data['image'].to(cfg.device)
            image_patches = patchify(image, patch_size=cfg.patch_size)
            predicted_patches, mask = model(image)
            loss = torch.sum(
                torch.mean(torch.square(image_patches - predicted_patches), dim=-1) * mask) / mask.sum()
            loss_meter.update(loss.item())
            mask = unpatchify(
                mask.unsqueeze(-1).tile(dims=(1, 1, cfg.patch_size * cfg.patch_size * 3)),
                patch_size=cfg.patch_size)
            predicted_val_image = unpatchify(predicted_patches, patch_size=cfg.patch_size)
            predicted_val_image = predicted_val_image * mask + image * (1 - mask)
            if i < 10:
                img = torch.cat([image * (1 - mask), predicted_val_image, image], dim=0)  # 3 x B, C, H, W
                img = einops.rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)  # 3 x H' x W'
                img = img.cpu()
                img = img * std + mean
                img = np.clip(img, a_min=0, a_max=1)
                img = (img * 255.0).numpy().astype(np.uint8)
                img = img.transpose(1, 2, 0)
                cv2.imwrite(os.path.join(save_dir, f"batch_{i}.png"), img)

    print(termcolor.colored(f"MSE average: {loss_meter.average()}"))
