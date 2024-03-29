import shutil

from trainer import *
from argparse import ArgumentParser
from config.config_finetune_vit import get_config
import traceback
from utils.utils import seed_everything

parser = ArgumentParser(description='Finetune ViT-UNETR for AFLW')
parser.add_argument("--resume", action='store_true', default=False, help='resume training from last checkpoint')
args = parser.parse_args()

if __name__ == "__main__":
    if args.resume:
        cfg = get_config(train=False)
    else:
        cfg = get_config(train=True)
    seed_everything(cfg.seed)
    trainer = FinetuneTrainer(cfg)
    try:
        trainer.train(args.resume)

    except Exception:
        print(traceback.print_exc())
        shutil.rmtree(cfg.snapshot_dir)

    except KeyboardInterrupt:
        shutil.rmtree(cfg.snapshot_dir)
        # trainer.save_checkpoint('checkpoint_last.pt')
