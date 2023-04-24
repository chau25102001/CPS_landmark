import shutil

from trainer import *
from argparse import ArgumentParser
from config.config import get_config
import traceback

parser = ArgumentParser(description='CPS for AFLW')
parser.add_argument("--resume", action='store_true', default=False, help='resume training from last checkpoint')
args = parser.parse_args()

if __name__ == "__main__":
    if args.resume:
        cfg = get_config(train=False)
    else:
        cfg = get_config(train=True)
    if cfg.mean_teacher:
        trainer = EMATrainer(cfg)
    else:
        trainer = Trainer(cfg)

    try:
        trainer.train(args.resume)
    except Exception:
        print(traceback.print_exc())
        shutil.rmtree(cfg.snapshot_dir)
    except FileNotFoundError as e:
        print(e)
        shutil.rmtree(cfg.snapshot_dir)

    except KeyboardInterrupt:
        shutil.rmtree(cfg.snapshot_dir)

        # trainer.save_checkpoint('checkpoint_last.pt')
