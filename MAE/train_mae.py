import sys

sys.path.append('..')
import shutil
from MAE.trainer.MAE_trainer import MAETrainer
from argparse import ArgumentParser
import MAE.config.config as aflw_config
import MAE.config.config_neck as neck_config
import MAE.config.config_side as side_config
import traceback
from utils.utils import seed_everything

parser = ArgumentParser(description='MAE for AFLW')
parser.add_argument("--resume", action='store_true', default=False, help='resume training from last checkpoint')
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
    if args.resume:
        cfg = get_cfg_func(train=False)
    else:
        cfg = get_cfg_func(train=True)

    seed_everything(cfg.seed)

    trainer = MAETrainer(cfg, args.neck)
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
