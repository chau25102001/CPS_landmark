import shutil

from trainer import *
from argparse import ArgumentParser
from config.config_ema_vit import get_config
import traceback
from utils.utils import seed_everything, merge_dict

parser = ArgumentParser(description="EMA CPS MAE for AFLW")
parser.add_argument("--resume", action='store_true', default=False, help='resume training from last checkpoint')
parser.add_argument("--mean_teacher", action='store_true', default=False, help="use mean teacher or not")
parser.add_argument("--fully_supervised", action="store_true", default=False, help="train fully supervised")
parser.add_argument("--train_text_labeled", type=str, help="path to labeled train text file")
parser.add_argument("--train_text_unlabeled", type=str, help="path to unlabeled train text file")
parser.add_argument("--test_text", type=str, help="path to test text file")
parser.add_argument("--train_annotations_path", type=str, help="path to the train annotation folder")
parser.add_argument("--test_annotations_path", type=str, help="path to the test annotation folder")
parser.add_argument("--train_images_path", type=str, help="path to the train images folder")
parser.add_argument("--test_images_path", type=str, help="path to the test images folder")
parser.add_argument("--num_classes", type=int,
                    help="number of keypoint + 1, either 20 for AFLW-19, 6 for AFLW-DA, or 5 for SideFace-DA")
parser.add_argument("--pretrained_path", type=str, help="path to the pretrained .pt weights for MAE model")
args = parser.parse_args()

if __name__ == "__main__":
    if args.resume:
        cfg = get_config(False)
    else:
        cfg = get_config(train=True)
    cfg = merge_dict(cfg, args)
    seed_everything(cfg.seed)
    trainer = EMAVitTrainer(cfg)
    try:
        trainer.train(args.resume)
    except Exception:
        print(traceback.print_exc())
        shutil.rmtree(cfg.snapshot_dir)

    except KeyboardInterrupt:
        shutil.rmtree(cfg.snapshot_dir)
