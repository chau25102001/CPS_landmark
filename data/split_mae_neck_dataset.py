import os.path
import shutil
from pathlib import Path

split_dir = "/home/s/chaunm/CPS_landmarks/data/split_mae_neck"
if os.path.exists(split_dir):
    shutil.rmtree(split_dir)
Path(split_dir).mkdir(parents=True, exist_ok=True)
train_split = open(os.path.join(split_dir, 'train.txt'), 'w')
test_split = open(os.path.join(split_dir, 'test.txt'), 'w')

lines = os.listdir("/home/s/chaunm/DATA/AFLW/MAE_images_neck")
lines = [l.strip() for l in lines]
split_index = int(len(lines) * 0.9)
for i, name in enumerate(lines):
    if i < split_index:
        train_split.write(name)
        train_split.write("\n")
    else:
        test_split.write(name)
        test_split.write("\n")

train_split.close()
test_split.close()
