import os
import shutil
from pathlib import Path
import random

train_annot_path = "/home/s/chaunm/DATA/AFLW/train_128_3/annotations"
test_annot_path = "/home/s/chaunm/DATA/AFLW/test_128_3/annotations"

save_text_dir = "./data/split"
if os.path.exists(save_text_dir):
    shutil.rmtree(save_text_dir)

Path(save_text_dir).mkdir(parents=True, exist_ok=True)

# test files
test_annot_files = os.listdir(test_annot_path)
test_txt = open(os.path.join(save_text_dir, 'test.txt'), 'w')
for f in test_annot_files:
    test_txt.write(f + "\n")
test_txt.close()
# train files
train_annot_files = os.listdir(train_annot_path)
len_train = len(train_annot_files)
names = ['1_2', '1_4', '1_8']
for i, ratio in enumerate([1 / 2, 1 / 4, 1 / 8]):
    train_labeled_txt = open(os.path.join(save_text_dir, f'train_labeled_{names[i]}.txt'), 'w')
    train_unlabeled_txt = open(os.path.join(save_text_dir, f'train_unlabeled_{names[i]}.txt'), 'w')
    l = int(ratio * len_train)
    random.shuffle(train_annot_files)
    labeled_part = train_annot_files[:l]
    unlabeled_part = train_annot_files[l:]
    for lb in labeled_part:
        train_labeled_txt.write(lb + '\n')
    for ulb in unlabeled_part:
        train_unlabeled_txt.writelines(ulb + '\n')
    train_labeled_txt.close()
    train_unlabeled_txt.close()
