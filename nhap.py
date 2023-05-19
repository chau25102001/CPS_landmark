import torch
pth_file = "/home/s/chaunm/DATA/AFLW/neck-new-sep/train.pth"
pth_test_file = "/home/s/chaunm/DATA/AFLW/neck-new-sep/test.pth"
path_annotation = torch.load(pth_file, map_location='cpu')
path_test_annotation = torch.load(pth_test_file, map_location='cpu')
for p in path_annotation:
    if p['label'] == 0:
        print(p)
        break