import xmltodict
import os
import json

root = "/home/s/chaunm/DATA/CPLFW/outputs"
annotation_files = os.listdir(root)

chosen_file = annotation_files[0]

with open(os.path.join(root, chosen_file), 'r') as f:
    data_dict = xmltodict.parse(f.read())

print(data_dict['doc']['outputs']['object']['item']['polygon'])
