import os
from img_feature.imgemb import get_img_emb
from tqdm import tqdm
import numpy as np
import argparse
import json

resnet_path = "./img_feature/ResNet50.pth"

for type in ['train']:
    img_dir = "../online_test_data/images_test/"
    img_list = []
    for i, img in enumerate(os.listdir(img_dir)):
        img_list.append(img)
    emb = get_img_emb(img_dir, img_list, resnet_path)
    label = np.argmax(emb, axis=1)
    res = {}
    for im, l in zip(img_list, label):
        res[im] = str(l)
    with open('./label.json', 'w', encoding='utf-8') as w:
        w.write(json.dumps(res, ensure_ascii=False, indent=2))
