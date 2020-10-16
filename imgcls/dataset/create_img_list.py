import sys
sys.path.append('../')
import os
import random
from config import config
from tqdm import tqdm
import json
import numpy as np
import cv2 as cv

random.seed(config.seed)

if not os.path.exists('./data'):
    os.makedirs('./data')

train_txt = open('./data/train.txt', 'w')
val_txt = open('./data/valid.txt', 'w')
label_txt = open('./data/label_list.txt', 'w')
cls_f = '../../data/images_classification.json'
tmp_txt = open('./tmp.txt', 'w')
data_dir ='/raid/zxy/project/jddc/imgcls/data/'
new_txt = './new_label.txt'

label_list = []

cls_list = []
with open(cls_f) as f:
    cls_list = json.load(f)
with open(new_txt) as f:
    lines = f.readlines()
new_label = {}
for line in lines:
    item = line.strip().split('\t')
    new_label[item[0]] = item[1]

data = {}
label = {}
label_cnt = {}
cnt = 0
for c in cls_list:
    data_ = {}
    if new_label[c['type']] not in label_list:
        label_list.append(new_label[c['type']])
        label_txt.write('{}\t{}\n'.format(new_label[c['type']], str(len(label_list)-1)))
        data[new_label[c['type']]] = [c['name']]
        label_cnt[new_label[c['type']]] = 1
    else:
        data[new_label[c['type']]].append(c['name'])
        label_cnt[new_label[c['type']]] += 1
    label[c['name']] = new_label[c['type']]
for key in label_cnt:
    print('{} {}'.format(key, label_cnt[key]))
for key in tqdm(data.keys()):
    train_list = random.sample(data[key], int(len(data[key])*0.8))
    for im in train_list:
        if os.path.exists('../../data/image/train/'+im):
            dir = '/raid/zxy/project/jddc/data/image/train/'+im
        elif os.path.exists('../../data/image/dev/'+im):
            dir = '/raid/zxy/project/jddc/data/image/dev/'+im
        else:
            print('miss: '+im)
            continue
        f_read = cv.imread(dir)
        '''
        if not os.path.exists(data_dir+label[im]):
            os.makedirs(data_dir+label[im])
            cnt = cnt+1
        '''
        f_out = data_dir + im
        cv.imwrite(f_out, f_read)
        train_txt.write('{} {}\n'.format(dir, str(label_list.index(key))))
        #tmp_txt.write(im +'\t'+ str(label[im])+'\n')
        if label_cnt[label[im]] < 100:
            flip_img = cv.flip(f_read, 1, dst=None)
            dir = data_dir + 'flip1_' + im
            cv.imwrite(dir, flip_img)
            train_txt.write('{} {}\n'.format(dir, str(label_list.index(key))))
        if label_cnt[label[im]] < 50:
            flip_img = cv.flip(f_read, 0, dst=None)
            dir = data_dir + 'flip0_' + im
            cv.imwrite(dir, flip_img)
            train_txt.write('{} {}\n'.format(dir, str(label_list.index(key))))
            flip_img = cv.flip(f_read, -1, dst=None)
            dir = data_dir + 'flip_1_' + im
            cv.imwrite(dir, flip_img)
            train_txt.write('{} {}\n'.format(dir, str(label_list.index(key))))
    for im in tqdm(data[key]):
        if im in train_list:
            continue
        else:
            if os.path.exists('../../data/image/train/'+im):
                dir = '/raid/zxy/project/jddc/data/image/train/'+im
            elif os.path.exists('../../data/image/dev/'+im):
                dir = '/raid/zxy/project/jddc/data/image/dev/'+im
            else:
                print('miss: '+im)
                continue
            f_read = cv.imread(dir)
            '''
            if not os.path.exists(data_dir+label[im]):
                os.makedirs(data_dir+label[im])
                cnt = cnt+1
            '''
            f_out = data_dir + im
            cv.imwrite(f_out, f_read)
            val_txt.write('{} {}\n'.format(dir, str(label_list.index(key))))
            #tmp_txt.write(im +'\t'+ str(label[im])+'\n')
            if label_cnt[label[im]] < 100:
                flip_img = cv.flip(f_read, 1, dst=None)
                dir = data_dir + 'flip1_' + im
                cv.imwrite(dir, flip_img)
                train_txt.write('{} {}\n'.format(dir, str(label_list.index(key))))
            if label_cnt[label[im]] < 50:
                flip_img = cv.flip(f_read, 0, dst=None)
                dir = data_dir + 'flip0_' + im
                cv.imwrite(dir, flip_img)
                train_txt.write('{} {}\n'.format(dir, str(label_list.index(key))))
                flip_img = cv.flip(f_read, -1, dst=None)
                dir = data_dir + 'flip_1_' + im
                cv.imwrite(dir, flip_img)
                train_txt.write('{} {}\n'.format(dir, str(label_list.index(key))))
print(cnt)
'''
for dir in tqdm(os.listdir(config.data_root)):
    if dir not in label_list:
        label_list.append(dir)
        label_txt.write('{} {}\n'.format(dir, str(len(label_list)-1)))
        data_path = os.path.join(config.data_root, dir)
        train_list = random.sample(os.listdir(data_path), 
                                   int(len(os.listdir(data_path))*0.8))
        for im in train_list:
            train_txt.write('{}/{} {}\n'.format(dir, im, str(len(label_list)-1)))
        for im in os.listdir(data_path):
            if im in train_list:
                continue
            else:
                val_txt.write('{}/{} {}\n'.format(dir, im, str(len(label_list)-1)))
'''


