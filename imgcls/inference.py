# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/gfx/Projects/remote_sensing_image_classification')
import os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.nn.parallel.data_parallel import data_parallel
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from dataset.dataset import *
from networks.network import *
from networks.lr_schedule import *
from metrics.metric import *
from utils.plot import *
from config import config


def inference():
    # model
    # load checkpoint
    model = torch.load(os.path.join('./checkpoints/ocr', config.checkpoint))
    print(model)
    # model = torch.nn.DataParallel(model)
    model.cuda()
    
    # validation data
    transform = transforms.Compose([transforms.Resize((config.width, config.height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])
    dst_valid = RSDataset('./data/label/valid.txt', width=config.width, 
                          height=config.height, transform=transform)
    dataloader_valid = DataLoader(dst_valid, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers)

    sum = 0
    val_top1_sum = 0
    val_recall_sum = 0
    val_f1_sum = 0
    labels = []
    preds = []
    model.eval()
    for ims, label in dataloader_valid:
        labels += label.numpy().tolist()

        input = Variable(ims).cuda()
        target = Variable(label).cuda()
        output = model(input)

        _, pred = output.topk(1, 1, True, True)
        preds += pred.t().cpu().numpy().tolist()[0]

        top1_val, recall, f1 = accuracy(output.data, target.data, topk=(1,))
        
        sum += 1
        val_top1_sum += top1_val[0]
        val_recall_sum += recall 
        val_f1_sum += f1
    avg_top1 = val_top1_sum / sum
    avg_recall = val_recall_sum / sum
    avg_f1 = val_f1_sum / sum
    print('acc: {} recall: {} f1: {}'.format(avg_top1.data, avg_recall, avg_f1))

    labels_=[str(i) for i in range(4)]   
    plot_confusion_matrix(labels, preds, labels_)
    t = classification_report(labels, preds, target_names=labels_)
    print(t)

if __name__ == '__main__':
    inference()
