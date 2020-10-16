import jieba
#from gensim.models import KeyedVectors
import numpy as np
import argparse
from tqdm import tqdm
import os 
import json
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
import PIL


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'eval': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class ImageDataset(Dataset):
    def __init__(self, img_dir, data, data_type='eval'):
        self.data = data
        self.data_type = data_type
        self.len = len(data)
        self.dir = img_dir

    def __getitem__(self, index):
        image = self.data[index]
        image = self.image_transform(image, self.data_type, self.dir)
        return image

    def __len__(self):
        return self.len

    @staticmethod
    def image_transform(image, data_type, dir):
        img = torch.zeros(3, 224, 224)
        image = dir + image
        try:
            img_tmp = PIL.Image.open(image)
            img = data_transforms[data_type](img_tmp)
        except Exception as err:
            print(err)

        return img

def get_loader(img_dir, data, batch_size=64, data_type='eval'):

    dataset = ImageDataset(img_dir=img_dir, data=data, data_type=data_type)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False)

    return data_loader

def get_imges_embedding_(img_data_loader, feature_extractor):

    data_s = list()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for batch_i, (images) in enumerate(tqdm(img_data_loader)):
        images = images.to(device)
        model_out = feature_extractor(images)
        model_data = model_out.cpu().data.numpy().squeeze()
        if batch_i == 0 :
            data_s = model_data
        else: 
            data_s = np.vstack((data_s, model_data))
    return data_s

def get_fasterrcnn_embedding(img_data_loader, feature_extractor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for batch_i, (images) in enumerate(tqdm(img_data_loader)):
        images = images.to(device)
        model_data = feature_extractor(images)
        #model_data = model_out.cpu().data.numpy().squeeze()
        import pdb;pdb.set_trace()
        if batch_i == 0 :
            boxes = model_data[0]['boxes'].cpu().data.numpy().squeeze()
            labels = model_data[0]['labels'].cpu().data.numpy().squeeze()
            scores = model_data[0]['scores'].cpu().data.numpy().squeeze()
        else: 
            boxes = np.vstack((boxes, model_data[0]['boxes'].cpu().data.numpy().squeeze()))
            labels = np.vstack((labels, model_data[0]['labels'].cpu().data.numpy().squeeze()))
            scores = np.vstack((boxes, model_data[0]['scores'].cpu().data.numpy().squeeze()))
    return boxes, labels, scores

def get_img_extractor(resnet_path):
    model_ft = torch.load(resnet_path)
    #feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1])
    feature_extractor = model_ft
    feature_extractor.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        feature_extractor = torch.nn.DataParallel(feature_extractor, device_ids=[0])
    feature_extractor.to(device)
    return feature_extractor

def get_img_emb(img_dir, img_list, resnet_path):
    fe = get_img_extractor(resnet_path)
    img_data_loader = get_loader(img_dir, img_list)
    ques_img_vec = get_imges_embedding_(img_data_loader, fe)
    return ques_img_vec#img_length*58

def get_fasterrcnn_emb(img_dir, img_list, resnet_path):
    #model = torch.load(resnet_path)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to('cuda')
    img_data_loader = get_loader(img_dir, img_list)
    boxes, labels, scores = get_fasterrcnn_embedding(img_data_loader, model)
    return boxes, labels, scores
    
class LayerActivations:
    features = None
 
    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)
 
    def hook_fn(self, module, input, output):
        self.features = output.cpu()
 
    def remove(self):
        self.hook.remove()