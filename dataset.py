# +
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import pandas as pd
import random
import glob

def get_transform(opt, transforms_type):
    if transforms_type == "train":
        transform = A.Compose([
            A.HorizontalFlip(p = 0.5),
            A.Resize(height = opt.imgH, width = opt.imgW), 
#             A.OneOf([
#                 A.Resize(height = opt.imgH, width = opt.imgW),
#                 A.CenterCrop(height = opt.imgH, width = opt.imgW)
#             ]), 
            A.OneOf([
                A.GaussNoise(p=0.75, var_limit=(100, 200)), 
                A.Cutout(p=0.75, num_holes=8, max_h_size=24, max_w_size=24), 
                A.CLAHE(p=0.75, clip_limit=2.0, tile_grid_size=(8, 8)), 
            ]), 
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            ToTensorV2()
        ])
    elif transforms_type == "valid":
        transform = A.Compose([
            A.Resize(height = opt.imgH, width = opt.imgW),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            ToTensorV2()
        ])
    elif transforms_type == "test":
        pass
    else:
        transform = None
        
    return transform

class CustomDataset(Dataset):
    def __init__(self, opt, img_path_list, label_list, transforms_type = ""):
        self.opt = opt
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = get_transform(opt, transforms_type)
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        image = cv2.imread(img_path)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)
