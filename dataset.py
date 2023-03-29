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

def get_transform(opt):
#     if opt.use_thumbnail:
#         print("transform for 2D")

#         """
#         transform = A.Compose([
#             A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)), 
#             A.Resize(height = opt.imgH, width = opt.imgW), 
# #             A.CenterCrop(height = opt.imgH, width = opt.imgW), 
#     #         A.ToTensorV2
#         ])
#         """
#         transform = A.Compose([
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
#             A.HorizontalFlip(p = 0.5),
# #             A.RandomCrop(height = opt.imgH, width = opt.imgW),
#             A.Resize(height = opt.imgH, width = opt.imgW), 
#         ])
        
#     else:
#         print("transform for 3D")
#         transform = A.Compose([
#             A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)), 
#             A.Resize(height=320, width = 480), 
#             A.CenterCrop(height = opt.imgH, width = opt.imgW), 
#     #         A.ToTensorV2
#         ])
    if opt.label_info == "weather":
        print("transform for weather")
        # not /255
        transform = A.Compose([
            A.HorizontalFlip(p = 0.5),
            A.Resize(height = opt.imgH*2, width = opt.imgW*2), 
            A.OneOf([
                A.Resize(height = opt.imgH, width = opt.imgW),
                A.CenterCrop(height = opt.imgH, width = opt.imgW)
            ]), 
            A.OneOf([
                A.GaussNoise(p=0.75, var_limit=(100, 200)), 
                A.Cutout(p=0.75, num_holes=8, max_h_size=24, max_w_size=24), 
                A.CLAHE(p=0.75, clip_limit=2.0, tile_grid_size=(8, 8)), 
            ]), 
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            
        ])
    elif opt.label_info == "timing":
        print("transform for timing")
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            A.Resize(height = opt.imgH, width = opt.imgW), 
        ])
    elif opt.label_info == "ego-involve":
        print("transform for ego-involve")
        transform = A.Compose([
            A.HorizontalFlip(p = 0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            A.Resize(height = 360, width = 640), 
            A.CenterCrop(height = opt.imgH, width = opt.imgW)
#             A.OneOf([
#                 A.Resize(height = opt.imgH, width = opt.imgW),
#                 A.CenterCrop(height = opt.imgH, width = opt.imgW)
#             ])
        ])
    return transform


class CustomDataset(Dataset):
    def __init__(self, opt, video_path_list, label_list, valid_mode = False):
        self.opt = opt
        self.video_path_list = video_path_list
        self.label_list = label_list
        self.transform = get_transform(opt)
        self.valid_mode = valid_mode
        
    def __getitem__(self, index):
        frames = self.get_video(self.video_path_list[index])
        
        if self.label_list is not None:
            label = self.label_list[index]
            return frames, label
        else:
            return frames
        
    def __len__(self):
        return len(self.video_path_list)
    
    def get_video(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        for _ in range(self.opt.video_length):
            _, img = cap.read()
#             img = cv2.resize(img, (self.opt.imgW, self.opt.imgH))
            img = self.transform(image = img)['image']
            

            if self.opt.label_info == 'ego-involve':
                img = img / 255.
            
            frames.append(img)
            
            if self.opt.use_thumbnail or self.valid_mode:
                return torch.FloatTensor(img).permute(2, 0, 1)
        if self.opt.image_input:
            img = frames[random.randrange(self.opt.video_length)]
            return torch.FloatTensor(np.array(img)).permute(2, 0, 1)
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)
"""
0: 사고를 관전함
1: 본인이 사고가 났음
2: 사고가 안났음
"""
def extract_ego_involve(v):
    if v == 0: # 사고가 안났으면 0
        return 2
    elif v >= 7: # 사고를 관전했으면 0
        return 0
    else:
        return 1 

def is_weather_normal(v):
    if v == 0:
        return -1
    elif v == 1 or v == 2 or v ==7 or v == 8:
        return 1
    else:
        return 0

def is_weather_snowy(v):
    if v == 0:
        return -1
    elif v == 3 or v == 4 or v == 9 or v == 10:
        return 1
    else:
        return 0
def is_weather_rainy(v):
    if v == 0:
        return -1
    elif v == 5 or v == 6 or v == 11 or v == 12:
        return 1
    else:
        return 0
"""
-1: 사고x
0: 야간
1: 주간
"""
def extract_timing(v):
    if v == 0:
        return -1
    elif v % 2 == 1:
        return 1
    else:
        return 0

"""
-1: 사고x
0: normal
1: snowy
2: rainy
"""
def extract_weather(v):
    if v == 0:
        return -1
    elif v == 1 or v == 2 or v ==7 or v == 8:
        return 0
    elif v == 3 or v == 4 or v == 9 or v == 10:
        return 1
    elif v == 5 or v == 6 or v == 11 or v == 12:
        return 2    

def preprocessing_df(file_path, ego_only = False):
    df = pd.read_csv(file_path)
    df["ego-involve"] = df['label'].apply(lambda x: extract_ego_involve(x))
#     df["normal"] = df['label'].apply(lambda x: is_weather_normal(x))
#     df["snowy"] = df['label'].apply(lambda x: is_weather_snowy(x))
#     df["rainy"] = df['label'].apply(lambda x: is_weather_rainy(x))
    df["weather"] = df['label'].apply(lambda x: extract_weather(x))
    df["timing"] = df['label'].apply(lambda x: extract_timing(x))
    
    if ego_only:
        return df[df.label > 0]
    else:
        return df
    

    
class ClassDataset(Dataset):
    def __init__(self, opt, path_list, label_list,valid_mode):
        self.opt = opt
        self.img_path_list = path_list
        self.label_list = label_list
        self.valid_mode = valid_mode
        self.transform = self.get_transform(self.opt)
        
    def __getitem__(self, index):
        img = cv2.imread(self.img_path_list[index])
        
        img = self.transform(image = img)['image']
        
        if self.valid_mode:
            return img
        else:
            return img, self.label_list[index]
        
        
    def __len__(self):
        return len(self.img_path_list)
    
    def get_transform(self, opt):
        transform = A.Compose([
            A.HorizontalFlip(p = 0.5),
            A.Resize(height = opt.imgH, width = opt.imgW), 
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            ToTensorV2(), 
        ])
        return transform