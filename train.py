# +
import os
import sys
import time
import random
import argparse
# from argparse import Namespace
import tqdm
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

import cv2
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn import metrics


from dataset import CustomDataset
from model import Model
from test import validation
from utils import BalancekFold, kfold_data_split, parse_yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_values(value):
    return value.values.reshape(-1, 1)

def train(opt):
    """ dataset preparation """
    df = pd.read_csv(opt.train_data)

    img_path_list = df.path.to_numpy()
    label_path_list = df.label.to_numpy()

    # def kfold_data_split(df, kfold = 0, n_splits = 10, random_state = 1):
    train_index, valid_index = kfold_data_split(img_path_list, kfold=opt.kfold, n_splits=opt.n_splits, random_state=opt.manualSeed)
    print(train_index.shape)

    train_dataset = CustomDataset(opt, img_path_list[[train_index]], label_path_list[[train_index]], transforms_type = "train")
    train_loader = DataLoader(train_dataset, batch_size = opt.batch_size, shuffle=True, num_workers=0)

    valid_dataset = CustomDataset(opt, img_path_list[[valid_index]], label_path_list[[valid_index]], transforms_type = "valid")
    valid_loader = DataLoader(valid_dataset, batch_size = opt.batch_size, shuffle=False, num_workers=0)
    
    with open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a') as log:
        dataset_log = '------------ Datasets -------------\n'
        dataset_log += f"train_df length: {img_path_list[[train_index]].shape}\n"
        dataset_log += f"valid_df length: {label_path_list[[valid_index]].shape}\n"

        print(dataset_log)
        log.write(dataset_log)

    print("------------ Model ------------")
    model = Model(opt).to(device)
        
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.9990))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)
        
    """ start training """
    start_iter = opt.start_iter
    start_time = time.time()
    best_accuracy: -1
    best_f1 = -1
#     iteration = start_iter
    scheduler = None

    info_dict = {
        'start_time': start_time,
        'iteration': start_iter,
        'best_accuracy': -1, 
        'best_f1': -1, 
    }
    
    criterion = nn.CrossEntropyLoss().to(device)
#     with open(f'./saved_models/{opt.exp_name}/log_train.csv', 'a') as f:
#         f.write("iteration,train_loss,valid_loss,valid_f1,valid_acc,elapsed_time\n")

    while True:
        model.train()
        train_loss = []
        true_labels = []
        pred_labels = []
        info_dict['epoch'] = info_dict['iteration'] / (np.ceil(len(img_path_list) / opt.batch_size))
    
        for images, labels in iter(train_loader):
            
            
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            pred_labels += output.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()
            
            # validation part
            info_dict = check_validation(opt, model, criterion, valid_loader, info_dict, train_loss, true_labels, pred_labels)
    
            # 스케줄러
            if scheduler is not None:
                scheduler.step(val_score)
            info_dict['iteration'] += 1
            info_dict['epoch'] = info_dict['iteration'] / (np.ceil(len(img_path_list) / opt.batch_size))
        # validation part
        info_dict = check_validation(opt, model, criterion, valid_loader, info_dict, train_loss, true_labels, pred_labels)
            
def check_validation(opt, model, criterion, valid_loader, info_dict, losses, trues, predes):
    run_validation = False
    if opt.valInterval > 0 and ((info_dict['iteration'] + 1) % opt.valInterval == 0 or info_dict['iteration'] == 0): # To see training progress, we also conduct validation when 'iteration == 0' 
        run_validation = True
    elif opt.valInterval < 0 and info_dict['epoch'].is_integer():
        run_validation = True
    
    if run_validation:
        # run validation
        info_dict = validation(opt, model, criterion, valid_loader, info_dict)  
        
        info_dict['train_loss'] = np.mean(losses)
        info_dict['train_f1_score'] = f1_score(trues, predes, average='macro')
        info_dict['train_accuracy_score'] = accuracy_score(trues, predes)
        
        save_log("saved_models/default/test.csv", info_dict)
    
    # 종료 조건
    if (info_dict['iteration'] + 1) == opt.num_iter and opt.valInterval > 0:
        print('end the training')
        sys.exit()
    elif info_dict['epoch'] >= opt.num_iter:
        print('end the training')
        sys.exit()
    
    return info_dict

def save_log(path: str, data: dict):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.append(data, ignore_index = True)
        df.to_csv(path)
    else:
        df = pd.DataFrame.from_records([data])
        df.to_csv(path)

def get_exp_name(opt, filename):
    return os.path.join("saved_models", opt.exp_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yml', help='path to config yaml file')
    opt = parser.parse_args()
    

    opt = parse_yaml(opt.config)

    if not opt.exp_name:
        opt.exp_name = "default"

    if os.path.isdir(f'./saved_models/{opt.exp_name}'):
        print(f"There already exist {opt.exp_name}!!")
        sys.exit()
    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)


    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)


    train(opt)
#     print(opt)
