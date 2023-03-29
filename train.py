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
from sklearn.model_selection import train_test_split

from sklearn import metrics


from dataset import CustomDataset, preprocessing_df, ClassDataset
from model import CrashModelV2_2D, CrashModelV2_3D
from test import validation
from utils import BalancekFold, kfold_data_split, parse_yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_values(value):
    return value.values.reshape(-1, 1)

def train(opt):
    """ dataset preparation """
    paper_path_list = glob.glob("data/paper/*")
    rock_path_list = glob.glob("data/rock/*")
    scissors_path_list = glob.glob("data/scissors/*")

    # len(paper_path_list), len(rock_path_list), len(scissors_path_list)
    path_list = np.array(paper_path_list + rock_path_list + scissors_path_list)
    label_list = np.array([0 for i in range(len(paper_path_list))] + [0 for i in range(len(rock_path_list))] + [0 for i in range(len(scissors_path_list))])
    original_data_shape = path_list.shape

    # def kfold_data_split(df, kfold = 0, n_splits = 10, random_state = 1):
    train_index, valid_index = kfold_data_split(path_list, kfold=opt.kfold, n_splits=opt.n_splits, random_state=opt.manualSeed)
    print(train_index.shape)
    # print(train_index[0])
    # print(len(path_list))
    # print(len(label_list))
    
    train_dataset = ClassDataset(opt, path_list[[train_index]], label_list[[train_index]], valid_mode = False)
    train_loader = DataLoader(train_dataset, batch_size = opt.batch_size, shuffle=True, num_workers=0)

    valid_dataset = ClassDataset(opt, path_list[[valid_index]], label_list[[valid_index]], valid_mode = False)
    valid_loader = DataLoader(valid_dataset, batch_size = opt.batch_size, shuffle=False, num_workers=0)
    
    with open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a') as log:
        dataset_log = '------------ Datasets -------------\n'
        dataset_log +=f"ego_only: {opt.ego_only}\nn_splits: {opt.n_splits}\nfold: {opt.kfold}"
        dataset_log += f"original data: {original_data_shape}\n"
        dataset_log += f"train_df length: {path_list[[train_index]].shape}\n"
        dataset_log += f"valid_df length: {label_list[[valid_index]].shape}\n"

        print(dataset_log)
        log.write(dataset_log)

    print("------------ Model ------------")
    if opt.video_input:
        print("3D model is loading...")
        model = CrashModelV2_3D(opt).to(device)
    else:
        print("2D model is loading...")
        model = CrashModelV2_2D(opt).to(device)
        
#     model = model
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))

#     print(model)

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

#     sys.exit()
        
    """ start training """
    start_iter = 0
    start_time = time.time()
    best_accuracy = -1
    best_f1 = -1
    iteration = start_iter
    scheduler = None

    criterion = nn.CrossEntropyLoss().to(device)
    with open(f'./saved_models/{opt.exp_name}/log_train.csv', 'a') as f:
        f.write("iteration,train_loss,valid_loss,valid_f1,valid_acc,elapsed_time\n")
    while True:
        model.train()
        train_loss = []
        true_labels = []
        pred_labels = []
        
        for videos, labels in iter(train_loader):
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(videos)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            if opt.verbose >= 1:
                print(f"train_loss: {np.mean(train_loss):.4f}")
            
            # validation part
            if opt.valInterval > 0 and ((iteration + 1) % opt.valInterval == 0 or iteration == 0): # To see training progress, we also conduct validation when 'iteration == 0' 
                elapsed_time = time.time() - start_time
                epoch = np.floor(iteration / (np.ceil(len(path_list) / opt.batch_size)))

                val_loss, val_f1_score, val_acc_score = validation(model, criterion, valid_loader, device)
                train_log = f'Epoch    : {epoch}, Iteration [{iteration:6d}], Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}]\n'
                train_log+= f"Valid F1 : [{val_f1_score:.5f}], Valid Acc : [{val_acc_score:.5f}] Elapsed_time : [{elapsed_time:0.5f}]"
                print(train_log)
                with open(f'./saved_models/{opt.exp_name}/log_train.csv', 'a') as f:
                    f.write(f"{iteration},{np.mean(train_loss):.6f},{val_loss:.6f},{val_f1_score:.6f},{val_acc_score:.6f},{elapsed_time:0.6f}\n")
                if best_f1 < val_f1_score:
                    best_f1 = val_f1_score
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_f1.pth')
                if best_accuracy < val_acc_score:
                    best_accuracy = val_acc_score
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                # # 초기화?
                train_loss = []

            # 스케줄러
            if scheduler is not None:
                scheduler.step(val_score)
            
            # 종료 조건
            if (iteration + 1) == opt.num_iter:
                print('end the training')
                sys.exit()
            iteration += 1

        # validation part
        if opt.valInterval <= -1:
            elapsed_time = time.time() - start_time
            epoch = np.floor(iteration / (np.ceil(len(path_list) / opt.batch_size)))

            val_loss, val_f1_score, val_acc_score = validation(model, criterion, valid_loader, device)
            train_log = f'Epoch    : {epoch}, Iteration [{iteration:6d}], Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}]\n'
            train_log+= f"Valid F1 : [{val_f1_score:.5f}], Valid Acc : [{val_acc_score:.5f}] Elapsed_time : [{elapsed_time:0.5f}]"
            print(train_log)
            with open(f'./saved_models/{opt.exp_name}/log_train.csv', 'a') as f:
                f.write(f"{iteration},{np.mean(train_loss):.6f},{val_loss:.6f},{val_f1_score:.6f},{val_acc_score:.6f},{elapsed_time:0.6f}\n")
            if best_f1 < val_f1_score:
                best_f1 = val_f1_score
                torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_f1.pth')
            if best_accuracy < val_acc_score:
                best_accuracy = val_acc_score
                torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
            
def print_validation(opt, model):
    elapsed_time = time.time() - start_time
    epoch = np.floor(iteration / (np.ceil(len(path_list) / opt.batch_size))).astype(np.int8)

    val_loss, val_f1_score, val_acc_score = validation(model, criterion, valid_loader, device)
    
    # print validation result.
    train_log = f'Epoch    : {epoch}, Iteration [{iteration:6d}], Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}]\n'
    train_log+= f"Valid F1 : [{val_f1_score:.5f}], Valid Acc : [{val_acc_score:.5f}] Elapsed_time : [{elapsed_time:0.5f}]"
    print(train_log)
    with open(f'./saved_models/{opt.exp_name}/log_train.csv', 'a') as f:
        f.write(f"{train_log}\n")

    # save current model.
    if best_f1 < val_f1_score:
        best_f1 = val_f1_score
        torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_f1.pth')

    if best_accuracy < val_acc_score:
        best_accuracy = val_acc_score
        torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yml', help='path to config yaml file')
    opt = parser.parse_args()
    

    opt = parse_yaml(opt.config)

    if not opt.exp_name:
        opt.exp_name = "default"

    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)


    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)


#     train(opt)
    print(opt)
