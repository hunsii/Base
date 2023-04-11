# +
import os
import time
import torch
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import Model
from utils import parse_yaml, mapping
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validation(opt, model, criterion, val_loader, loss_dict):
    elapsed_time = time.time() - loss_dict['start_time']
    loss_dict['elapsed_time'] = elapsed_time
    
    model.eval()
    val_loss = []
    preds, trues = [], []
    
    with torch.no_grad():
        for videos, labels in tqdm(iter(val_loader)):
            videos = videos.to(device)
            labels = labels.to(device)
            
            logit = model(videos)
            
            loss = criterion(logit, labels)
            
            val_loss.append(loss.item())
            
            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
        
        _val_loss = np.mean(val_loss)
    
    _val_f1_score = f1_score(trues, preds, average='macro')
    _val_acc_score = accuracy_score(trues, preds)
    
    loss_dict["validation_loss"]           = _val_loss
    loss_dict["validation_f1_score"]       = _val_f1_score
    loss_dict["validation_accuracy_score"] = _val_acc_score
    loss_dict["validation_running_time"]   = loss_dict['elapsed_time'] - time.time()
    
    # save current model.
    if loss_dict['best_f1'] < loss_dict['validation_f1_score']:
        loss_dict['best_f1'] = loss_dict['validation_f1_score']
        torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_f1.pth')

    if loss_dict['best_accuracy'] < loss_dict['validation_accuracy_score']:
        loss_dict['best_accuracy'] = loss_dict['validation_accuracy_score']
        torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
    
    return loss_dict

def inference(model, test_loader, device):
    # model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for videos in tqdm(iter(test_loader)):
            videos = videos.to(device)
            
            logit = model(videos)

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
    return preds

def test(opt):
    """ dataset preparation """
    df = pd.read_csv(opt.test_data)

    img_path_list = df.path.to_numpy()
    
    test_dataset = CustomDataset(opt, img_path_list, None, transforms_type="valid")
    test_loader = DataLoader(test_dataset, batch_size = opt.batch_size, shuffle=False, num_workers=0)
    
    print("------------ Model ------------")
    model = Model(opt).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device), strict=True)
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./result', exist_ok=True)
#     os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')


    """ evaluation """
    preds = inference(model, test_loader, device)
    preds = [mapping(pred, inverse=True) for pred in preds]
    
    submit = pd.read_csv('sample_submission.csv')
    if opt.test_data == "train.csv":
        submit = pd.read_csv(opt.test_data)

        
    
    submit['label'] = preds
    submit.to_csv('./submission.csv', index=False)
    if opt.output == "":
        submit.to_csv(f'./result/{opt.label_info}.csv', index=False)
    else:
        submit.to_csv(f'./result/{opt.output}.csv', index=False)
    print("done!")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yml', help='path to config yaml file')
    opt = parser.parse_args()
    opt = parse_yaml(opt.config)
    
    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    # random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    test(opt)
