# +
import os
import torch
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import CrashModelV2_2D, CrashModelV2_3D
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validation(model, criterion, val_loader, device):
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
    return _val_loss, _val_f1_score, _val_acc_score

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
    test_df = pd.read_csv(opt.test_data)
    
    test_dataset = CustomDataset(opt, test_df['video_path'].values, None)
    test_loader = DataLoader(test_dataset, batch_size = opt.batch_size, shuffle=False, num_workers=0)
    
    print("------------ Model ------------")
    if opt.video_input:
        print("3D model is loading...")
        model = CrashModelV2_3D(opt).to(device)
    else:
        print("2D model is loading...")
        model = CrashModelV2_2D(opt).to(device)
    model = model.to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./result', exist_ok=True)
#     os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')


    """ evaluation """
    preds = inference(model, test_loader, device)
    
    submit = pd.read_csv('sample_submission.csv')
    if opt.test_data == "train.csv":
        submit = pd.read_csv(opt.test_data)

    submit['label'] = preds
    submit.to_csv('./submission.csv', index=False)
    if opt.output == "":
        submit.to_csv(f'./result/{opt.label_info}.csv', index=False)
    else:
        submit.to_csv(f'./result/{opt.output}.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', default='test.csv', help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    
    parser.add_argument('--video_length', type=int, default=50, help='the height of the input image')
    parser.add_argument('--imgH', type=int, default=128, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=128, help='the width of the input image')

    parser.add_argument('--model_name', default='', help='"" | inception_v3')
    parser.add_argument('--video_input', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--feature_dim', type=int, default=1024, help='the width of the input image')
    parser.add_argument('--num_classes', type=int, default=2, help='the width of the input image')

    parser.add_argument('--use_thumbnail', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--label_info', default='', help='the width of the input image')
    parser.add_argument('--output', default='', help='the width of the input image')

    
    opt = parser.parse_args()
    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    # random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    test(opt)

