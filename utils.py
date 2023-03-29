# +
import yaml
from argparse import Namespace
import pandas as pd
from sklearn.model_selection import KFold

def parse_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    opt = Namespace(**data)
    return opt

def print_and_log(log_path, log_messages):
    print(log_messages)
    

def kfold_data_split(df, kfold = 0, n_splits = 10, random_state = 1):
    if kfold > n_splits:
        print("wrong")
        return None
    # split 개수, 셔플 여부 및 seed 설정
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = random_state)
    
    # split 개수 스텝 만큼 train, test 데이터셋을 매번 분할
    for idx, (train_index, test_index) in enumerate(kf.split(df)):
        if idx != kfold:
            continue
        # X_train, X_test = df.iloc[train_index], df.iloc[test_index]
        break
    return train_index, test_index

def BalancekFold(df, column_name, fold_idx = 0, n_splits = 5, random_state = 1):
    x_list = []
    y_list = []
    for label in sorted(df[column_name].unique()):
        sub_df = df[df[column_name] == label]
        x, y = kfold_data_split(sub_df, kfold = fold_idx, n_splits = n_splits, random_state=random_state)
        x_list.append(x)
        y_list.append(y)
    X_train = pd.concat(x_list)
    y_train = pd.concat(y_list)
    return X_train, y_train

