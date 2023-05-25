import os
import pandas as pd
import torch.utils.data as data
import numpy as np
import torch

from tqdm import tqdm

NUM_USERS = 53897
NUM_ITEMSETS = 27694

class ContrastiveCFloader(data.Dataset):
    
    def __init__(self, data_path, ans_path=None):
        self.data_path = data_path
        self.data = self.data_preprocess()
        self.ans_path = ans_path
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        return self.data[index]
    
    def data_preprocess(self):
        
        df = pd.read_csv(self.data_path, names=['user_id', 'itemset_id'])
        
        users = list(set(df['user_id'].tolist()))
        data_dict = dict()
        for user in tqdm(users):
            items = df[df['user_id']==user]['itemset_id'].tolist()
            data_dict.update({user:items})
            
        user_itemset = np.zeros((NUM_USERS, NUM_ITEMSETS))
        for i in range(len(user_itemset)):
            user_itemset[i,dict[i]] = 1

        return user_itemset
    
    

if __name__=="__main__":
    train_dataset = ContrastiveCFloader('/home/nas3_userL/sohyunjeong/work_dir/etc/23spring/hw2/dataset/user_itemset_training.csv')
    valid_dataset = ContrastiveCFloader('/home/nas3_userL/sohyunjeong/work_dir/etc/23spring/hw2/dataset/user_itemset_valid_query.csv')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=8)
    
    for d, v in zip(train_loader, valid_loader):
        d = d
        v = v
        print('a')
        