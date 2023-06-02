import os
import pandas as pd
import torch.utils.data as data
import numpy as np
import torch

from tqdm import tqdm

NUM_USERS = 53897
NUM_ITEMSETS = 27694
NUM_ITEMS = 42653

class Task1Loader(data.Dataset):
    
    def __init__(self, data_path): 
        self.data_path = '/dataset/user_itemset_training.csv'
        self.data = self.data_preprocess()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        return self.data[index], index
    
    def data_preprocess(self):
        
        df = pd.read_csv(self.data_path, names=['user_id', 'itemset_id'])
        
        users = list(set(df['user_id'].tolist()))
        data_dict = dict()
        for user in tqdm(users):
            itemsets = df[df['user_id']==user]['itemset_id'].tolist() 
            data_dict.update({user:itemsets})
            
        user_itemset = np.zeros((NUM_USERS, NUM_ITEMSETS))
        for i in range(len(user_itemset)):
            user_itemset[i,data_dict[i]] = 1
            
        return user_itemset


class Task2Loader(data.Dataset):

    def __init__(self, data_path):
        self.data_path = '/home/nas3_userL/sohyunjeong/work_dir/etc/23spring/hw2/dataset/itemset_item_training.csv'
        self.data = self.data_preprocess()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index][:-1]
        itemset_id = self.data[index][-1]
        
        return data, itemset_id

    def data_preprocess(self):

        df = pd.read_csv(self.data_path, names=['itemset_id', 'item_id'])

        itemsets = list(set(df['itemset_id'].tolist()))
        data_dict = dict()
        for itemset in tqdm(itemsets):
            items = df[df['itemset_id']==itemset]['item_id'].tolist()
            data_dict.update({itemset:items})

        itemset_item = np.zeros((len(data_dict.keys()), NUM_ITEMS+1))
        for i, (k, v) in enumerate(data_dict.items()) :
            itemset_item[i,NUM_ITEMS] = k
            itemset_item[i,v] = 1

        return itemset_item



if __name__=="__main__":


    task1_dataset = Task1Loader('/home/nas3_userL/sohyunjeong/work_dir/etc/23spring/hw2/dataset/user_itemset_training.csv')
    task2_dataset = Task2Loader('/home/nas3_userL/sohyunjeong/work_dir/etc/23spring/hw2/dataset/itemset_item_training.csv')
    
    task1_loader = torch.utils.data.DataLoader(task1_dataset, batch_size=8)
    task2_loader = torch.utils.data.DataLoader(task2_dataset, batch_size=8)
    
    for d, v in zip(task1_loader, task2_loader):
        d = d
        v = v
        print('a')
        