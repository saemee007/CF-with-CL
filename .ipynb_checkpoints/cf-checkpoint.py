import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from data.cf_loader import Task1Loader, Task2Loader

task1_dataset = Task1Loader('/home/nas3_userL/sohyunjeong/work_dir/etc/23spring/hw2/dataset/user_itemset_training.csv')
task2_dataset = Task2Loader('/home/nas3_userL/sohyunjeong/work_dir/etc/23spring/hw2/dataset/itemset_item_training.csv')
valid_df = pd.read_csv('/home/nas3_userL/sohyunjeong/work_dir/etc/23spring/hw2/dataset/user_itemset_valid_query.csv', names=['user_id', 'itemset_id'])
valid_ans = pd.read_csv('/home/nas3_userL/sohyunjeong/work_dir/etc/23spring/hw2/dataset/user_itemset_valid_answer.csv', names=['answer'])
print("Loading data is done.")

NUM_USERS = 53897
NUM_ITEMSETS = 27694

x = {}
for user in tqdm(range(NUM_USERS)):

    checkpoint_path = f'/home/nas4_user/saemeechoi/course/DataMining/WCL/data/dataset/features/{user}.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    x[user] = checkpoint
    
    
# similarity function
def sim(A, B):
    return torch.dot(A, B) / (torch.norm(A) * torch.norm(B))

valid_df['score'] = 0
for i in valid_df['user_id'].tolist():
    numer = 0
    denom = 0
    
    for j in range(NUM_USERS):
        similarity = sim(x[i], x[j])
        denom += similarity
        numer += similarity * task1_dataset[i]

        # import pdb
        # pdb.set_trace()
    valid_df['score'][i] = denom / numer


