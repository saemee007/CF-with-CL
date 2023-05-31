import numpy as np
import pandas as pd
from tqdm import tqdm
from dotmap import DotMap

import torch
import argparse
from data.cf_loader import Task1Loader, Task2Loader

"""
Run:
python cf.py result_save_path='/home/nas4_user/hawonjeong/course/ai506/results/task1_valid_result/task1_results.csv'
"""


def run(args):

    args = DotMap(args)
    num_users = args.num_users
    num_itemsets = args.num_itemsets
    latent_dim = args.latent_dim
    result_save_path = args.result_save_path

    # Load dataset

    task1_dataset = Task1Loader('/home/nas3_userL/sohyunjeong/work_dir/etc/23spring/hw2/dataset/user_itemset_training.csv').data_preprocess()
    task2_dataset = Task2Loader('/home/nas3_userL/sohyunjeong/work_dir/etc/23spring/hw2/dataset/itemset_item_training.csv').data_preprocess()
    valid_df = pd.read_csv('/home/nas3_userL/sohyunjeong/work_dir/etc/23spring/hw2/dataset/user_itemset_valid_query.csv', names=['user_id', 'itemset_id'])
    valid_ans = pd.read_csv('/home/nas3_userL/sohyunjeong/work_dir/etc/23spring/hw2/dataset/user_itemset_valid_answer.csv', names=['answer'])
    print("Loading data is done.")
    print()


    # Load user checkpoints from WCL training
    print("[Load user checkpoints]")
    user_rep = torch.zeros((num_users, latent_dim))
    
    for user in tqdm(range(num_users)):
        checkpoint_path = f'/home/nas4_user/saemeechoi/course/DataMining/WCL/data/dataset/features_task1/{user}.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        user_rep[user] = checkpoint
    print("Checkpoints are loaded.")
    print()
        
    # Get similarity matrix
    mul_user_rep = torch.mm(user_rep, user_rep.t())
    norm = torch.sqrt(torch.diag(mul_user_rep)).unsqueeze(1)
    user_sim = torch.div(mul_user_rep, norm)
    user_sim = torch.div(user_sim, norm.t())


    # Get scores of valid set
    print("Calculating scores on valid set...")

    task1_dataset_tensor = torch.tensor(task1_dataset).float()
    valid_df['score'] = torch.zeros(len(valid_df))

    for i in tqdm(range(len(valid_df))):
        user_id = valid_df['user_id'][i]
        itemset_id = valid_df['itemset_id'][i]

        numer = torch.matmul(user_sim[user_id].unsqueeze(0), task1_dataset_tensor[:, itemset_id].unsqueeze(1))
        denom = torch.sum(user_sim[user_id])

        valid_df['score'][i] = numer.item() / denom.item()


    # save the dataframe
    # result_save_path = '/home/nas4_user/hawonjeong/course/ai506/results/task1_valid_result/task1_results.csv'
    valid_df.to_csv(result_save_path, index=False)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_users', type=int, default=53897)
    parser.add_argument('--num_itemsets', type=int, default=27694)
    parser.add_argument('--num_items', type=int, default=0)
    parser.add_argument('--latent_dim', type=int, default=4096)
    parser.add_argument('result_save_path', type=str)
    args = parser.parse_args()
    
    run(vars(args))

