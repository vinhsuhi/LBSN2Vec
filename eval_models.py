import argparse
from scipy.io import loadmat
import os
import numpy as np
from utils import friendship_pred_ori

def parse_args():
    parser = argparse.ArgumentParser(description="LBSN configurations")
    parser.add_argument('--emb_path', type=str, default="")
    parser.add_argument('--dataset_name', type=str, default="")
    parser.add_argument('--model', type=str, default="")
    args = parser.parse_args()
    return args

def read_emb(path, model):
    embs = None
    if model == "node2vec":
        file = open(path, 'r', encoding='utf-8')
        count = 0
        embs = []
        for line in file:
            if count == 0:
                count += 1
                continue
            data_line = line.split()
            embs.append([float(ele) for ele in data_line])
        embs = np.array(embs)
        embs = embs[np.argsort(embs[:, 0])][:, 1:]
    
    return embs 


def read_input(path):
    mat = loadmat('dataset/cleaned_{}.mat'.format(args.dataset_name))
    friendship_old = mat['friendship_old']
    friendship_new = mat['friendship_new']
    # friendship_old -= 1
    # friendship_new -= 1
    # nodes = np.unique(friendship_old)
    # print("Min: {}, Max: {}, Len: {}".format(np.min(nodes), np.max(nodes), np.len(nodes)))
    friendship_old = friendship_old[np.argsort(friendship_old[:, 0])]
    return friendship_old, friendship_new


args = parse_args()
print(args)
embs = read_emb(args.emb_path, args.model)
friendship_old, friendship_new = read_input(args.dataset_name)
friendship_pred_ori(embs, friendship_old, friendship_new)
