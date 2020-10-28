import argparse
from scipy.io import loadmat
import os
import numpy as np
from evaluation import friendship_pred_persona, friendship_pred_ori



def parse_args():
    parser = argparse.ArgumentParser(description="LBSN configurations")
    parser.add_argument('--emb_path', type=str, default="")
    parser.add_argument('--dataset_name', type=str, default="")
    parser.add_argument('--model', type=str, default="")
    args = parser.parse_args()
    return args

def read_emb(path, model):
    embs = None
    if model == "node2vec" or model == "deepwalk" or model == "line":
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

"""

for data in hongzhi NYC TKY
do
python eval_models.py --emb_path line_emb/${data}.embeddings --dataset_name ${data} --model line
done 



for data in NYC TKY hongzhi 
do     
deepwalk --format edgelist --input ../LBSN2Vec/edgelist_graph/${data}.edgelist     --max-memory-data-size 0 --number-walks 10 --representation-size 128 --walk-length 80 --window-size 10     --workers 16 --output ../LBSN2Vec/deepwalk_emb/${data}.embeddings
done

for data in NYC TKY hongzhi
do 
python run_node2vec --dataset_name ${data}
done

for data in NYC TKY hongzhi
do
python -m openne --method line --input ../../LBSN2Vec/edgelist_graph/${data}.edgelist --graph-format edgelist --output ../../LBSN2Vec/line_emb/${data}.embeddings 
done

"""