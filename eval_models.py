import argparse
from scipy.io import loadmat
import os
import numpy as np
from evaluation import friendship_pred_persona, friendship_pred_ori, location_prediction
from utils import renumber_checkins

def parse_args2():
    parser = argparse.ArgumentParser(description="LBSN configurations")
    parser.add_argument('--emb_path', type=str, default="")
    parser.add_argument('--dataset_name', type=str, default="")
    parser.add_argument('--model', type=str, default="")
    parser.add_argument('--POI', action='store_true')
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
    elif model == "dhne":
        embs = np.load(path, allow_pickle=True)
        if not args.POI:
            embs = embs[0]
    return embs 


def read_input2(path):
    mat = loadmat('dataset/cleaned_{}.mat'.format(args.dataset_name))
    friendship_old = mat['friendship_old']
    friendship_new = mat['friendship_new']

    friendship_old = friendship_old[np.argsort(friendship_old[:, 0])]
    return friendship_old, friendship_new



def read_input(path):
    mat = loadmat('dataset/cleaned_{}.mat'.format(args.dataset_name))
    friendship_old = mat['friendship_old']
    selected_checkins = mat['selected_checkins']
    friendship_old -= 1
    nodes = np.unique(friendship_old)
    print("Min: {}, Max: {}, Len: {}".format(np.min(nodes), np.max(nodes), len(nodes)))
    friendship_old = friendship_old[np.argsort(friendship_old[:, 0])]
    return friendship_old, selected_checkins


def preprocess_selected_checkins(selected_checkins):
    selected_checkins = np.delete(selected_checkins, 1, 1)
    selected_checkins = selected_checkins[np.argsort(selected_checkins[:, 0])]
    unique_location = np.unique(selected_checkins[:, 1])
    unique_cate = np.unique(selected_checkins[:, 2])
    location_id2idx = {unique_location[i]: i for i in range(len(unique_location))}
    cate_id2idx = {unique_cate[i]: i for i in range(len(unique_cate))}
    for i in range(len(selected_checkins)):
        selected_checkins[i, 0] = selected_checkins[i, 0] - 1
        selected_checkins[i, 1] = location_id2idx[selected_checkins[i, 1]]
        selected_checkins[i, 2] = cate_id2idx[selected_checkins[i, 2]]
    new_location = len(unique_location)
    new_cate = len(cate_id2idx)
    num_user = np.max(selected_checkins[:, 0]) + 1
    unique_user = np.unique(selected_checkins[:, 0]).tolist()
    additional_checkins = []
    for i in range(num_user):
        if i not in unique_user:
            additional_checkins.append([i, new_location, new_cate])
            new_location += 1
            new_cate += 1
    additional_checkins = np.array(additional_checkins)
    if len(additional_checkins) > 0:
        selected_checkins = np.concatenate((selected_checkins, additional_checkins), axis=0)
    
    return selected_checkins



def preprocess_selected_checkins2(selected_checkins):
    """
    What does this function do???
    1. sort selected checkins according to user ID
    2. renumberring phase 1
    """
    # selected_checkins = np.delete(selected_checkins, 1, 1)
    selected_checkins = selected_checkins[np.argsort(selected_checkins[:, 0])]
    unique_location = np.unique(selected_checkins[:, 2])
    unique_cate = np.unique(selected_checkins[:, 3])
    unique_time = np.unique(selected_checkins[:, 1])
    location_id2idx = {unique_location[i]: i for i in range(len(unique_location))}
    cate_id2idx = {unique_cate[i]: i for i in range(len(unique_cate))}
    time_id2idx = {unique_time[i]: i for i in range(len(unique_time))}
    for i in range(len(selected_checkins)):
        selected_checkins[i, 0] = selected_checkins[i, 0] - 1
        selected_checkins[i, 1] = time_id2idx[selected_checkins[i, 1]]
        selected_checkins[i, 2] = location_id2idx[selected_checkins[i, 2]]
        selected_checkins[i, 3] = cate_id2idx[selected_checkins[i, 3]]
    new_location = len(unique_location)
    new_cate = len(cate_id2idx)
    new_time = len(unique_time)
    num_user = np.max(selected_checkins[:, 0]) + 1
    unique_user = np.unique(selected_checkins[:, 0]).tolist()
    additional_checkins = []
    for i in range(num_user):
        if i not in unique_user:
            additional_checkins.append([i,new_time, new_location, new_cate])
            new_location += 1
            new_cate += 1
            new_time += 1
    additional_checkins = np.array(additional_checkins)
    if len(additional_checkins) > 0:
        selected_checkins = np.concatenate((selected_checkins, additional_checkins), axis=0)
    
    return selected_checkins



if __name__ == "__main__":
    args = parse_args2()
    print(args)
    model = args.model 
    if args.POI:
        embs = read_emb(args.emb_path, args.model)
        friendship, selected_checkins = read_input(args.dataset_name)
        friendship = friendship.astype(int)
        if model.lower() != "dhne":
            selected_checkins = preprocess_selected_checkins2(selected_checkins)
            selected_checkins, o1, o2, o3, nt, nu = renumber_checkins(selected_checkins)
            max_node = selected_checkins.max()
            if args.POI:
                n_trains = int(0.8 * len(selected_checkins))
                selected_checkins = selected_checkins[:n_trains]
                sorted_time = np.argsort(selected_checkins[:, 1])
                train_indices = sorted_time[:n_trains]
                test_indices = np.array([ele for ele in range(len(selected_checkins)) if ele not in train_indices])
                train_checkins = selected_checkins[train_indices]
                test_checkins = selected_checkins[test_indices]

            embs_user = embs[:o1]
            embs_time = embs[o1:o2]
            embs_venue = embs[o2:o3]

            test_checkins[:, 2] -= o2

            location_prediction(test_checkins, embs, embs_venue, k=10)
    else:
        # train_checkins, test_checkins = read_input_POI(args.path)

        friendship_old, friendship_new = read_input(args.dataset_name)
        n_users = max(np.max(friendship_old), np.max(friendship_new))
        embs = embs[:n_users]

        friendship_pred_ori(embs, friendship_old, friendship_new)

"""
####################### eval ###############################
for data in Istanbul
do
python eval_models.py --emb_path line_emb/${data}_M_POI.embeddings --dataset_name ${data} --model line --POI
done 


for data in hongzhi NYC TKY
do
python eval_models.py --emb_path line_emb/${data}.embeddings --dataset_name ${data} --model line
done 
#############################################################3

######################## gen embedding #########################


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

for data in NYC hongzhi TKY
do
python src/hypergraph_embedding.py --data_path ../LBSN2Vec/dhne_graph/${data} --save_path ../LBSN2Vec/dhne_emb/${data} -s 16 16 16
done

#################################################################
"""
