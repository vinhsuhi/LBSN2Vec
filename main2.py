import pdb
import numpy as np
import random
from tqdm import tqdm
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.io import loadmat
from tqdm import tqdm
import numpy as np
import random
import pdb
import math
import networkx as nx 
import os
import multiprocessing
import numpy as np
import multiprocessing
import networkx as nx
from gensim.models import Word2Vec
from evaluation import *
import argparse
import learn
from embedding_model import EmbModel
import torch
import time
import json
from utils import save_info, sample_neg, read_embs, initialize_emb, random_walk
import torch.nn as nn
from link_pred_model import StructMLP
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score


def parse_args():
    parser = argparse.ArgumentParser(description="LBSN configurations")
    parser.add_argument('--num_walks', type=int, default=10)
    parser.add_argument('--walk_length', type=int, default=80)
    parser.add_argument('--workers', type=int, default=26)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--mobility_ratio', type=float, default=0.2)
    parser.add_argument('--K_neg', type=int, default=10)
    parser.add_argument('--win_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--dim_emb', type=int, default=128)
    parser.add_argument('--batchsize', type=int, default=512)
    parser.add_argument('--mode', type=str, default='friend', help="friend or POI")
    parser.add_argument('--input_type', type=str, default="mat", help="mat or persona") 
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--load', action='store_true') 
    parser.add_argument('--py', action='store_true') 
    parser.add_argument('--dataset_name', type=str, default='NYC')
    parser.add_argument('--clean', action='store_true', help='use cleaned dataset')
    parser.add_argument('--num_embs', type=int, default=10)
    args = parser.parse_args()
    return args


def load_ego(path1, path2, path3=None, path4=None):
    maps = dict()
    new_maps = dict()
    max_node = 0
    with open(path2, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.strip().split(',')
            persona_node = int(data_line[0]) + 1
            ori_node = int(data_line[1])
            if ori_node not in new_maps:
                new_maps[ori_node] = set([persona_node])
            else:
                new_maps[ori_node].add(persona_node)
            maps[persona_node] = ori_node
            if persona_node > max_node:
                max_node = persona_node

    additional_edges = []

    center_ori_dict = dict()
    for key, value in new_maps.items():
        max_node += 1
        maps[max_node] = key
        center_ori_dict[max_node] = key
        new_maps[key].add(max_node)
        for ele in value:
            additional_edges.append([max_node, ele])

    edges = []
    with open(path1, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.strip().split()
            edges.append([int(ele) + 1 for ele in data_line[:2]])
    print("Number of edges before: {}".format(len(edges)))
    edges = edges + additional_edges
    edges = np.array(edges)
    print("Number of edges after: {}".format(len(edges)))

    user_POI = dict() # persona user to POI of input of persona
    if path3 is not None:
        with open(path3, 'r', encoding='utf-8') as file:
            for line in file:
                data_line = line.strip().split(',')
                user = int(data_line[0]) + 1
                location = int(data_line[1])
                if user not in user_POI:
                    user_POI[user] = set([location])
                else:
                    user_POI[user].add(location)
    POI_dict = dict() # POI of input of persona to original POI 
    if path4 is not None:
        with open(path4, 'r', encoding='utf-8') as file:
            for line in file:
                data_line = line.split()
                POI_dict[int(data_line[0])] = int(data_line[1])
    if path3 is not None:
        return edges, maps, user_POI, POI_dict, new_maps, center_ori_dict
    else:
        return edges, maps, new_maps


def load_data2(args):
    maps = None
    new_maps = None
    friendship_old_ori = None

    if args.clean:
        mat = loadmat('dataset/cleaned_{}.mat'.format(args.dataset_name))
    else:
        mat = loadmat('dataset/dataset_connected_{}.mat'.format(args.dataset_name))
    edges, maps, persona_POI, POI_dict, new_mapss, center_ori_dict = load_ego('Suhi_output/edgelist_{}'.format(args.dataset_name), 
                                                  'Suhi_output/ego_net_{}'.format(args.dataset_name), 
                                                  'Suhi_output/edgelistPOI_{}'.format(args.dataset_name),
                                                  'Suhi_output/location_dict_{}'.format(args.dataset_name))
    friendship_old_ori = mat['friendship_old']
    friendship_old = edges
    friendship_n = mat["friendship_new"]
    new_maps = dict()
    for key, value in maps.items():
        if value not in new_maps:
            new_maps[value] = set([key])
        else:
            new_maps[value].add(key)

    def create_new_checkins2(old_checkins, new_maps, persona_POI, POI_dict, center_ori_dict):
        ori_center_dict = {v:k for k,v in center_ori_dict.items()}
        new_checkins = []
        for i in tqdm(range(len(old_checkins))):
            old_checkini = old_checkins[i]
            user = old_checkini[0]
            center_user = ori_center_dict[user]
            new_checkins.append([center_user, old_checkini[1], old_checkini[2], old_checkini[3]])
            location = old_checkini[2]
            location_image = POI_dict[location]
            for ele in new_maps[user]:
                if ele not in persona_POI:
                    continue
                if location_image in persona_POI[ele]:
                    new_checkins.append([ele, old_checkini[1], old_checkini[2], old_checkini[3]])
        new_checkins = np.array(new_checkins)
        return new_checkins

    selected_checkins = create_new_checkins2(mat['selected_checkins'], new_maps, persona_POI, POI_dict, center_ori_dict)
    friendship_new = friendship_n


    offset1 = max(selected_checkins[:, 0])
    _, n = np.unique(selected_checkins[:, 1], return_inverse=True)  #
    selected_checkins[:, 1] = n + offset1 + 1
    offset2 = max(selected_checkins[:, 1])
    _, n = np.unique(selected_checkins[:, 2], return_inverse=True)
    selected_checkins[:, 2] = n + offset2 + 1
    offset3 = max(selected_checkins[:, 2])
    _, n = np.unique(selected_checkins[:, 3], return_inverse=True)
    selected_checkins[:, 3] = n + offset3 + 1
    n_nodes_total = np.max(selected_checkins)

    n_users = selected_checkins[:, 0].max()  # user
    print(f"""Number of users: {n_users}
    Number of nodes total: {n_nodes_total}""")

    n_data = selected_checkins.shape[0]
    if args.mode == "friend":
        n_train = n_data
    else:
        n_train = int(n_data * 0.8)

    sorted_checkins = selected_checkins[np.argsort(selected_checkins[:, 1])]
    train_checkins = sorted_checkins[:n_train]
    val_checkins = sorted_checkins[n_train:]

    print("Build user checkins dictionary...")
    train_user_checkins = {}
    for user_id in range(1, n_users + 1):
        inds_checkins = np.argwhere(train_checkins[:, 0] == user_id).flatten()
        checkins = train_checkins[inds_checkins]
        train_user_checkins[user_id] = checkins
    val_user_checkins = {}
    for user_id in range(1, n_users + 1):
        inds_checkins = np.argwhere(val_checkins[:, 0] == user_id).flatten()
        checkins = val_checkins[inds_checkins]
        val_user_checkins[user_id] = checkins
    # everything here is from 1
    return train_checkins, val_checkins, n_users, n_nodes_total, train_user_checkins, val_user_checkins, friendship_old, friendship_new, selected_checkins, offset1, offset2, offset3, new_maps, maps, friendship_old_ori


def load_data(args):
    maps = None
    new_maps = None
    friendship_old_ori = None
    if args.clean:
        mat = loadmat('dataset/cleaned_{}.mat'.format(args.dataset_name))
    else:
        mat = loadmat('dataset/dataset_connected_{}.mat'.format(args.dataset_name))
    edges, maps, new_maps = load_ego('Suhi_output/edgelist_{}'.format(args.dataset_name), 'Suhi_output/ego_net_{}'.format(args.dataset_name))
    friendship_old_ori = mat['friendship_old']
    friendship_old = edges 
    friendship_new = mat["friendship_new"] 
    
    def create_new_checkins(old_checkins, new_maps):
        new_checkins = []
        for i in range(len(old_checkins)):
            checkins_i = old_checkins[i]
            user = old_checkins[i][0]
            for ele in new_maps[user]:
                new_checkins.append([ele, checkins_i[1], checkins_i[2], checkins_i[3]])
        new_checkins = np.array(new_checkins)
        return new_checkins
            
    selected_checkins = create_new_checkins(mat['selected_checkins'], new_maps)

    offset1 = max(selected_checkins[:,0])
    _, n = np.unique(selected_checkins[:,1], return_inverse=True) # 
    selected_checkins[:,1] = n + offset1 + 1
    offset2 = max(selected_checkins[:,1])
    _, n = np.unique(selected_checkins[:,2], return_inverse=True)
    selected_checkins[:,2] = n + offset2 + 1
    offset3 = max(selected_checkins[:,2])
    _, n = np.unique(selected_checkins[:,3], return_inverse=True)
    selected_checkins[:,3] = n + offset3 + 1
    n_nodes_total = np.max(selected_checkins)

    n_users = selected_checkins[:,0].max() # user
    print(f"""Number of users: {n_users}
        Number of nodes total: {n_nodes_total}""")

    n_data = selected_checkins.shape[0]
    if args.mode == "friend":
        n_train = n_data
    else:
        n_train = int(n_data * 0.8)

    sorted_checkins = selected_checkins[np.argsort(selected_checkins[:,1])]
    train_checkins = sorted_checkins[:n_train]
    val_checkins = sorted_checkins[n_train:]


    print("Build user checkins dictionary...")
    train_user_checkins = {}
    for user_id in range(1, n_users+1): 
        inds_checkins = np.argwhere(train_checkins[:,0] == user_id).flatten()
        checkins = train_checkins[inds_checkins]
        train_user_checkins[user_id] = checkins
        break
    val_user_checkins = {}
    for user_id in range(1, n_users+1): 
        inds_checkins = np.argwhere(val_checkins[:,0] == user_id).flatten()
        checkins = val_checkins[inds_checkins]
        val_user_checkins[user_id] = checkins
        break
    # everything here is from 1
    return train_checkins, val_checkins, n_users, n_nodes_total, train_user_checkins, val_user_checkins, friendship_old, friendship_new, selected_checkins, offset1, offset2, offset3, new_maps, maps, friendship_old_ori

def sample_edges_non_edges(edges, num_samples, n_nodes):
    num_edges = edges.shape[0]
    edges_sampled = edges[np.random.randint(num_edges, size=num_samples)]
    source = np.random.randint(0, n_nodes, num_samples)
    target = np.random.randint(0, n_nodes, num_samples)
    non_edges = np.array([source, target])
    return edges_sampled, non_edges

def eval_acc(mlp, embs, friendship_new):
            friendship_new = torch.FloatTensor(friendship_new)
            friendship_new = friendship_new.cuda()
            pred = mlp.forward(embs, friendship_new)
            pred = F.log_softmax(pred)
            pred = pred.detach().cpu().numpy()
            pred = np.argmax(pred, axis=1)
            t_test = np.ones(len(pred))
            print("Test Micro F1 Score: ", f1_score(t_test, pred, average='micro'))
            print("Test Weighted F1 Score: ", f1_score(t_test, pred, average='weighted'))
            print("Test Accuracy Score: ", accuracy_score(t_test, pred))

if __name__ == "__main__":
    # maps: {key: value}; key in [1,..,n], value in [1,...,m] (also new_maps)
    args = parse_args()
    if args.input_type == "persona2":
        train_checkins, val_checkins, n_users, n_nodes_total, train_user_checkins, val_user_checkins, friendship_old, friendship_new, selected_checkins, offset1, offset2, offset3, new_maps, maps, friendship_old_ori = load_data2(args)
    else:
        train_checkins, val_checkins, n_users, n_nodes_total, train_user_checkins, val_user_checkins, friendship_old, friendship_new, selected_checkins, offset1, offset2, offset3, new_maps, maps, friendship_old_ori = load_data(args)

    sentences = random_walk(friendship_old, n_users, args)
    neg_user_samples, neg_checkins_samples = sample_neg(friendship_old, selected_checkins)
    embs_ini = initialize_emb(args, n_nodes_total)
    save_info(args, sentences, embs_ini, neg_user_samples, neg_checkins_samples, train_user_checkins)
    for i in tqdm(range(args.num_embs)):
        """
        learn.apiFunction("temp/processed/", args.learning_rate, args.K_neg, args.win_size, args.num_epochs, args.workers, args.mobility_ratio)
        embs_file = "temp/processed/embs.txt"
        embs = read_embs(embs_file)
        embs_user = embs[:offset1]
        embs_time = embs[offset1:offset2]
        embs_venue = embs[offset2:offset3]
        embs_cate = embs[offset3:]
        """
        # predict link here
        embs_user = np.random.rand(n_users, 128)
        mlp = StructMLP(embs_user.shape[1], 256)
        mlp = mlp.cuda()
        mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

        """
        embs = torch.FloatTensor(embs_user)
        embs = embs.cuda()
        """

        for ep in range(10):
            mlp_optimizer.zero_grad()
            edges, non_edges = sample_edges_non_edges(friendship_old, 2000, n_users)
            edges = torch.LongTensor(edges)
            non_edges = torch.LongTensor(non_edges)
            edges = edges.cuda()
            non_edges = non_edges.cuda()
            samples = torch.cat((edges, non_edges), dim=0)
            labels = torch.cat((torch.ones(len(edges)), torch.zeros(len(non_edges))), dim = 0)
            samples = samples.cuda()
            labels = labels.cuda()
            loss = mlp.compute_loss(embs, samples, labels)
            loss.backward()
            print("Loss: {:.4f}".format(loss.item()))
            mlp_optimizer.step()

        eval_acc(mlp, embs, friendship_new)
    

        # evaluate here
    exit()
    """
    if args.mode == 'friend':
        # maps and new_maps must be from 1
        # input friendship must be from 0
        if np.min(friendship_old_ori) == 1:
            friendship_old_ori -= 1
        if np.min(friendship_old) == 1: # cpp
            friendship_linkprediction(embs_user, friendship_old-1, friendship_new-1, k=10, new_maps=new_maps, maps=maps, friendship_old_ori=friendship_old_ori)
        else:
            friendship_linkprediction(embs_user, friendship_old, friendship_new, k=10, new_maps=new_maps, maps=maps, friendship_old_ori=friendship_old_ori)
    else:
        # import pdb; pdb.set_trace()
        val_checkins[:,0] -= 1
        val_checkins[:,1] -= (offset1+1)
        val_checkins[:,2] -= (offset2+1)
        location_prediction(val_checkins[:,:3], embs, embs_venue, k=10)
    """