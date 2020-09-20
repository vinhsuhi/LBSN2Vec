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
    args = parser.parse_args()
    return args


def learn_emb(sentences, n_nodes, emb_dim, n_epochs, win_size, \
        selected_checkins, user_checkins_dict, alpha=0.2, num_neg=10, args=None, maps=None, new_maps=None):
    min_user = np.min(selected_checkins[:,0])
    max_user = np.max(selected_checkins[:,0])
    sentences = np.array(sentences)
    embedding_model = EmbModel(n_nodes, emb_dim)
    embedding_model = embedding_model.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, embedding_model.parameters()), lr=args.learning_rate)
    sentence_length = sentences.shape[1]
    BATCH_SIZE = args.batchsize
    N_ITERS = len(sentences) // BATCH_SIZE
    if N_ITERS % BATCH_SIZE > 0:
        N_ITERS += 1
    for _ in range(n_epochs):
        np.random.shuffle(sentences)
        for iter in tqdm(range(N_ITERS)):
            this_sentences = sentences[iter * BATCH_SIZE: (iter + 1) * BATCH_SIZE]
            loss1s = []
            loss2s = []
            loss3s = []
            for j in range(sentence_length):
                
                words = this_sentences[:, j]
                edges = []
                for k in range(1, win_size + 1):
                    if np.random.rand() > alpha:
                        if j >= k:
                            this_edges1 = this_sentences[:, [j, j - k]]
                            edges.append(this_edges1)
                        if j + k < sentence_length:
                            this_edges2 = this_sentences[:, [j, j + k]]
                            edges.append(this_edges2)
                if len(edges) > 0:
                    edges = np.concatenate(edges, axis=0)
                    edges = torch.LongTensor(edges)
                    edges = edges.cuda()
                    neg = np.random.randint(min_user, max_user, num_neg)
                    neg = torch.LongTensor(neg).cuda()
                    optimizer.zero_grad()
                    loss1 = embedding_model.edge_loss(edges, neg)
                    loss1.backward()
                    optimizer.step()
                    loss1s.append(loss1.item())

                this_user_checkins = []
                for w in words:
                    try:
                        this_checkins = user_checkins_dict[w]
                        this_user_checkins.append(this_checkins)
                    except Exception as err:
                        print(err)
                        print(w)
                        continue
                this_user_checkins = np.concatenate(this_user_checkins, axis=0)
                num_checkins_to_sample = int(alpha * len(this_user_checkins))
                if num_checkins_to_sample > 0:
                    checkin_indices = np.random.randint(0, len(this_user_checkins), min(num_checkins_to_sample, 2*win_size*BATCH_SIZE))
                    checkins = this_user_checkins[checkin_indices]
                    checkins = torch.LongTensor(checkins).cuda()
                    neg_ind = np.random.randint(0, len(selected_checkins), 10)
                    neg = selected_checkins[neg_ind]
                    neg = torch.LongTensor(neg).cuda()
                    optimizer.zero_grad()
                    loss2 = embedding_model.hyperedge_loss(checkins, neg)
                    loss2.backward()
                    optimizer.step()
                    loss2s.append(loss2.item())
                if args.input_type == "persona":
                    groups = [maps[ele] for ele in words]
                    toconnect = np.array([np.random.choice(new_maps[ele]) for ele in groups])
                    edges = np.array([words, toconnect])
                    edges = torch.LongTensor(edges)
                    if args.cuda:
                        edges = edges.cuda()
                    neg = np.random.randint(min_user, max_user, num_neg)
                    neg = torch.LongTensor(neg)
                    if args.cuda:
                        neg = neg.cuda()
                    optimizer.zero_grad()
                    loss_persona = embedding_model.edge_loss(edges, neg)
                    loss_persona.backward()
                    optimizer.step()
                    loss3s.append(loss_persona.item())
            print("Loss1: {:.4f}".format(np.mean(loss1s)))
            print("Loss2: {:.4f}".format(np.mean(loss2s)))
            if args.input_type == "persona":
                print("Loss3: {:.4f}".format(np.mean(loss3s)))
    embeddings = embedding_model.node_embedding(torch.LongTensor(np.arange(n_nodes)).cuda())
    embeddings = embeddings.detach().cpu().numpy()
    return embeddings


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

    for key, value in new_maps.items():
        max_node += 1
        maps[max_node] = key
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

    maps = dict()
    with open(path2, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.strip().split(',')
            maps[int(data_line[0]) + 1] = int(data_line[1])

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
        return edges, maps, user_POI, POI_dict, new_maps
    else:
        return edges, maps, new_maps


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
    val_user_checkins = {}
    for user_id in range(1, n_users+1): 
        inds_checkins = np.argwhere(val_checkins[:,0] == user_id).flatten()
        checkins = val_checkins[inds_checkins]
        val_user_checkins[user_id] = checkins
    # everything here is from 1
    return train_checkins, val_checkins, n_users, n_nodes_total, train_user_checkins, val_user_checkins, friendship_old, friendship_new, selected_checkins, offset1, offset2, offset3, new_maps, maps, friendship_old_ori


if __name__ == "__main__":
    # maps: {key: value}; key in [0,..,n], value in [1,...,m]
    args = parse_args()
    train_checkins, val_checkins, n_users, n_nodes_total, train_user_checkins, val_user_checkins, friendship_old, friendship_new, selected_checkins, offset1, offset2, offset3, new_maps, maps, friendship_old_ori = load_data(args)

    sentences = random_walk(friendship_old, n_users, args)
    neg_user_samples, neg_checkins_samples = sample_neg(friendship_old, selected_checkins)
    embs_ini = initialize_emb(args, n_nodes_total)
    save_info(args, sentences, embs_ini, neg_user_samples, neg_checkins_samples, train_user_checkins)
    
    learn.apiFunction("temp/processed/", args.learning_rate, args.K_neg, args.win_size, args.num_epochs, args.workers, args.mobility_ratio)
    embs_file = "temp/processed/embs.txt"
    embs = read_embs(embs_file)
    # train_checkins -= 1
    # val_checkins -= 1
    # train_user_checkins = {key - 1: value - 1 for key, value in train_user_checkins.items()}
    # val_user_checkins = {key - 1: value - 1 for key, value in val_user_checkins.items()}
    # friendship_new -= 1
    # friendship_old -= 1
    # maps_suhi = dict()
    # new_maps_suhi = dict()
    # if maps is not None:
    #     maps_suhi = dict()
    #     new_maps_suhi = dict()
    #     for key, value in maps.items():
    #         maps_suhi[key -1] = value - 1
    #     for key, value in new_maps.items():
    #         new_maps_suhi[key - 1] = [ele - 1 for ele in value]
    # for i in range(len(sentences)):
    #     sentences[i] = [x-1 for x in sentences[i]]
    # embs = learn_emb(sentences, n_nodes_total, args.dim_emb, args.num_epochs, args.win_size, \
    #     train_checkins, train_user_checkins, alpha=args.mobility_ratio, num_neg = args.K_neg, args=args, maps=maps_suhi, new_maps = new_maps_suhi)
    # evaluate
    embs_user = embs[:offset1]
    embs_time = embs[offset1:offset2]
    embs_venue = embs[offset2:offset3]
    embs_cate = embs[offset3:]

    if np.min(friendship_old_ori) == 1:
        friendship_old_ori -= 1
    if np.min(friendship_old) == 1: # cpp
        friendship_linkprediction(embs_user, friendship_old-1, friendship_new-1, k=10, new_maps=new_maps, maps=maps, friendship_old_ori=friendship_old_ori)
    else:
        friendship_linkprediction(embs_user, friendship_old, friendship_new, k=10, new_maps=new_maps, maps=maps, friendship_old_ori=friendship_old_ori)

