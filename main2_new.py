import pdb
import random
from scipy.sparse import csr_matrix
from scipy.io import loadmat
from tqdm import tqdm
import math
import os
import numpy as np
import multiprocessing
import networkx as nx
from gensim.models import Word2Vec
from evaluation import *
import argparse
import learn
from embedding_model import EmbModel
import time
import json
from utils import save_info, sample_neg, read_embs, initialize_emb, random_walk

import torch
import torch.nn as nn
from torch.nn import init
from random import shuffle, randint
import torch.nn.functional as F
from itertools import combinations, combinations_with_replacement
from sklearn.metrics import f1_score, accuracy_score
from sklearn.decomposition import TruncatedSVD

"""
SVD

"""

num_neurons = 256
input_rep = num_neurons + data.num_features
device = torch.device('cuda')

class StructMLP(nn.Module):
    def __init__(self, node_set_size=2):
        super(StructMLP, self).__init__()

        self.node_set_size = node_set_size
        #Deepsets MLP

        self.ds_layer_1 = nn.Linear(input_rep, num_neurons)
        self.ds_layer_2 = nn.Linear(num_neurons, num_neurons)
        self.rho_layer_1 = nn.Linear(num_neurons, num_neurons)
        self.rho_layer_2 = nn.Linear(num_neurons, num_neurons)

        #One Hidden Layer
        self.layer1 = nn.Linear(num_neurons, num_neurons)
        self.layer2 = nn.Linear(num_neurons, 2) # has edge or not
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, samples):
        """
        input_tensor: TruncatedSVD approximations of adj, 1 x num_nodes x svd_dim
        samples: 200 x 2: Matrix of edges and nonedges
        """
        #Deepsets initially on each of the samples
        # input_tensor (svd_representation of ADJ)
        num_nodes = input_tensor.shape[1]
        num_examples = samples.shape[0]
        sum_tensor = torch.zeros(num_examples, num_neurons).to(device) # 200 x 256
        for i in range(input_tensor.shape[0]):
            #Process the input tensor to form n choose k combinations and create a zero tensor
            set_init_rep = input_tensor[i].view(-1, input_rep) # 1
            x = self.ds_layer_1(set_init_rep)
            x = self.relu(x)
            x = self.ds_layer_2(x)
            x = x[samples]
            x = torch.sum(x, dim=1)
            x = self.rho_layer_1(x)
            sum_tensor += x

        x = sum_tensor / input_tensor.shape[0]

        #One Hidden Layer for predictor
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

    def compute_loss(self, input_tensor, samples, target):
        pred = self.forward(input_tensor, samples)
        return F.cross_entropy(pred, target)

def corrupt_adj(adj_mat, percent=2):
    """ Returns the corrupted version of the adjacency matrix """
    """
    adj_mat_corrupted = adj after randomly removing edges (percent) and adding randomly edges
    false_edges = edges that are added
    false_non_edges = edges that are removed
    """
    edges = adj_mat.triu().nonzero()
    num_edges = edges.shape[0]
    num_to_corrupt = int(percent/100.0 * num_edges)
    random_corruption = np.random.randint(num_edges, size=num_to_corrupt)
    adj_mat_corrupted = adj_mat.clone()
    false_edges, false_non_edges = [], []
    #Edge Corruption
    for ed in edges[random_corruption]:
        adj_mat_corrupted[ed[0], ed[1]] = 0
        adj_mat_corrupted[ed[1], ed[0]] = 0
        false_non_edges.append(ed.tolist())
    #Non Edge Corruption
    random_non_edge_corruption = list(np.random.randint(adj_mat.shape[0], size = 6*num_to_corrupt))
    non_edge_to_corrupt = []
    for k in range(len(random_non_edge_corruption)-1):
        to_check = [random_non_edge_corruption[k], random_non_edge_corruption[k+1]]
        if to_check not in edges.tolist():
            non_edge_to_corrupt.append(to_check)
        if len(non_edge_to_corrupt) == num_to_corrupt:
            break
    non_edge_to_corrupt = torch.Tensor(non_edge_to_corrupt).type(torch.int16)
    for n_ed in non_edge_to_corrupt:
        adj_mat_corrupted[n_ed[0], n_ed[1]] = 1
        adj_mat_corrupted[n_ed[1], n_ed[0]] = 1
        false_edges.append(n_ed.tolist())
    return adj_mat_corrupted, false_edges, false_non_edges



def sample_equal_number_edges_non_edges(adj_mat, false_non_edges, false_edges, small_samples):
    """
    return: final_edges: ~200 true edges + false_non_edges
            final_non_edges: ~200 true non_edges + false_edges
    """
    edges = adj_mat.nonzero()
    num_edges = edges.shape[0]
    inverse_adj_mat = 1 - adj_mat
    non_edges = inverse_adj_mat.nonzero()
    num_non_edges  = non_edges.shape[0]
    edges_sampled = edges[np.random.randint(num_edges, size=small_samples)]
    non_edges_sampled = non_edges[np.random.randint(num_non_edges, size=small_samples)]
    final_edges = []
    final_non_edges = []
    for ed in edges_sampled.tolist():
        if ed not in false_edges:
            final_edges.append(ed)
    final_edges += false_non_edges
    for n_ed in non_edges_sampled.tolist():
        if n_ed not in false_non_edges:
            final_non_edges.append(n_ed)
    final_non_edges += false_edges # final non_edge

    return final_edges, final_non_edges


if __name__ == "__main__":

    adj_train, adj_test, adj_validation = None, None, None

    adj_train_corrupted, train_false_edges, train_false_non_edges = corrupt_adj(adj_train, 'link', percent=2)
    adj_val_corrupted, val_false_edges, val_false_non_edges = corrupt_adj(adj_validation, 'link', percent=2)
    adj_test_corrupted, test_false_edges, test_false_non_edges  = corrupt_adj(adj_test, 'link', percent=2)

    mlp = StructMLP().to(torch.device('cuda'))
    mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
    mlp_model = 'best_mlp_model.model'



    epochs = 50
    validation_loss = 10000.0
    small_samples = 200
    for num_epoch in range(epochs):
        mlp_optimizer.zero_grad()
        numbers = list(np.random.randint(500, size=1))
        hidden_samples_train = [] # list of svd transform of adj
        for number in numbers :
            svd = TruncatedSVD(n_components=256, n_iter=10, random_state=number)
            u_train = svd.fit_transform(adj_train_corrupted) # matrix that lacks of edges and ....
            hidden_samples_train.append(torch.Tensor(u_train).to(device))
        # for i in range(1):
        #     hidden_samples_train[i] = torch.cat((hidden_samples_train[i].to(device), data.x[data.train_mask].to(device)),1)
        input_ = torch.stack(hidden_samples_train) # low rank representation of A
        input_ = input_.detach()
        edges, non_edges = sample_equal_number_edges_non_edges(adj_train_corrupted, false_non_edges=train_false_non_edges, false_edges=train_false_edges, small_samples=small_samples)
        samples = torch.cat((torch.Tensor(edges), torch.Tensor(non_edges)),dim=0).type(torch.long).to(device)
        target = torch.cat((torch.ones(len(edges)), torch.zeros(len(non_edges))),dim=0).type(torch.long).to(device)
        loss = mlp.compute_loss(input_, samples, target=target)
        print("Training Loss: ", loss.item())
        ### VALIDATION STEP ###
        with torch.no_grad():
            #Do Validation and check if validation loss has gone down
            numbers = list(np.random.randint(500, size=1))
            hidden_samples_validation = []
            for number in numbers :
                svd = TruncatedSVD(n_components=256, n_iter=10, random_state=number)
                u_validation = svd.fit_transform(adj_val_corrupted)
                hidden_samples_validation.append(torch.Tensor(u_validation).to(device))
            for i in range(1):
                hidden_samples_validation[i] = torch.cat((hidden_samples_validation[i].to(device), data.x[data.val_mask].to(device)),1)
            input_val = torch.stack(hidden_samples_validation)
            input_val = input_val.detach()
            edges, non_edges = sample_equal_number_edges_non_edges(adj_val_corrupted, false_non_edges=val_false_non_edges, false_edges=val_false_edges, small_samples=small_samples)
            samples = torch.cat((torch.Tensor(edges), torch.Tensor(non_edges)),dim=0).type(torch.long).to(device)
            target_val = torch.cat((torch.ones(len(edges)), torch.zeros(len(non_edges))),dim=0).type(torch.long).to(device)
            compute_val_loss = mlp.compute_loss(input_val, samples, target=target_val)
            if compute_val_loss < validation_loss:
                validation_loss = compute_val_loss
                print("Validation Loss: ", validation_loss)
                #Save Model
                torch.save(mlp.state_dict(), mlp_model)
        loss.backward()
        mlp_optimizer.step()

    mlp = StructMLP(node_set_size).to(torch.device("cuda"))
    mlp.load_state_dict(torch.load(mlp_model))

    numbers = list(np.random.randint(500, size=1))
    hidden_samples_test = []
    for number in numbers :
        svd = TruncatedSVD(n_components=256, n_iter=10, random_state=number)
        u_test = svd.fit_transform(adj_test_corrupted)
        hidden_samples_test.append(torch.Tensor(u_test).to(device))
    for i in range(1):
        hidden_samples_test[i] = torch.cat((hidden_samples_test[i].to(device), data.x[data.test_mask].to(device)),1)

    edges, non_edges = sample_equal_number_edges_non_edges(adj_test_corrupted, false_non_edges=test_false_non_edges, false_edges=test_false_edges, small_samples=200)
    samples = torch.cat((torch.Tensor(edges), torch.Tensor(non_edges)),dim=0).type(torch.long).to(device)
    target = torch.cat((torch.ones(len(edges)), torch.zeros(len(non_edges))),dim=0).type(torch.long).to(device)

    t_test = target.to("cpu").numpy()
    input_test = torch.stack(hidden_samples_test)
    input_test = input_test.detach()

    with torch.no_grad():
        test_pred = mlp.forward(input_test, samples)
        pred = F.log_softmax(test_pred, dim=1)
    pred = pred.detach().to("cpu").numpy()
    pred = np.argmax(pred, axis=1)


"""
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
    val_user_checkins = {}
    for user_id in range(1, n_users+1): 
        inds_checkins = np.argwhere(val_checkins[:,0] == user_id).flatten()
        checkins = val_checkins[inds_checkins]
        val_user_checkins[user_id] = checkins
    # everything here is from 1
    return train_checkins, val_checkins, n_users, n_nodes_total, train_user_checkins, val_user_checkins, friendship_old, friendship_new, selected_checkins, offset1, offset2, offset3, new_maps, maps, friendship_old_ori


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
    
    learn.apiFunction("temp/processed/", args.learning_rate, args.K_neg, args.win_size, args.num_epochs, args.workers, args.mobility_ratio)
    embs_file = "temp/processed/embs.txt"
    embs = read_embs(embs_file)
    embs_user = embs[:offset1]
    embs_time = embs[offset1:offset2]
    embs_venue = embs[offset2:offset3]
    embs_cate = embs[offset3:]


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