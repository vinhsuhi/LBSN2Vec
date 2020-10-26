import pdb
import numpy as np
import random
from tqdm import tqdm
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.io import loadmat
import random
import pdb
import math
import os
import multiprocessing
from evaluation import *
import argparse
import learn
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import save_info, sample_neg, read_embs, initialize_emb, random_walk
from link_pred_model import StructMLP
from sklearn.metrics import f1_score, accuracy_score


def parse_args():
    parser = argparse.ArgumentParser(description="LBSN configurations")
    parser.add_argument('--num_walks', type=int, default=10)
    parser.add_argument('--walk_length', type=int, default=80)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--mobility_ratio', type=float, default=0.2)
    parser.add_argument('--K_neg', type=int, default=10)
    parser.add_argument('--win_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001) # 0.001 for code c
    parser.add_argument('--dim_emb', type=int, default=128)
    # often change parameters
    parser.add_argument('--dataset_name', type=str, default='NYC')
    parser.add_argument('--input_type', type=str, default="persona_ori", help="persona_ori or persona_POI") 
    parser.add_argument('--bias_randomwalk', action='store_true')
    args = parser.parse_args()
    return args


def load_ego_ori_dict(path2):
    """
    load file containing information about persona --> ori maps
    """
    maps_PtOri = dict() 
    maps_OritP = dict() 
    max_node = 0 
    with open(path2, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.strip().split(',')
            persona_node = int(data_line[0]) + 1
            ori_node = int(data_line[1])
            if ori_node not in maps_OritP:
                maps_OritP[ori_node] = set([persona_node])
            else:
                maps_OritP[ori_node].add(persona_node)
            maps_PtOri[persona_node] = ori_node
            if persona_node > max_node:
                max_node = persona_node
    return maps_PtOri, maps_OritP, max_node


def create_pseudo_edges(maps_OritP, maps_PtOri, max_node):
    """
    create pseudo_edges
    parameters: 
        maps_OritP: maps from ori node to persona node
        maps_PtOri: maps from persona node to ori node
        max_node: current max index
    """
    additional_edges = []
    center_ori_dict = dict()
    for key, value in maps_OritP.items():
        max_node += 1
        maps_PtOri[max_node] = key
        center_ori_dict[max_node] = key
        maps_OritP[key].add(max_node)
        for ele in value:
            additional_edges.append([max_node, ele])
    return additional_edges, center_ori_dict, maps_OritP, maps_PtOri


def load_persona_graph(path1):
    """
    load persona graph
    parameters:
        path1: path to persona graph
    return edges
    """
    edges = []
    with open(path1, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.strip().split()
            edges.append([int(ele) + 1 for ele in data_line[:2]])
    file.close()
    return edges


def read_poi_map(path4):
    """

    """
    POI_dict = dict() # POI_index ---> POI_id
    if path4 is not None:
        with open(path4, 'r', encoding='utf-8') as file:
            for line in file:
                data_line = line.split()
                POI_dict[int(data_line[0])] = int(data_line[1])
    return POI_dict


def load_ego(path1, path2, path3=None, path4=None):
    """
    load ego graph
    parameters:
        path1: edgeslist: Friendgraph after splitting
        path2: ego_net: ego_node --> ori_node
        path3: edgelist_POI: ego_node --> POI node
        path4: location_dict: to_ori_location
    """
    maps_PtOri, maps_OritP, max_node = load_ego_ori_dict(path2)
    additional_edges, center_ori_maps, maps_OritP, maps_PtOri = create_pseudo_edges(maps_OritP, maps_PtOri, max_node)
    persona_edges = load_persona_graph(path1)
    print("Number of edges before: {}".format(len(persona_edges)))
    persona_edges += additional_edges
    print("Number of edges after: {}".format(len(persona_edges)))
    persona_edges = np.array(persona_edges)

    if path3 is not None:
        persona_POI = allocate_poi_to_user(path3)
    
    if path4 is not None:
        POI_maps = read_poi_map(path4)

    if path3 is not None:
        return persona_edges, maps_PtOri, persona_POI, POI_maps, maps_OritP, center_ori_maps
    return persona_edges, maps_PtOri, maps_OritP, center_ori_maps


def mat_to_numpy_array(matt):
    return np.array([[int(matt[i, 0]), int(matt[i, 1])] for i in range(len(matt))])


def create_persona_checkins(ori_checkins, maps_OritP):
    persona_checkins = []
    for i in range(len(ori_checkins)):
        checkins_i = ori_checkins[i]
        user = checkins_i[0]
        for persona_user in maps_OritP[user]:
            persona_checkins.append([persona_user, checkins_i[1], checkins_i[2], checkins_i[3]])
    persona_checkins = np.array(persona_checkins)
    return persona_checkins


def create_personaPOI_checkins(old_checkins, maps_OritP, persona_POI, POI_maps, center_ori_dict):
    """
    center_ori_dict: center_node --> ori_node (> 1)
    persona_POI: persona_node --> location_of_splitter (> 1)
    POI_maps: location_ori --> location_of_splitter
    maps_OritP: user_ori --> set_of_persona (not center)
    """
    ori_center_dict = {v:k for k,v in center_ori_dict.items()}
    personaPOI_checkins = []
    for i in tqdm(range(len(old_checkins))):
        old_checkini = old_checkins[i]
        user_ori = old_checkini[0]
        center_user = ori_center_dict[user_ori] # center user will have all checkins
        new_checkins.append([center_user, old_checkini[1], old_checkini[2], old_checkini[3]])
        location_ori = old_checkini[2]
        location_index = POI_maps[location]
        for persona_user in maps_OritP[user_ori]:
            if persona_user not in persona_POI:
                continue
            if location_index in persona_POI[persona_user]:
                new_checkins.append([persona_user, old_checkini[1], old_checkini[2], old_checkini[3]])
    new_checkins = np.array(new_checkins)
    return new_checkins


def renumber_checkins(checkins_matrix):
    offset1 = max(checkins_matrix[:,0])
    _, n = np.unique(checkins_matrix[:,1], return_inverse=True) # 
    checkins_matrix[:,1] = n + offset1 + 1
    offset2 = max(checkins_matrix[:,1])
    _, n = np.unique(checkins_matrix[:,2], return_inverse=True)
    checkins_matrix[:,2] = n + offset2 + 1
    offset3 = max(checkins_matrix[:,2])
    _, n = np.unique(checkins_matrix[:,3], return_inverse=True)
    checkins_matrix[:,3] = n + offset3 + 1
    n_nodes_total = np.max(checkins_matrix)
    n_users = checkins_matrix[:, 0].max()

    print(f"""Number of users: {n_users}
        Number of nodes total: {n_nodes_total}""")
    return checkins_matrix, offset1, offset2, offset3, n_nodes_total, n_users



def load_data(args):
    """
    this is for cleaned data

    There are two types of persona graph:
    1. persona_ori: original_persona
    2. persona_POI: persona with separated POIs

    use args.input_type to change between these types
    """
    mat = loadmat('dataset/cleaned_{}.mat'.format(args.dataset_name))
    friendship_old_ori = mat_to_numpy_array(mat['friendship_old'])
    friendship_new = mat_to_numpy_array(mat["friendship_new"]) 

    edgelist_path = 'Suhi_output/edgelist_{}'.format(args.dataset_name)
    persona_to_ori_path = 'Suhi_output/ego_net_{}'.format(args.dataset_name)
    edgelistPOI_path = 'Suhi_output/edgeslistPOI_{}'.format(args.dataset_name)
    location_map_path = 'Suhi_output/location_dict_{}'.format(args.dataset_name)

    if args.input_type == "persona_ori":
        friendship_old_persona, maps_PtOri, maps_OritP, center_ori_maps  = load_ego(edgelist_path, persona_to_ori_path)
        persona_checkins = create_persona_checkins(mat['selected_checkins'], maps_OritP)
    elif args.input_type == "persona_POI":
        persona_edges, maps_PtOri, persona_POI, POI_maps, maps_OritP, center_ori_maps = load_ego(edgelist_path, persona_to_ori_path, edgelistPOI_path, location_map_path)
        persona_checkins = create_personaPOI_checkins(mat['selected_checkins'], maps_OritP, persona_POI, POI_maps, center_ori_maps)

    persona_checkins, offset1, offset2, offset3, n_nodes_total, n_users = renumber_checkins(persona_checkins)
    
    ############## Train Test split for POI prediction ##################
    n_data = persona_checkins.shape[0]
    if args.mode == "friend":
        n_train = n_data
    else:
        n_train = int(n_data * 0.8)
    
    sorted_checkins = persona_checkins[np.argsort(persona_checkins[:,1])]
    train_checkins = sorted_checkins[:n_train]
    val_checkins = sorted_checkins[n_train:]
    #####################################################################
    
    print("Build user checkins dictionary...")
    train_user_checkins = {}
    user_location = dict()
    for user_id in range(1, n_users+1): 
        inds_checkins = np.argwhere(train_checkins[:,0] == user_id).flatten()
        checkins = train_checkins[inds_checkins]
        train_user_checkins[user_id] = checkins
        user_location[user_id] = set(np.unique(checkins[:, 2]).tolist())
    
    # val_user_checkins = {}
    # for user_id in range(1, n_users+1): 
    #     inds_checkins = np.argwhere(val_checkins[:,0] == user_id).flatten()
    #     checkins = val_checkins[inds_checkins]
    #     val_user_checkins[user_id] = checkins
    # everything here is from 1

    offsets = [offset1, offset2, offset3]
    checkins = [train_checkins, val_checkins, train_user_checkins, user_location]
    count_nodes = [n_users, n_nodes_total]
    friendships = [friendship_old_ori, friendship_old_persona, friendship_new]
    maps = [maps_PtOri, maps_OritP]

    return offsets, checkins, count_nodes, friendships, maps, train_user_checkins, persona_checkins, center_ori_maps

if __name__ == "__main__":
    args = parse_args()
    print(args)

    ######################################### load data ##########################################
    offsets, checkins, count_nodes, friendships, maps, train_user_checkins, persona_checkins, center_ori_maps = load_data(args)

    offset1, offset2, offset3 = offsets
    train_checkins, val_checkins, train_user_checkins, user_location = checkins
    n_users, n_nodes_total = count_nodes
    friendship_old_ori, friendship_old_persona, friendship_new = friendships
    maps_PtOri, maps_OritP = maps
    ###############################################################################################

    sentences = random_walk(friendship_old_persona, n_users, args, user_location, center_ori_maps)
    neg_user_samples, neg_checkins_samples = sample_neg(friendship_old_persona, persona_checkins)
    embs_ini = initialize_emb(args, n_nodes_total)
    save_info(args, sentences, embs_ini, neg_user_samples, neg_checkins_samples, train_user_checkins)

    learn.apiFunction("temp/processed/", args.learning_rate, args.K_neg, args.win_size, args.num_epochs, args.workers, args.mobility_ratio)
    embs_file = "temp/processed/embs.txt"
    embs = read_embs(embs_file)
    embs_user = embs[:offset1]

    friendship_linkprediction(embs_user, friendship_old_persona, friendship_new, k=10, new_maps=maps_OritP, maps=maps_PtOri, friendship_old_ori=friendship_old_ori)

"""
scripts:

for data in NYC hongzhi TKY
do 
    python -u CMan.py --input_type persona_ori --dataset_name ${data} 
done


"""



    """
        else:
            # import pdb; pdb.set_trace()
            val_checkins[:,0] -= 1
            val_checkins[:,1] -= (offset1+1)
            val_checkins[:,2] -= (offset2+1)
            location_prediction(val_checkins[:,:3], embs, embs_venue, k=10)
    """
    """
    ############################## Trash ###########################################
    else:
        mlp = StructMLP(args.dim_emb, 256)
        mlp = mlp.cuda()
        mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
        first_emb = None
        for i in tqdm(range(args.num_embs)):
            if not os.path.exists('embs_{}_{}.npy'.format(args.dataset_name, i)):
                learn.apiFunction("temp/processed/", args.learning_rate, args.K_neg, args.win_size, args.num_epochs, args.workers, args.mobility_ratio)
                embs_file = "temp/processed/embs.txt"
                embs = read_embs(embs_file)
                embs_user = embs[:offset1]
                embs_time = embs[offset1:offset2]
                embs_venue = embs[offset2:offset3]
                embs_cate = embs[offset3:]
                np.save('embs_{}_{}.npy'.format(args.dataset_name, i), embs_user)
            else:
                embs_user = np.load('embs_{}_{}.npy'.format(args.dataset_name, i))
            
            if i == 0:
                first_emb = embs_user
            else:
                embs_user = map_to_old_embs(first_emb, embs_user)
            # predict link here
            
            embs = torch.FloatTensor(embs_user)
            embs = embs.cuda()

            for ep in range(100):
                mlp_optimizer.zero_grad()
                edges, non_edges = sample_edges_non_edges(friendship_old-1, 2000, n_users)
                edges = torch.LongTensor(edges) 
                non_edges = torch.LongTensor(non_edges)
                edges = edges.cuda()
                non_edges = non_edges.cuda()
                samples = torch.cat((edges, non_edges), dim=0)
                labels = torch.cat((torch.ones(len(edges)), torch.zeros(len(non_edges))), dim = 0).long()
                samples = samples.cuda()
                labels = labels.cuda()
                loss = mlp.compute_loss(embs, samples, labels)
                loss.backward()
                if ep % 25 == 0:
                    print("Loss: {:.4f}".format(loss.item()))
                mlp_optimizer.step()

            eval_acc(mlp, embs, friendship_new - 1, friendship_old - 1, k=10, new_maps=new_maps, maps=maps, friendship_old_ori=friendship_old_ori)
    """

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



"""
######################################## Trash ########################################
def sample_edges_non_edges(edges, num_samples, n_nodes):
    num_edges = edges.shape[0]
    edges_sampled = edges[np.random.randint(0, num_edges, size=num_samples)]
    source = np.random.randint(0, n_nodes, num_samples)
    target = np.random.randint(0, n_nodes, num_samples)
    non_edges = np.array([source, target]).T
    return edges_sampled, non_edges

def eval_acc(mlp, embs, friendship_new, friendship_old, k=10, new_maps=None, maps=None, friendship_old_ori=None):
    friendship_new = torch.LongTensor(friendship_new)
    friendship_new = friendship_new.cuda()
    pred = mlp.forward(embs, friendship_new)
    pred = pred.detach().cpu().numpy()
    pred = np.argmax(pred, axis=1)
    t_test = np.ones(len(pred))
    print("Test Micro F1 Score: ", f1_score(t_test, pred, average='micro'))
    print("Test Weighted F1 Score: ", f1_score(t_test, pred, average='weighted'))
    print("Test Accuracy Score: ", accuracy_score(t_test, pred))
    return

    simi_matrix = np.zeros((embs.shape[0], embs.shape[0]))

    for i in tqdm(range(embs.shape[0])):
        for j in range(embs.shape[0]):
            input = torch.LongTensor([[i, j]]).cuda()
            pred = F.softmax(mlp.forward(embs, input))
            pred = pred.detach().cpu().numpy()
            simi_matrix[i,j] = pred[0, -1]
    
    friendship_linkprediction(embs, friendship_old, friendship_new, k=k, new_maps=new_maps, maps=maps, friendship_old_ori=friendship_old_ori, simi=simi_matrix)


class MappingModel(nn.Module):
    def __init__(self, dim):
        super(MappingModel, self).__init__()
        self.weight = nn.Linear(dim, dim)
    
    def forward(self, emb):
        return self.weight(emb)

    def loss(self, source, target):
        sub = (source - target) ** 2
        losses = torch.sum(sub, dim=1)
        loss = torch.mean(losses)
        return loss


def map_to_old_embs(first_emb, to_map):
    model = MappingModel(first_emb.shape[1])
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    source_emb = torch.FloatTensor(to_map).cuda()
    target_emb = torch.FloatTensor(first_emb).cuda()

    mapp_epochs = 100
    for epoch in range(mapp_epochs):
        optimizer.zero_grad()

        new_source = model(source_emb)
        loss = model.loss(new_source, target_emb)
        print("Mapping Loss: {:.4f}".format(loss.item()))
        loss.backward()
        optimizer.step()

    new_source = model(source_emb)
    return new_source.detach().cpu().numpy()
"""