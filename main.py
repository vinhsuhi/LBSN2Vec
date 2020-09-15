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

def train_persona(embedding_model, optimizer, maps, new_maps, this_sentences, j, min_user, max_user, num_neg, args):
    print("x"*100)
    words = this_sentences[:, j]
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
    print("loss3: {:.4f}".format(loss_persona))
    optimizer.step()



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




def deepwalk_walk_wrapper(class_instance, walk_length, start_node):
    class_instance.deepwalk_walk(walk_length, start_node)

def deepwalk_walk(params):
    '''
    Simulate a random walk starting from start node.
    '''
    walk_length = params["walk_length"]
    neibs = params["neibs"]
    nodes = params["nodes"]
    # if args["iter"] % 5 == 0:
    print("Iter:", params["iter"]) # keep printing, avoid moving process to swap

    walks = []
    for node in nodes:
        walk = [node]
        if len(neibs[node]) == 0:
            walks.append(walk)
            continue
        while len(walk) < walk_length:
            cur = int(walk[-1])
            cur_nbrs = neibs[cur]
            if len(cur_nbrs) == 0: break
            walk.append(np.random.choice(cur_nbrs))
        walks.append(walk)
    return walks


class BasicWalker:
    def __init__(self, G, start_nodes=None):
        self.G = G
        if hasattr(G, 'neibs'):
            self.neibs = G.neibs
        else:
            self.build_neibs_dict()
        if start_nodes is not None:
            self.start_nodes = start_nodes
        else:
            self.start_nodes = list(self.G.nodes())


    def build_neibs_dict(self):
        self.neibs = {}
        for node in self.G.nodes():
            self.neibs[node] = list(self.G.neighbors(node))

    def simulate_walks(self, num_walks, walk_length, num_workers):
        pool = multiprocessing.Pool(processes=num_workers)
        walks = []
        print('Walk iteration:')
        nodes = self.start_nodes
        nodess = [np.random.shuffle(nodes)]
        for i in range(num_walks):
            _ns = nodes.copy()
            np.random.shuffle(_ns)
            nodess.append(_ns)
        params = list(map(lambda x: {'walk_length': walk_length, 'neibs': self.neibs, 'iter': x, 'nodes': nodess[x]},
            list(range(1, num_walks+1))))
        walks = pool.map(deepwalk_walk, params)
        pool.close()
        pool.join()
        # walks = np.vstack(walks)
        while len(walks) > 1:
            walks[-2] = walks[-2] + walks[-1]
            walks = walks[:-1]
        walks = walks[0]
        return walks


def read_embs(embs_file):
    embs = []
    with open(embs_file, "r") as fp:
        for line in fp.readlines()[1:]:
            embs.append([float(x) for x in line.strip().split()])
    embs = np.array(embs)
    return embs

def load_ego(path1, path2, path3=None, path4=None):
    edges = []
    with open(path1, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.strip().split()
            edges.append([int(ele) + 1 for ele in data_line[:2]])
    edges = np.array(edges)

    maps = dict()
    with open(path2, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.strip().split(',')
            maps[int(data_line[0]) + 1] = int(data_line[1])

    user_POI = [] # persona user to POI of input of persona
    if path3 is not None:
        with open(path3, 'r', encoding='utf-8') as file:
            for line in file:
                data_line = line.strip().split()
                user_POI.append([int(data_line[0]) + 1, int(data_line[1])])
    user_POI = np.array(user_POI)

    POI_dict = dict() # POI of input of persona to original POI 
    if path4 is not None:
        with open(path4, 'r', encoding='utf-8') as file:
            for line in file:
                data_line = line.split()
                POI_dict[int(data_line[0])] = int(data_line[1])
    if path3 is not None:
        return edges, maps, user_POI, POI_dict
    else:
        return edges, maps


def load_data(args):
    maps = None
    new_maps = None
    if args.input_type == "mat":
        if args.clean:
            mat = loadmat('dataset/cleaned_{}.mat'.format(args.dataset_name))
        else:
            mat = loadmat('dataset/dataset_connected_{}.mat'.format(args.dataset_name))
        selected_checkins = mat['selected_checkins'] 
        friendship_old = mat["friendship_old"] # edge index from 0
        friendship_new = mat["friendship_new"] 
    elif args.input_type == "persona":
        if args.clean:
            mat = loadmat('dataset/cleaned_{}.mat'.format(args.dataset_name))
        else:
            mat = loadmat('dataset/dataset_connected_{}.mat'.format(args.dataset_name))
        edges, maps = load_ego('Suhi_output/edgelist_{}'.format(args.dataset_name), 'Suhi_output/ego_net_{}'.format(args.dataset_name))
        friendship_old = edges 
        friendship_n = mat["friendship_new"] 
        new_maps = dict()
        for key, value in maps.items():
            if value not in new_maps:
                new_maps[value] = set([key])
            else:
                new_maps[value].add(key)
        
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
        friendship_new = friendship_n

    elif args.input_type == "persona2":
        if args.clean:
            mat = loadmat('dataset/cleaned_{}.mat'.format(args.dataset_name))
        else:
            mat = loadmat('dataset/dataset_connected_{}.mat'.format(args.dataset_name))
        edges, maps, persona_POI, POI_dict = load_ego('Suhi_output/edgelist_{}'.format(args.dataset_name), \
            'Suhi_output/ego_net_{}.txt'.format(args.dataset_name), \
                'Suhi_output/edgelistPOI_{}'.format(args.dataset_name), 'Suhi_output/location_dict_{}'.format(args.dataset_name))

        friendship_old = edges 
        friendship_n = mat["friendship_new"] 
        new_maps = dict()
        for key, value in maps.items():
            if value not in new_maps:
                new_maps[value] = set([key])
            else:
                new_maps[value].add(key)

        def create_new_checkins2(old_checkins, new_maps, persona_POI, POI_dict):
            new_checkins = []
            for i in range(len(old_checkins)):
                old_checkini = old_checkins[i]
                user = old_checkini[0]
                location = old_checkins[2]
                location_image = POI_dict[location]
                for ele in new_maps[user]:
                    if location_image in persona_POI[ele]:
                        new_checkins.append([ele, old_checkini[1], old_checkini[2], old_checkini[3]])
            new_checkins = np.array(new_checkins)
            return new_checkins

        selected_checkins = create_new_checkins2(mat['selected_checkins'], new_maps, persona_POI, POI_dict)
        friendship_new = friendship_n
        

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

    return train_checkins, val_checkins, n_users, n_nodes_total, train_user_checkins, val_user_checkins, friendship_old, friendship_new, selected_checkins, offset1, offset2, offset3, new_maps, maps


def random_walk(friendship_old, n_users, args):
    print("Performing random walks on hypergraph...")
    # graph: undirected, edges = friendship_old
    adj = csr_matrix((np.ones(len(friendship_old)), (friendship_old[:,0]-1, friendship_old[:,1]-1)), shape=(n_users, n_users), dtype=int)
    adj = adj + adj.T
    G = nx.from_scipy_sparse_matrix(adj)
    walker = BasicWalker(G)
    sentences = walker.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length, num_workers=args.workers)
    for i in range(len(sentences)):
        sentences[i] = [x+1 for x in sentences[i]]
    # sentences: args.walk_length of each walk may be different
    return sentences

def sample_neg(friendship_old, selected_checkins):
    # for negative sampling
    user_ids, counts = np.unique(friendship_old.flatten(), return_counts=True)
    freq = (100*counts/counts.sum()) ** 0.75
    neg_user_samples = np.repeat(user_ids, np.round(1000000 * freq/sum(freq)).astype(np.int64)).astype(np.int64)
    neg_checkins_samples = {}
    for i in range(selected_checkins.shape[1]):
        values, counts = np.unique(selected_checkins[:,i], return_counts=True)
        freq = (100*counts/counts.sum()) ** 0.75
        neg_checkins_samples[i] = np.repeat(values, np.round(1000000 * freq/sum(freq)).astype(np.int64))
    return neg_user_samples, neg_checkins_samples


def initialize_emb(args, n_nodes_total):
    # TODO: initialize here!
    embs_ini = (np.random.uniform(size=(n_nodes_total, args.dim_emb)) -0.5)/args.dim_emb
    embs_len = np.sqrt(np.sum(embs_ini**2, axis=1)).reshape(-1,1)
    embs_ini = embs_ini / embs_len
    return embs_ini


def save_info(args, sentences, embs_ini, neg_user_samples, neg_checkins_samples):
    input_dir = "temp/processed/"
    if not os.path.isdir(input_dir):
        os.makedirs(input_dir)
    print("Write walks")
    with open(f"{input_dir}/walk.txt", "w+") as fp:
        fp.write(f"{len(sentences)} {args.walk_length}\n")
        for sent in sentences:
            fp.write(" ".join(map(str, sent)) + "\n")

    print("Write user_checkins")
    with open(f"{input_dir}/user_checkins.txt", "w+") as fp:
        fp.write(f"{len(train_user_checkins)}\n") # num users
        for id in sorted(train_user_checkins.keys()):
            checkins = train_user_checkins[id]
            fp.write(f"{checkins.shape[0]}\n")
            for checkin in checkins:
                fp.write(" ".join(map(str, checkin)) + "\n")

    print("Write embs_ini")
    with open(f"{input_dir}/embs_ini.txt", "w+") as fp:
        fp.write(f"{embs_ini.shape[0]} {embs_ini.shape[1]}\n") # num users
        for emb in embs_ini:
            fp.write(" ".join([f"{x:.5f}" for x in emb]) + "\n")

    print("Write neg_user_samples")
    with open(f"{input_dir}/neg_user_samples.txt", "w+") as fp:
        fp.write(f"{neg_user_samples.shape[0]}\n") # num users
        for neg in neg_user_samples:
            fp.write(f"{neg}\n")

    print("Write neg_checkins_samples")
    with open(f"{input_dir}/neg_checkins_samples.txt", "w+") as fp:
        keys = sorted(neg_checkins_samples.keys())
        for key in keys:
            neg_table = neg_checkins_samples[key]
            fp.write(f"{neg_table.shape[0]}\n")
            fp.write("\n".join(map(str, neg_table)) + "\n")


if __name__ == "__main__":
    args = parse_args()
    train_checkins, val_checkins, n_users, n_nodes_total, train_user_checkins, val_user_checkins, friendship_old, friendship_new, selected_checkins, offset1, offset2, offset3, new_maps, maps = load_data(args)
    if not args.load:
        sentences = random_walk(friendship_old, n_users, args)
        if not args.py:
            neg_user_samples, neg_checkins_samples = sample_neg(friendship_old, selected_checkins)
            embs_ini = initialize_emb(args, n_nodes_total)
            save_info(args, sentences, embs_ini, neg_user_samples, neg_checkins_samples)
            
            learn.apiFunction("temp/processed/", args.learning_rate, args.K_neg, args.win_size, args.num_epochs, args.workers, args.mobility_ratio)
            embs_file = "temp/processed/embs.txt"
            embs = read_embs(embs_file)
        else:
            train_checkins -= 1
            val_checkins -= 1
            train_user_checkins = {key - 1: value - 1 for key, value in train_user_checkins.items()}
            val_user_checkins = {key - 1: value - 1 for key, value in val_user_checkins.items()}
            friendship_new -= 1
            friendship_old -= 1
            maps_suhi = dict()
            new_maps_suhi = dict()
            if maps is not None:
                maps_suhi = dict()
                new_maps_suhi = dict()
                for key, value in maps.items():
                    maps_suhi[key -1] = value - 1
                for key, value in new_maps.items():
                    new_maps_suhi[key - 1] = [ele - 1 for ele in value]
            for i in range(len(sentences)):
                sentences[i] = [x-1 for x in sentences[i]]
            embs = learn_emb(sentences, n_nodes_total, args.dim_emb, args.num_epochs, args.win_size, \
                train_checkins, train_user_checkins, alpha=args.mobility_ratio, num_neg = args.K_neg, args=args, maps=maps_suhi, new_maps = new_maps_suhi)
    else:
        embs_file = "temp/processed/embs.txt"
        embs = read_embs(embs_file)


    # evaluate
    embs_user = embs[:offset1]
    embs_time = embs[offset1:offset2]
    embs_venue = embs[offset2:offset3]
    embs_cate = embs[offset3:]

    if args.mode == 'friend':
        if not args.py:
            friendship_linkprediction(embs_user, friendship_old-1, friendship_new-1, k=10, new_maps=new_maps, maps=maps)
        else:
            friendship_linkprediction(embs_user, friendship_old, friendship_new, k=10, new_maps=new_maps, maps=maps)

    else:
        # import pdb; pdb.set_trace()
        val_checkins[:,0] -= 1
        val_checkins[:,1] -= (offset1+1)
        val_checkins[:,2] -= (offset2+1)
        location_prediction(val_checkins[:,:3], embs, embs_venue, k=10)
"""
import numpy as np
import networkx as nx
from tqdm import tqdm
import pdb
import os
from evaluation import *
import argparse
from embedding_model import EmbModel
import torch
import time
from utils import save_info, load_data, sample_neg, read_embs, initialize_emb, random_walk


def parse_args():
    parser = argparse.ArgumentParser(description="LBSN configurations")
    parser.add_argument('--num_walks', type=int, default=10)
    parser.add_argument('--walk_length', type=int, default=80)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--mobility_ratio', type=float, default=0.2)
    parser.add_argument('--K_neg', type=int, default=10)
    parser.add_argument('--win_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001) # ten times for python would return better result
    parser.add_argument('--dim_emb', type=int, default=128)
    parser.add_argument('--mode', type=str, default='friend', help="friend or POI")
    parser.add_argument('--input_type', type=str, default="mat", help="mat or persona") 
    parser.add_argument('--load', action='store_true') 
    parser.add_argument('--cuda', action='store_true') 
    parser.add_argument('--py', action='store_true') 
    parser.add_argument('--dataset_name', type=str, default='NYC')
    parser.add_argument('--clean', action='store_true', help='use cleaned dataset')
    parser.add_argument('--batchsize', type=int, default=512)
    args = parser.parse_args()
    return args


def train_social(embedding_model, optimizer, win_size, alpha, this_sentences, \
                    j, loss1s, min_user, max_user, num_neg, args):
    edges = []
    for k in range(1, win_size + 1):
        if np.random.rand() > alpha:
            if j >= k:
                this_edges1 = this_sentences[:, [j, j - k]]
                edges.append(this_edges1)
            if j + k < this_sentences.shape[1]:
                this_edges2 = this_sentences[:, [j, j + k]]
                edges.append(this_edges2)
    if len(edges) > 0:
        edges = np.concatenate(edges, axis=0)
        edges = torch.LongTensor(edges)
        if args.cuda:
            edges = edges.cuda()
        neg = np.random.randint(min_user, max_user, num_neg)
        neg = torch.LongTensor(neg)
        if args.cuda:
            neg = neg.cuda()
        optimizer.zero_grad()
        loss1 = embedding_model.edge_loss(edges, neg)
        loss1s.append(loss1.item())
        loss1.backward()
        optimizer.step()


def train_poi(user_checkins_dict, this_sentences, embedding_model, loss2s, alpha, optimizer, j, win_size, BATCH_SIZE, args):
    words = this_sentences[:, j]
    this_user_checkins = []
    for w in words:
        try:
            this_checkins = user_checkins_dict[w]
            this_user_checkins.append(this_checkins)
        except:
            continue
    this_user_checkins = np.concatenate(this_user_checkins, axis=0)
    num_checkins_to_sample = int(alpha * len(this_user_checkins))
    if num_checkins_to_sample > 0:
        checkin_indices = np.random.randint(0, len(this_user_checkins), min(num_checkins_to_sample, 2*win_size*BATCH_SIZE))
        checkins = this_user_checkins[checkin_indices]
        checkins = torch.LongTensor(checkins)
        if args.cuda:
            checkins = checkins.cuda()
        neg_ind = np.random.randint(0, len(selected_checkins), 10)
        neg = selected_checkins[neg_ind]
        neg = torch.LongTensor(neg)
        if args.cuda:
            neg = neg.cuda()
        optimizer.zero_grad()
        loss2 = embedding_model.hyperedge_loss(checkins, neg)
        loss2s.append(loss2.item())
        loss2.backward()
        optimizer.step()


def train_persona(embedding_model, optimizer, maps, new_maps, this_sentences, j, min_user, max_user, num_neg, args):
    print("x"*100)
    words = this_sentences[:, j]
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
    print("loss3: {:.4f}".format(loss_persona))
    optimizer.step()


def learn_emb(args, sentences, n_nodes, emb_dim, n_epochs, win_size, \
        selected_checkins, user_checkins_dict, alpha=0.2, num_neg=10, maps=None, new_maps=None):
    
    import pdb 
    pdb.set_trace()

    print("Sentences: Shape: {}, Max: {}, Min: {}".format(sentences.shape, sentences.max(), sentences.min()))
    print("Number of nodes: {}".format(n_nodes))
    print("Embedding dim: {}".format(emb_dim))
    print("Num epochs: {}".format(num))

    if maps is not None:
        mymaps = {k - 1: v - 1 for k, v in maps.items()}
        mynew_maps = {k - 1: [ele - 1 for ele in v] for k, v in new_maps.items()}
    min_user = np.min(selected_checkins[:,0])
    max_user = np.max(selected_checkins[:,0])
    sentences = np.array(sentences)
    embedding_model = EmbModel(n_nodes, emb_dim)
    if args.cuda:
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
            for j in range(sentence_length):
                train_social(embedding_model, optimizer, win_size, \
                    alpha, this_sentences, j, loss1s, min_user, max_user, num_neg, args)
                train_poi(user_checkins_dict, this_sentences, embedding_model, loss2s, alpha, optimizer, j, win_size, BATCH_SIZE, args)
                if maps is not None: 
                    train_persona(embedding_model, optimizer, mymaps, mynew_maps, this_sentences, j, min_user, max_user, num_neg, args)
            print("loss1: {:.4f}".format(np.mean(loss1s)))
            print("loss2: {:.4f}".format(np.mean(loss2s)))
            print("-"*100)
    if args.cuda:
        embeddings = embedding_model.node_embedding(torch.LongTensor(np.arange(n_nodes)).cuda())
    else:
        embeddings = embedding_model.node_embedding(torch.LongTensor(np.arange(n_nodes)))
    embeddings = embeddings.detach().cpu().numpy()
    return embeddings


if __name__ == "__main__":
    args = parse_args()
    train_checkins, val_checkins, n_users, n_nodes_total, train_user_checkins, val_user_checkins, friendship_old, friendship_new, selected_checkins, offset1, offset2, offset3, new_maps, maps = load_data(args)
    if not args.load:
        sentences = random_walk(friendship_old, n_users, args)
        if not args.py:
            import learn
            neg_user_samples, neg_checkins_samples = sample_neg(friendship_old, selected_checkins)
            embs_ini = initialize_emb(args, n_nodes_total)
            save_info(args, sentences, embs_ini, neg_user_samples, neg_checkins_samples, train_user_checkins)
            
            learn.apiFunction("temp/processed/", args.learning_rate, args.K_neg, args.win_size, args.num_epochs, args.workers, args.mobility_ratio)
            embs_file = "temp/processed/embs.txt"
            embs = read_embs(embs_file)
        else:
            train_checkins -= 1
            val_checkins -= 1
            train_user_checkins = {key - 1: value - 1 for key, value in train_user_checkins.items()}
            val_user_checkins = {key - 1: value - 1 for key, value in val_user_checkins.items()}
            friendship_new -= 1
            friendship_old -= 1
            for i in range(len(sentences)):
                sentences[i] = [x-1 for x in sentences[i]]
            sentences = np.array(sentences)
            embs = learn_emb(args, sentences, n_nodes_total, args.dim_emb, args.num_epochs, args.win_size, \
                train_checkins, train_user_checkins, alpha=args.mobility_ratio, num_neg = args.K_neg, maps=maps, new_maps=new_maps)
    else:
        embs_file = "temp/processed/embs.txt"
        embs = read_embs(embs_file)

    # evaluate
    embs_user = embs[:offset1]
    embs_time = embs[offset1:offset2]
    embs_venue = embs[offset2:offset3]
    embs_cate = embs[offset3:]

    if args.mode == 'friend':
        if not args.py:
            friendship_linkprediction(embs_user, friendship_old-1, friendship_new-1, k=10, new_maps=new_maps, maps=maps)
        else:
            friendship_linkprediction(embs_user, friendship_old, friendship_new, k=10, new_maps=new_maps, maps=maps)
    else:
        val_checkins[:,0] -= 1
        val_checkins[:,1] -= (offset1+1)
        val_checkins[:,2] -= (offset2+1)
        location_prediction(val_checkins[:,:3], embs, embs_venue, k=10)
"""
