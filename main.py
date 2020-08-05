from scipy.io import loadmat
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

def deepwalk_walk_wrapper(class_instance, walk_length, start_node):
    class_instance.deepwalk_walk(walk_length, start_node)

def deepwalk_walk(args):
    '''
    Simulate a random walk starting from start node.
    '''
    walk_length = args["walk_length"]
    neibs = args["neibs"]
    nodes = args["nodes"]
    # if args["iter"] % 5 == 0:
    print("Iter:", args["iter"]) # keep printing, avoid moving process to swap

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
    def __init__(self, G, workers, start_nodes=None):
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

num_walks = 10
walk_length = 80
workers = 2 
num_epoch = 1
mobility_ratio = 0.2
K_neg = 10
win_size = 10
learning_rate = 0.001
dim_emb = 128

mat = loadmat('dataset/dataset_connected_NYC.mat')
# print(mat.keys())
selected_checkins = mat['selected_checkins'] 
friendship_old = mat["friendship_old"] # edge index from 0
friendship_new = mat["friendship_new"] 

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

n_users = selected_checkins[:,0].max()
print(f"""Number of users: {n_users}
    Number of nodes total: {n_nodes_total}""")


n_data = selected_checkins.shape[0]
# n_train = int(n_data * 0.8) # for location prediction
n_train = n_data # for friendship prediction
sorted_checkins = selected_checkins[np.argsort(selected_checkins[:,1])]
train_checkins = sorted_checkins[:n_train]
val_checkins = sorted_checkins[n_train:]

print("1")
import pdb
pdb.set_trace()

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

print("2")
import pdb
pdb.set_trace()

print("Performing random walks on hypergraph...")
# graph: undirected, edges = friendship_old
adj = csr_matrix((np.ones(len(friendship_old)), (friendship_old[:,0]-1, friendship_old[:,1]-1)), shape=(n_users, n_users), dtype=int)
adj = adj + adj.T
G = nx.from_scipy_sparse_matrix(adj)
walker = BasicWalker(G, workers=workers)
sentences = walker.simulate_walks(num_walks=num_walks, walk_length=walk_length, num_workers=workers)
for i in range(len(sentences)):
    sentences[i] = [x+1 for x in sentences[i]]
# sentences: walk_length of each walk may be different

# for negative sampling
user_ids, counts = np.unique(friendship_old.flatten(), return_counts=True)
freq = (100*counts/counts.sum()) ** 0.75
neg_user_samples = np.repeat(user_ids, np.round(1000000 * freq/sum(freq)).astype(np.int64)).astype(np.int64)
neg_checkins_samples = {}
for i in range(selected_checkins.shape[1]):
    values, counts = np.unique(selected_checkins[:,i], return_counts=True)
    freq = (100*counts/counts.sum()) ** 0.75
    neg_checkins_samples[i] = np.repeat(values, np.round(1000000 * freq/sum(freq)).astype(np.int64))


embs_ini = (np.random.uniform(size=(n_nodes_total, dim_emb)) -0.5)/dim_emb
embs_len = np.sqrt(np.sum(embs_ini**2, axis=1)).reshape(-1,1)
embs_ini = embs_ini / embs_len

input_dir = "temp/processed/"
if not os.path.isdir(input_dir):
    os.makedirs(input_dir)
print("Write walks")
with open(f"{input_dir}/walk.txt", "w+") as fp:
    fp.write(f"{len(sentences)} {walk_length}\n")
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

import learn

learn.apiFunction("temp/processed", learning_rate, K_neg, win_size, num_epoch, workers,
    mobility_ratio)
embs_file = "temp/processed/embs.txt"
embs = read_embs(embs_file)


# evaluate
embs_user = embs[:offset1]
embs_time = embs[offset1:offset2]
embs_venue = embs[offset2:offset3]
embs_cate = embs[offset3:]

print("3")
import pdb
pdb.set_trace()

# import pdb; pdb.set_trace()
val_checkins[:,0] -= 1
val_checkins[:,1] -= (offset1+1)
val_checkins[:,2] -= (offset2+1)

print("4")
import pdb
pdb.set_trace()

location_prediction(val_checkins[:,:3], embs, embs_venue, k=10)
friendship_linkprediction(embs_user, friendship_old-1, friendship_new-1, k=10)
