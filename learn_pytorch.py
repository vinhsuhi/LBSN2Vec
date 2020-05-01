from scipy.io import loadmat
from tqdm import tqdm
import numpy as np
import random
import pdb
import math
import networkx as nx 

import multiprocessing
import numpy as np
import multiprocessing
import networkx as nx
from gensim.models import Word2Vec

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
    def __init__(self, G, workers):
        self.G = G
        if hasattr(G, 'neibs'):
            self.neibs = G.neibs
        else:
            self.build_neibs_dict()


    def build_neibs_dict(self):
        self.neibs = {}
        for node in self.G.nodes():
            self.neibs[node] = list(self.G.neighbors(node))

    def simulate_walks(self, num_walks, walk_length, num_workers):
        pool = multiprocessing.Pool(processes=num_workers)
        walks = []
        print('Walk iteration:')
        nodes = list(self.G.nodes())
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

import torch 
import torch.nn as nn
import torch.nn.functional as F

def learn_user_user(embs, edges, neg_samples):
    # learn a batch of edges
    B, H = edges.shape
    _, D = embs.shape
    pos_embs = torch.stack([embs[edge] for edge in edges]) # shape BxHxD, H: length of hyperedges
    vbs = pos_embs.sum(dim=1).view(B, D, 1)  # shape BxDx1

    neg_embs = torch.stack([embs[neg_sample] for neg_sample in neg_samples]) # BxKxD
    pos_cosin = torch.bmm(pos_embs, vbs).view(B, H) 
    pos_cosin = pos_cosin.sum(dim=1)

    neg_cosin = torch.bmm(neg_embs, vbs).view(B, -1) 
    neg_cosin = neg_cosin.sum(dim=1) 

    loss = -(pos_cosin + neg_cosin).mean()
    return loss

def learn_checkins(embs, edges, neg_checkins_samples, num_neg=10):
    B, H = edges.shape
    _, D = embs.shape
    pos_embs = torch.stack([embs[edge] for edge in edges]) # shape BxHxD, H: length of hyperedges
    vbs = pos_embs.sum(dim=1).view(B, D, 1)  # shape BxDx1
    loss = 0
    for i in range(H):
        pos_embs = embs[edges[:, i]].view(B, 1, D)
        neg_samples = np.random.choice(neg_checkins_samples[i], B*num_neg, replace=True)
        neg_embs = embs[neg_samples].view(B, num_neg, D)
        pos_cosin = torch.bmm(pos_embs, vbs).view(B)
        neg_cosin = torch.bmm(neg_embs, vbs).view(B, num_neg).mean(dim=1)
        loss += (pos_cosin + neg_cosin).mean()
    loss = -loss/B
    return loss

def learn(walks, user_checkins, n_nodes_total, neg_user_samples, neg_checkins_samples):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim_size = 128
    n_epochs = 3
    win_size = 10
    num_neg = 10
    alpha = 0.2

    embs = nn.Embedding(n_nodes_total, dim_size).to(device)
    embs.weight.data = F.normalize(embs.weight, dim=1)
    optim = torch.optim.Adam(embs.parameters(), lr=0.001)

    for epoch in range(1, n_epochs + 1):
        for iter, sentence in tqdm(zip(np.arange(len(walks)), walks)):
            total_loss = 0
            for i, word in enumerate(sentence):
                embs.zero_grad()
                embs.weight.data = F.normalize(embs.weight, dim=1)
                # learn user-user
                targets = sentence[max(i-win_size, 0):i] + sentence[i+1:i+win_size] # not including center node
                edges = torch.LongTensor([[word, target] for target in targets]).to(device)
                neg_samples = np.random.choice(neg_user_samples, num_neg*edges.shape[0], replace=True)
                loss = learn_user_user(embs.weight, edges, neg_samples.reshape(len(edges), -1))
                # learn user-checkin
                # samples 2k user checkins
                checkins = user_checkins[word]
                if len(checkins) > 0:
                    sample_inds = np.random.choice(np.arange(len(checkins)), win_size*2, replace=True)
                    sampled_checkins = checkins[sample_inds]
                    loss += alpha*learn_checkins(embs.weight, sampled_checkins, neg_checkins_samples)

                loss.backward()
                optim.step()
                total_loss += loss.item()
            if iter % 100 == 0:
                print(f"Epoch {epoch} - Iter {iter}/{len(walks)}: loss: {total_loss:.3f}")


from scipy.io import loadmat
import pdb
import numpy as np
import random
from tqdm import tqdm
from scipy.sparse import csr_matrix
import networkx as nx

mat = loadmat('dataset/dataset_connected_NYC.mat')
# print(mat.keys())
selected_checkins = mat['selected_checkins'] - 1
friendship_old = mat["friendship_old"] - 1 # edge index from 0
friendship_new = mat["friendship_new"] - 1

offset1 = max(selected_checkins[:,0])
_, n = np.unique(selected_checkins[:,1], return_inverse=True) # 
selected_checkins[:,1] = n + offset1 + 1
offset2 = max(selected_checkins[:,1])
_, n = np.unique(selected_checkins[:,2], return_inverse=True)
selected_checkins[:,2] = n + offset2 + 1
offset3 = max(selected_checkins[:,2])
_, n = np.unique(selected_checkins[:,3], return_inverse=True)
selected_checkins[:,3] = n + offset3 + 1
n_nodes_total = np.max(selected_checkins) + 1

n_users = selected_checkins[:,0].max() + 1
n_times = selected_checkins[:,1].max() + 1
n_venues = selected_checkins[:,2].max() + 1
n_cates = selected_checkins[:,3].max() + 1

print(f"""Number of users: {n_users}
    Number of times: {n_times}
    Number of venues: {n_venues}
    Number of cates: {n_cates}
    Number of nodes total: {n_nodes_total}""")

print("Build user checkins dictionary...")
user_checkins = {}
for user_id in range(n_users): 
    inds_checkins = np.argwhere(selected_checkins[:,0] == user_id).flatten()
    checkins = selected_checkins[inds_checkins]
    user_checkins[user_id] = checkins

print("Performing random walks on hypergraph...")
num_walks = 20
walk_length = 80
workers = 2
# graph: undirected, edges = friendship_old
adj = csr_matrix((np.ones(len(friendship_old)), (friendship_old[:,0], friendship_old[:,1])), shape=(n_users, n_users), dtype=int)
adj = adj + adj.T
G = nx.from_scipy_sparse_matrix(adj)
walker = BasicWalker(G, workers=workers)
sentences = walker.simulate_walks(num_walks=num_walks, walk_length=walk_length, num_workers=workers)
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

learn(sentences, user_checkins, n_nodes_total=n_nodes_total, neg_user_samples=neg_user_samples, neg_checkins_samples=neg_checkins_samples)
