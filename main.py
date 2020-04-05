from scipy.io import loadmat
import pdb
import numpy as np
import random

mat = loadmat('dataset_connected_NYC.mat')
print(mat.keys())
selected_checkins = mat['selected_checkins']
selected_users_IDs =  mat['selected_users_IDs']
friendship_old = mat["friendship_old"] - 1 # edge index from 0

# 1. rebuild node index

offset1 = max(selected_checkins[:,0])
_, n = np.unique(selected_checkins[:,1], return_inverse=True) # 
selected_checkins[:,1] = n + offset1 + 1
offset2 = max(selected_checkins[:,1])
_, n = np.unique(selected_checkins[:,2], return_inverse=True)
selected_checkins[:,2] = n + offset2 + 1
offset3 = max(selected_checkins[:,2])
_, n = np.unique(selected_checkins[:,3], return_inverse=True)
selected_checkins[:,3] = n + offset3 + 1
num_node_total = np.max(selected_checkins)

# 2. prepare checkins per user (fast)
user_checkins = np.zeros((len(selected_users_IDs)), dtype=np.object)
temp_checkins = selected_checkins[selected_checkins[:,0].argsort(),:]  # a[a[:,0].argsort(),], different with matlab due to multiple values are equals
 	
u, m, n, counters = np.unique(temp_checkins[:,0], return_index=True, return_inverse=True, return_counts=True)

# 
# counters = m - np.array([0] + m[:-1].tolist(), dtype=np.int32)
user_checkins[u-1] = np.split(temp_checkins, np.cumsum(counters))[:-1]
# OK here  
def transpose(x):
    if type(x) != int:
        return x.T
    else:
        return x

user_checkins = np.array([transpose(x) for x in user_checkins])

user_checkins_counter = np.zeros((len(selected_users_IDs)), dtype=np.int64)
user_checkins_counter[u-1] = counters.astype(np.int64)
# 3. random walk
num_node = len(selected_users_IDs)

from scipy.sparse import csr_matrix
network = csr_matrix((np.ones(len(friendship_old)), (friendship_old[:,0], friendship_old[:,1])), shape=(num_node, num_node), dtype=int)
network = network + network.T

node_list = np.zeros((num_node), dtype=np.object)
node_list_len = np.zeros((num_node), dtype=np.int32)
num_walk = 2
len_walk = 80

indx, indy = network.nonzero()
temp, m, n, counters = np.unique(indx, return_index=True, return_inverse=True, return_counts=True)
node_list_len[temp] = counters
node_list[temp] = np.split(indy, np.cumsum(node_list_len[temp]))[:-1]

walks = np.zeros((num_walk*num_node, len_walk), dtype=np.int64)
for ww in tqdm(range(num_walk)):
    for ii in range(num_node):
        seq = np.zeros((len_walk), dtype=np.int)
        seq[0] = ii
        current_e = ii
        for jj in range(len_walk-1):
            rand_ind = random.randint(0, node_list_len[seq[jj]]-1)
            node_list[seq[jj]]
            seq[jj+1] = node_list[seq[jj]][rand_ind]
        walks[ii + ww*num_node, :] = seq

# Tested here pdb.set_trace()


_, r =  network.nonzero()
# tab_degree = 
# 4. prepare negative sample table in advance (fast)
unique, counts = np.unique(r, return_counts=True)
freq = (counts/len(r)) ** 0.75
neg_sam_table_social = np.repeat(unique, np.round(100000*freq/sum(freq))).astype(np.int64)

# checkins
neg_sam_table_mobility_norm = np.zeros((4,1))
for ii in range(len(neg_sam_table_mobility_norm)):
    unique, counts = np.unique(temp_checkins[:,ii], return_counts=True)
    freq = (counts/len(temp_checkins)) ** 0.75
    neg_sam_table_mobility_norm[ii] = np.repeat(unique, np.round(100000*freq/sum(freq))).astype(np.int64)

# LBSN2vec
dim_emb = 128
num_epoch = 1
num_threads =  1
K_neg = 10
win_size = 10
learning_rate = 0.001

embs_ini = (np.random.rand(size=(num_node_total, dim_emb)) -0.5)/dim_emb
embs_len = np.sqrt(np.sum(embs_ini**2, axis=1))
embs_ini = embs_ini / (np.tile(embs_len, (1, dim_emb)))

mobility_ratio = 0.2

# tic;
# [embs] = learn_LBSN2Vec_embedding(walks',user_checkins, user_checkins_counter,...
#     embs_ini', learning_rate, K_neg,...
#     neg_sam_table_social, win_size, neg_sam_table_mobility_norm,...
#     num_epoch,num_threads,mobility_ratio);
# toc;
