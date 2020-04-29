from scipy.io import loadmat
from utils import random_walk, create_social_graph, get_neg_sample, load_data
from tqdm import tqdm
import numpy as np
import random
import pdb
import math

# need initializing
dim_emb = 0
emb_n = []
neg_sam_table_social = None
alpha = None
neg_sam_table_mobility1 = None
neg_sam_table_mobility2 = None
neg_sam_table_mobility3 = None
neg_sam_table_mobility4 = None
table_size_mobility1 = None
table_size_mobility2 = None
table_size_mobility3 = None
table_size_mobility4 = None

def fixed_unigram_candidate_sampler(num_sampled, unique, range_max, distortion, unigrams):
    weights = unigrams**distortion
    prob = weights/weights.sum()
    sampled = np.random.choice(range_max, num_sampled, p=prob, replace=~unique)
    return sampled

def get_a_neg_sample_Kless1():
    if np.random.randint(1024) <= num_neg:
        return 1
    else:
        return 0
 
def get_a_social_decision():
    if np.random.uniform() <= mobility_ratio:
        return 0
    else:
        return 1

def get_a_mobility_decision():
    if np.random.uniform() <= mobility_ratio:
        return 1
    else:
        return 0
        
def learn_a_pair_loc_loc_cosine(flag, loc1, loc2):
    norm1 = np.linalg.norm(emb_n[loc1])
    norm2 = np.linalg.norm(emb_n[loc2])
    f = np.sum(emb_n[loc1]*emb_n[loc2])
    c1 = 1/(norm1*norm2)*alpha
    c2 = f/(norm1**3 * norm2) * alpha
    c3 = f/(norm1*norm2**3) * alpha

    if flag == 1:
        emb_n[loc2] += c1*emb_n[loc1] - c3*emb_n[loc2]
        emb_n[loc1] += c1*emb_n[loc2] - c2*emb_n[loc1]
    else:
        emb_n[loc2] -= c1*emb_n[loc1] - c3*emb_n[loc2]
        emb_n[loc1] -= c1*emb_n[loc2] - c2*emb_n[loc1]

def learn_a_pair_loc_pr_cosine(flag, word, best_fit):
    f = np.sum(emb_n[word] * best_fit)
    norm1 = np.linalg.norm(emb_n[word])
    g = f/norm1
    c1 = 1/(norm1)*alpha
    c2 = f/(norm1*norm1*norm1)*alpha
    if flag == 1:
        emb_n[word] += c1*best_fit - c2*emb_n[word]
    else:
        emb_n[word] -= c1*best_fit - c2*emb_n[word]

def learn_an_edge(word, target_e):
    learn_a_pair_loc_loc_cosine(1, word, target_e)
    if num_neg < 1:
        if get_a_neg_sample_Kless1() == 1:
            target_n = random.choice(neg_sam_table_social)
            if target_n != target_e and target_n != word:
                learn_a_pair_loc_loc_cosine(0, word, target_n)
    else:
        for i in range(num_neg):
            target_n = random.choice(neg_sam_table_social)
            if target_n != target_e and target_n != word:
                learn_a_pair_loc_loc_cosine(0, word, target_n)


def learn_an_edge_with_BFT(word, target_e, best_fit):
    best_fit = emb_n[word] + emb_n[target_e]
    best_fit /= np.linalg.norm(best_fit)
    
    learn_a_pair_loc_pr_cosine(1, word, best_fit)
    learn_a_pair_loc_pr_cosine(1, target_e, best_fit)

    if num_neg < 1:
        if get_a_neg_sample_Kless1() == 1:
            target_n = random.choice(neg_sam_table_social)
            if target_n != target_e and target_n != word:
                learn_a_pair_loc_pr_cosine(0, target_n, best_fit)
    else:
        for n in range(num_neg):
            target_n = random.choice(neg_sam_table_social)
            if target_n != target_e and target_n != word:
                learn_a_pair_loc_pr_cosine(0, target_n, best_fit)

def learn_a_hyperedge(edge, best_fit):
    best_fit = np.sum(
        emb_n[edge] / np.linalg.norm(emb_n[edge], axis=1).reshape(-1,1), 
        axis=0)
    best_fit /= np.linalg.norm(best_fit)

    for i in range(edge.shape[0]):
        node = edge[i]
        learn_a_pair_loc_pr_cosine(1, node, best_fit)  

        if num_neg < 1:
            if get_a_neg_sample_Kless1() == 1:
                if i == 0:
                    target_neg = random.choice(neg_sam_table_mobility1)
                elif i == 1:
                    target_neg = random.choice(neg_sam_table_mobility2)
                elif i == 2:
                    target_neg = random.choice(neg_sam_table_mobility3)
                elif i == 3:
                    target_neg = random.choice(neg_sam_table_mobility4)
                if target_neg != node:
                    learn_a_pair_loc_pr_cosine(0, target_neg, best_fit)
        else:
            for n in range(num_neg):
                if i == 0:
                    target_neg = random.choice(neg_sam_table_mobility1)
                elif i == 1:
                    target_neg = random.choice(neg_sam_table_mobility2)
                elif i == 2:
                    target_neg = random.choice(neg_sam_table_mobility3)
                elif i == 3:
                    target_neg = random.choice(neg_sam_table_mobility4)
                if target_neg != node: 
                    learn_a_pair_loc_pr_cosine(0, target_neg, best_fit)


# mexFunction
def api(walk, user_checkins, user_checkins_count, embs_ini, learning_rate,
    num_neg, neg_sam_table_social, win_size, neg_sam_table_mobility_norm,
    num_epoch, num_threads, mobility_ratio):
    num_w, num_wl = walk.shape
    num_u = user_checkins.shape[0]

    global emb_n
    emb_n = embs_ini
    num_n, dim_emb = embs_ini.shape

    starting_alpha = learning_rate
    
    table_size_social = neg_sam_table_social.shape[0]

    neg_sam_table_mobility = neg_sam_table_mobility_norm
    table_num_mobility = neg_sam_table_mobility.shape[0]
    if table_num_mobility != 4:
        raise Exception("four negative sample tables are required in neg_sam_table_mobility")
    
    global neg_sam_table_mobility1
    global neg_sam_table_mobility2
    global neg_sam_table_mobility3
    global neg_sam_table_mobility4
    global table_size_mobility1
    global table_size_mobility2
    global table_size_mobility3
    global table_size_mobility4
    temp = neg_sam_table_mobility[0]
    neg_sam_table_mobility1 = temp
    table_size_mobility1 = temp.shape[0]
    temp = neg_sam_table_mobility[1]
    neg_sam_table_mobility2 = temp
    table_size_mobility2 = temp.shape[0]
    temp = neg_sam_table_mobility[2]
    neg_sam_table_mobility3 = temp
    table_size_mobility3 = temp.shape[0]
    temp = neg_sam_table_mobility[3]
    neg_sam_table_mobility4 = temp
    table_size_mobility4 = temp.shape[0]

    print("walk size = %d %d" % (num_w,num_wl))
    print("user checkins, user count = %d" % num_u)
    print("num of nodes: %d; embedding dimension: %d"%(num_n,dim_emb))
    print("learning rate: %f"%(starting_alpha))
    print("negative sample number: %f"%(num_neg))
    print("social neg table size: %d"%(table_size_social))
    print("mobility neg table num: %d"%(table_num_mobility))
    print("mobility neg table sizes: %d,%d,%d,%d"%(table_size_mobility1,table_size_mobility2,table_size_mobility3,table_size_mobility4))
    print("num_epoch: %d"%num_epoch)
    print("num_threads: %d"%num_threads)

    best_fit = np.zeros((dim_emb))
    progress=0
    progress_old=0
    global alpha
    alpha = starting_alpha

    loc_walk = None

    for pp in range(num_epoch):
        print(f"Epoch {pp}")
        for w in tqdm(range(num_w)):
            progress = (pp*num_w+w) / (num_w*num_epoch)
            if (progress-progress_old > 0.001):
                alpha = starting_alpha * (1 - progress)
                if (alpha < starting_alpha * 0.001):
                    alpha = starting_alpha * 0.001
                progress_old = progress

            for i in range(num_wl):
                word = walk[w, i]
                for j in range(1, win_size+1):
                    if (get_a_social_decision()==1):
                        if (i-j>=0):
                            target_e = walk[w, i-j]
                            if word!=target_e:
                                learn_an_edge_with_BFT(word, target_e, best_fit)
                                learn_an_edge(word, target_e)
                        if (i+j<num_wl):
                            target_e = walk[w, i+j]
                            if word!=target_e:
                                learn_an_edge_with_BFT(word, target_e, best_fit)

                if ((user_checkins_count[word]>0) ):
                    for m in range(min(win_size*2, user_checkins_count[word])):
                        if (get_a_mobility_decision()==1):
                            a_user_checkins = user_checkins[word]
                            edge = a_user_checkins[:, np.random.randint(0, a_user_checkins.shape[1])]
                            learn_a_hyperedge(edge, best_fit)

if __name__ == "__main__":
    from scipy.io import loadmat
    import pdb
    import numpy as np
    import random
    from tqdm import tqdm

    mat = loadmat('dataset/dataset_connected_NYC.mat')
    # print(mat.keys())
    selected_checkins = mat['selected_checkins'] - 1
    selected_users_IDs =  mat['selected_users_IDs'] - 1
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
    num_node_total = np.max(selected_checkins) + 1

    # 2. prepare checkins per user (fast)
    user_checkins = np.zeros((len(selected_users_IDs)), dtype=np.object)
    temp_checkins = selected_checkins[selected_checkins[:,0].argsort(),:]  # a[a[:,0].argsort(),], different with matlab due to multiple values are equals
        
    u, m, n, counters = np.unique(temp_checkins[:,0], return_index=True, return_inverse=True, return_counts=True)

    # 
    # counters = m - np.array([0] + m[:-1].tolist(), dtype=np.int32)
    user_checkins[u] = np.split(temp_checkins, np.cumsum(counters))[:-1]
    # OK here  
    def transpose(x):
        if type(x) != int:
            return x.T
        else:
            return x

    user_checkins = np.array([transpose(x) for x in user_checkins])

    user_checkins_counter = np.zeros((len(selected_users_IDs)), dtype=np.int64)
    user_checkins_counter[u] = counters.astype(np.int64)
    # 3. random walk
    num_node = len(selected_users_IDs)

    from scipy.sparse import csr_matrix
    network = csr_matrix((np.ones(len(friendship_old)), (friendship_old[:,0], friendship_old[:,1])), shape=(num_node, num_node), dtype=int)
    network = network + network.T

    node_list = np.zeros((num_node), dtype=np.object)
    node_list_len = np.zeros((num_node), dtype=np.int32)
    num_walk = 10
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

    # Tested here 
    _, r =  network.nonzero()
    # tab_degree = 
    # 4. prepare negative sample table in advance (fast)
    unique, counts = np.unique(r, return_counts=True)
    freq = (100*counts/counts.sum()) ** 0.75
    neg_sam_table_social = np.repeat(unique, np.round(1000000*freq/sum(freq)).astype(np.int64))
    table_size_social = neg_sam_table_social.shape[0]
    print('neg_sam_table_social', neg_sam_table_social.shape)
    # checkins
    neg_sam_table_mobility_norm = np.zeros((4), dtype=np.object) # cell(4,1)
    for ii in range(len(neg_sam_table_mobility_norm)):
        unique, counts = np.unique(temp_checkins[:,ii], return_counts=True)
        freq = (100*counts/counts.sum()) ** 0.75
        neg_sam_table_mobility_norm[ii] = np.repeat(unique, np.round(100000*freq/sum(freq)).astype(np.int64))


    # LBSN2vec
    dim_emb = 128
    num_epoch = 5
    num_threads =  1
    num_neg = 10
    win_size = 10
    learning_rate = 0.001

    embs_ini = (np.random.uniform(size=(num_node_total, dim_emb)) -0.5)/dim_emb
    embs_len = np.sqrt(np.sum(embs_ini**2, axis=1)).reshape(-1,1)
    embs_ini = embs_ini / embs_len
    mobility_ratio = 0.2

    print("Embs init shape: ", embs_ini.shape)
    print("user_checkins_counter shape: ", user_checkins_counter.shape)
    print("user_checkins shape: ", user_checkins.shape)
    print("neg_sam_table_social shape: ", neg_sam_table_social.shape)
    print("neg_sam_table_mobility_norm shape: ", neg_sam_table_mobility_norm.shape)

    api(walks, user_checkins, user_checkins_counter, embs_ini, learning_rate,
        num_neg, neg_sam_table_social, win_size, neg_sam_table_mobility_norm, num_epoch,
        num_threads, mobility_ratio)
    
    embs = embs_n;
    embs_len = np.sqrt(np.sum(embs**2, axis=1))
    embs = embs / embs_len

    embs_user = embs[:offset1]
    embs_time = embs[offset1:offset2]
    embs_venue = embs[offset2:offset3]
    embs_cate = embs[offset3:]

