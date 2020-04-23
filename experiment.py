from scipy.io import loadmat
from utils import random_walk, create_social_graph, get_neg_sample, load_data
from tqdm import tqdm
import numpy as np
import random
import pdb
import math

RAND_MULTIPLIER = 25214903917
RAND_INCREMENT = 11
MAX_EXP = 6
EXP_TABLE_SIZE = 1000
ULONG_MAX = 4294967295

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


def main():
    """
    TODO: 
    """
    pass
    # walk = random_walk(num_walk, walk_len)

def get_a_neg_sample(neg_sam_table, table_size):
    """
    TODO: IMPLEMENT THIS
    """
    ind = np.random.randint(table_size)
    target_n = neg_sam_table[ind]
    return target_n

def get_a_checkin_sample(table_size):
    return np.random.randint(table_size)

def sigmoid(f):
    if f >= MAX_EXP:
        return 1
    elif f <= -MAX_EXP: return 0
    else:
        return expTable[int((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2 ))]

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

def get_norm_l2_loc(loc_node):
    norm = 0
    for d in range(dim_emb):
        norm = norm + emb_n[loc_node, d] * emb_n[loc_node, d]
    return math.sqrt(norm)

def get_norm_l2_pr(vec):
    norm = 0
    for d in range(dim_emb):
        norm = norm + vec[d]**2
    return math.sqrt(norm)
        
def learn_a_pair_loc_loc_cosine(flag, loc1, loc2, loss):
    f = 0
    tmp1, tmp2, c1, c2, c3 = 0, 0, 0, 0, 0
    norm1 = get_norm_l2_loc(loc1)
    norm2 = get_norm_l2_loc(loc2)

    for d in range(dim_emb):
        f += emb_n[loc1+d] * emb_n[loc2+d]
    c1 = 1/(norm1*norm2)*alpha
    c2 = f/(norm1**3 * norm2) * alpha
    c3 = f/(norm1*norm2**3) * alpha

    if flag == 1:
        for d in range(dim_emb):
            tmp1 = emb_n[loc1+d]
            tmp2 = emb_n[loc2+d]
            emb_n[loc2+d] += c1*tmp1 - c3*tmp2
            emb_n[loc1+d] += c1*tmp2 - c2*tmp1
    else:
        for d in range(dim_emb):
            tmp1 = emb_n[loc1+d]
            tmp2 = emb_n[loc2+d]
            emb_n[loc2+d] -= c1*tmp1 - c3*tmp2
            emb_n[loc1+d] -= c1*tmp2 - c2*tmp1

def learn_a_pair_loc_pr_cosine(flag, loc1, best_fit, loss):
    f, g, a, c1, c2 = 0, 0, 0, 0, 0
    norm1 = get_norm_l2_loc(loc1)
    for d in range(dim_emb):
        f += emb_n[loc1, d]*best_fit[d]

    g = f/norm1
    a = alpha
    c1 = 1/(norm1)*a
    c2 = f/(norm1*norm1*norm1)*a
    if flag == 1:
        for d in range(dim_emb):
            emb_n[loc1, d] += c1*best_fit[d] - c2*emb_n[loc1, d]
    else:
        for d in range(dim_emb):
            emb_n[loc1, d] -= c1*best_fit[d] - c2*emb_n[loc1, d]

def learn_an_edge(word, target_e, counter):
    target_n, loc_neg = 0,0 
    loc_w = (word-1)
    loc_e = (target_e-1)
    learn_a_pair_loc_loc_cosine(1, loc_w, loc_e, counter)

    if num_neg < 1:
        if get_a_neg_sample_Kless1() == 1:
            target_n = get_a_neg_sample( neg_sam_table_social, table_size_social)
            if target_n != target_e and target_n != word:
                loc_neg = (target_n-1)
                learn_a_pair_loc_loc_cosine(0, loc_w, loc_neg, counter)
    else:
        for i in range(num_neg):
            target_n = get_a_neg_sample( neg_sam_table_social, table_size_social)
            if target_n != target_e and target_n != word:
                loc_neg = (target_n-1)
                learn_a_pair_loc_loc_cosine(0, loc_w, loc_neg, counter)

# def get_a_social_decision():
#     rand = np.random.rand()
#     return rand > MOBILITYRATIO

def learn_an_edge_with_BFT(word, target_e, best_fit, counter):
    """
    TODO: IMPLEMENT THIS
    WHAT DOES THIS FUNCTION DO?
    """
    # negs = fixed_unigram_candidate_sampler(KNEG, False, range_max, 0.75, unigrams):

    # for n in range(KNEG):
    #     # TODO: implement this function
    #     target_n = get_a_neg_sample()
    target_n, loc_neg = 0,0
    norm = 0
    loc_w = (word-1)
    loc_e = (target_e-1)
    for d in range(dim_emb):
        best_fit[d] = emb_n[loc_w, d] + emb_n[loc_e, d]
    norm = get_norm_l2_pr(best_fit)
    for d in range(dim_emb):
        best_fit[d] = best_fit[d]/norm
    
    learn_a_pair_loc_pr_cosine(1, loc_w, best_fit, counter)
    learn_a_pair_loc_pr_cosine(1, loc_e, best_fit, counter)

    if num_neg < 1:
        if get_a_neg_sample_Kless1() == 1:
            target_n = get_a_neg_sample( neg_sam_table_social, table_size_social)
            if target_n != target_e and target_n != word:
                loc_neg = (target_n-1)
                learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, counter)
    else:
        for n in range(num_neg):
            target_n = get_a_neg_sample( neg_sam_table_social, table_size_social)
            if target_n != target_e and target_n != word:
                loc_neg = (target_n-1)
                learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, counter)

def learn_a_hyperedge(edge, edge_len, best_fit, counter):
    node, target_neg = 0,0
    loc_n, loc_neg = 0,0 
    norm = 0
    for d in range(dim_emb):
        best_fit[d] = 0
    
    for i in range(edge_len):
        norm = get_norm_l2_pr(emb_n[edge[i]])
        for d in range(dim_emb):
            best_fit[d] += emb_n[edge[i], d] / norm
    norm = get_norm_l2_pr(best_fit)
    for d in range(dim_emb):
        best_fit[d] = best_fit[d]/norm
    
    for i in range(edge_len):
        node = edge[i]
        learn_a_pair_loc_pr_cosine(1, node, best_fit, counter)  

        if num_neg < 1:
            if get_a_neg_sample_Kless1() == 1:
                if i == 0:
                    target_neg = get_a_neg_sample( neg_sam_table_mobility1, table_size_mobility1)
                elif i == 1:
                    target_neg = get_a_neg_sample( neg_sam_table_mobility2, table_size_mobility2)
                elif i == 2:
                    target_neg = get_a_neg_sample( neg_sam_table_mobility3, table_size_mobility3)
                elif i == 3:
                    target_neg = get_a_neg_sample( neg_sam_table_mobility4, table_size_mobility4)
                
                if target_neg != node:
                    learn_a_pair_loc_pr_cosine(0, target_neg, best_fit, counter)
        else:
            for n in range(num_neg):
                if i == 0:
                    target_neg = get_a_neg_sample( neg_sam_table_mobility1, table_size_mobility1)
                elif i == 1:
                    target_neg = get_a_neg_sample( neg_sam_table_mobility2, table_size_mobility2)
                elif i == 2:
                    target_neg = get_a_neg_sample( neg_sam_table_mobility3, table_size_mobility3)
                elif i == 3:
                    target_neg = get_a_neg_sample( neg_sam_table_mobility4, table_size_mobility4)
                if target_neg != node:
                    learn_a_pair_loc_pr_cosine(0, target_neg, best_fit, counter)


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

    # num_epoch = 
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
    # # call learn multithread
    # for a in range(num_threads):
    #     learn(num_threads, a)

    # def learn() below
    id = 0
    word, target_e, a_checkin_ind, a_checkin_loc = None, None, None, None
    best_fit = np.zeros((dim_emb))
    counter = 0;
    user_pr = None
    a_user_checkins = None
    edge = None
    edge_len = 4; # here 4 is a checkin node number user-time-POI-category

    ind_start = int(num_w/num_threads * id)
    ind_end = int(num_w/num_threads * (id+1))

    ind_len = ind_end-ind_start;
    progress=0
    progress_old=0
    global alpha
    alpha = starting_alpha

    loc_walk = None

    for pp in range(num_epoch):
        counter = 0
        for w in range(ind_start, ind_end):
            progress = ((pp*ind_len)+(w-ind_start)) / (ind_len*num_epoch)
            if (progress-progress_old > 0.001):
                alpha = starting_alpha * (1 - progress)
                if (alpha < starting_alpha * 0.001):
                    alpha = starting_alpha * 0.001
                progress_old = progress

            print(num_wl, w)
            for i in range(num_wl):
                word = walk[w, i]
                for j in range(1, win_size+1):
                    if (get_a_social_decision()==1):
                        if (i-j>=0):
                            target_e = walk[w, i-j]
                            if (word!=target_e):
                                learn_an_edge_with_BFT(word, target_e, best_fit, counter)
                                learn_an_edge(word, target_e, counter)
                        if (i+j<num_wl):
                            target_e = walk[w, i+j];
                            if (word!=target_e):
                                learn_an_edge_with_BFT(word, target_e, best_fit, counter)

                if ((user_checkins_count[word]>0) ):
                    for m in range(min(win_size*2, user_checkins_count[word])):
                        if (get_a_mobility_decision()==1):
                            user_pr = user_checkins[word]
                            a_user_checkins = user_pr
                            a_checkin_ind = get_a_checkin_sample(user_checkins_count[word]);
                            print("sampled checkin index is %d" % a_checkin_ind);
                            edge = a_user_checkins[:, a_checkin_ind]
                            learn_a_hyperedge(edge, edge_len, best_fit, counter)



if __name__ == "__main__":
    from scipy.io import loadmat
    import pdb
    import numpy as np
    import random
    from tqdm import tqdm

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

    # Tested here 
    _, r =  network.nonzero()
    # tab_degree = 
    # 4. prepare negative sample table in advance (fast)
    unique, counts = np.unique(r, return_counts=True)
    freq = (100*counts/counts.sum()) ** 0.75
    neg_sam_table_social = np.repeat(unique, np.round(1000000*freq/sum(freq)).astype(np.int64))
    table_size_social = neg_sam_table_social.shape[0]
    # checkins
    neg_sam_table_mobility_norm = np.zeros((4), dtype=np.object) # cell(4,1)
    for ii in range(len(neg_sam_table_mobility_norm)):
        unique, counts = np.unique(temp_checkins[:,ii], return_counts=True)
        freq = (100*counts/counts.sum()) ** 0.75
        neg_sam_table_mobility_norm[ii] = np.repeat(unique, np.round(100000*freq/sum(freq)).astype(np.int64))


    # LBSN2vec
    dim_emb = 128
    num_epoch = 1
    num_threads =  1
    num_neg = 10
    win_size = 10
    learning_rate = 0.001

    embs_ini = (np.random.uniform(size=(num_node_total, dim_emb)) -0.5)/dim_emb
    embs_len = np.sqrt(np.sum(embs_ini**2, axis=1)).reshape(-1,1)
    embs_ini = embs_ini / (np.tile(embs_len, (1, dim_emb)))
    mobility_ratio = 0.2

    # for i,x in enumerate(user_checkins):
    #     try:
    #         if x == 0:
    #             user_checkins[i] = np.zeros()
    #     except:
    #         continue
    api(walks, user_checkins, user_checkins_counter, embs_ini, learning_rate,
        num_neg, neg_sam_table_social, win_size, neg_sam_table_mobility_norm, num_epoch,
        num_threads, mobility_ratio)


    # NUMWALK = 1
    # WALKLEN = 8
    # EMBDIM = 128
    # NUMEPOCH = 1
    # KNEG = 10
    # WINSIZE = 10
    # LEARNINGRATE = 0.001 # stating alpha
    # MOBILITYRATIO = 0.2
    # edge_len = 4 # here 4 is a checkin node number user-time-POI-category

    # # CONSTANTS
    # MAX_EXP = 6
    # EXP_TABLE_SIZE = 1000

    # path = "dataset_connected_NYC.mat"
    # selected_checkins, SocialGraph, num_users, user_checkins_count, user_checkin_dict = load_data(path)
    # degree_social = np.array([SocialGraph.degree(node) for node in SocialGraph.nodes()])
    # # exit()
    # walk = random_walk(NUMWALK, WALKLEN, SocialGraph)

    # # neg_sam_table_social

    # # neg_sam_table_mobility_norm

    # # neg_sam_table_mobility1, 2, 3, 4 ( in code c )

    # # learn here
    # num_w = NUMWALK * num_users
    # progress_old = 0
    # for epoch_index in range(NUMEPOCH):
    #     counter = 0
    #     for w_index in tqdm(range(num_w), desc="Training..."):

    #         # for learning rate regularization
    #         progress = (epoch_index * num_w + w_index) / (num_w * NUMEPOCH)
    #         if progress - progress_old > 0.001:
    #             alpha = LEARNINGRATE * (1 - progress)
    #             if alpha < LEARNINGRATE * 0.001:
    #                 alpha = LEARNINGRATE * 0.001
    #             progress_old = progress
            
    #         loc_walk = w_index * WALKLEN
    #         for i in range(WALKLEN):
    #             word = walk[w_index + i]
    #             for j in range(1, WINSIZE + 1):
    #                 if get_a_social_decision():
    #                     if i >= j:
    #                         target_e = walk[w_index + i - j]
    #                         if word != target_e:
    #                             # TODO: IMPLEMENT THIS FUNCTION
    #                             learn_an_edge_with_BFT(word, target_e, best_fit, counter)
    #                     if i + j < WALKLEN:
    #                         target_e = walk[loc_walk + i + j]
    #                         if word != target_e:
    #                             learn_an_edge_with_BFT(word, target_e, best_fit, counter)


    #             if user_checkins_count[word - 1] > 0:
    #                 for m in range(min(WINSIZE * 2, user_checkins_count[word - 1])):
    #                     # TODO: IMPLEMENT THIS FUNCTION
    #                     if get_a_mobility_decision():
    #                         a_user_checkins = user_checkin_dict[word]
    #                         # TODO: IMPLEMENT THIS FUNCTION
    #                         a_checkin_ind = get_a_checkin_sample(user_checkins_count[word-1])
    #                         # a_checkin_loc = e_checkin_ind * edge_len
    #                         edge = a_user_checkins[a_checkin_ind]
    #                         # TODO: IMPLEMENT THIS FUNCTION
    #                         learn_a_hyperedge(edge, edge_len, best_fit)

