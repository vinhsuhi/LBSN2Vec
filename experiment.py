from scipy.io import loadmat
from utils import random_walk, create_social_graph, get_neg_sample, load_data
from tqdm import tqdm
import numpy as np
import 
import random

RAND_MULTIPLIER = 25214903917
RAND_INCREMENT = 11
MAX_EXP = 6
EXP_TABLE_SIZE = 1000
ULONG_MAX = 18446744073709551615

# need initializing
mobility_ratio = 0.
dim_emb = 0
emb_n = []
neg_sam_table_social = None
table_size_social = None
num_neg = 0

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


def getNextRand():
    """
    TODO: Define global next_random
    """
    return RAND_MULTIPLIER + RAND_INCREMENT

def get_a_neg_sample(next_random, neg_sam_table, table_size):
    """
    TODO: IMPLEMENT THIS
    """
    ind = (next_random >> 16) % table_size
    target_n = neg_sam_table[ind]
    return target_n

def get_a_checkin_sample(next_random, table_size):
    return (next_random >> 16) % table_size

def sigmoid(f):
    if f >= MAX_EXP:
        return 1
    elif f <= -MAX_EXP: return 0
    else:
        return expTable[int((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2 ))]

def get_a_neg_sample_Kless1(next_random):
    v_rand_uniform = next_random / ULONG_MAX
    if v_rand_uniform <= num_neg:
        return 1
    else:
        return 0

def get_a_social_decision(next_random):
    v_rand_uniform = next_random / ULONG_MAX
    if v_rand_uniform <= mobility_ratio:
        return 0
    else:
        return 1

def get_a_mobility_decision(next_random):
    v_rand_uniform = next_random / ULONG_MAX
    if v_rand_uniform <= mobility_ratio:
        return 1
    else:
        return 0

def get_norm_l2_loc(loc_node):
    norm = 0
    for d in range(dim_emb):
        norm = norm + emb_n[loc_node+d] * emb_n[loc_node+d]
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
        f += emb_n[loc1+d]*best_fit[d]

    g = f/norm1
    a = alpha
    c1 = 1/(norm1)*a
    c2 = f/(norm1*norm1*norm1)*a
    if flag == 1:
        for d in range(dim_emb):
            emb_n[loc1+d] += c1*best_fit[d] - c2*emb_n[loc1+d]
    else:
        for d in range(dim_emb):
            emb_n[loc1+d] -= c1*best_fit[d] - c2*emb_n[loc1+d]

def learn_an_edge(word, target_e, next_random, counter):
    target_n, loc_neg = 0,0 
    loc_w = (word-1)*dim_emb
    loc_e = (target_e-1)*dim_emb
    learn_a_pair_loc_loc_cosine(1, loc_w, loc_e, counter)

    if num_neg < 1:
        next_random = getNextRand(next_random)
        if get_a_neg_sample_Kless1(next_random) == 1:
            next_random = getNextRand(next_random)
            target_n = get_a_neg_sample(next_random, neg_sam_table_social, table_size_social)
            if target_n != target_e and target_n != word:
                loc_neg = (target_n-1) * dim_emb
                learn_a_pair_loc_loc_cosine(0, loc_w, loc_neg, counter)
    else:
        for i in range(num_neg):
            next_random = getNextRand(next_random)
            target_n = get_a_neg_sample(next_random, neg_sam_table_social, table_size_social)
            if target_n != target_e and target_n != word:
                loc_neg = (target_n-1)*dim_emb
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
    loc_w = (word-1) * dim_emb
    loc_e = (target_e-1)*dim_emb
    for d in range(dim_emb):
        best_fit[d] = emb_n[loc_w+d] + emb_n[loc_e+d]
    norm = get_norm_l2_pr(best_fit)
    for d in range(dim_emb):
        best_fit[d] = best_fit[d]/norm
    
    learn_a_pair_loc_pr_cosine(1, loc_w, best_fit, counter)
    learn_a_pair_loc_pr_cosine(1, loc_e, best_fit, counter)

    if num_neg < 1:
        next_random = getNextRand(next_random)
        if get_a_neg_sample_Kless1(next_random) == 1:
            next_random = getNextRand(next_random)
            target_n = get_a_neg_sample(next_random, neg_sam_table_social, table_size_social)
            if target_n != target_e and target_n != word:
                loc_neg = (target_n-1)*dim_emb
                learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, counter)
    else:
        for n in range(num_neg):
            next_random = getNextRand(next_random)
            target_n = get_a_neg_sample(next_random, neg_sam_table_social, table_size_social)
            if target_n != target_e and target_n != word:
                loc_neg = (target_n-1)*dim_emb
                learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, counter)

def learn_a_hyperedge(edge, edge_len, best_fit, counter):
    node, target_neg = 0,0
    loc_n, loc_neg = 0,0 
    norm = 0
    for d in range(dim_emb):
        best_fit[d] = 0
    for i in range(edge_len):
        loc_n = (edge[i]-1)*dim_emb
        norm = get_norm_l2_pr(emb_n[loc_n])
        for d in range(dim_emb):
            best_fit[d] += emb_n[loc_n+d] / norm
    norm = get_norm_l2_pr(best_fit)
    for d in range(dim_emb):
        best_fit[d] = best_fit[d]/norm
    
    for i in range(edge_len):
        node = edge[i]
        loc_n = (node-1)*dim_emb
        learn_a_pair_loc_pr_cosine(1, loc_n, best_fit, counter)

        if num_neg < 1:
            next_random = getNextRand(next_random)
            if get_a_neg_sample_Kless1(next_random) == 1:
                next_random = getNextRand(next_random)
                if i == 0:
                    target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility1, table_size_mobility1)
                elif i == 1:
                    target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility2, table_size_mobility2)
                elif i == 2:
                    target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility3, table_size_mobility3)
                elif i == 3:
                    target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility4, table_size_mobility4)
                
                if target_neg != node:
                    loc_neg = (target_neg-1)*dim_emb
                    learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, counter)
        else:
            for n in range(num_neg):
                next_random = getNextRand(next_random)
                if i == 0:
                    target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility1, table_size_mobility1)
                elif i == 1:
                    target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility2, table_size_mobility2)
                elif i == 2:
                    target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility3, table_size_mobility3)
                elif i == 3:
                    target_neg = get_a_neg_sample(next_random, neg_sam_table_mobility4, table_size_mobility4)
                if target_neg != node:
                    loc_neg = (target_neg-1)*dim_emb
                    learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, counter)

# mexFunction
def api(walk, user_checkins, user_checkins_count, embs_ini, learning_rate,
    num_neg, neg_sam_table_social, win_size, neg_sam_table_mobility_norm,
    num_epoch, num_threads, mobility_ratio):
    num_w, num_wl = walk.shape
    num_u = user_checkins.shape[0]

    emb_n = embs_ini
    num_n, dim_emb = embs_ini.shape

    starting_alpha = learning_rate
    
    table_size_social = neg_sam_table_social.shape[0]

    neg_sam_table_mobility = neg_sam_table_mobility_norm
    table_num_mobility = neg_sam_table_mobility.shape[0]
    if table_num_mobility != 4:
        raise Exception("four negative sample tables are required in neg_sam_table_mobility")
    
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
    print("walk size = %d %d\n", num_w,num_wl)
    print("user checkins, user count = %d\n", num_u)
    print("num of nodes: %lld; embedding dimension: %lld\n",num_n,dim_emb)
    print("learning rate: %f\n",starting_alpha)
    print("negative sample number: %f\n",num_neg)
    print("social neg table size: %lld\n",table_size_social)
    print("mobility neg table num: %lld\n",table_num_mobility)
    print("mobility neg table sizes: %lld,%lld,%lld,%lld\n",table_size_mobility1,table_size_mobility2,table_size_mobility3,table_size_mobility4)
    print("num_epoch: %lld\n",num_epoch)
    print("num_threads: %lld\n",num_threads)

    a = 0
    # call learn multithread




if __name__ == "__main__":
    NUMWALK = 1
    WALKLEN = 8
    EMBDIM = 128
    NUMEPOCH = 1
    KNEG = 10
    WINSIZE = 10
    LEARNINGRATE = 0.001 # stating alpha
    MOBILITYRATIO = 0.2
    edge_len = 4 # here 4 is a checkin node number user-time-POI-category

    # CONSTANTS
    MAX_EXP = 6
    EXP_TABLE_SIZE = 1000
    next_random = 0

    path = "dataset_connected_NYC.mat"
    selected_checkins, SocialGraph, num_users, user_checkins_count, user_checkin_dict = load_data(path)
    degree_social = np.array([SocialGraph.degree(node) for node in SocialGraph.nodes()])
    # exit()
    walk = random_walk(NUMWALK, WALKLEN, SocialGraph)

    # neg_sam_table_social

    # neg_sam_table_mobility_norm

    # neg_sam_table_mobility1, 2, 3, 4 ( in code c )

    # learn here
    num_w = NUMWALK * num_users
    progress_old = 0
    for epoch_index in range(NUMEPOCH):
        counter = 0
        for w_index in tqdm(range(num_w), desc="Training..."):

            # for learning rate regularization
            progress = (epoch_index * num_w + w_index) / (num_w * NUMEPOCH)
            if progress - progress_old > 0.001:
                alpha = LEARNINGRATE * (1 - progress)
                if alpha < LEARNINGRATE * 0.001:
                    alpha = LEARNINGRATE * 0.001
                progress_old = progress
            
            loc_walk = w_index * WALKLEN
            for i in range(WALKLEN):
                word = walk[w_index + i]
                for j in range(1, WINSIZE + 1):
                    if get_a_social_decision():
                        if i >= j:
                            target_e = walk[w_index + i - j]
                            if word != target_e:
                                # TODO: IMPLEMENT THIS FUNCTION
                                learn_an_edge_with_BFT(word, target_e, best_fit, counter)
                        if i + j < WALKLEN:
                            target_e = walk[loc_walk + i + j]
                            if word != target_e:
                                learn_an_edge_with_BFT(word, target_e, best_fit, counter)


                if user_checkins_count[word - 1] > 0:
                    for m in range(min(WINSIZE * 2, user_checkins_count[word - 1])):
                        # TODO: IMPLEMENT THIS FUNCTION
                        if get_a_mobility_decision():
                            a_user_checkins = user_checkin_dict[word]
                            # TODO: IMPLEMENT THIS FUNCTION
                            a_checkin_ind = get_a_checkin_sample(user_checkins_count[word-1])
                            # a_checkin_loc = e_checkin_ind * edge_len
                            edge = a_user_checkins[a_checkin_ind]
                            # TODO: IMPLEMENT THIS FUNCTION
                            learn_a_hyperedge(edge, edge_len, best_fit)

