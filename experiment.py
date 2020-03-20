from scipy.io import loadmat
from utils import random_walk, create_social_graph, get_neg_sample, load_data
from tqdm import tqdm
import numpy as np


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
    pass


def get_a_social_decision():
    rand = np.random.rand()
    return rand > MOBILITYRATIO


def get_a_neg_sample():
    """
    TODO: IMPLEMENT THIS
    """
    rand = np.random.rand()
    # if 
    pass


def learn_an_edge_with_BFT(word, target_e, best_fit):
    """
    TODO: IMPLEMENT THIS
    WHAT DOES THIS FUNCTION DO?
    """
    negs = fixed_unigram_candidate_sampler(KNEG, False, range_max, 0.75, unigrams):

    for n in range(KNEG):
        # TODO: implement this function
        target_n = get_a_neg_sample()



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

