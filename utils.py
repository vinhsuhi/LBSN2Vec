import numpy as np 
import os 
import networkx as nx
from scipy.io import loadmat
from collections import Counter
from tqdm import tqdm

def random_walk(num_walk, walk_len, Graph):
    """
    TODO: DONE!
    parameters:
    * num_walk: Number of walks per node
    * walk_len: The length of each walk
    * Graph: The input graph

    return:
    * walks: Matrix m x n, m = num_walk * num_nodes, n = walk_len
    """
    walks = []
    num_nodes = Graph.number_of_nodes()
    for i in tqdm(range(num_walk), desc="Walking"):
        for node in tqdm(Graph.nodes()):
            walk = [node]
            curr_node = node
            for j in range(walk_len - 1):
                if Graph.degree(curr_node) > 0:
                    neighbors = [node for node in Graph[curr_node]]
                    rand_node = np.random.choice(neighbors)
                else:
                    rand_node = curr_node
                curr_node = rand_node
                walk.append(curr_node)
            walks.append(walk)
    
    return np.array(walks).flatten()


def create_social_graph(users_IDs, old_friendship):
    """
    TODO: Create social graph: DONE!
    parameters:
    * users_IDs: array of user_IDs
    * old_friendship: edge list of graph

    return:
    * SocialGraph: The undirected networkx graph
    """

    SocialGraph = nx.Graph()
    SocialGraph.add_nodes_from(users_IDs)
    SocialGraph.add_edges_from(old_friendship)

    return SocialGraph


def get_neg_sample(Graph):
    """
    TODO: latter
    """
    print("have not implement yet!")
    exit()


def load_data(path):
    data = loadmat(path)
    selected_checkins = data['selected_checkins']
    # selected_users_IDs = data['selected_users_IDs']

    offset1 = np.max(selected_checkins[:, 0])
    unique1 = np.unique(selected_checkins[:, 1])
    unique1_dict = {ele: i + 1 for i, ele in enumerate(unique1)}
    selected_checkins[:, 1] = np.array([unique1_dict[id] + offset1 for id in selected_checkins[:, 1]])
    offset2 = np.max(selected_checkins[:, 1])
    unique2 = np.unique(selected_checkins[:, 2])
    unique2_dict = {ele: i + 1 for i, ele in enumerate(unique2)}
    selected_checkins[:, 2] = np.array([unique2_dict[id] + offset2 for id in selected_checkins[:, 2]])
    offset3 = np.max(selected_checkins[:, 2])
    unique3 = np.unique(selected_checkins[:, 3])
    unique3_dict = {ele: i + 1 for i, ele in enumerate(unique3)}
    selected_checkins[:, 3] = np.array([unique3_dict[id] + offset3 for id in selected_checkins[:, 3]])

    num_node_total = np.max(selected_checkins)
    num_users = len(data['selected_users_IDs'])

    # preprocessing data
    users = np.arange(num_users, dtype=int) + 1
    old_social_edges = data['friendship_old']
    new_social_edges = data['friendship_new']

    SocialGraph = create_social_graph(users, old_social_edges)
    counter_checkin_dict = Counter(selected_checkins[:, 0])
    user_checkins_counter = np.zeros(num_users)
    for i in range(num_users):
        user_checkins_counter[i] = counter_checkin_dict[i + 1]

    user_checkin_dict = dict() # i: list of [i, x, y, z]s

    for i in range(len(selected_checkins)):
        user = selected_checkins[i, 0]
        if user not in user_checkin_dict:
            user_checkin_dict[user] = [selected_checkins[i]]
        else:
            user_checkin_dict[user].append(selected_checkins[i])
    

    return selected_checkins, SocialGraph, num_users, user_checkins_counter, user_checkin_dict