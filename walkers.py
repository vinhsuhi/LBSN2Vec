import numpy as np 
import networkx as nx 
import torch 
import multiprocessing


class BasicWalker:
    def __init__(self, G, start_nodes=None, user_poi_dict={}, bias=False, thresh=0):
        self.G = G
        if hasattr(G, 'neibs'):
            self.neibs = G.neibs
        else:
            self.build_neibs_dict()
        if start_nodes is not None:
            self.start_nodes = start_nodes
        else:
            self.start_nodes = list(self.G.nodes())
        
        self.user_poi_dict = user_poi_dict
        self.bias = bias
        self.thresh = thresh


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
        params = list(map(lambda x: {'walk_length': walk_length, 'neibs': self.neibs, 'iter': x, 'nodes': nodess[x], 'bias': self.bias, 'user_poi_dict': self.user_poi_dict, 'thresh': thresh},
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


def deepwalk_walk_wrapper(class_instance, walk_length, start_node):
    class_instance.deepwalk_walk(walk_length, start_node)


def deepwalk_walk(params):
    '''
    Simulate a random walk starting from start node.
    '''
    bias = params["bias"]
    user_poi_dict = params["user_poi_dict"]
    walk_length = params["walk_length"]
    neibs = params["neibs"]
    nodes = params["nodes"]
    thresh = params["thresh"]
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
            if not bias:
                walk.append(np.random.choice(cur_nbrs))
            else:
                walk.append(bias_walk(cur, cur_nbrs, user_poi_dict, thresh))
        walks.append(walk)
    return walks


def bias_walk(cur, cur_nbrs, user_poi_dict, thresh):
    this_poi = user_poi_dict[cur]
    prob = []
    for i in range(len(cur_nbrs)):
        nb = cur_nbrs[i]
        nb_poi = user_poi_dict[nb]
        if nb_poi >= thresh:
            prob.append(0)
            continue
        common = nb_poi.intersection(this_poi)
        union = nb_poi.union(this_poi)
        if len(union) == 0:
            prob.append(0)
        else:
            prob.append(len(common) / len(union))
    prob = np.array(prob, dtype=float)
    prob += np.max(prob) / 10
    if np.max(prob) == 0:
        prob += 1
    return np.random.choice(cur_nbrs, p=prob/prob.sum())