import numpy as np 
import networkx as nx 
import torch 
import multiprocessing


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

