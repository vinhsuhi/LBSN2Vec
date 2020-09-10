"""Ego-Splitter class"""

import community
import networkx as nx
from tqdm import tqdm


class EgoNetSplitter(object):
    """An implementation of `"Ego-Splitting" see:
    https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p145.pdf
    From the KDD '17 paper "Ego-Splitting Framework: from Non-Overlapping to Overlapping Clusters".
    The tool first creates the egonets of nodes.
    A persona-graph is created which is clustered by the Louvain method.
    The resulting overlapping cluster memberships are stored as a dictionary.
    Args:
        resolution (float): Resolution parameter of Python Louvain. Default 1.0.
    """
    def __init__(self, resolution=1.0):
        self.resolution = resolution

    def _create_egonet(self, node):
        """
        Creating an ego net, extracting personas and partitioning it.
        Args:
            node: Node ID for egonet (ego node).
        """
        ego_net_minus_ego = self.graph.subgraph(self.graph.neighbors(node))
        # print("ego_net_minus_ego   ",ego_net_minus_ego.nodes)
        ############### AVG DEGREE #########################
        connected_components_list = []
        component_combie = []
        avg_degree_threshold = 1
        for i in nx.connected_components(ego_net_minus_ego):
            sub_graph_component = self.graph.subgraph(i)
            # print(self.graph.subgraph(i).copy().edges)
            # print(self.graph.subgraph(i).copy().edges)
            avg_degree = len(sub_graph_component.edges)/len(sub_graph_component.nodes)
            if avg_degree <= avg_degree_threshold:
                component_combie.extend(i)
            else:
                connected_components_list.append(i)
        connected_components_list.append(component_combie)

        # components = {i: n for i, n in enumerate(nx.connected_components(ego_net_minus_ego))}
        components = {i: n for i, n in enumerate(connected_components_list)}
        #######################################################
        ###############  NUMBER NODE  #########################
        new_mapping = {}
        personalities = []
        # print("components   ",components)
        components_new = {}
        node_combine = []
        k_new = 0
        node_threshold = 1
        for k, v in components.items():
            # print(type(v))
            # print(v)
            if len(v)<=node_threshold:
                node_combine.extend(list(v))
            else:
                components_new[k_new] = v
                k_new+=1
        components_new[k_new] = node_combine
        components = components_new

        #######################################################
        ################# ORIGINAL CODE #######################

        # new_mapping = {}
        # personalities = []
        # components = {i: n for i, n in enumerate(nx.connected_components(ego_net_minus_ego))}

        ########################################
        for k, v in components.items():
            # print("k:  ",k)
            # print("v:  ",v)
            personalities.append(self.index)
            for other_node in v:
                new_mapping[other_node] = self.index
            self.index = self.index+1
        # print("new_mapping   ",new_mapping)
        # print("personalities   ",personalities)
        self.components[node] = new_mapping
        self.personalities[node] = personalities

    def _create_egonets(self):
        """
        Creating an egonet for each node.
        """
        self.components = {}
        self.personalities = {}
        self.index = 0
        print("Creating egonets.")
        for node in tqdm(self.graph.nodes()):
            # print("nodes   ",node)
            self._create_egonet(node)

    def _map_personalities(self):
        """
        Mapping the personas to new nodes.
        """
        self.personality_map = {p: n for n in self.graph.nodes() for p in self.personalities[n]}
        # print("personality_map    ",self.personality_map)
    def _get_new_edge_ids(self, edge):
        """
        Getting the new edge identifiers.
        Args:
            edge: Edge being mapped to the new identifiers.
        """
        return (self.components[edge[0]][edge[1]], self.components[edge[1]][edge[0]])

    def _create_persona_graph(self):
        """
        Create a persona graph using the egonet components.
        """
        print("Creating the persona graph.")
        self.persona_graph_edges = [self._get_new_edge_ids(e) for e in tqdm(self.graph.edges())]
        self.persona_graph = nx.from_edgelist(self.persona_graph_edges)

    def _create_partitions(self):
        """
        Creating a non-overlapping clustering of nodes in the persona graph.
        """
        print("Clustering the persona graph.")
        self.partitions = community.best_partition(self.persona_graph, resolution=self.resolution)
        self.overlapping_partitions = {node: [] for node in self.graph.nodes()}
        for node, membership in self.partitions.items():
            self.overlapping_partitions[self.personality_map[node]].append(membership)

    def fit(self, graph):
        """
        Fitting an Ego-Splitter clustering model.
        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self.graph = graph
        self._create_egonets()
        self._map_personalities()
        self._create_persona_graph()
        self._create_partitions()

    def get_memberships(self):
        r"""Getting the cluster membership of nodes.
        Return types:
            * **memberships** *(dictionary of lists)* - Cluster memberships.
        """
        return self.overlapping_partitions
