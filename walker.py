import itertools
import math
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import trange
from .alias import alias_sample, create_alias_table
from .utils import partition_num

class Random_walker():
    def __init__(self,G,p,q):
        self.G = G
        self.p = p
        self.q = q

    def preprocessing_transition_probs(self):
        G = self.G
        alias_nodes = {}
        alias_edges = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbs].get('weight',1.0) for nbs in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(un_prob/norm_const) for un_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)

        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0],edge[1])
        
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
    
    def get_alias_edge(self,pre_node,_node):
        G = self.G
        p = self.p 
        q = self.q 
        unnormalized_probs = []
        for nbs in G.neighbors(_node):
            weight = G[_node][nbs].get('weight', 1.0)
            if nbs == pre_node:
                unnormalized_probs.append(weight/p)
            elif G.has_edge(pre_node,nbs):
                unnormalized_probs.append(weight)
            else:unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(un_probs/norm_const) for un_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)
    
    def node2vec_walk(self,walk_length,start_node):
        G = self.G 
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        walk = [start_node]
        while len(walk)<walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs)>0:
                if len(walk)==1:
                    walk.append(cur_nbrs(alias_sample(alias_nodes[cur][0],alias_nodes[cur][1])))
                else:
                    prev = walk[-2]
                    edge = (prev,cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0],alias_edges[edge][1])]
                    walk.append(next_node)
            else: break
        
        return walk 


