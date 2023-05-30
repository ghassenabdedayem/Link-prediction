import random
from random import randint
from datetime import datetime
import pandas as pd
import numpy as np
from time import time
import networkx as nx
from random import choice
from urllib.request import urlopen
import requests

def read_train_val_graph(path='https://www.lix.polytechnique.fr/~nikolentzos/files/aai/challenge/edgelist.txt', val_ratio=0.1):
    # We gets the data from the file on the distant server
    G = nx.read_edgelist(urlopen(path), delimiter=',', create_using=nx.Graph(), nodetype=int)
    nodes = list(G.nodes())
    edges = list(G.edges())
    n = G.number_of_nodes()
    m = G.number_of_edges()


    print('Number of nodes:', n, 'number of edges:', m,'in the Complete set')

    node_to_idx = dict()
    for i, node in enumerate(nodes):
        node_to_idx[node] = i

    val_edges = list()
    G_train = G.copy()

    for edge in edges:
        if random.random() < val_ratio and edge[0] < n and edge[1] < n:
            val_edges.append(edge)
            G_train.remove_edge(edge[0], edge[1]) # We remove the val edges from the graph G

   
    #for edge in val_edges:
        

    n = G_train.number_of_nodes()
    m = G_train.number_of_edges()
    train_edges = list(G_train.edges())

    print('Number of nodes:', n, 'number of edges:', m, 'in the Training set')
    print('len(nodes)', len(nodes))

    y_val = [1]*len(val_edges)

    n_val_edges = len(val_edges)
    
    # print('Creating random val_edges...')
    # for i in range(n_val_edges):
    #     n1 = nodes[randint(0, n-1)]
    #     n2 = nodes[randint(0, n-1)]
    #     (n1, n2) = (min(n1, n2), max(n1, n2))
    #     while n2 >= n: #or (n1, n2) in train_edges:
    #         if (n1, n2) in train_edges:
    #             print((n1, n2), 'in train_edges:')
    #         n1 = nodes[randint(0, n-1)]
    #         n2 = nodes[randint(0, n-1)]
    #         (n1, n2) = (min(n1, n2), max(n1, n2))
    #     val_edges.append((n1, n2))

    y_val.extend([0]*(n_val_edges))
    
    ### From Giannis /!\
    val_indices = np.zeros((2,len(val_edges)))
    for i,edge in enumerate(val_edges):
        val_indices[0,i] = node_to_idx[edge[0]]
        val_indices[1,i] = node_to_idx[edge[1]]
    
    print('Returned G_train, train_edges, val_edges, y_val, nodes and node_to_idx objects')
    print('Loaded from', path[path.rfind('/')+1:], 'and with a training validation split ratio =', val_ratio)
    
    
    
    return G, G_train, train_edges, val_edges, val_indices, y_val, nodes, node_to_idx