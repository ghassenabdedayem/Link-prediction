import networkx as nx
from random import randint
from random import random
from time import time
from random import choice
from gensim.models import Word2Vec
import networkx as nx
import numpy as np
from scipy.sparse import identity, diags


def read_train_val_graph(path='../input_data/edgelist.txt', val_ratio=0.1):
    G = nx.read_edgelist(path, delimiter=',', create_using=nx.Graph(), nodetype=int)
    nodes = list(G.nodes())
    print('max of nodes=',max(nodes))
    n = G.number_of_nodes()
    print('number of nodes of G', n)
    m = G.number_of_edges()
    edges = list(G.edges())

    print('Number of nodes of total set:', n)
    print('Number of edges of total set:', m)

    node_to_idx = dict()
    for i, node in enumerate(nodes):
        node_to_idx[node] = i

    val_edges = list()
    G_train = G

    for edge in edges:
        if random() < val_ratio:
            val_edges.append(edge)

    # We remove the val edges from the graph G
    for edge in val_edges:
        G_train.remove_edge(edge[0], edge[1])

    n = G_train.number_of_nodes()
    m = G_train.number_of_edges()
    train_edges = list(G_train.edges())

    print('Number of nodes of training set:', n)
    print('Number of edges of training set:', m)
    print('max of nodes of G_train', max(G_train.nodes))

    y_val = [1]*len(val_edges)

    n_val_edges = len(val_edges)

    # Create random pairs of nodes (testing negative edges)
#     for i in range(n_val_edges):
#         n1 = nodes[randint(0, n-1)]
#         n2 = nodes[randint(0, n-1)]
#         (n1, n2) = (min(n1, n2), max(n1, n2))
#         val_edges.append((n1, n2))

    # Remove from val_edges edges that exist in both train and val
    # for edge in list(set(val_edges) & set(train_edges)):
    #     val_edges.remove(edge)

    # n_val_edges = len(val_edges) # - len(y_val) #because we removed from val_edges edges that exist in both
    y_val.extend([0]*(n_val_edges))
    print('Returned G_train, train_edges, val_edges, y_val and nodes objects')
    print('Loaded from', path[path.rfind('/')+1:], 'and with a training validation split ratio =', val_ratio)
    return G_train, train_edges, val_edges, y_val, nodes

def random_walk(G, node, walk_length):
    walk = [node]
  
    for i in range(walk_length-1):
        neibor_nodes = list(G.neighbors(walk[-1]))
        if len(neibor_nodes) > 0:
            next_node = choice(neibor_nodes)
            walk.append(next_node)
    walk = [str(node) for node in walk] # in case the nodes are in string format, we don't need to cast into string, but if the nodes are in numeric or integer, we need this line to cast into string
    return walk


def generate_walks(G, num_walks, walk_length):
  # Runs "num_walks" random walks from each node, and returns a list of all random walk
    t = time()
    print('Start generating walks....')
    walks = list()  
    for i in range(num_walks):
        for node in G.nodes():
            walk = random_walk(G, node, walk_length)
            walks.append(walk)
        #print('walks : ', walks)
    print('Random walks generated in in {}s!'.format(round(time()-t)))
    return walks


def apply_word2vec_on_features(features, nodes, vector_size=128, window=5, min_count=0, sg=1, workers=8):
    t = time()
    print('Start applying Word2Vec...')
    wv_model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, sg=sg, workers=workers)
    wv_model.build_vocab(features)
    wv_model.train(features, total_examples=wv_model.corpus_count, epochs=5) 
    print('Word2vec model trained on features in {} min!'.format(round((time()-t)/60)))
    features_np = []
    for node in nodes:
        features_np.append(wv_model.wv[str(node)])

    features_np = np.array(features_np)
    print(features_np.shape, 'features numpy array created in {} min!'.format(round((time()-t)/60)))
    return features_np

def normalize_adjacency(A):
    n = A.shape[0]
    A = A + identity(n)
    degs = A.dot(np.ones(n))
    inv_degs = np.power(degs, -1)
    D_inv = diags(inv_degs)
    A_hat = D_inv.dot(A)
    return A_hat

def create_and_normalize_adjacency(G):
    adj = nx.adjacency_matrix(G) # Obtains the adjacency matrix of the training graph
    adj = normalize_adjacency(adj)
    print('Created a normalized adjancency matrix of shape', adj.shape)
    indices = np.array(adj.nonzero()) # Gets the positions of non zeros of adj into indices
    print('Created indices', indices.shape, 'with the positions of non zeros in adj matrix')
    return adj, indices
