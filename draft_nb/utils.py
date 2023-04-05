import networkx as nx
from random import randint
from random import random
from time import time
from random import choice
from gensim.models import Word2Vec
import networkx as nx
import numpy as np
from scipy.sparse import identity, diags
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss, accuracy_score
from unidecode import unidecode
import pandas as pd

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def text_to_list(text):
    return unidecode(text).split(',')

def intersection(lst1, lst2): # a function that returns the number of common items of two lists and 1 or 0 if there are common. This function will be used in add_authors_to_pairs to add this features to the pairs.
    lst3 = [value for value in lst1 if value in lst2]
    is_common = 1 if len(lst3)>0 else 0
    return len(lst3), is_common


def save_subgraph_in_file(nbr_nodes, source_path='../input_data/edgelist.txt', destination_path='../input_data/small_edgelist.txt'):
    G = nx.read_edgelist(source_path, delimiter=',', create_using=nx.Graph(), nodetype=int)
    G = G.subgraph(range(nbr_nodes))
    nx.write_edgelist(G, path=destination_path, delimiter=',')
    print(G.number_of_nodes(), 'nodes,', G.number_of_edges(), 'edges Graph extracted from', source_path[source_path.rfind('/')+1:])
    G = nx.read_edgelist(destination_path, delimiter=',', create_using=nx.Graph(), nodetype=int)
    print(G.number_of_nodes(), 'nodes,', G.number_of_edges(), 'edges Graph saved in', destination_path[destination_path.rfind('/')+1:])
    print(max(G.nodes))
    return




def read_train_val_graph(path='../input_data/edgelist.txt', val_ratio=0.1):
    G = nx.read_edgelist(path, delimiter=',', create_using=nx.Graph(), nodetype=int)
    nodes = list(G.nodes())
    n = G.number_of_nodes()
    m = G.number_of_edges()
    edges = list(G.edges())

    print('Number of nodes:', n, 'number of edges:', m,'in the Complete the set')

    node_to_idx = dict()
    for i, node in enumerate(nodes):
        node_to_idx[node] = i

    val_edges = list()
    G_train = G.copy()

    for edge in edges:
        if random() < val_ratio and edge[0] < n and edge[1] < n:
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
    
    print('Creating random val_edges...')
    for i in range(n_val_edges):
        n1 = nodes[randint(0, n-1)]
        n2 = nodes[randint(0, n-1)]
        (n1, n2) = (min(n1, n2), max(n1, n2))
        while n2 >= n: #or (n1, n2) in train_edges:
            if (n1, n2) in train_edges:
                print((n1, n2), 'in train_edges:')
            n1 = nodes[randint(0, n-1)]
            n2 = nodes[randint(0, n-1)]
            (n1, n2) = (min(n1, n2), max(n1, n2))
        val_edges.append((n1, n2))

    y_val.extend([0]*(n_val_edges))
    
    ### From Giannis
    val_indices = np.zeros((2,len(val_edges)))
    for i,edge in enumerate(val_edges):
        val_indices[0,i] = node_to_idx[edge[0]]
        val_indices[1,i] = node_to_idx[edge[1]]
    
    print('Returned G_train, train_edges, val_edges, y_val, nodes and node_to_idx objects')
    print('Loaded from', path[path.rfind('/')+1:], 'and with a training validation split ratio =', val_ratio)
    
    
    
    return G, G_train, train_edges, val_edges, val_indices, y_val, nodes, node_to_idx

def random_walk(G, node, walk_length):
    walk = [node]
  
    for i in range(walk_length-1):
        neibor_nodes = list(G.neighbors(walk[-1]))
        if len(neibor_nodes) > 0:
            next_node = choice(neibor_nodes)
            walk.append(next_node)
    walk = [node for node in walk] # in case the nodes are in string format, we don't need to cast into string, but if the nodes are in numeric or integer, we need this line to cast into string
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
        features_np.append(wv_model.wv[node])

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

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
 

def add_authors_to_pairs (pairs, authors):
    authors = pd.DataFrame(authors)

    pairs_df = pd.DataFrame(np.transpose(pairs)).rename(columns={0: "paper_1", 1: "paper_2"})
    pairs_df = pairs_df.merge(authors, left_on='paper_1', right_on='paper_id', how='left').rename(columns={'authors': "authors_1"})
    pairs_df = pairs_df.merge(authors, left_on='paper_2', right_on='paper_id', how='left').rename(columns={'authors': "authors_2"})
    pairs_df.drop(['paper_id_x', 'paper_id_y'], axis=1, inplace=True)

    pairs_df['nb_common_author'] = pairs_df.apply(lambda row: intersection(row['authors_1'], row['authors_2'])[0], axis=1)
    pairs_df['is_common_author'] = pairs_df.apply(lambda row: intersection(row['authors_1'], row['authors_2'])[1], axis=1)

    pairs_tensor = torch.LongTensor(np.transpose(pairs_df[["paper_1", "paper_2", 'is_common_author', 'nb_common_author']].values.tolist()))
    
    return pairs_tensor
