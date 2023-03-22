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


def save_subgraph_in_file(nbr_nodes, source_path='../input_data/edgelist.txt', destination_path='../input_data/small_edgelist.txt'):
    G = nx.read_edgelist(source_path, delimiter=',', create_using=nx.Graph(), nodetype=int)
    G = G.subgraph(range(nbr_nodes))
    nx.write_edgelist(G, path=destination_path, delimiter=',')
    print(G.number_of_nodes(), 'nodes,', G.number_of_edges(), 'edges Graph extracted from', source_path[source_path.rfind('/')+1:])
    G = nx.read_edgelist(destination_path, delimiter=',', create_using=nx.Graph(), nodetype=int)
    print(G.number_of_nodes(), 'nodes,', G.number_of_edges(), 'edges Graph saved in', destination_path[destination_path.rfind('/')+1:])
    print(max(G.nodes))
    return

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def read_train_val_graph(path='../input_data/edgelist.txt', val_ratio=0.1):
    G = nx.read_edgelist(path, delimiter=',', create_using=nx.Graph(), nodetype=int)
    nodes = list(G.nodes())
    n = G.number_of_nodes()
    m = G.number_of_edges()
    edges = list(G.edges())

    print('Number of nodes:', n, 'number of edges:', m,'in All the set')

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
    
    print('Returned G_train, train_edges, val_edges, y_val, nodes and node_to_idx objects')
    print('Loaded from', path[path.rfind('/')+1:], 'and with a training validation split ratio =', val_ratio)
    
    return G, G_train, train_edges, val_edges, y_val, nodes, node_to_idx

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

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def train_model(model, optimizer, features, adj, indices, y, epochs):
    # Train model
    model.train()
    start_time = time()
    for epoch in range(epochs):
        t = time()
        optimizer.zero_grad()
        rand_indices = torch.randint(0, features.size(0), (indices.size(0),indices.size(1)), device=adj.device)# We take random indices each time we run an epoch
        pairs = torch.cat((indices, rand_indices), dim=1) # Concatenate the edges indices and random indices.   
        output = model(features, adj, pairs) # we run the model that gives the output.
        loss_train = F.nll_loss(output, y) # we are using nll_loss as loss to optimize, we store it in loss_train. We compare to y which is stable and contains the tag ones and zeros.
        acc_train = accuracy_score(torch.argmax(output, dim=1).detach().cpu().numpy(), y.cpu().numpy())# just to show it in the out put message of the training
        loss_train.backward() # The back propagation ? --> Computes the gradient of current tensor w.r.t. graph leaves
        optimizer.step() # Performs a single optimization step (parameter update).

        if epoch % 5 == 0:
            print('Epoch: {:03d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'time: {:.4f} s'.format(time() - t),
                 'total_time: {} min'.format(round((time() - start_time)/60)))

    print("Optimization Finished in {} min!".format(round((time() - start_time)/60)))
    return model
    