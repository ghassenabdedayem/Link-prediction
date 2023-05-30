import numpy as np
import scipy.sparse as sp

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
    #adj = normalize_adj(adj)
    print('Created a normalized adjancency matrix of shape', adj.shape)
    indices = np.array(adj.nonzero()) # Gets the positions of non zeros of adj into indices
    print('Created indices', indices.shape, 'with the positions of non zeros in adj matrix')
    return adj, indices


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

