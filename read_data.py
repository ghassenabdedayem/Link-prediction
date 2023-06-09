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
from urllib.request import urlopen
from unidecode import unidecode
import re
from tqdm.notebook import tqdm
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem.wordnet import WordNetLemmatizer

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

    
    ### From Giannis /!\
    val_indices = np.zeros((2,len(val_edges)))
    for i,edge in enumerate(val_edges):
        val_indices[0,i] = node_to_idx[edge[0]]
        val_indices[1,i] = node_to_idx[edge[1]]
    
    print('Returned G_train, train_edges, val_edges, y_val, nodes and node_to_idx objects')
    print('Loaded from', path[path.rfind('/')+1:], 'and with a training validation split ratio =', val_ratio)
    
    
    
    return G, G_train, train_edges, val_indices, nodes, node_to_idx



def read_and_clean_abstracts (nodes, sample_length=-1, abstracts_path = 'https://www.lix.polytechnique.fr/~nikolentzos/files/aai/challenge/abstracts.txt'):
    # The sample_length variable is used to determine if we load only a subset of the data
    t = time()
    abstracts = dict()
    abstracts_list = list()
    f = urlopen(abstracts_path)
    
    for i, line in tqdm(enumerate(f)):
        if i == sample_length:
            break
        if i in nodes:
            node, abstract = str(line).lower().split('|--|')
            abstract = remove_stopwords(abstract)
            #abstract = re.sub(r"[,.;@#?!&$()-]", " ", abstract)
            abstract = re.sub(r"[^a-zA-Z0-9\s]", "", abstract)
            #abstract = re.sub(r"\\", " ", abstract)
            abstract = remove_stopwords(abstract)

            for word in abstract.split()[:-1]:
                #abstract = abstract.replace(word, stemmer.stem(word))
                abstract = abstract.replace(word, lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word), pos='s'), pos='a'), pos='n'), pos='v'), pos='r'))
            
            node = re.sub("[^0-9]", "", node)
            if i != int(node):
                print('i and node not the same', i, node)
            abstracts[int(node)] = abstract
            abstracts_list.append(abstract)
        
    print('Text loaded and cleaned in {:.0f} min'.format((time()-t)/60))
    return abstracts

def text_to_list(text):
    text = unidecode(text)
    text = re.sub(r"[^a-zA-Z\s.,]", "", text)
    return text.split(',')

def read_and_clean_authors (path = 'https://www.lix.polytechnique.fr/~nikolentzos/files/aai/challenge/authors.txt'):
    authors = pd.read_csv(urlopen('https://www.lix.polytechnique.fr/~nikolentzos/files/aai/challenge/authors.txt'), sep = '|', header=None)
    authors = authors.rename(columns={0: "paper_id", 2: "authors"})
    authors['authors'] = authors['authors'].apply(text_to_list)
    authors = authors[["paper_id", "authors"]]
    return authors

