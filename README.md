# Link prediction in Graph
 
 ## Introduction
 The aim of this project is to apply advanced Deep Learning techniques to solve a link prediction problem. The problem involves analyzing a graph network with over a hundred thousand nodes, representing scientific papers. The edges in the graph represent links between scientific papers, indicating if one paper cites another. In addition to the graph structure, we are given the abstracts and authors of each paper.
 
 ## Data exploration
 The files are 
 ### Graph
 - Number of nodes: 138,499
 - Number of the edges: 1,091,955

<img width="450" alt="image" src="https://github.com/ghassenabdedayem/Link-prediction/assets/56557440/0fef38c6-6bd4-40ea-82b1-d59bcf5a1b52">

Figure 1: 1,000 first nodes of the graph

<img width="450" alt="image" src="https://github.com/ghassenabdedayem/Link-prediction/assets/56557440/2da14f09-c1c4-4ee5-8abd-495c3239d4e2">

Figure 2: 1,000 second nodes of the graph

### Abstracts
#### Before normalization
Characteristics of the abstracts before normalization:
- Longest abstract: 1,462 words
- Number of words: 345,570 words
- Empty abstracts: 7,249
- Long abstracts more than 128 words: 82,394
- Very long abstracts more than 256 words: 4,171
- Huge abstracts more than 512 words: 65
#### After text normalization
We normalized the text by removing special characters and applying Lemmatization using the WordNet lemmatizer three times with different part-of-speech tags: the first time as a noun,
the second as an adjective, the third as a verb. The output of the text cleaning of the abstracts has the below characteristics:
- Longest sentence = 915 words
- Number of words = 188,891 words
- Empty abstracts = 7249
- Long abstracts more than 128 words = 11,217
- Very long abstracts more than 256 words = 95
- Huge abstracts more than 512 words = 12

### Authors
Authors are cleaned by removing spacial characters and removing accents from letters with accents (using unicode function) and saving them as a list of cleaned authors for each paper.

<img width="400" alt="image" src="https://github.com/ghassenabdedayem/Link-prediction/assets/56557440/95721e9a-a5e3-4bbb-aaf2-bfbd01058550">


## Features engineering
### Training and validation split
function: read_train_val_graph
Using the function read_train_val_graph we load the graph and then we remove randomly some edges (10% of the edges of the graph) that will represent the validation set. The remaining edges are kept as the training set. This function also creates pairs (edges and negative edges) labels y and y_val with 1 to indicate that there is an edge between the pair of nodes and 0 if the pair of nodes is not connected.
### Graph adjacency matrix
Then create and normalize an adjacency matrix from the graph using the functions create_adjacency and normalize_adjacency. The adjacency matrix is a fundamental representation of a graph that encodes the relationships between its nodes. It is particularly useful in the context of graph neural networks (GNNs) because it provides a concise and efficient way to store and manipulate the graph structure. However, the adjacency matrix can have varying degrees of sparsity, which can lead to numerical instability and poor performance when used directly in GNNs. To address this, we normalized the adjacency matrix.
### Random walks features
Creating random walks features allows us to capture more nuanced relationships between nodes in the graph, as the random walks provide a way to sample different paths and capture the context in which nodes appear. The dimensionality of the resulting embeddings is reduced using word2vec, allowing us to efficiently incorporate these features into our model. Overall, this technique provides a way to enrich our model with more information about the graph structure, potentially improving its performance.
### Authors features
The lists of authors by paper are transformed into a one hot vector sparse representation. And assuming that only authors occusing on at least two papers have effect on the link prediction problem, we got rid of authors occuring only once. This sparse representation is then densified using TruncatedSVD.
### Abstracts text






