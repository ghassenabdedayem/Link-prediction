# Link prediction in Graph
 
 ## Introduction
 The aim of this project is to apply advanced Deep Learning techniques to solve a link prediction problem. The problem involves analyzing a graph network with over a hundred thousand nodes, representing scientific papers. The edges in the graph represent links between scientific papers, indicating if one paper cites another. In addition to the graph structure, we are given the abstracts and authors of each paper.
 
 <p align="center"><img width="320" alt="image" src="https://github.com/ghassenabdedayem/Link-prediction/assets/56557440/a320bce2-cd87-46c7-89d4-8d9d62e72086"></p>
 <p align="center">Figure 1: link prediction in graph</p><br>
 
 ## Data exploration
 The files are 
 ### Graph
 The size of the graph and its characteristics are:
 - Number of nodes: 138,499
 - Number of the edges: 1,091,955

<p align="center"><img width="400" alt="image" src="https://github.com/ghassenabdedayem/Link-prediction/assets/56557440/0fef38c6-6bd4-40ea-82b1-d59bcf5a1b52"></p>
<p align="center">Figure 2: 1,000 first nodes of the graph</p><br>
 
<p align="center"><img width="400" alt="image" src="https://github.com/ghassenabdedayem/Link-prediction/assets/56557440/2da14f09-c1c4-4ee5-8abd-495c3239d4e2"></p>
<p align="center">Figure 3: 1,000 second nodes of the graph</p><br>


### Abstracts
#### Before normalization:
Characteristics of the abstracts before normalization:
- Longest abstract: 1,462 words
- Number of words: 345,570 words
- Empty abstracts: 7,249
- Long abstracts more than 128 words: 82,394
- Very long abstracts more than 256 words: 4,171
- Huge abstracts more than 512 words: 65
#### After normalization:
We normalized the text by removing special characters and stop words and by applying Lemmatization using the WordNet lemmatizer (three times with different part-of-speech tags: the first time as a noun,
the second as an adjective, the third as a verb). The output of the text cleaning of the abstracts has the below characteristics:
- Longest sentence = 915 words
- Number of words = 188,891 words
- Empty abstracts = 7,249
- Long abstracts more than 128 words = 11,217
- Very long abstracts more than 256 words = 95
- Huge abstracts more than 512 words = 12

### Authors
Authors are cleaned by removing spacial characters and removing accents from letters with accents (using unicode function) and saving them as a list of cleaned authors for each paper.

<p align="center"><img width="400" alt="image" src="https://github.com/ghassenabdedayem/Link-prediction/assets/56557440/95721e9a-a5e3-4bbb-aaf2-bfbd01058550"></p>
<p align="center">Figure 4: extract from the cleaned authors dataset</p><br>

## Features engineering
In this section we will explain the features engineering from the Graph, the Authors and the Abstracts. These precalculated features are then stored on Google Cloud Platform (GCP) and used directly by the model of next section.
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
Many techniques have been applied to get abstracts embeddings. These resulted embeddings are then texted as input in the implemeted model.
#### TF-IDF:
From the Python library scikit-learn we used the function TfidfVectorizer to generate the tf-idf matrix with normalization and log frequency. Normalization ensures that the values in the matrix are scaled between 0 and 1, while log frequency scales down the importance of high-frequency words. This approach results in a sparse matrix, where each row represents a document and each column represents a term in the corpus vocabulary. Then we applied the dimensionality reduction by using the TruncatedSVD function.

*Another variant is made by concatenating the authors to the cleaned abstracts of the papers and then creating the logarithmic TF-IDF matrix.*

*And because our task is a link prediction between abstracts, another variant has been tested by keeping only words that occur in at least two abstracts.*

#### Word2vec:
A local word2vec model is trained on the vocabulary of the abstracts to ensure that each word had its embedded representation. Then I applied mean pooling across the word embeddings of the abstracts to obtain a single representation for each abstract. The size of the output vector of this approach was set to 300.
<p align="center"><img width="700" alt="image" src="https://github.com/ghassenabdedayem/Link-prediction/assets/56557440/9a37152b-faf6-4336-9baf-e50d7c3cc252"></p>
<p align="center">Figure: word2vec traning and abstracts encoding</p><br>

#### Goog300 word2vec:
The pre-trained Google News 300-dimensional word2vec model (goog300) was used to obtain words embeddings. Each word in the abstracts' vocabulary is checked against the vocabulary of the pre-trained model, and the corresponding word embeddings for those found are retreived. For those not found, they are just omitted. Then, averaging the word embeddings for all words in each abstract allowed us to obtain a single vector representation that captured its semantic meaning.
*NB. The unfound words in the goog300 vocab were just ommitted and not replaced by random vectors because of the required calculation time.*<br>
#### BART:
The pre-trained BART (Bidirectional and Auto-Regressive Transformer) model and tokenizer provided by the Hugging Face transformers library is a transformer-based sequence-to-sequence model that excels at tasks such as text generation, summarization, and translation. Our implementation employed the get_bart_embeddings function, which generates embeddings using the BART model on input text. By default, the maximum text length accepted by the BART model is 1024 tokens. This can be changed by modifying the max_length parameter in the encoded_input dictionary when tokenizing the text. If the input text is shorter than 1024 tokens, it is padded with zeros up to the maximum length of 1024 tokens. The resulting tensor is obtained by averaging the hidden states of the last layer of the BART model and flattening it into a one-dimensional vector.<br>
#### BERT:
We used the pretrained BERT model ‘bert-base-nli-mean-tokens’ which is more suitable for text similarity. The model 'bert-base-nli-mean-tokens' refers to a specific variant of the BERT (Bidirectional Encoder Representations from Transformers) model that has been fine-tuned for natural language inference (NLI) tasks. This variant is trained to generate sentence-level embeddings by taking the mean of the token embeddings produced by the BERT model. The specificity of 'bert-base-nli-mean-tokens' lies in its ability to capture the contextual information of sentences and generate fixed-length vector representations (embeddings) that encode the meaning of the entire sentence.

## Model
<p align="center"><img width="900" alt="image" src="https://github.com/ghassenabdedayem/Link-prediction/assets/56557440/a80766fa-a085-4982-90b1-d8b989899ae2"></p>
<p align="center">Figure: architecture with sparse authors, abstract features and random walks

 
<p align="center"><img width="900" alt="image" src="https://github.com/ghassenabdedayem/Link-prediction/assets/56557440/00515438-4b51-4369-85ad-b0c6b2d471c7"></p>
<p align="center">Figure: architecture with TF-IDF with authors and walks features
 

<p align="center"><img width="500" alt="image" src="https://github.com/ghassenabdedayem/Link-prediction/assets/56557440/af426332-899a-4b19-a782-3e104d6a9775"></p>
<p align="center">Figure: log loss over epoch




