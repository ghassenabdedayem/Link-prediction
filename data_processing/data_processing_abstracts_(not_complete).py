from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from time import time


def read_and_clean_abstracts (sample_length=-1, abstracts_path = 'https://www.lix.polytechnique.fr/~nikolentzos/files/aai/challenge/abstracts.txt'):
    t = time()
    abstracts = dict()
    abstracts_list = list()
    f = urlopen(abstracts_path)
    
    for i, line in enumerate(f):
        if i == sample_length:
            break
        node, abstract = str(line).lower().split('|--|')
        abstract = remove_stopwords(abstract)
        abstract = re.sub(r"[,.;@#?!&$()-]", " ", abstract)

        for word in abstract.split()[:-1]:
            #abstract = abstract.replace(word, stemmer.stem(word))
            abstract = abstract.replace(word, lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word), pos='s'), pos='a'), pos='n'), pos='v'), pos='r'))
        
        node = re.sub("[^0-9]", "", node)
        abstracts[int(node)] = abstract
        abstracts_list.append(abstract)
        
    print('Text loaded and cleaned in {:.0f} sec'.format(time()-t))
    return abstracts


class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.words_list = []
        self.num_words = 0
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.words_list.append(word)
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1
            
    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence.split()[:-1]:
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]

    def words(self):
        return self.words_list

def doc_counter (documents, word): #a function that return the number of documents containing a word
    counter = 0
    for i in documents:
        if word in documents[i]:
            counter += 1
    return counter


def replace_missing_words_with_random(abstract_list_of_words, wv_model):
    if len(abstract_list_of_words) == 0:
        embedded_abstracts[node] = []
    else:
        missing_word_mask = ~np.isin(abstract_list_of_words, wv_model.index_to_key)
        missing_word_indices = np.where(missing_word_mask)[0]
        missing_word_embeddings = np.random.rand(len(missing_word_indices), wv_model.vector_size)
        abstract_list_of_words_clean = np.compress(~missing_word_mask, abstract_list_of_words)
        result_embeddings = np.vstack((wv_model[abstract_list_of_words_clean], missing_word_embeddings))
    return result_embeddings


def word_embeddings_replace_missing_words_with_random(wv_model, abstracts):
    t = time()
    embedded_abstracts = dict()
    
    with Pool(processes=cpu_count()) as p:
        results = list(tqdm(p.imap(partial(replace_missing_words_with_random, wv_model=wv_model), abstracts), total=len(abstracts)))
        
    for node, result_embeddings in enumerate(results):
        embedded_abstracts[node] = result_embeddings
        
    print('list of words embeddings generated for each node in {:.0f} min'.format((time()-t)/60))
    return embedded_abstracts


def list_words_to_one_sentence_wv_vector(wv_model, sentences_list_words):
    t = time()

    embedded_abstracts = dict()
    for node, abstract in enumerate(sentences_list_words):
        #some abstracts are null
        cleaned_abstract=[]
        for word in abstract:
            try: 
                wv_model[word] #we try to find the word in the Vocabulary
                cleaned_abstract.append(word)
            except: pass
        if len(cleaned_abstract) > 0:
            wv_model[cleaned_abstract]
            embedded_abstracts[node] = np.mean(wv_model[cleaned_abstract], axis=0)
            for quartile in np.percentile(wv_model[cleaned_abstract], [25, 50, 75], axis=0):
                embedded_abstracts[node] = np.concatenate((embedded_abstracts[node], quartile), axis=0)
        else: #if the abstract text is null, we fill the embedded text vector by random numbers (it could help to prevent overfittiing)
            embedded_abstracts[node] = np.random.uniform(wv_model.vectors.min(), wv_model.vectors.max(), size=embedded_abstracts[0].shape)
        if (node % 10000 == 0):
            print('Procssed at {:.0f} % in {:.0f} min'.format((node / len(abstracts))*100, (time()-t)/60))
    print('nodes embeddings generated based on words embeddings in {:.0f}'.format((time()-t)/60)) #206 sec
    return embedded_abstracts, cleaned_abstract


def list_words_to_list_embeddings(sentences_list_words, wv_model):
    t = time()

    embedded_abstracts = dict()
    for node, abstract in enumerate(sentences_list_words):
        #some abstracts are null
        cleaned_abstract=[]
        for word in abstract:
            try: 
                wv_model[word] #we try to find the word in the Vocabulary
                cleaned_abstract.append(word)
            except: pass
        if len(cleaned_abstract) > 0:
            embedded_abstracts[node] = wv_model[cleaned_abstract]
        else: #if the abstract text is null, we fill the embedded text vector by random numbers (it could help to prevent overfittiing)
            embedded_abstracts[node] = np.zeros(shape=embedded_abstracts[0].shape)
        if (node % 10000 == 0):
            print('Procssed at {:.0f} % in {:.0f} min'.format((node / len(abstracts))*100, (time()-t)/60))
    print('nodes embeddings generated based on words embeddings in {:.0f} min in embedded_abstracts'.format((time()-t)/60)) #206 sec
    return embedded_abstracts, cleaned_abstract


def train_wv_on_vocab (voc, vector_size):
    t = time()
    wv_model = Word2Vec(vector_size=vector_size, window=5, min_count=1, sg=1, workers=8)
    wv_model.build_vocab(voc.sentences_list_words)
    wv_model.train(voc.sentences_list_words, total_examples=wv_model.corpus_count, epochs=5) 
    print('word2vec trained in {:.0f} sec'.format(time()-t)) #219 sec
    return wv_model

