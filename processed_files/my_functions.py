from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
from time import time
from tqdm.notebook import tqdm


def replace_words_with_embeddings(abstract_list_of_words, wv_model):
    if len(abstract_list_of_words) == 0:
        result_embeddings = []
    else:
        result_embeddings = wv_model[abstract_list_of_words]
    return result_embeddings


def words_embeddings_multithread_3(wv_model, abstracts):
    from time import time
    t = time()
    embedded_abstracts = dict()
    
    with Pool(processes=cpu_count()) as p:
        results = list(tqdm(p.imap(partial(replace_words_with_embeddings, wv_model=wv_model), abstracts), total=len(abstracts)))
        
    for node, result_embeddings in tqdm(enumerate(results)):
        embedded_abstracts[node] = result_embeddings
        
    print('list of words embeddings generated for each node in {:.0f} hours'.format((time()-t)/360))
    return embedded_abstracts


#         missing_word_mask = ~np.isin(abstract_list_of_words, wv_model.index_to_key)
#         missing_word_indices = np.where(missing_word_mask)[0]
#         missing_word_embeddings = np.random.rand(len(missing_word_indices), wv_model.vector_size)
#         abstract_list_of_words_clean = np.compress(~missing_word_mask, abstract_list_of_words)
#         result_embeddings = np.vstack((wv_model[abstract_list_of_words_clean], missing_word_embeddings))