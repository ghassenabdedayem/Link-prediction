import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def authors_to_sparse_mx (authors):
    # get the unique list of authors
    authors_lst_ppr = list(set([a for authors_list in (authors['authors']) for a in authors_list]))

    # create a mapping of author to index
    author_to_index = {author: i for i, author in (enumerate(authors_lst_ppr))}

    # create an empty sparse matrix
    nrows = len(authors)
    ncols = len(authors_lst_ppr)
    data = np.ones(nrows)
    row_ind = np.arange(nrows)
    col_ind = np.zeros(nrows)

    # fill in the sparse matrix with 1 where authors appear
    for i, authors_list in (enumerate(authors['authors'])):
        for author in authors_list:
            col_ind[i] = author_to_index[author]
            row_ind[i] = i
            data[i] = 1
    auth_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(nrows, ncols))
    return auth_matrix

