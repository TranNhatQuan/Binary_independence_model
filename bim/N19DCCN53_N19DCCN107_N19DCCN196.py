"""
    N19DCCN153 - Tran Nhat Quan
    N19DCCN107 - Pham Minh Manh
    N19DCCN196 - Nguyen Duc Thinh
"""

from collections import defaultdict
from functools import reduce
from math import log
import string
from nltk.corpus import stopwords
from copy import deepcopy


def import_dataset():
    docs = None
    with open('dataset/doc-text.txt', 'r') as f:
        docs = f.readlines()

        f.close()
    return docs


def remove_stop_word(sentence):
    '''
        Remove stop word in a sentence
    '''
    english_stop_words = stopwords.words('english')
    new_docs = []
    for word in sentence:
        if word not in english_stop_words:
            new_docs.append(word)

    return new_docs


def remove_stop_words(docs):
    '''
        Remove stop word in many sentences
    '''
    english_stop_words = stopwords.words('english')
    new_docs = []
    for doc in docs:
        new_docs.append(
            [word for word in doc if word not in english_stop_words])
    return new_docs


def docs_processing(docs):
    '''
        Process and split sentences by ' '
    '''
    docs_after_process = []
    # removing digit and break line ('\n)
    pre_docs = list(filter(lambda y: not y.isdigit(),
                              map(lambda x: x[:-1], docs)))
    # print(pre_docs)
    # splitting allow forward flash
    table_remove_punctuation = str.maketrans(
        dict.fromkeys(string.punctuation))
    # print(table_remove_punctuation)
    docs_tmp = []
    for doc in pre_docs:
        if doc.endswith('/'):
            docs_after_process.append(' '.join(docs_tmp))
            docs_tmp.clear()
        else:
            docs_tmp.append(
                ' '.join([w.translate(table_remove_punctuation) for w in doc.split()]))
    # print(docs_after_process)
    docs_after_process = [w.split() for w in docs_after_process]
    
    return docs_after_process


def make_inverted_index(corpus):
    """
    This function builds an inverted index as an hash table (dictionary)
    where the keys are the terms and the values are ordered lists of
    docIDs containing the term.
    """
    corpus = remove_stop_words(corpus)
    
    index = defaultdict(set)
    for docid, article in enumerate(corpus):
        for term in article:
            index[term].add(docid)
    # print(index['equal'])
    return index

# ### Union of two posting lists


def posting_lists_union(pl1, pl2):
    """
        Returns a new posting list resulting from the union of the
        two lists passed as arguments.
        """
    
   

    # print(union)
    return list(set(pl1).union(set(pl2)))


# ## Precomputing weights


def DF(term, index):
    '''
    Function computing Document Frequency for a term.
    '''
    return len(index[term])


def IDF(term, index, corpus):
    '''
    Function computing Inverse Document Frequency for a term.
    '''
    return log(len(corpus)/DF(term, index))


def RSV_weights(corpus, index):
    '''
    This function precomputes the Retrieval Status Value weights
    for each term in the index
    '''
    N = len(corpus)
    w = {}
    for term in index.keys():
        p = DF(term, index)/(N+0.5)
        w[term] = IDF(term, index, corpus) + log(p/(1-p))
    return w


# ## BIM Class

class BIM():
    '''
    Binary Independence Model class
    '''

    def __init__(self, corpus):
        self.original_corpus = deepcopy(corpus)
        self.articles = corpus
        self.index = make_inverted_index(self.articles)
        self.weights = RSV_weights(self.articles, self.index)
        self.ranked = []
        self.query_text = ''
        

    def RSV_doc_query(self, doc_id, query):
        '''
        This function computes the Retrieval Status Value for a given couple document - query
        using the precomputed weights
        '''
        score = 0
        doc = self.articles[doc_id]
        
        for term in doc:
            if term in query:
                score += self.weights[term]
        
        return score

    def ranking(self, query):
        '''
        Auxiliary function for the function answer_query. Computes the score only for documents
        that are in the posting list of al least one term in the query
        '''

        docs = []
        for term in self.index:
            if term in query:
                
                docs = posting_lists_union(docs, self.index[term])
        # print(docs)
        scores = []
        for doc in docs:
            scores.append((doc, self.RSV_doc_query(doc, query)))

        self.ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        return self.ranked

    def recompute_weights(self, relevant_idx, query):
        '''
        Auxiliary function for relevance_feedback function and
        for the pseudo relevance feedback in answer_query function.
        Recomputes the weights, only for the terms in the query
        based on a set of relevant documents.
        '''

        relevant_docs = []
        for idx in relevant_idx:
            doc_id = self.ranked[idx-1][0]
            relevant_docs.append(self.articles[doc_id])

        N = len(self.articles)
        N_rel = len(relevant_idx)

        for term in query:
            if term in self.weights.keys():
                vri = 0
                for doc in relevant_docs:
                    if term in doc:
                        vri += 1
                p = (vri + 0.5) / (N_rel + 1)
                u = (DF(term, self.index) - vri + 0.5) / (N - N_rel + 1)
                self.weights[term] = log((1-u)/u) + log(p/(1-p))        

    def answer_query(self, query_text, n_retrived):
        '''
        Function to answer a free text query. Shows the first 30 words of the
        n_retrived most relevant documents. 
        Also implements the pseudo relevance feedback with k = 5
        '''

        self.query_text = query_text
        query = remove_stop_word(query_text.lower().split(' '))
        
        ranking = self.ranking(query)
        
        # pseudo relevance feedback
        i = 0
        new_ranking = []
        while i < 10 and ranking != new_ranking:
            self.recompute_weights([1, 2, 3, 4, 5], query)
            new_ranking = self.ranking(query)
            i += 1

        ranking = new_ranking

        for i in range(0, n_retrived):
            article = self.original_corpus[ranking[i][0]]
            text = " ".join(article)
            print(f"Article {i + 1}, score: {ranking[i][1]}")
            print(text, '\n')

        self.weights = RSV_weights(self.articles, self.index)

   

    


articles = docs_processing(import_dataset())

# print(articles[0])
bim = BIM(articles)
query = 'USE OF PROGRAMS IN ENGINEERING TESTING OF COMPUTERS'
bim.answer_query(query,2)

