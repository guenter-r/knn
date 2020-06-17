import re, nltk, bs4
import pandas as pd
import numpy as np
from numpy.linalg import norm
import scipy as sp
from scipy.sparse import csr_matrix as csr
import random
from string import punctuation
from nltk.corpus import stopwords
from collections import defaultdict

file_path = 'C:\\path\\to\\both_files\\'

def tf(n=5, source='reuters'):
    # define number of relevant words n
    nltk.download('stopwords')
    punc = list(punctuation)
    stop_words = set(stopwords.words('english'))

    if source == 'reuters':
        file_path = './reut.html'

        def extract_reuters_news(path_file):
            file = open(path_file , 'r').read()
            soup = bs4.BeautifulSoup(file)
            all_bodies = [el.text for el in soup.find_all('content')] # replace body with content, otherwise bs4 wont find any body other then main body
            return all_bodies

        data = extract_reuters_news(file_path)
        NO_OF_ENTRIES = len(data)

    if source == 'imdb':
        data = pd.read_csv('./imdb.csv')
        NO_OF_ENTRIES = len(data) //2
        NO_OF_ENTRIES
        data = data.review.iloc[:NO_OF_ENTRIES]

    ## GLOBAL DICTS
    ## countains overall count of the term among all documents
    maximum_per_document = defaultdict(int) # maximum a term occurs in one doc. denominator for first equation
    number_docs_containing_term = defaultdict(int) ## How many documents contain a term --> denominator for second equation

    # bow_count will clean the input, create sets for every sentence and return a dict {word:count} & int(maximum count per doc)
    def bow_count(sentences):
        new_sentence = ''
        sentences = re.sub(r'<\s*br\s*\/s*>', '', sentences)
        sentences = re.sub(r'\n>', ' ', sentences)
        sentences = re.sub(r'\s+', ' ', sentences)
        sentences = re.sub(r'\.+\s*', '.', sentences)
        sentences = re.sub(r'who\'ll', 'who will', sentences)
        sentences = re.sub(r'[IiyouYousheSHE]\'ll', 'i will', sentences)
        sentences = re.sub(r'[wW]ouldn\'t', 'would not', sentences)
        sentences = re.sub(r'[mM]mustn\'t', 'must not', sentences)
        sentences = re.sub(r'[tT]hat\'s', 'that is', sentences)

        for el in sentences:
            if el.isspace() or el.isalpha() or el == '.': #or el.isnumeric():
                new_sentence += el.lower()

        new_sentences = new_sentence.split('.')
        new_sentences = [set(e for e in el.split() if e not in stop_words) for el in new_sentence.split('.')]
        temp_set = set()
        temp_count = defaultdict(int)

        for el in new_sentences:
            for l in el:

                temp_count[l] += 1
                temp_set.add(l)

        doc_max_term_count = [v for k,v in sorted(temp_count.items(), key= lambda x : x[1], reverse=True)][0]

        for term in temp_set:
            number_docs_containing_term[term] += 1

        return temp_count, doc_max_term_count ## returning a list of sets, where every set is a sentence

    docs = []
    for i,doc in enumerate(data):
        counted_terms, m = bow_count(doc)
        maximum_per_document[i] = m
        docs.append(counted_terms)

    def get_tf_idf(w,doc_index):
        tf_idf = {}
        tf = {}

        for k,v in w.items():
            tf[k] = v / maximum_per_document[doc_index]
            ni = number_docs_containing_term[k]
            from math import log
            idf = log(NO_OF_ENTRIES / ni)
            tf_idf[k] = tf[k] * idf

        return tf_idf

    result = []
    words_vector = set()

    for ind, words in enumerate(docs):
        ranked_words = get_tf_idf(words, ind)
        top_n = {k:v for k,v in sorted(ranked_words.items(), key=lambda x: (-x[1]) )[:n] }
        result.append(top_n)
        top_set = set([el for el in top_n.keys()])
        words_vector |= top_set

    all_word_vector = np.zeros(len(words_vector))

    ## global list that will then be stacked to sparse matrix
    similarity_to_stack = []
    ## create a similariy vector of all words -> which is then used to create per-result-datapoint-vectors --> stacking those to matrix
    def similarity_vector(words):
        doc_vec = all_word_vector.copy()

        for i,word in enumerate(words_vector):
            if word in words:
                doc_vec[i] = 1

        doc_vec_norm = np.linalg.norm(doc_vec)
        doc_vec /= doc_vec_norm

        return doc_vec # which is a vector that is normalized and can be compared to all others

    # iterate over all entries in result (dictonary with n entries of top words)
    for progress,r in enumerate(result):
        similarity_to_stack.append(similarity_vector(list(r.keys())))
        if progress%1000 == 0:
            print(progress, ' records analysed.')

    # stack all results similarity vectors to one matrix
    m = csr(np.vstack(similarity_to_stack))
    m.shape

    # print the stacked matrix:
    # m: number of datapoints, n: number of words in all_word_vector
    plt.spy(m, marker='.', markersize=1)

    # create a similarity vector, by multiplying each element with all others
    ref = m.dot(m.T).toarray()

    return ref, data

ref,data = tf()




# quick & dirty: identify similar articles/blogs/reviews
for ind,ary in enumerate(ref):
    for i,el in enumerate(ary):
        if el > .6 and ind != i :
            print(ind, ' and ', i)
            break
#
# comparing_matrix = m.dot(m.T)
#
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9, 9))
# plt.spy(comparing_matrix, marker='.', markersize=1)
# #comparing_matrix[20].toarray()
#
# random_entry = np.random.randint(0,NO_OF_ENTRIES-1)
# print(random_entry) #508
# #random_entry = 3
# similar = [i for i, el in enumerate(comparing_matrix[random_entry].toarray()[0]) if el > .2 and i != random_entry]
# print(similar)
#
# ### Interesting for Medium blog!!!! NOT a duplicate
# ## data for reuters
# data[924]
# data[369]
#
# ## data for imdb
# data[random_entry]
# data[similar[0]]
# result[random_entry]
# result[similar[0]]
#
# # from scipy.sparse.linalg import svds
# # from numpy import abs
# # u, s, v = svds(comparing_matrix, k=1, which='LM', return_singular_vectors=True)
# # s,u,v = s, abs(u.reshape(comparing_matrix.shape[0])), abs(v.reshape(comparing_matrix.shape[1]))
#
# for i,el in enumerate(words_vector):
#     if i == 644:
#         print(el)
# np.argsort(abs(u.reshape(comparing_matrix.shape[0])))[::-1]
#
# similar.append(random_entry)
#
#
# import os
# os.getcwd()
#
# with open('test.txt', 'a') as f:
#     for el in similar:
#         f.write(str(data[el]))
#         f.write('\n\n')
#         #f.write(str(doc_and_words[el]))
#         f.write('\n\n ########################################\n')
