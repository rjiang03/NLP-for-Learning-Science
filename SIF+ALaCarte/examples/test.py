import numpy as np
import sys
import os
sys.path.append('../src')
import data_io, params, SIF_embedding
from t_sne_plot import plot_embedding
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

wordfile = '../data/glove.6B.300d.txt' # word vector file, can be downloaded from GloVe website
weightfile = '../auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme

(words, We) = data_io.getWordmap(wordfile)
# load word weights
word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
weight4ind = data_io.getWeight(words, word2weight)

#import data
list_set = []
list_speaker = []
orig = []
dic = {'TA': 0, 'S1': 1, 'S2': 2, 'S3': 3, 'S4': 4, 'S5': 5, 'S6': 6, 'S7': 7}

path = "../data/txt data"
files= os.listdir(path)
doc_range = [0]

for file in files:
     if not os.path.isdir(file):
          f = open(path+"/"+file);
          next(f)
          next(f)
          doc_range.append(doc_range[-1])
          list_set.append([])
          list_speaker.append([])
          line = f.readline()
          while line:
              if len(line[4:-2].split()) != 1 and line is not '':
                  doc_range[-1] += 1
                  list_set[-1].append(line[4:-2])
                  orig.append(line[4:-2])
                  list_speaker[-1].append(line[0:2])
              line = f.readline()


x, m = data_io.sentences2idx(orig, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
w = data_io.seq2weight(x, m, weight4ind) # get word weights

matrixfile = open('840B.300d.bin', 'r')
M = np.fromfile(matrixfile, dtype=np.float32)
A = np.reshape(M,(300,300))

params = params.params()
params.rmpc = rmpc
# get SIF embedding
embedding = SIF_embedding.SIF_embedding_alac(We, x, w, 5, A, params) # embedding[i,:] is the embedding for sentence i

X_embedded = TSNE(n_components=2).fit_transform(embedding)