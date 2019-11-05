import sys
sys.path.append('../src')
import data_io, params, SIF_embedding
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# input
wordfile = '../data/glove.6B.300d.txt' # word vector file, can be downloaded from GloVe website
weightfile = '../auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme
#sentences = ['this is an example sentence asdascasx', 'this is also an example', 'this is another sentence that is slightly longer','hello my best friend']


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

list_set = []
list_speaker = []
orig = []
dic = {'TA': 0, 'S1': 1, 'S2': 2, 'S3': 3, 'S4': 4, 'S5': 5, 'S6': 6, 'S7': 7}

f = open("../data/Gold_4_4_Tran.txt")  # 返回一个文件对象
next(f)
next(f)
line = f.readline()  # 调用文件的 readline()方法
while line:
    orig.append(line)
    if len(line[4:-2].split()) != 1 and line is not '':
        list_set.append(line[4:-2])
        list_speaker.append(dic.get(line[0:2], 8))
    # print(line, end = '')　      # 在 Python 3 中使用
    line = f.readline()

f.close()

def label(N):
    lab = np.zeros(N)
    for i in range(5):
        lab[i*N//5 : ] += 1
    return lab

(words, We) = data_io.getWordmap(wordfile)
# load word weights
word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word
# load sentences
x, m = data_io.sentences2idx(list_set, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
w = data_io.seq2weight(x, m, weight4ind)
matrixfile = open('840B.300d.bin', 'r')
M = np.fromfile(matrixfile, dtype=np.float32)
A = np.reshape(M,(300,300))
embedding = SIF_embedding.SIF_embedding_pc(We, x, w)



X_embedded = TSNE(n_components=2).fit_transform(embedding)
