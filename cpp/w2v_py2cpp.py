# -*- coding: gbk -*-

import numpy as np
import sys
from gensim import models
from struct import pack

def w2v_py2cpp(w2v_model_file, output):
    m = models.Word2Vec.load(w2v_model_file)
    with open(output, 'w') as fout:
        for w in m.vocab:
            fout.write(w + '\t')
            fout.write(' '.join(str(e) for e in m[w]) + '\n')
def w2v_py2cpp_bin(w2v_model_file, vocab_file, syn0_file):
    m = models.Word2Vec.load(w2v_model_file)
    m.syn0 = np.fromfile('./data/syn0_cdssm', dtype='float32')
    with open(vocab_file, 'w') as fout_vocab, open(syn0_file, 'wb') as fout_syn0:
        for w in m.vocab:
            fout_vocab.write(w + '\t')
            fout_vocab.write(str(m.vocab[w].index) + '\n')

        fout_syn0.write(pack('I', len(m.vocab)))
        print 'm.layer1 size: %d' % m.layer1_size
        fout_syn0.write(pack('I', m.layer1_size))
        m.syn0 = m.syn0.astype('float32')
        fout_syn0.write(m.syn0.tobytes())

def main():
    model_file = 'word2vec.model_200_2_5_ns.baike'

    vocab_file = './data/query_side/vocab.txt'
    syn0_file = './data/query_side/w2v_baike.bin'
    w2v_py2cpp_bin(model_file, vocab_file, syn0_file)

if __name__ == '__main__':
    main()
