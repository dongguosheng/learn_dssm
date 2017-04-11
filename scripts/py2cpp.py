# -*- coding: gbk -*-

import numpy as np
from gensim import models
from struct import pack
import sys

def to_binary_model(input, output_model, key_list):
    params = np.load(input)
    with open(output_model, 'wb') as fout_model:
        for k in key_list:
            param = params[k]
            param = param.astype('float32')
            fout_model.write(param.tobytes())
def gen_doc_params(doc_params):
    with open(doc_params, 'w') as fout:
        fout.write('n_dims\t4096,2048,2048,128\n')
def gen_query_params(query_params):
    n_feat_map = 100
    with open(query_params, 'w') as fout:
        fout.write('n_dim\t200\n')
        fout.write('n_dim_output\t128\n')
        fout.write('n_feat_map\t{}\n'.format(n_feat_map))
        fout.write('filter_widths\t1,2,3\n')
def w2v_py2cpp_bin(w2v_model_file, syn0, vocab_file, syn0_file):
    m = models.Word2Vec.load(w2v_model_file)
    m.syn0 = syn0
    with open(vocab_file, 'w') as fout_vocab, open(syn0_file, 'wb') as fout_syn0:
        for w in m.vocab:
            fout_vocab.write(w + '\t')
            fout_vocab.write(str(m.vocab[w].index) + '\n')
        fout_vocab.write("#" + '\t' + str(m.syn0.shape[0] - 1) + '\n')
        fout_syn0.write(pack('I', len(m.vocab) + 1))
        print 'm.layer1 size: %d' % m.layer1_size
        fout_syn0.write(pack('I', m.layer1_size))
        m.syn0 = m.syn0.astype('float32')
        fout_syn0.write(m.syn0.tobytes())

def py2cpp(cdssm_model, w2v_model, output_dir):
    print cdssm_model
    print w2v_model
    print output_dir
    if not output_dir.endswith('/'):
        output_dir += '/'
    doc_model = output_dir + 'cdssm_doc.model'
    doc_params = output_dir + 'cdssm_doc.params'
    # doc model
    print 'convert doc model'
    to_binary_model(cdssm_model, doc_model, key_list=['d_h1_w', 'd_h1_b', 'd_h2_w', 'd_h2_b', 'd_h3_w', 'd_h3_b'])
    gen_doc_params(doc_params)
    # query model
    print 'convert query model'
    query_model = output_dir + 'cdssm_query.model'
    query_params = output_dir + 'cdssm_query.params'
    to_binary_model(cdssm_model, query_model, key_list=['conv_1_w', 'conv_1_b', 'conv_2_w', 'conv_2_b', 'conv_3_w', 'conv_3_b', 'q_h1_w', 'q_h1_b'])
    gen_query_params(query_params)
    syn0 = np.load(cdssm_model)['syn0'].astype('float32')
    # w2v model
    print 'convert w2v model'
    vocab_file = output_dir + 'vocab.txt'
    syn0_file = output_dir + 'w2v_baike.bin'
    w2v_py2cpp_bin(w2v_model, syn0, vocab_file, syn0_file)

def main():
    if len(sys.argv) != 4:
        print 'python py2cpp.py cdssm_model w2v_model output_dir'
    else:
        cdssm_model = sys.argv[1]
        w2v_model = sys.argv[2]
        output_dir = sys.argv[3]
        py2cpp(cdssm_model, w2v_model, output_dir)

if __name__ == '__main__':
    main()
