# -*- coding: gbk -*-

import numpy as np
import theano
import theano.tensor as T
import random
from gensim import models
from theano.tensor.signal import downsample
from util import *
from conv_net_classes import LeNetConvPoolLayer
from datetime import datetime
import os

batch_size = 1
lr = 0.01
n_neg = 5
n_dim_query = 200  # maxpooling
n_dim_conv = 10
n_feature_map = 20
n_dim_doc = 4096
input_size = 25
PAD_IDX = -1

def gen_batch_index():
    query_index = []
    doc_index = []
    for i in range(batch_size):
        query_index.append(i)   # query
        for j in range(n_neg + 1):           
            doc_index.append(i * (n_neg + 1) + j)   # doc
    return (np.array([query_index]).reshape(batch_size, 1), np.array([doc_index]).reshape(batch_size, n_neg + 1))
query_index, doc_index = gen_batch_index()
print query_index
print doc_index

# load word2vec model
W2V_MODEL = models.Word2Vec.load('w2v_model/word2vec.model_200_2_5_ns.baike')
W2V_MODEL.syn0 = np.append(W2V_MODEL.syn0, np.zeros((1, 200), dtype='float32'), axis=0)
print W2V_MODEL.syn0[-1]
syn0 = theano.shared(value=W2V_MODEL.syn0, name='syn0', borrow=True)

# use rand word vec
# rng = np.random.RandomState(1234)
# syn0_weights = np.asarray(rng.uniform(low=-0.25, high=0.25, size=W2V_MODEL.syn0.shape), dtype=theano.config.floatX)
# syn0 = theano.shared(value=syn0_weights, name='syn0', borrow=True)

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        if W is None:
            W_vals = np.asarray(
                rng.uniform(
                    low  = -np.sqrt(6. / (n_in + n_out)),
                    high = np.sqrt(6. / (n_in + n_out)),
                    size = (n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_vals *= 4
            W = theano.shared(value=W_vals, name='W', borrow=True)
            
        if b is None:
            b_vals = np.zeros((n_out, ), dtype=theano.config.floatX)
            b = theano.shared(value=b_vals, name='b', borrow=True)
        self.W = W
        self.b = b
        linear_output = T.dot(self.input, self.W) + self.b
        self.output = linear_output if activation is None else activation(linear_output)
        self.params = [self.W, self.b]

class ConvLayer(object):
    def __init__(self, rng, idx_list, syn0=syn0):
        self.idx_list = idx_list[0]
        # fine-tune word2vec syn0
        self.syn0 = syn0
        # self.input=self.syn0[T.cast(idx_list.flatten(), dtype="int32")].reshape( (1, 1, 4, 200) )
        filter_w = n_dim_query
        filter_h_list = [1, 2, 3, 4]
        self.input=self.syn0[T.cast(self.idx_list.flatten(), dtype="int32")].reshape( (1, 1, input_size, filter_w) )
        conv_layers = []
        layer1_inputs = []
        for filter_h in filter_h_list:
            filter_shape = (n_feature_map, 1, filter_h, filter_w)  # (feature_maps, 1, filter_h, filter_w)
            pool_size = (input_size - filter_h + 1, 1)  # (img_h-filter_h+1, img_w-filter_w+1)
            conv_layer = LeNetConvPoolLayer(rng, input=self.input, image_shape=(1, 1, input_size, filter_w),
                                            filter_shape=filter_shape, poolsize=pool_size, non_linear='tanh')
            layer1_input = conv_layer.output.flatten(2)
            conv_layers.append(conv_layer)
            layer1_inputs.append(layer1_input)
        self.output = T.concatenate(layer1_inputs,1)
        self.params = [self.syn0]
        for conv_layer in conv_layers:
            self.params += conv_layer.params
        self.output_dim = n_feature_map * len(filter_h_list)

class CosineLayer(object):
    def __init__(self, query_rep, pos_neg_docs_rep, query_index, doc_index):
        self.output, _ = theano.scan(self.cal_cos, sequences=[query_index, doc_index], non_sequences=[query_rep, pos_neg_docs_rep])
        self.output = T.reshape(self.output, (batch_size, n_neg + 1))
        
    def cal_cos(self, query_idx, doc_idx, Q, D):
        norm_q = T.sqrt(T.sum(T.sqr(Q[query_idx]), 1))
        norm_d = T.sqrt(T.sum(T.sqr(D[doc_idx]), 1))
        return T.dot( Q[query_idx], D[doc_idx].T) / T.outer(norm_q, norm_d)  
        
class Representation(object):
    def __init__(self, rng, input, n_in, n_hidden1, n_hidden2, n_out, params_list=None, activation=T.tanh):
        if params_list is None:
            hidden_layer1 = HiddenLayer(rng, input, n_in, n_hidden1, activation=activation)
            hidden_layer2 = HiddenLayer(rng, hidden_layer1.output, n_hidden1, n_hidden2, activation=activation)
            hidden_layer3 = HiddenLayer(rng, hidden_layer2.output, n_hidden2, n_out, activation=activation)
        else:
            W1, b1, W2, b2, W3, b3 = params_list
            hidden_layer1 = HiddenLayer(rng, input, n_in, n_hidden1, W=W1, b=b1, activation=activation)
            hidden_layer2 = HiddenLayer(rng, hidden_layer1.output, n_hidden1, n_hidden2, W=W2, b=b2, activation=activation)
            hidden_layer3 = HiddenLayer(rng, hidden_layer2.output, n_hidden2, n_out, W=W3, b=b3, activation=activation)
        self.output = hidden_layer3.output
        self.params = hidden_layer1.params + hidden_layer2.params + hidden_layer3.params
        
class DSSM(object):
    def __init__(self, rng, query_idx_list, pos_neg_docs, query_index, doc_index, lr=lr):
        # query_embedding = EmbeddingLayer(rng, query_idx_list)
        query_conv = ConvLayer(rng, query_idx_list)
        # query_rep = Representation(rng, query_conv.output, n_dim_conv, 256, 256, 200)
        query_rep = HiddenLayer(rng, query_conv.output, query_conv.output_dim, 128, activation=T.tanh)
        query_output = query_rep.output
        pos_neg_docs_rep = Representation(rng, pos_neg_docs, n_dim_doc, 2048, 2048, 128)
        pos_neg_docs_output = pos_neg_docs_rep.output
        self.sim_rs = CosineLayer(query_output, pos_neg_docs_output, query_index, doc_index).output
        self.st = T.nnet.softmax(self.sim_rs)
        self.cost = -T.sum( T.log( self.st[:, 0] ) ) / self.st.shape[0]
        
        self.params = pos_neg_docs_rep.params + query_rep.params + query_conv.params
        self.grad_params = T.grad(self.cost, self.params)
        self.updates = [
            (param, param - lr * grad_param) for param, grad_param in zip(self.params, self.grad_params)
        ]

def compile_func():
    # Compile Theano Function
    T_query_idx_list = T.matrix('query_idx_list', dtype='int32')
    T_pos_neg_doc = T.matrix('pos_neg_doc', dtype='float32')
    T_query_index = T.matrix('query_index', dtype='int32')
    T_doc_index = T.matrix('doc_index', dtype='int32')
    # T_lr = T.scalar('lr', dtype='float32')
    rng = np.random.RandomState(1234)
    dssm = DSSM(rng, T_query_idx_list, T_pos_neg_doc, T_query_index, T_doc_index, lr=lr)

    train = theano.function(
        inputs  = [T_query_idx_list, T_pos_neg_doc, T_query_index, T_doc_index],
        outputs = [dssm.sim_rs, dssm.st, dssm.cost, 
                   dssm.params[0], dssm.params[1], dssm.params[2], dssm.params[3], dssm.params[4], dssm.params[5],
                   dssm.params[6], dssm.params[7], 
                   dssm.params[8], 
                   dssm.params[9], dssm.params[10], dssm.params[11], dssm.params[12], dssm.params[13], dssm.params[14]],
        updates = dssm.updates,
        allow_input_downcast=True,
        on_unused_input='ignore'
    )
    return train

class DataIter(object):
    def __init__(self, idx_file, feat_file):
        self.idx_file = idx_file
        self.feat_mat = np.load(feat_file)['feat_mat']
    def __iter__(self):
        with open(self.idx_file) as fin:
            for line in fin:
                query_idx_str, feat_idx_str = line.rstrip().split('\t')
                query_idx_list = [int(e) for e in query_idx_str.split(',')]
                feat_idx_list = [int(e) for e in feat_idx_str.split(',')]
                yield (query_idx_list, self.feat_mat[feat_idx_list])

dssm_train = compile_func()
def train_dssm(idx_file, feat_file, i, epoch_num):
    results = ''
    n_epoch = 1
    cnt = 0
    cost_each = 0
    model_dir = './400w_cnn_models_w2v_init_4-20/'
    for epoch in range(n_epoch):
        print 'epoch %d' % epoch
        cost = 0.0
        for query_idx, doc_feat in DataIter(idx_file, feat_file):
            # print i, query_idx
            if len(query_idx) == 0 or len(query_idx) > input_size - 4:
                continue
            cnt += 1
            # print query_idx_list
            n_words = len(query_idx)
            # print n_words
            query_idx = [int(e) for e in query_idx]
            n_left = (input_size - n_words) / 2
            n_right = input_size - n_words - n_left
            query_idx = [PAD_IDX for _ in range(n_left)] + query_idx + [PAD_IDX for _ in range(n_right)]
            query_idx = np.array( [ query_idx ] )
            # print query_idx
            results = dssm_train(query_idx, doc_feat, query_index, doc_index)
            cost += results[2]
            cost_each += results[2]
            # print 'cos: ' + str(results[0])
            # print 'st: ' + str(results[1])
            # print 'cost: ' + str(results[2])
            # print 'conv -> ' + str(np.asarray(results[12]).shape)
            # print 'conv -> ' + str(np.asarray(results[13]).shape)
            # print 'conv -> ' + str(np.asarray(results[14]).shape)
            # print 'conv -> ' + str(np.asarray(results[15]).shape)
            # print 'conv -> ' + str(np.asarray(results[16]).shape)
            # print 'conv -> ' + str(np.asarray(results[17]).shape)
            if cnt > 0 and cnt % 10000 == 0:
                print '[%s] Cost(1w): %f, lr: %f' % (str(datetime.now()), cost_each, lr)
                cost_each = 0
                # if lr > 0.0001:
                #     lr *= 0.999
        if epoch > 0 and epoch % 5 == 0:
            np.savez(model_dir + 'params_%d' % epoch, d_h1_w=results[3], d_h1_b=results[4], 
                                          d_h2_w=results[5], d_h2_b=results[6], 
                                          d_h3_w=results[7], d_h3_b=results[8],
                                          q_h1_w=results[9], q_h1_b=results[10],
                                          syn0=results[11],
                                          conv_1_w=results[12], conv_1_b=results[13], 
                                          conv_2_w=results[14], conv_2_b=results[15],
                                          conv_3_w=results[16], conv_3_b=results[17])
        print 'Cost(%d): %f' % (cnt, cost)
    np.savez(model_dir + 'params_%d_%d' % (epoch_num, i), d_h1_w=results[3], d_h1_b=results[4], 
                                  d_h2_w=results[5], d_h2_b=results[6], 
                                  d_h3_w=results[7], d_h3_b=results[8],
                                  q_h1_w=results[9], q_h1_b=results[10],
                                  syn0=results[11],
                                  conv_1_w=results[12], conv_1_b=results[13], 
                                  conv_2_w=results[14], conv_2_b=results[15],
                                  conv_3_w=results[16], conv_3_b=results[17])

def main():
    idx_file = './dssm_train_data/400w/train_data/{}_idx'
    feat_file = './dssm_train_data/400w/imgs_feat_valid/{}_valid.npz'
    num = len(os.listdir('./dssm_train_data/400w/imgs_feat_valid/'))
    n_epoch = 10
    for epoch in range(n_epoch):
        idx_list = range(num)
        random.shuffle(idx_list)
        print idx_list
        for i in idx_list:
            print feat_file.format(i)
            train_dssm(idx_file.format(i), feat_file.format(i), i, epoch)
    
if __name__ == '__main__':
    main()                                    
