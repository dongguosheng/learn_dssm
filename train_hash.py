# -*- coding: gbk -*-

import numpy as np
import theano
import theano.tensor as T
from dssm import Representation, DataIter
from dssm import query_index, doc_index
from lr_policy import lr_decay

hash_input_dim = 200
n_bit = 48
lr = 0.01
batch_size = 1
n_neg = 5
n_dim_query = 200
n_dim_doc = 4096

def init_hash_params(rng, n_in, n_bit):
    params = []
    W_vals = np.asarray(
            rng.randn(n_in, n_bit),
            dtype=theano.config.floatX
            )
    W_vals *= 0.01
    params.append(theano.shared(value=W_vals, name='W', borrow=True))
    b_vals = np.asarray(0.01 * rng.randn(n_bit, ), dtype=theano.config.floatX)
    params.append(theano.shared(value=b_vals, name='b', borrow=True))
    return params

rng = np.random.RandomState(1234)
hash_params = init_hash_params(rng, hash_input_dim, n_bit)

class HashLaryer(object):
    def __init__(self, input, hash_params):
        self.W, self.b = hash_params
        self.output = T.dot(input, self.W) + self.b
        self.hash_params = hash_params

class DSSMHash(object):
    def __init__(self, rng, querys, pos_neg_docs, query_index, doc_index, lr, params_list):
        # query_rep = Representation(rng, querys).output
        query_output = querys
        pos_neg_docs_rep = Representation(rng, pos_neg_docs, n_dim_doc, 2048, 2048, 200, params_list=params_list, activation=T.nnet.sigmoid)
        pos_neg_docs_output = pos_neg_docs_rep.output
        hash_layer_query = HashLaryer(query_output, hash_params)
        hash_layer_doc = HashLaryer(pos_neg_docs_output, hash_params)
        relax_bits_query = hash_layer_query.output
        relax_bits_doc = hash_layer_doc.output
        
        self.sim_rs, _ = theano.scan(self.cal_dot, sequences=[query_index, doc_index], non_sequences=[relax_bits_query, relax_bits_doc])
        self.sim_rs = T.reshape(self.sim_rs, (batch_size, n_neg + 1))
        
        self.st = T.nnet.softmax(self.sim_rs)
        self.cost = -T.sum( T.log( self.st[:, 0] ) ) / self.st.shape[0]
        
        self.params = pos_neg_docs_rep.params + hash_params
        self.params_update = hash_params   # only update hash layer weight
        self.grad_params_hash = T.grad(self.cost, self.params_update)
        self.updates = [
            (param, param - lr * grad_param) for param, grad_param in zip(self.params, self.grad_params_hash)
        ]

    def cal_dot(self, query_idx, doc_idx, Q, D):
  	    return T.dot( Q[query_idx], D[doc_idx].T) / 2

def compile_func():
    # Compile Theano Function
    T_query = T.matrix('query', dtype='float32')
    T_pos_neg_doc = T.matrix('pos_neg_doc', dtype='float32')
    T_query_index = T.matrix('query_index', dtype='int32')
    T_doc_index = T.matrix('doc_index', dtype='int32')
    T_lr = T.scalar('lr', dtype='float32')
    rng = np.random.RandomState(1234)
    params = np.load('params_50.npz')

    params_list = [params['d_h1_w'], params['d_h1_b'], params['d_h2_w'], params['d_h2_b'], params['d_h3_w'], params['d_h3_b']]
    params_list = [theano.shared(param, borrow = True) for param in params_list]
    dssm_hash = DSSMHash(rng, T_query, T_pos_neg_doc, T_query_index, T_doc_index, T_lr, params_list)

    train_hash = theano.function(
        inputs  = [T_query_index, T_doc_index, T_query, T_pos_neg_doc, T_lr],
        outputs = [dssm_hash.sim_rs, dssm_hash.st, dssm_hash.cost, 
                   dssm_hash.params[0], dssm_hash.params[1], dssm_hash.params[2], dssm_hash.params[3], dssm_hash.params[4], dssm_hash.params[5], 
                   dssm_hash.params[6], dssm_hash.params[7]],
        updates = dssm_hash.updates,
        allow_input_downcast=True,
        on_unused_input='ignore'
    )
    return train_hash

def train_dssm_hash():
    results = ''
    n_epoch = 50
    npz_file = 'tmp1_4_1.npz'
    dssm_hash_train = compile_func()
    data = np.load(npz_file)
    feat_mat = data['train_mat']
    print 'train data size: %d' % feat_mat.shape[0]
    global lr
    for epoch in range(n_epoch):
        print 'epoch %d' % epoch
        cost = 0.0
        for query_feat, doc_feat in DataIter(feat_mat, batch_size, n_neg, n_dim_query, n_dim_doc):
            results = dssm_hash_train(query_index, doc_index, query_feat, doc_feat, lr)
            cost += results[2]
            # print 'cos: ' + str(results[0])
            # print 'st: ' + str(results[1])
            # print 'cost: ' + str(results[2])
        if epoch > 0 and epoch % 1 == 0:
            np.savez('params_%d' % epoch, d_h1_w=results[3], d_h1_b=results[4], 
                                          d_h2_w=results[5], d_h2_b=results[6], 
                                          d_h3_w=results[7], d_h3_b=results[8],
                                          hash_w=results[9], hash_b=results[10])
        print 'Cost: %f' % cost
        lr_new = lr_decay(lr, epoch, 5)
        if lr_new != lr:
            lr = lr_new
            print 'lr: %f' % lr
    np.savez('params_%d' % epoch, d_h1_w=results[3], d_h1_b=results[4], 
                                  d_h2_w=results[5], d_h2_b=results[6], 
                                  d_h3_w=results[7], d_h3_b=results[8],
                                  hash_w=results[9], hash_b=results[10])

def main():
    train_dssm_hash()
    
if __name__ == '__main__':
    main()
