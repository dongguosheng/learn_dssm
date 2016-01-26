# -*- coding: gbk -*-

import numpy as np
import theano
import theano.tensor as T
import random

batch_size = 1
lr = 0.01
n_neg = 5
n_dim_query = 200
n_dim_doc = 4096

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

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.nnet.sigmoid):
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
        
class CosineLayer(object):
    def __init__(self, query_rep, pos_neg_docs_rep, query_index, doc_index):
        self.output, _ = theano.scan(self.cal_cos, sequences=[query_index, doc_index], non_sequences=[query_rep, pos_neg_docs_rep])
        self.output = T.reshape(self.output, (batch_size, n_neg + 1))
        
    def cal_cos(self, query_idx, doc_idx, Q, D):
        norm_q = T.sqrt(T.sum(T.sqr(Q[query_idx]), 1))
        norm_d = T.sqrt(T.sum(T.sqr(D[doc_idx]), 1))
        return T.dot( Q[query_idx], D[doc_idx].T) / T.outer(norm_q, norm_d)  
        
class Representation(object):
    def __init__(self, rng, input, n_in, n_hidden1, n_hidden2, n_out):
        hidden_layer1 = HiddenLayer(rng, input, n_in, n_hidden1, activation=T.nnet.sigmoid)
        hidden_layer2 = HiddenLayer(rng, hidden_layer1.output, n_hidden1, n_hidden2, activation=T.nnet.sigmoid)
        hidden_layer3 = HiddenLayer(rng, hidden_layer2.output, n_hidden2, n_out, activation=T.nnet.sigmoid)
        self.output = hidden_layer3.output
        self.params = hidden_layer1.params + hidden_layer2.params + hidden_layer3.params
        
class DSSM(object):
    def __init__(self, rng, querys, pos_neg_docs, query_index, doc_index, lr):
        # query_rep = Representation(rng, querys).output
        query_output = querys
        pos_neg_docs_rep = Representation(rng, pos_neg_docs, n_dim_doc, 2048, 2048, 200)
        pos_neg_docs_output = pos_neg_docs_rep.output
        self.sim_rs = CosineLayer(query_output, pos_neg_docs_output, query_index, doc_index).output
        self.st = T.nnet.softmax(self.sim_rs)
        self.cost = -T.sum( T.log( self.st[:, 0] ) ) / self.st.shape[0]
        
        self.params = pos_neg_docs_rep.params
        self.grad_params_doc = T.grad(self.cost, self.params)
        self.updates = [
            (param, param - lr * grad_param) for param, grad_param in zip(self.params, self.grad_params_doc)
        ]

class HashLaryer(object):
    def __init__(self, rng, input, hash_params):
        self.W, self.b = hash_params
        self.output = (T.dot(input, self.W) + self.b) / 2
        self.hash_params = hash_params

def init_hash_params(n_in, n_bit):
    params = []
    W_vals = np.asarray(
            np.random.randn(n_in, n_bit),
            dtype=theano.config.floatX
            )
    params.append(theano.shared(value=W_vals, name='W', borrow=True))
    b_vals = np.zeros((n_bit, ), dtype=theano.config.floatX)
    params.append(theano.shared(value=b_vals, name='b', borrow=True))
    return params
hash_params = init_hash_params(200, 48)

class DSSMHash(object):
    def __init__(self, rng, querys, pos_neg_docs, query_index, doc_index, lr):
        # query_rep = Representation(rng, querys).output
        query_output = querys
        pos_neg_docs_rep = Representation(rng, pos_neg_docs, n_dim_doc, 2048, 2048, 200)
        pos_neg_docs_output = pos_neg_docs_rep.output
        hash_layer_query = HashLaryer(rng, query_output, hash_params)
        hash_layer_doc = HashLaryer(rng, pos_neg_docs_output, hash_params)
        relax_bits_query = hash_layer_query.output
        relax_bits_doc = hash_layer_doc.output
        
        self.sim_rs, _ = theano.scan(self.cal_dot, sequences=[query_index, doc_index], non_sequences=[relax_bits_query, relax_bits_doc])
        self.sim_rs = T.reshape(self.sim_rs, (batch_size, n_neg + 1))
        
        self.st = T.nnet.softmax(self.sim_rs)
        self.cost = -T.sum( T.log( self.st[:, 0] ) ) / self.st.shape[0]
        
        self.params = pos_neg_docs_rep.params + hash_params
        self.grad_params_doc = T.grad(self.cost, self.params)
        self.updates = [
            (param, param - lr * grad_param) for param, grad_param in zip(self.params, self.grad_params_doc)
        ]
    
    def cal_dot(self, query_idx, doc_idx, Q, D):
        return T.dot( Q[query_idx], D[doc_idx].T)
        
def compile_func():
    # Compile Theano Function
    T_query = T.matrix('query', dtype='float32')
    T_pos_neg_doc = T.matrix('pos_neg_doc', dtype='float32')
    T_query_index = T.matrix('query_index', dtype='int32')
    T_doc_index = T.matrix('doc_index', dtype='int32')
    rng = np.random.RandomState(1234)
    # dssm = DSSM(rng, T_query, T_pos_neg_doc, T_query_index, T_doc_index, lr=lr)
    dssm = DSSMHash(rng, T_query, T_pos_neg_doc, T_query_index, T_doc_index, lr=lr)     # DSSM + Hash

    train = theano.function(
        inputs  = [T_query_index, T_doc_index, T_query, T_pos_neg_doc],
        outputs = [dssm.sim_rs, dssm.st, dssm.cost, 
                   dssm.params[0], dssm.params[1], dssm.params[2], dssm.params[3], dssm.params[4], dssm.params[5], dssm.params[6]],
        updates = dssm.updates,
        allow_input_downcast=True,
        on_unused_input='ignore'
    )
    return train

class DataIter(object):
    def __init__(self, feat_mat, batch_size, n_neg, n_dim_query, n_dim_doc):
        self.batch_size = batch_size
        self.n_neg = n_neg
        self.n_dim_query = n_dim_query
        self.n_dim_doc = n_dim_doc
        self.train_mat = feat_mat
        self.shuffle_index_list = [i for i in range(self.train_mat.shape[0])]
        random.shuffle(self.shuffle_index_list)
        self.n_train_batches = self.train_mat.shape[0] / self.batch_size

    def __iter__(self):
        for i in range(self.n_train_batches):
            # print '__iter__ i: %d' % i
            indexes = self.shuffle_index_list[i : i + self.batch_size]
            query_doc_feat = self.train_mat[indexes]
            query_feat = query_doc_feat[:, : self.n_dim_query]
            doc_feat = query_doc_feat[:, self.n_dim_query :]
            doc_feat = doc_feat.reshape((self.n_neg+1) * self.batch_size, self.n_dim_doc)
            yield (query_feat, doc_feat)

def train_dssm():
    results = ''
    n_epoch = 50
    npz_file = 'tmp1_4_1.npz'
    dssm_train = compile_func()
    data = np.load(npz_file)
    feat_mat = data['train_mat']
    print 'train data size: %d' % feat_mat.shape[0]
    for epoch in range(n_epoch):
        print 'epoch %d' % epoch
        cost = 0.0
        for query_feat, doc_feat in DataIter(feat_mat, batch_size, n_neg, n_dim_query, n_dim_doc):
            results = dssm_train(query_index, doc_index, query_feat, doc_feat)
            cost += results[2]
            # print 'cos: ' + str(results[0])
            # print 'st: ' + str(results[1])
            # print 'cost: ' + str(results[2])
        if epoch > 0 and epoch % 10 == 0:
            np.savez('params_%d' % epoch, d_h1_w=results[3], d_h1_b=results[4], 
                                          d_h2_w=results[5], d_h2_b=results[6], 
                                          d_h3_w=results[7], d_h3_b=results[8], 
                                          hash_w=result[9], hash_b=result[10])
        print 'Cost: %f' % cost
    np.savez('params_%d' % epoch, d_h1_w=results[3], d_h1_b=results[4], 
                                  d_h2_w=results[5], d_h2_b=results[6], 
                                  d_h3_w=results[7], d_h3_b=results[8],
                                  hash_w=result[9], hash_b=result[10])

def main():
    train_dssm()
    
if __name__ == '__main__':
    main()                                    
