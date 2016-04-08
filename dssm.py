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
        
class CosineLayer(object):
    def __init__(self, query_rep, pos_neg_docs_rep, query_index, doc_index):
        self.output, _ = theano.scan(self.cal_cos, sequences=[query_index, doc_index], non_sequences=[query_rep, pos_neg_docs_rep])
        self.output = T.reshape(self.output, (batch_size, n_neg + 1))
        
    def cal_cos(self, query_idx, doc_idx, Q, D):
        norm_q = T.sqrt(T.sum(T.sqr(Q[query_idx]), 1))
        norm_d = T.sqrt(T.sum(T.sqr(D[doc_idx]), 1))
        return T.dot( Q[query_idx], D[doc_idx].T) / T.outer(norm_q, norm_d)  
        
class Representation(object):
    def __init__(self, rng, input, n_in, n_hidden1, n_hidden2, n_out, params_list=None, activation=T.nnet.sigmoid):
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
    def __init__(self, rng, querys, pos_neg_docs, query_index, doc_index, lr=lr):
        query_rep = Representation(rng, querys, n_dim_query, 256, 256, 200)
        query_output = query_rep.output
        pos_neg_docs_rep = Representation(rng, pos_neg_docs, n_dim_doc, 2048, 2048, 200)
        pos_neg_docs_output = pos_neg_docs_rep.output
        self.sim_rs = CosineLayer(query_output, pos_neg_docs_output, query_index, doc_index).output
        self.st = T.nnet.softmax(self.sim_rs)
        self.cost = -T.sum( T.log( self.st[:, 0] ) ) / self.st.shape[0]
        
        self.params = pos_neg_docs_rep.params + query_rep.params
        self.grad_params = T.grad(self.cost, self.params)
        self.updates = [
            (param, param - lr * grad_param) for param, grad_param in zip(self.params, self.grad_params)
        ]

def compile_func():
    # Compile Theano Function
    T_query = T.matrix('query', dtype='float32')
    T_pos_neg_doc = T.matrix('pos_neg_doc', dtype='float32')
    T_query_index = T.matrix('query_index', dtype='int32')
    T_doc_index = T.matrix('doc_index', dtype='int32')
    rng = np.random.RandomState(1234)
    dssm = DSSM(rng, T_query, T_pos_neg_doc, T_query_index, T_doc_index, lr=lr)

    train = theano.function(
        inputs  = [T_query_index, T_doc_index, T_query, T_pos_neg_doc],
        outputs = [dssm.sim_rs, dssm.st, dssm.cost, 
                   dssm.params[0], dssm.params[1], dssm.params[2], dssm.params[3], dssm.params[4], dssm.params[5],
                   dssm.params[6], dssm.params[7], dssm.params[8], dssm.params[9], dssm.params[10], dssm.params[11]],
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


dssm_train = compile_func()
def train_dssm(mat_file, cnt, i, epoch_num):
    results = ''
    n_epoch = 1
    feat_mat = np.fromfile(mat_file, dtype='float32')
    feat_mat = feat_mat.reshape(cnt, n_dim_query + n_dim_doc*(n_neg + 1))
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
        if epoch > 0 and epoch % 1 == 0:
            np.savez('params_%d' % epoch, d_h1_w=results[3], d_h1_b=results[4], 
                                          d_h2_w=results[5], d_h2_b=results[6], 
                                          d_h3_w=results[7], d_h3_b=results[8],
                                          q_h1_w=results[9], q_h1_b=results[10],
                                          q_h2_w=results[11], q_h2_b=results[12],
                                          q_h3_w=results[13], q_h3_b=results[14])
        print 'Cost: %f' % cost
    np.savez('params_%d_%d' % (epoch_num, i), d_h1_w=results[3], d_h1_b=results[4], 
                                  d_h2_w=results[5], d_h2_b=results[6], 
                                  d_h3_w=results[7], d_h3_b=results[8],
                                  q_h1_w=results[9], q_h1_b=results[10],
                                  q_h2_w=results[11], q_h2_b=results[12],
                                  q_h3_w=results[13], q_h3_b=results[14])

def main():
    mat_file = './dssm_train_data/150w{i}_train_mat'
    cnt_file = './dssm_train_data/150w{i}_cnt'
    n_epoch = 20
    for epoch in range(n_epoch):
        idx_list = range(14)
        random.shuffle(idx_list)
        print idx_list
        for i in idx_list:
            print mat_file.format(i=i)
            cnt = 0
            with open(cnt_file.format(i=i)) as fin:
                cnt = int(fin.next().strip())
            train_dssm(mat_file.format(i=i), cnt, i, epoch)
    
if __name__ == '__main__':
    main()                                    
