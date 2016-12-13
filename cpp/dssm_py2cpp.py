# -*- coding: gbk -*-

import numpy as np
import sys

def to_binary_model(input, output, key_list):
    params = np.load(input)
    with open(output, 'wb') as fout:
        for k in key_list:
            param = params[k]
            param = param.astype('float32')
            fout.write(param.tobytes())

def main():
    if len(sys.argv) != 4:
        print 'python dssm_py2cpp.py params_xyz.npz ./data/doc_side/cdssm_doc.model ./data/query_side/cdssm_query.model'
    else:
        input = sys.argv[1]
        output_doc = sys.argv[2]
        output_query = sys.argv[3]
        to_binary_model(input, output_doc, key_list=['d_h1_w', 'd_h1_b', 'd_h2_w', 'd_h2_b', 'd_h3_w', 'd_h3_b'])
        to_binary_model(input, output_query, key_list=['conv_1_w', 'conv_1_b', 'conv_2_w', 'conv_2_b', 'conv_3_w', 'conv_3_b', 'q_h1_w', 'q_h1_b'])
        to_binary_model(input, './data/syn0_cdssm', key_list=['syn0'])

if __name__ == '__main__':
    main()
