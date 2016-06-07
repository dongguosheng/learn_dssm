# -*- coding: gbk -*-

import numpy as np
import sys
from lsh.itqlsh import ITQLSH

def to_binary_model(input, output, key_list=['d_h1_w', 'd_h1_b', 'd_h2_w', 'd_h2_b', 'd_h3_w', 'd_h3_b']):
    params = np.load(input)
    h1_w = params[ key_list[0] ]
    h1_b = params[ key_list[1] ]
    h2_w = params[ key_list[2] ]
    h2_b = params[ key_list[3] ]
    h3_w = params[ key_list[4] ]
    h3_b = params[ key_list[5] ]
    # print h1_w.dtype
    # print h1_b.dtype
    h1_w = h1_w.astype('float32')
    h1_b = h1_b.astype('float32')
    h2_w = h2_w.astype('float32')
    h2_b = h2_b.astype('float32')
    h3_w = h3_w.astype('float32')
    h3_b = h3_b.astype('float32')
    with open(output, 'wb') as fout:
        # print h1_w[: 20]
        # print h2_w[: 20]
        # print h3_w[: 20]
        fout.write(h1_w.tobytes())
        fout.write(h1_b.tobytes())
        fout.write(h2_w.tobytes())
        fout.write(h2_b.tobytes())
        fout.write(h3_w.tobytes())
        fout.write(h3_b.tobytes())


def to_binary_lsh(model_str, output):
    lsh = ITQLSH(64, 200, n_table=1)
    lsh.load(model_str)
    with open(output, 'wb') as fout:
        fout.write(lsh.pca_list[0].astype('float32').tobytes())
        fout.write(lsh.R_list[0].astype('float32').tobytes())

def main():
    if len(sys.argv) != 6:
        print 'python to_binary_model.py params_xyz.npz dssm_doc.model dssm_query.model 200_64_for_dssm 200_64_for_dssm.model'
    else:
        input = sys.argv[1]
        output_doc = sys.argv[2]
        output_query = sys.argv[3]
        lsh_model_str = sys.argv[4]
        lsh_binary_model = sys.argv[5]
        to_binary_file(input, output_doc, key_list=['d_h1_w', 'd_h1_b', 'd_h2_w', 'd_h2_b', 'd_h3_w', 'd_h3_b'])
        to_binary_model(input, output_query, key_list=['q_h1_w', 'q_h1_b', 'q_h2_w', 'q_h2_b', 'q_h3_w', 'q_h3_b'])

        to_binary_lsh(lsh_model_str, lsh_binary_model)

if __name__ == '__main__':
    main()
