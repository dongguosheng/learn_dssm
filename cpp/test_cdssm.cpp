#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include "itq_atlas.h"
#include "bitarray.h"
#include "w2v.h"
#include "cdssm.h"
#include "dssm.h"
#include <sys/time.h>

using namespace std;

int main() {
    // ---------------- Doc Side (Index)---------------------
    size_t dims[4] = {4096, 2048, 2048, 128};
    vector<size_t> _dims(dims, dims + 4);
    const char *doc_model_file = "./data/doc_side/cdssm_doc.model";
    DSSM doc_dssm;
    doc_dssm.SetDims(_dims);
    bool is_success = doc_dssm.LoadModel(doc_model_file);
    if(!is_success) {
        std::cerr << "Load Doc Side Model Fail." << std::endl;
        return -1;
    }
    std::cerr << "DSSM Model(Doc Side) Load Success." << std::endl;
    
    mat::Mat h1_output(NULL, 1, dims[1]), h2_output(NULL, 1, dims[2]), h3_output(NULL, 1, dims[3]);
    mat::Mat input(NULL, 1, dims[0]);
    h1_output.dptr = new float[dims[1]];
    h2_output.dptr = new float[dims[2]];
    h3_output.dptr = new float[dims[3]];
    input.dptr = new float[dims[0]];
    // construct doc side input
    FILE *fp = fopen("img_feat", "rb");
    if(fp) {
        fread(input.dptr, 1, sizeof(float) * dims[0], fp);
        fclose(fp);
    } else {
        return -1;
    }
    cout << "input: \n" << input.ToString() << endl;
    doc_dssm.Forward(input, h1_output, h2_output, h3_output);
    cout << "d_h3_output: \n" << h3_output.ToString() << endl;

    // ---------------- Query Side (QO)---------------------
    const char *vocab_file = "./data/query_side/vocab.txt";
    const char *syn0_file = "./data/query_side/w2v_baike.bin";
    const char *rewrite_file = "./data/query_side/dssm_rewrite.txt";
    W2V w2v_model;
    size_t n_dim = 200;
    w2v_model.SetDim(n_dim);
    bool flag = w2v_model.LoadModel(vocab_file, syn0_file, rewrite_file);
    if (!flag) {
        std::cerr << "w2v model load fail." << std::endl;
        return -1;
    }
    std::string words[] = {"£°£´£°£²£²£¸", "£°£´£°£²£²£¸", "Á½Ö»", "ÀÏ»¢", "¼òÆ×", "£°£´£°£²£²£¸", "£°£´£°£²£²£¸"};
    // How to Do Padding ??? 
    std::vector<std::string> words_vec(words, words + 3 + 4);
    mat::Mat words_mat(NULL, words_vec.size(), n_dim);
    words_mat.dptr = new float[words_vec.size() * n_dim];
    words_mat.Reset();
    // mat::Mat sub_mat = words_mat.SubMat(2, 5);
    w2v_model.GetMat(words_vec, words_mat, false);
    size_t filter_widths[] = {1, 2, 3};
    std::vector<size_t> filter_width_vec(filter_widths, filter_widths + 3);
    size_t n_feat_dim = 10;
    std::vector<size_t> fc_dims;
    fc_dims.push_back(filter_width_vec.size() * n_feat_dim);
    fc_dims.push_back(128);
    CDSSM cdssm(n_dim, filter_width_vec, n_feat_dim, fc_dims);
    const char *cdssm_model = "./data/query_side/cdssm_query.model";
    cdssm.LoadModel(cdssm_model);
    mat::Mat conv_output(NULL, filter_width_vec.size(), n_feat_dim), h_output(NULL, 1, 128);
    conv_output.dptr = new float[filter_width_vec.size() * n_feat_dim];
    h_output.dptr = new float[128];
    conv_output.Reset();
    h_output.Reset();
    // std::cerr << "words_mat: \n" << words_mat.ToString() << std::endl;
    struct timeval st; gettimeofday( &st, NULL );
    cdssm.Forward(words_mat, conv_output, h_output);
    struct timeval et; gettimeofday( &et, NULL );
    printf("timeval: %f ms.\n", ((et.tv_sec - st.tv_sec) * 1000 + (et.tv_usec - st.tv_usec)/1000) / 1.0f);
    std::cerr << "conv_output: \n" << conv_output.ToString() << "h_output: \n" << h_output.ToString() << std::endl;

    float cos_sim = 0.0f;
    float sum = .0f;
    float left = .0f;
    float right = .0f;
    cout << "d_h3_output shape: " << h3_output.Shape() << ", query output shape: " << h_output.Shape() << endl;
    for(size_t i = 0; i < 128; ++ i) {
        sum += h3_output.dptr[i] * h_output.dptr[i];
        left += h_output.dptr[i] * h_output.dptr[i];
        right += h3_output.dptr[i] * h3_output.dptr[i];
    }
    cos_sim = sum / sqrt(left) / sqrt(right);
    cout << "cos sim: " << cos_sim << endl;

    // ---------------- ITQ Hash ---------------------
    using namespace lsh;
    ITQ itq(64, 128, 1);
    mat::Mat pca_rs(NULL, 1, 64), relax(NULL, 1, 64);
    pca_rs.dptr = new float[64 * 1];
    relax.dptr = new float[64 * 1];
    itq.LoadModel("64_128_lsh_for_cdssm.model");
    vector<vector<bool> > doc_hash = itq.Hash(h3_output, pca_rs, relax);
    vector<vector<bool> > query_hash = itq.Hash(h_output, pca_rs, relax);
    size_t dist = itq.hamming_dist(doc_hash[0], query_hash[0]);
    cout << "hamming dist " << dist << endl;

    vector<unsigned char> bits_doc = BitArray::Bool2uchar(doc_hash[0]);
    vector<unsigned char> bits_query = BitArray::Bool2uchar(query_hash[0]);
    int dist_tmp = BitArray::CalHammingDist(bits_doc, bits_query);
    cout << "hamming dist test: " << dist_tmp << endl;
    uint64_t doc_hash_num, query_hash_num;
    copy(bits_doc.begin(), bits_doc.end(), (unsigned char*)(&doc_hash_num));
    copy(bits_query.begin(), bits_query.end(), (unsigned char*)(&query_hash_num));
    dist_tmp = BitArray::CalHammingDist(&doc_hash_num, &query_hash_num, 1);
    cout << "hamming dist test unit64_t: " << dist_tmp << endl;

    delete [] h1_output.dptr;
    delete [] h2_output.dptr;
    delete [] h3_output.dptr;
    delete [] input.dptr;

    delete [] words_mat.dptr;;
    delete [] conv_output.dptr;
    delete [] h_output.dptr;

    delete [] pca_rs.dptr;
    delete [] relax.dptr;

    return 0;
}
