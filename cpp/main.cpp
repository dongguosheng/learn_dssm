#include <iostream>
#include "dssm.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include "itq_atlas.h"
#include "bitarray.h"

using namespace std;

const int m = 3;
const int k = 3;
const int n = 2;
int main() {
    // test gemm
    float mat_a[m * k] = {1, 2, 4, 1, 2, 4, 1, 2, 4};
    float mat_b[k * n] = {1, 2, 3, 4, 5, 6};
    float mat_rs[m * n] = {3, 1, 3, 2, 3, 3};
    mat::Mat A(mat_a, m, k), B(mat_b, k, n), C(mat_rs, m, n);
    cout << "A: (" << A.n_row << ", " << A.n_col << ")\n" << A.to_string()
         << "B: (" << B.n_row << ", " << B.n_col << ")\n" << B.to_string()
         << "C: (" << C.n_row << ", " << C.n_col << ")\n" << C.to_string();
    // C = gemm(A, B);  // dot(A, B) + C
    mat::sgemm(A, B, C);
    cout << "C: (" << C.n_row << ", " << C.n_col << ")\n" << C.to_string() << endl;
   
    // test dssm & hash 
    size_t dims[4] = {4096, 2048, 2048, 200};
    vector<size_t> _dims(dims, dims + 4);
    const char *model_file = "/search/dongguosheng/demos/dssm_demo/cpp/dssm_doc.model";
    DSSM dssm(_dims);
    bool is_success = dssm.LoadModel(model_file);
    // cout << _dims[0] << endl;
    if(!is_success) {
        std::cerr << "Load Model Fail." << std::endl;
        return -1;
    }
    std::cerr << "DSSM Model Load Success." << std::endl;

    float *p_1 = new float[dims[1]];
    float *p_2 = new float[dims[2]];
    float *p_3 = new float[dims[3]];
    float *input_p = new float[dims[0]];

    float *word_p = new float[dims[3]];
    // construct doc side input
    FILE *fp = fopen("/search/dongguosheng/demos/dssm_demo/cpp/input_feat", "rb");
    if(fp) {
        fread(input_p, 1, sizeof(float) * dims[0], fp);
        fclose(fp);
    } else {
        return -1;
    }

    fp = fopen("/search/dongguosheng/demos/dssm_demo/cpp/word_feat", "rb");
    if(fp) {
        fread(word_p, 1, sizeof(float) * dims[3], fp);
        fclose(fp);
    } else {
        return -1;
    }

    mat::Mat h1_output(p_1, 1, dims[1]), h2_output(p_2, 1, dims[2]), h3_output(p_3, 1, dims[3]);
    mat::Mat input(input_p, 1, dims[0]);
    // cout << "input shape: " << input.shape() << endl;
    // cout << "h1_output shape: " << h1_output.shape() << endl;
    // cout << "h2_output shape: " << h2_output.shape() << endl;
    // cout << "h3_output shape: " << h3_output.shape() << endl;

    dssm.Predict(input, h1_output, h2_output, h3_output);

    // cout << h1_output.to_string() << endl;
    // cout << h2_output.to_string() << endl;
    // cout << h3_output.to_string() << endl;

    mat::Mat wordvec(word_p, 1, dims[3]);
    
    float cos_sim = 0.0f;
    float sum = .0f;
    float left = .0f;
    float right = .0f;
    for(size_t i = 0; i < dims[3]; ++ i) {
        sum += h3_output.dptr[i] * wordvec.dptr[i];
        left += h3_output.dptr[i] * h3_output.dptr[i];
        right += wordvec.dptr[i] * wordvec.dptr[i];
    }
    cos_sim = sum / sqrt(left) / sqrt(right);
    cout << "cos sim: " << cos_sim << endl;
    delete [] p_1;
    delete [] p_2;
    delete [] input_p;

    // hash
    using namespace lsh;
    ITQ itq(64, 200, 1);
    p_1 = NULL;
    p_2 = NULL;
    p_1 = new float[200 * 64];
    p_2 = new float[64 * 1];
    mat::Mat pca_rs(p_1, 200, 64), relax(p_2, 1, 64);
    itq.LoadModel("/search/dongguosheng/demos/dssm_demo/cpp/200_64_dssm.model");
    vector<vector<bool> > doc_hash = itq.Hash(h3_output, pca_rs, relax);
    vector<vector<bool> > query_hash = itq.Hash(wordvec, pca_rs, relax);
    size_t dist = itq.hamming_dist(doc_hash[0], query_hash[0]);
    cout << "hamming dist " << dist << endl;

    vector<unsigned char> bits_doc = BitArray::Bool2uchar(doc_hash[0]);
    vector<unsigned char> bits_query = BitArray::Bool2uchar(query_hash[0]);
    int dist_tmp = BitArray::CalHammingDist(bits_doc, bits_query);
    cout << "hamming dist test: " << dist_tmp << endl;

    delete [] p_3;
    delete [] word_p;
    delete [] p_1;
    delete [] p_2;

    return 0;
}
