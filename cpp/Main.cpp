#include <iostream>
#include "DSSM.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include "ITQ.h"

using namespace std;

const int m = 3;
const int k = 3;
const int n = 2;
int main() {
    float mat_a[m * k] = {1, 2, 4, 1, 2, 4, 1, 2, 4};
    float mat_b[k * n] = {1, 2, 3, 4, 5, 6};
    float mat_rs[m * n] = {3, 1, 3, 2, 3, 3};
    Mat A(mat_a, m, k), B(mat_b, k, n), C(mat_rs, m, n);
    cout << "A: (" << A.n_row << ", " << A.n_col << ")\n" << A.to_string()
         << "B: (" << B.n_row << ", " << B.n_col << ")\n" << B.to_string();
    C = gemm(A, B);  // dot(A, B) + C
    cout << "C: (" << C.n_row << ", " << C.n_col << ")\n" << C.to_string() << endl;
    // for(int i = 0; i < n; ++i) {
    //     printf("%f\n", C.dptr[i]);
    // }
    
    size_t dims[4] = {4096, 2048, 2048, 200};
    vector<size_t> _dims(dims, dims + 4);
    const char *model_file = "dssm.model";
    DSSM dssm(_dims);
    bool is_success = dssm.LoadModel(model_file);
    cout << _dims[0] << endl;
    if(!is_success) {
        cout << "Load Model Fail." << endl;
        return -1;
    }
    cout << "DSSM Model Load Success." << endl;

    float *p_1 = new float[dims[1]];
    float *p_2 = new float[dims[2]];
    float *p_3 = new float[dims[3]];
    float *input_p = new float[dims[0]];

    float *word_p = new float[dims[3]];
    // construct input
    FILE *fp = fopen("input_feat", "rb");
    if(fp) {
        fread(input_p, 1, sizeof(float) * dims[0], fp);
    } else {
        return -1;
    }

    fclose(fp);
    fp = fopen("word_feat", "rb");
    if(fp) {
        fread(word_p, 1, sizeof(float) * dims[3], fp);
    } else {
        return -1;
    }
    fclose(fp);

    Mat h1_output(p_1, 1, dims[1]), h2_output(p_2, 1, dims[2]), h3_output(p_3, 1, dims[3]);
    Mat input(input_p, 1, dims[0]);
    cout << "input shape: " << input.shape() << endl;
    cout << "h1_output shape: " << h1_output.shape() << endl;
    cout << "h2_output shape: " << h2_output.shape() << endl;
    cout << "h3_output shape: " << h3_output.shape() << endl;

    dssm.Predict(input, h1_output, h2_output, h3_output);

    // cout << h1_output.to_string() << endl;
    // cout << h2_output.to_string() << endl;
    // cout << h3_output.to_string() << endl;

    Mat wordvec(word_p, 1, dims[3]);
    
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

    ITQ itq(64, 200, 1);
    p_1 = NULL;
    p_2 = NULL;
    p_1 = new float[sizeof(float) * 200 * 64];
    p_2 = new float[sizeof(float) * 64 * 1];
    Mat pca_rs(p_1, 200, 64), relax(p_2, 1, 64);
    itq.LoadModel("itq.model");
    vector<vector<bool> > doc_hash = itq.hash(h3_output, pca_rs, relax);
    vector<vector<bool> > query_hash = itq.hash(wordvec, pca_rs, relax);
    size_t dist = itq.hamming_dist(doc_hash[0], query_hash[0]);
    cout << "hamming dist " << dist << endl;

    delete [] p_3;
    delete [] word_p;
    delete [] p_1;
    delete [] p_2;

    return 0;
}
