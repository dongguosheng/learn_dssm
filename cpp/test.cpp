#include <iostream>
#include "dssm.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include "itq_atlas.h"
#include "bitarray.h"

using namespace std;

int main() {
    // test dssm & hash 
    size_t dims[4] = {4096, 2048, 2048, 200};
    vector<size_t> _dims(dims, dims + 4);
    const char *model_file = "/search/dongguosheng/demos/dssm_demo/cpp/dssm_doc.model";
    DSSM dssm(_dims);
    bool is_success = dssm.LoadModel(model_file);
    if(!is_success) {
        std::cerr << "Load Model Fail." << std::endl;
        return -1;
    }
    std::cerr << "DSSM Model Load Success." << std::endl;

    float *input_ptr = new float[dims[0]];
    float *dssm_ptr_1 = new float[dims[1]];
    float *dssm_ptr_2 = new float[dims[2]];
    float *dssm_ptr_3 = new float[dims[3]];

    // construct doc side input, 4096 features
    FILE *fp = fopen("/search/dongguosheng/demos/dssm_demo/cpp/input_feat", "rb");
    if(fp) {
        fread(input_ptr, 1, sizeof(float) * dims[0], fp);
        fclose(fp);
    } else {
        return -1;
    }

    mat::Mat input(input_ptr, 1, dims[0]);
    mat::Mat h1_output(dssm_ptr_1, 1, dims[1]), h2_output(dssm_ptr_2, 1, dims[2]), h3_output(dssm_ptr_3, 1, dims[3]);
    dssm.Predict(input, h1_output, h2_output, h3_output);

    // hash
    using namespace lsh;
    ITQ itq(64, 200, 1);
    float *lsh_ptr_1 = new float[200 * 64];
    float *lsh_ptr_2 = new float[64 * 1];
    mat::Mat pca_rs(lsh_ptr_1, 200, 64), relax(lsh_ptr_2, 1, 64);
    itq.LoadModel("/search/dongguosheng/demos/dssm_demo/cpp/200_64_dssm.model");
    vector<vector<bool> > doc_hash = itq.Hash(h3_output, pca_rs, relax);
    vector<unsigned char> bits_doc = BitArray::Bool2uchar(doc_hash[0]);
    cout << "bits_doc len: " << bits_doc.size() << endl;

    delete [] input_ptr;
    delete [] dssm_ptr_1;
    delete [] dssm_ptr_2;
    delete [] dssm_ptr_3;
    delete [] lsh_ptr_1;
    delete [] lsh_ptr_2;

    return 0;
}
