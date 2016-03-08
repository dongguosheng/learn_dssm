#ifndef DSSM_H
#define DSSM_H

#include "Mat.h"
#include <cstdio>
#include <vector>

class DSSM {
    public:
        DSSM() {}
        DSSM(std::vector<size_t> _dims) : dims(_dims) {}
        virtual ~DSSM() {
            for(size_t i = 0; i < p_params_vec.size(); ++ i) {
                delete [] p_params_vec[i];
            }
        }
        void SetDims(std::vector<size_t> _dims) {
            dims = _dims;
        }
        bool LoadModel(const char *model_file) {
            float *p_params = NULL;
            params_vec.reserve(N * 2);
            for(int i = 0; i < N; ++ i) {
                p_params = new float[dims[i] * dims[i+1]];
                p_params_vec.push_back(p_params);
                p_params = NULL;
                p_params = new float[dims[i+1]];
                p_params_vec.push_back(p_params);
            }
            FILE *fp = fopen(model_file, "rb");
            // TODO: check model file valid
            if(fp) {
                for(int i = 0; i < N; ++ i) {
                    fread(p_params_vec[i*2], 1, sizeof(float) * dims[i] * dims[i+1], fp);
                    fread(p_params_vec[i*2+1], 1, sizeof(float) * dims[i+1], fp);
                    params_vec[i*2].Set(p_params_vec[i*2], dims[i], dims[i+1]);
                    params_vec[i*2+1].Set(p_params_vec[i*2+1], 1, dims[i+1]);
                }
                return true;
            }
            fclose(fp);
            return false;
        }

        std::vector<Mat> GetParams() {
            return params_vec;
        }

        bool Predict(const Mat &input, Mat &h1_output, Mat &h2_output, Mat &h3_output) {
            // users have to make sure h1_output, h2_output, h3_output have enough memory.
            // set bias
            h1_output = params_vec[1];
            h2_output = params_vec[3];
            h3_output = params_vec[5];
            h1_output = gemm(input, params_vec[0]);
            h1_output.sigmoid();
            h2_output = gemm(h1_output, params_vec[2]);
            h2_output.sigmoid();
            h3_output = gemm(h2_output, params_vec[4]);
            h3_output.sigmoid();
            return true;
        }

    private:
        const static int N = 3;
        std::vector<Mat> params_vec;
        std::vector<float*> p_params_vec;
        std::vector<size_t> dims;
};
#endif /*DSSM_H*/
