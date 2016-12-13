#ifndef LAYERS_H
#define LAYERS_H

#include "mat.h"
#include <cstdio>
#include <vector>
#include <limits>

class ConvLayer {     // nlp conv only
    public:
        ConvLayer() {}
        ConvLayer(size_t n_dim, std::vector<size_t> filter_width_vec, size_t n_feat_map)
            : n_dim(n_dim), filter_width_vec(filter_width_vec), n_feat_map(n_feat_map) {
        }
        virtual ~ConvLayer() {
            for(size_t i = 0; i < filter_vec.size(); ++ i) {
                for(size_t j = 0; j < filter_vec[i].size(); ++ j) {
                    delete [] filter_vec[i][j].dptr;
                }
            }
            for(size_t i = 0; i < bias_vec.size(); ++ i) {
                delete [] bias_vec[i].dptr;
            }
        }
        inline void LoadParams(FILE *fp) {
            filter_vec.resize(filter_width_vec.size());
            for(size_t i = 0; i < filter_vec.size(); ++ i) {
                filter_vec[i].resize(n_feat_map);
            }
            bias_vec.resize(filter_width_vec.size());
            for(size_t i = 0; i < filter_width_vec.size(); ++ i) {
                for(size_t j = 0; j < n_feat_map; ++ j) {
                    float *weights_ptr = new float[n_dim * filter_width_vec[i]];
                    fread(weights_ptr, 1, sizeof(float) * n_dim * filter_width_vec[i], fp);
                    filter_vec[i][j].Set(weights_ptr, filter_width_vec[i], n_dim);
                }
                float *bias_ptr = new float[n_feat_map];
                fread(bias_ptr, 1, sizeof(float) * n_feat_map, fp);
                bias_vec[i].Set(bias_ptr, 1, n_feat_map);
            }
        }
        inline float Conv2d(const mat::Mat &patch, const mat::Mat &weights) {
            size_t n_total = patch.n_row * patch.n_col;
            // fprintf(stderr, "patch shape: %s\n", patch.Shape().c_str());
            // fprintf(stderr, "weights shape: %s\n", weights.Shape().c_str());
            float result = 0.0f;
            for(size_t i = 0; i < n_total; ++ i) {
                result += patch.dptr[i] * weights.dptr[n_total - 1 - i];    // flip filter
            }
            return result;
        }
        inline void ForwardNoGemm(mat::Mat &input, mat::Mat &output) {
            // input already do padding
            // output had enough memory   
            for(size_t i = 0; i < filter_width_vec.size(); ++ i) {
                for(size_t j = 0; j < n_feat_map; ++ j) {
                    float max_num = -std::numeric_limits<float>::max();
                    for(size_t k = 0; k < input.n_row - filter_width_vec[i] + 1; ++ k) {  
                        mat::Mat patch = input.SubMat(k, k + filter_width_vec[i]);
                        float result = Conv2d(patch, filter_vec[i][j]);
                        // fprintf(stderr, "%f\t", result);
                        if (result > max_num)   max_num = result;
                    }
                    // fprintf(stderr, "\n");
                    output[i][j] = max_num;
                }
                // fprintf(stderr, "filter_width(%lu), output: %s\n", filter_width_vec[i], output.ToString().c_str());
                output.SubMat(i, i+1) += bias_vec[i];
                // fprintf(stderr, "weights : %s\n", filter_vec[i][9].ToString().c_str());
                // fprintf(stderr, "b (OK): %s\n", bias_vec[i].ToString().c_str());
                // fprintf(stderr, "output + b: %s\n", output.ToString().c_str());
            }
            output.Reshape(1, filter_width_vec.size() * n_feat_map);
        }
        inline void Forward(mat::Mat &input, mat::Mat &output) {
            // TODO
        }
    private:
        size_t n_dim;
        std::vector<size_t> filter_width_vec;
        size_t n_feat_map;
        std::vector<std::vector<mat::Mat> > filter_vec;
        std::vector<mat::Mat> bias_vec;
};
class FullyConnectedLayer {
    public:
        FullyConnectedLayer() {}
        FullyConnectedLayer(size_t n_row, size_t n_col) : n_row(n_row), n_col(n_col) {}
        virtual ~FullyConnectedLayer() {
            delete [] weights.dptr;
            delete [] bias.dptr;
        }
        inline void LoadParams(FILE *fp) {
            float *weights_ptr = new float[n_row * n_col];
            float *bias_ptr = new float[n_col];
            fread(weights_ptr, 1, sizeof(float) * n_row * n_col, fp);
            fread(bias_ptr, 1, sizeof(float) * n_col, fp);
            weights.Set(weights_ptr, n_row, n_col);
            bias.Set(bias_ptr, 1, n_col);
        }
        inline void Forward(mat::Mat &input, mat::Mat &output) {
            // doesnot care input and output
            output.deepcopy(bias);
            mat::sgemm(input, weights, output);  // output = dot(input, weights) + bias;
        }
    private:
        size_t n_row;
        size_t n_col;
        mat::Mat weights;
        mat::Mat bias;
};

enum Activation {Sigmoid=0, Tanh, Relu};
static inline void sigmoid(mat::Mat &m) {
    size_t n_total = m.n_row * m.n_col;
    for(size_t i = 0; i < n_total; ++ i) {
        m.dptr[i] = 1.0f / ( 1.0 + exp(-m.dptr[i]) );
    }
}
static inline void tanh_mat(mat::Mat &m) {
    size_t n_total = m.n_row * m.n_col;
    for(size_t i = 0; i < n_total; ++ i) {
        m.dptr[i] = tanh(m.dptr[i]);
    }
}
static inline void relu(mat::Mat &m) {
    size_t n_total = m.n_row * m.n_col;
    for(size_t i = 0; i < n_total; ++ i) {
        m.dptr[i] = (m.dptr[i] >= 0.0f ? m.dptr[i] : 0.0f);
    }
}
class ActivationLayer {
    public:
        ActivationLayer(Activation type) : type(type) {}
        inline void Forward(mat::Mat &m) {
            switch (type) {
                case Sigmoid:
                    sigmoid(m);
                    break;
                case Tanh:
                    tanh_mat(m);
                    break;
                case Relu:
                    relu(m);
                    break;
                default:
                    break;
            }
        }
    private:
        Activation type;
};

#endif /*LAYERS_H*/
