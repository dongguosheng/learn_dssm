#ifndef CDSSM_H
#define CDSSM_H

#include "mat.h"
#include "layers.h"
#include <cstdio>
#include <vector>

class CDSSM {
    public:
        CDSSM() {}
        CDSSM(size_t _n_dim, std::vector<size_t> &_filter_width_vec, size_t _n_feat_map, std::vector<size_t> &_fc_dims)
            : n_dim(_n_dim), filter_width_vec(_filter_width_vec), n_feat_map(_n_feat_map), fc_dims(_fc_dims) {}
        virtual ~CDSSM() {
            delete conv_layer;
            delete fc_layer;
            delete activation_layer;
        }
        inline void SetConvParams(size_t _n_dim, std::vector<size_t> &_filter_width_vec, size_t _n_feat_map) {
            n_dim = _n_dim;
            filter_width_vec = _filter_width_vec;
            n_feat_map = _n_feat_map;
        }
        inline void SetFCDims(std::vector<size_t> &_fc_dims) {
            fc_dims = _fc_dims;
        }
        bool LoadModel(const char *model_file) {
            conv_layer = new ConvLayer(n_dim, filter_width_vec, n_feat_map);
            fc_layer = new FullyConnectedLayer(fc_dims[0], fc_dims[1]);
            activation_layer = new ActivationLayer(Tanh);
            FILE *fp = fopen(model_file, "rb");
            // TODO: check model file valid
            if(fp) {
                conv_layer->LoadParams(fp);
                fc_layer->LoadParams(fp);
                return true;
                fclose(fp);
            }
            return false;
        }

        bool Forward(mat::Mat &input, mat::Mat &conv_output, mat::Mat &h_output) const {
            // users have to make sure h1_output, h2_output, h3_output have enough memory.
            conv_layer->ForwardNoGemm(input, conv_output);
            activation_layer->Forward(conv_output);
            fc_layer->Forward(conv_output, h_output);
            activation_layer->Forward(h_output);
            
            return true;
        }

    private:
        ConvLayer *conv_layer;
        FullyConnectedLayer *fc_layer;
        ActivationLayer *activation_layer;
        size_t n_dim;
        std::vector<size_t> filter_width_vec;
        size_t n_feat_map;
        std::vector<size_t> fc_dims;

        CDSSM(const CDSSM &other);
        CDSSM& operator=(const CDSSM &other);
};
#endif /*CDSSM_H*/
