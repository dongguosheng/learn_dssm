#ifndef CDSSM_H
#define CDSSM_H

#include "mat.h"
#include "layers.h"
#include <cstdio>
#include <vector>
#include <fstream>
#include <sstream>

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
        inline bool Init(const char *params_file, const char *model_file) {
            return LoadParams(params_file) && LoadModel(model_file);
        }
        inline bool LoadParams(const char *params_file) {
            std::ifstream fin(params_file);
            if(fin.fail()) {
                fprintf(stderr, "%s open failed\n", params_file);
                return false;
            }
            std::string line;
            size_t n_dim, n_feat_map, n_dim_output;
            std::vector<size_t> filter_width_vec;
            while(std::getline(fin, line)) {
                std::stringstream ss(line);
                std::string key;
                ss >> key;
                if (key == "n_dim")             ss >> n_dim;
                else if (key == "n_feat_map")   ss >> n_feat_map;
                else if (key == "n_dim_output") ss >> n_dim_output;
                else if (key == "filter_widths") {
                    size_t val;
                    while(ss >> val) {
                        filter_width_vec.push_back(val);
                        ss.ignore(32, ',');
                    }
                }
            }
            SetConvParams(n_dim, filter_width_vec, n_feat_map);
            std::vector<size_t> fc_dims;
            fc_dims.push_back(filter_width_vec.size() * n_feat_map);
            fc_dims.push_back(n_dim_output);
            SetFCDims(fc_dims);
            return true;
        }
        inline std::string ToString() {
            std::ostringstream ss;
            ss << "n_dim: " << n_dim << ", n_feat_map: " << n_feat_map << ", filter_widths: (";
            for(size_t i = 0; i < filter_width_vec.size(); ++ i) {
                ss << filter_width_vec[i] << ",";
            }
            ss << "), (";
            for(size_t i = 0; i < fc_dims.size(); ++ i) {
                ss << fc_dims[i] << ",";
            }
            ss << ")";
            return ss.str();
        }
        inline void SetConvParams(size_t _n_dim, std::vector<size_t> &_filter_width_vec, size_t _n_feat_map) {
            n_dim = _n_dim;
            filter_width_vec = _filter_width_vec;
            n_feat_map = _n_feat_map;
        }
        inline size_t GetDimNum() {
            return n_dim;
        }
        inline void SetFCDims(std::vector<size_t> &_fc_dims) {
            fc_dims = _fc_dims;
        }
        inline ConvLayer* GetConvLayerPtr() const {
            return this->conv_layer;
        }
        inline FullyConnectedLayer* GetFCLayerPtr() const {
            return this->fc_layer;
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

        bool Forward(mat::Mat &input, mat::Mat &conv_output, mat::Mat &h_output, std::map<size_t, size_t> &max_idx_dict) const {
            // users have to make sure h1_output, h2_output, h3_output have enough memory.
            conv_layer->ForwardNoGemm(input, conv_output, max_idx_dict);
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
