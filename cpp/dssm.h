#ifndef DSSM_H
#define DSSM_H

#include "mat.h"
#include "layers.h"
#include <cstdio>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

class DSSM {
    public:
        DSSM() {}
        DSSM(std::vector<size_t> _dims) : dims(_dims) {}
        virtual ~DSSM() {
            delete fc_layer1;
            delete fc_layer2;
            delete fc_layer3;
            delete activation_layer;
        }
        inline void SetDims(std::vector<size_t> _dims) {
            dims = _dims;
        }
        inline std::vector<size_t> GetDims() {
            return dims;
        }
        bool Init(const char *params_file, const char *model_file) {
            return LoadParams(params_file) && LoadModel(model_file);
        }
        bool LoadParams(const char *params_file) {
            std::ifstream fin(params_file);
            if(fin.fail()) {
                fprintf(stderr, "%s open failed.\n", params_file);
                return false;
            }
            std::string line, key;
            size_t val;
            while(std::getline(fin, line)) {
                std::istringstream iss(line);
                iss >> key;
                if (key == "n_dims") {
                    while(iss >> val) {
                        dims.push_back(val);
                        iss.ignore(32, ',');
                    }
                }
            }
            return true;
        }
        std::string ToString() {
            std::ostringstream oss;
            oss << "fc dims: (";
            for(size_t i = 0; i < dims.size(); ++ i) {
                oss << dims[i] << ",";
            }
            oss << ")";
            return oss.str();
        }
        bool LoadModel(const char *model_file) {
            fc_layer1 = new FullyConnectedLayer(dims[0], dims[1]);
            fc_layer2 = new FullyConnectedLayer(dims[1], dims[2]);
            fc_layer3 = new FullyConnectedLayer(dims[2], dims[3]);
            activation_layer = new ActivationLayer(Tanh);
            FILE *fp = fopen(model_file, "rb");
            // TODO: check model file valid
            if(fp) {
                fc_layer1->LoadParams(fp);
                fc_layer2->LoadParams(fp);
                fc_layer3->LoadParams(fp);
                return true;
                fclose(fp);
            }
            return false;
        }

        bool Forward(mat::Mat &input, mat::Mat &h1_output, mat::Mat &h2_output, mat::Mat &h3_output) const {
            // users have to make sure h1_output, h2_output, h3_output have enough memory.
            fc_layer1->Forward(input, h1_output);
            activation_layer->Forward(h1_output);
            fc_layer2->Forward(h1_output, h2_output);
            activation_layer->Forward(h2_output);
            fc_layer3->Forward(h2_output, h3_output);
            activation_layer->Forward(h3_output);
            
            return true;
        }

    private:
        FullyConnectedLayer *fc_layer1;
        FullyConnectedLayer *fc_layer2;
        FullyConnectedLayer *fc_layer3;
        ActivationLayer *activation_layer;
        std::vector<size_t> dims;

        DSSM(const DSSM &other);
        DSSM& operator=(const DSSM &other);
};
#endif /*DSSM_H*/
