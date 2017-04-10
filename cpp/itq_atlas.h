#ifndef ITQ_ATLAS_H
#define ITQ_ATLAS_H

#include "mat.h"
#include "lsh.h"
#include <vector>
#include <fstream>
#include <sstream>

namespace lsh {
class ITQ : public LSH {
    public:
        ITQ() {}
        ITQ(size_t _n_bit, size_t _n_dim, size_t _n_table)
            : n_bit(_n_bit), n_dim(_n_dim), n_table(_n_table) {}
        virtual ~ITQ() {
            for(size_t i = 0; i < n_table; ++ i) {
                delete [] pca_vec[i].dptr;
                delete [] r_vec[i].dptr;
            }
        }
        inline void Set(size_t _n_bit, size_t _n_dim, size_t _n_table) {
            n_bit = _n_bit;
            n_dim = _n_dim;
            n_table = _n_table;
        }
        inline size_t GetBitNum() const {
            return this->n_bit;
        }
        inline size_t GetDim() const {
            return this->n_dim;
        }
        inline size_t GetTableNum() const {
            return this->n_table;
        }
        bool Init(const char *params_file, const char *model_file) {
            return LoadParams(params_file) && LoadModel(model_file);
        }
        bool LoadParams(const char *params_file) {
            std::ifstream fin(params_file);
            if (fin.fail()) {
                fprintf(stderr, "%s open failed.\n", params_file);
                return false;
            }
            std::string line;
            while(getline(fin, line)) {
                std::istringstream iss(line);
                std::string key;
                size_t val;
                iss >> key >> val;
                if (key == "n_bit")         n_bit = val;
                else if (key == "n_dim")    n_dim = val;
                else if (key == "n_table") n_table = val;
            }
            return true;
        }
        bool LoadModel(const char *model_file) {
            pca_vec.resize(n_table);
            r_vec.resize(n_table);
            FILE *fp = fopen(model_file, "rb");
            // TODO: check model file valid
            if(fp) {
                for(size_t i = 0; i < n_table; ++ i) {
                    float *pca_ptr = new float[n_dim * n_bit];
                    float *r_ptr = new float[n_bit * n_bit];
                    fread(pca_ptr, 1, sizeof(float) * n_dim * n_bit, fp);
                    fread(r_ptr, 1, sizeof(float) * n_bit * n_bit, fp);
                    pca_vec[i].Set(pca_ptr, n_dim, n_bit);
                    r_vec[i].Set(r_ptr, n_bit, n_bit);
                }
                fclose(fp);
                return true;
            }
            return false;
        }
        inline const std::vector<std::vector<bool> > Hash(const mat::Mat &input, mat::Mat &pca_rs, mat::Mat &relax) {
            std::vector<std::vector<bool> > rs;
            for(size_t i = 0; i < n_table; ++ i) {
                std::vector<bool> hash_rs(n_bit);
                pca_rs.Reset();
                relax.Reset();
                // pca_rs = gemm(input, pca_vec[i]);
                // relax = gemm(pca_rs, r_vec[i]);
                sgemm(input, pca_vec[i], pca_rs);
                sgemm(pca_rs, r_vec[i], relax);
                for(size_t i = 0; i < relax.n_col; ++ i) {
                    hash_rs[i] = relax.dptr[i] > 0 ? true : false;
                }
                rs.push_back(hash_rs);
            }
            return rs;
        }
        inline std::string ToString() {
            std::ostringstream oss;
            oss << "n_bit: " << n_bit << ", n_dim: " << n_dim << ", n_table: " << n_table;
            return oss.str();
        }

    private:
        size_t n_bit;
        size_t n_dim;
        size_t n_table;
        std::vector<mat::Mat> pca_vec;
        std::vector<mat::Mat> r_vec;
        ITQ(const ITQ &other);
        ITQ& operator=(const ITQ &other);
};
}
#endif /*ITQ_ATLAS_H*/
