#ifndef ITQ_ATLAS_H
#define ITQ_ATLAS_H

#include "mat.h"
#include "lsh.h"
#include <vector>

namespace lsh {
class ITQ : public LSH {
    public:
        ITQ() {}
        ITQ(size_t n_bit, size_t n_dim, size_t n_table)
            : n_bit(n_bit), n_dim(n_dim), n_table(n_table) {}
        virtual ~ITQ() {
            for(size_t i = 0; i < n_table; ++ i) {
                delete [] pca_vec[i].dptr;
                delete [] r_vec[i].dptr;
            }
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
