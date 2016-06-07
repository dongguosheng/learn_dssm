#ifndef ITQLSH_H
#define ITQLSH_H

#include "Eigen/Core"
#include "Eigen/Eigen"
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <tr1/random>
#include <omp.h>
#include <cstdio>
#include <string>
#include <sstream>
#include <cstdlib>

namespace lsh {
using namespace Eigen;
class ITQLSH {
    public:
        ITQLSH() {}
        ITQLSH(int n_bit, int n_dim, int n_table, float sample_rate, int n_iter)
            : n_bit(n_bit), n_dim(n_dim), n_table(n_table), sample_rate(sample_rate), n_iter(n_iter) {}
        virtual ~ITQLSH() {}
        inline void Train(const MatrixXf &data_mat) {
            srand(time(NULL));
            int n_sample = int(data_mat.rows() * sample_rate);
            std::cout << "Sample Num: " << n_sample << std::endl;
            for(int i = 0; i < n_table; ++ i) {
                MatrixXf sample_mat = GetSampleMat(data_mat, n_sample);
                std::cout << "Sample Done." << std::endl;
                pca(sample_mat);
                std::cout << "PCA Done." << std::endl;
                MatrixXf r_mat(n_bit, n_bit);
                InitR(r_mat);
                std::cout << "Init R Done." << std::endl;
                MatrixXf v_mat = sample_mat * pca_vec.back();
                MatrixXf b_mat(n_sample, n_bit);
                for(int j = 0; j < n_iter; ++ j) {
                    std::cout << "n_table: " << i+1 << ", n_iter: " << j+1 << "(" << n_iter << ")\r" << std::flush;
                    b_mat = Sign(v_mat * r_mat);
                    JacobiSVD<MatrixXf> svd(b_mat.transpose() * v_mat, ComputeThinU | ComputeThinV);
                    r_mat = svd.matrixV() * svd.matrixU().transpose();
                }
                r_vec.push_back(r_mat);
                std::cout << std::endl;
            }
        }
        inline MatrixXf GetSampleMat(const MatrixXf &data_mat, int n_sample) {
            MatrixXf sample_mat(n_sample, data_mat.cols());
            std::tr1::uniform_int<int> ud(0, data_mat.rows() - 1);
            std::set<int> row_idx_set;
            std::tr1::mt19937 rng(rand());
            int row_idx;
            while(static_cast<int>(row_idx_set.size()) < n_sample) {
                row_idx = ud(rng);
                if(row_idx_set.find(row_idx) == row_idx_set.end()) {
                    sample_mat.row(row_idx_set.size()) = data_mat.row(row_idx);
                    row_idx_set.insert(row_idx);
                }
            }
            return sample_mat;
        }
        inline void InitR(MatrixXf &r_mat) {
            std::tr1::mt19937 rng(rand());
            std::tr1::normal_distribution<float> nd;
            std::tr1::variate_generator<std::tr1::mt19937, std::tr1::normal_distribution<float> > gen(rng, nd);
            for(int i = 0; i < r_mat.rows(); ++ i) {
                for(int j = 0; j < r_mat.cols(); ++ j) {
                    r_mat(i, j) = gen();
                }
            }
        }
        inline MatrixXf Sign(const MatrixXf &mat) {
            MatrixXf rs_mat(mat.rows(), mat.cols());
            # pragma omp parallel for collapse(2)
            for(int i = 0; i < mat.rows(); ++ i) {
                for(int j = 0; j < mat.cols(); ++ j) {
                    rs_mat(i, j) = (mat(i, j) > 0 ? 1.0 : -1.0);
                }
            }
            return rs_mat;
        }
        // pay attention to std::vector<bool>
        inline std::vector<MatrixXf> Hash(const MatrixXf &input) {
            std::vector<MatrixXf> hash_mats;
            hash_mats.reserve(n_table);
            assert(input.cols() == n_dim);
            assert(pca_vec.size() == r_vec.size());
            assert(pca_vec.size() == static_cast<size_t>(n_table));
            for(size_t j = 0; j < pca_vec.size(); ++ j) {
                MatrixXf b_mat = (input * pca_vec[j]) * r_vec[j];
                hash_mats.push_back(b_mat);
            }
            return hash_mats;
        }
        inline std::vector<std::vector<bool> > Quant(const std::vector<MatrixXf> &hash_mats) {
            std::vector<std::vector<bool> > bits_vec(hash_mats[0].rows(), std::vector<bool>(n_bit * hash_mats.size(), false));
            for(size_t i = 0; i < hash_mats.size(); ++ i) {
                # pragma omp parallel for
                for(int j = 0; j < hash_mats[i].rows(); ++ j) {
                    for(int k = 0; k < hash_mats[i].row(j).cols(); ++ k) {
                        bits_vec[j][k + i*n_bit] = (hash_mats[i](j, k) > 0 ? true : false);
                    }
                }
            }   
            return bits_vec;
        }
        inline void SaveText(const char *filename) {
            std::ofstream fout(filename);
            if(fout.fail()) {
                std::cerr << filename << " open failed." << std::endl;
                return;
            }
            fout << n_bit << " " << n_dim << " " << n_table << " " << sample_rate << " " << n_iter << std::endl;
            for(size_t i = 0; i < pca_vec.size(); ++ i) {
                fout << pca_vec[i] << std::endl;
            }
            for(size_t i = 0; i < r_vec.size(); ++ i) {
                fout << r_vec[i] << std::endl;
            }
        }
        inline bool LoadText(const char *filename) {
            std::ifstream fin(filename);
            if(fin.fail()) {
                std::cerr << filename << " open failed." << std::endl;
                return false;
            }
            std::string line;
            std::getline(fin, line);
            std::stringstream ss(line);
            ss >> n_bit >> n_dim >> n_table >> sample_rate >> n_iter;
            printf("n_bit: %d, n_dim: %d, n_table: %d, sample_rate: %f, n_iter: %d\n", n_bit, n_dim, n_table, sample_rate, n_iter);
            int table_cnt = 0;
            pca_vec.clear();
            r_vec.clear();
            pca_vec.reserve(n_table);
            r_vec.reserve(n_table);
            while(table_cnt < n_table) {
                MatrixXf pca_tmp(n_dim, n_bit);
                int n_line = 0;
                while(n_line < n_dim) {
                    std::getline(fin, line);
                    float val;
                    std::stringstream ss(line);
                    int idx = 0;
                    while(!ss.eof()) {
                        if(!(ss >> val))    break;
                        // ss.ignore(32, ',');
                        pca_tmp(n_line, idx) = val;
                        idx ++;
                    }
                    n_line ++;
                }
                assert(n_line == n_dim);
                pca_vec.push_back(pca_tmp);
                table_cnt ++;
            }
            std::cout << "Load PCA Mat Complete." << std::endl;
            table_cnt = 0;
            while(table_cnt < n_table) {
                MatrixXf r_tmp(n_bit, n_bit);
                int n_line = 0;
                while(n_line < n_bit) {
                    std::getline(fin, line);
                    float val;
                    std::stringstream ss(line);
                    int idx = 0;
                    while(!(ss.eof())) {
                        if(!(ss >> val))    break;
                        // ss.ignore(32, ',');
                        r_tmp(n_line, idx) = val;
                        idx ++;
                    }
                    n_line ++;
                }
                assert(n_line == n_bit);
                r_vec.push_back(r_tmp);
                table_cnt ++;
            }
            std::cout << "Load R Mat Complete." << std::endl;
            std::cout << "ITQ Model Load Complete." << std::endl;
            return true;
        }
        inline int GetBitNum() {
        	return n_bit;	
        }
        inline int GetTableNum() {
        	return n_table;
        }
    private:
        inline void pca(const MatrixXf &sample_mat) {
            MatrixXf centered = sample_mat.rowwise() - sample_mat.colwise().mean();
            MatrixXf cov = (centered.transpose() * centered) / static_cast<float>(sample_mat.rows() - 1);
            SelfAdjointEigenSolver<MatrixXf> eig(cov);
            MatrixXf pca_mat = eig.eigenvectors().rightCols(n_bit);
            pca_vec.push_back(pca_mat);
        }

        int n_bit;
        int n_dim;
        int n_table;
        float sample_rate;
        int n_iter;
        std::vector<MatrixXf> pca_vec;
        std::vector<MatrixXf> r_vec;
};
};

#endif /*ITQLSH_H*/
