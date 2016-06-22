#ifndef RHPLSH_H
#define RHPLSH_H

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
class RHPLSH {
    public:
        RHPLSH() {}
        RHPLSH(int n_bit, int n_dim, int n_table)
            : n_bit(n_bit), n_dim(n_dim), n_table(n_table) {}
        virtual ~RHPLSH() {}
        inline void Train(void) {
			srand(time(NULL));
            for(int i = 0; i < n_table; ++ i) {
                Eigen::MatrixXf r_mat(n_dim, n_bit);
                InitR(r_mat);
                std::cout << "Init R Done." << std::endl;
            	std::cout << "r_vec: " << r_mat.rows() << ", " << r_mat.cols() << std::endl;
                r_vec.push_back(r_mat);
                std::cout << std::endl;
            }
        }
        inline void InitR(Eigen::MatrixXf &r_mat) {
            std::tr1::mt19937 rng(rand());
			std::tr1::normal_distribution<float> nd;
    	    std::tr1::variate_generator<std::tr1::mt19937, std::tr1::normal_distribution<float> > gen(rng, nd);
            for(int i = 0; i < r_mat.rows(); ++ i) {
                for(int j = 0; j < r_mat.cols(); ++ j) {
                    r_mat(i, j) = gen();
                }
            }
        }
        inline Eigen::MatrixXf Sign(const Eigen::MatrixXf &mat) {
            Eigen::MatrixXf rs_mat(mat.rows(), mat.cols());
            # pragma omp parallel for collapse(2)
            for(int i = 0; i < mat.rows(); ++ i) {
                for(int j = 0; j < mat.cols(); ++ j) {
                    rs_mat(i, j) = (mat(i, j) > 0 ? 1.0 : -1.0);
                }
            }
            return rs_mat;
        }
        // pay attention to std::vector<bool>
        inline std::vector<Eigen::MatrixXf> Hash(const Eigen::MatrixXf &input) {
            std::vector<Eigen::MatrixXf> hash_mats;
            hash_mats.reserve(n_table);
            assert(input.cols() == n_dim);
            assert(r_vec.size() == static_cast<size_t>(n_table));
            for(size_t j = 0; j < r_vec.size(); ++ j) {
                Eigen::MatrixXf b_mat = input * r_vec[j];
                hash_mats.push_back(b_mat);
            }
            return hash_mats;
        }
        inline std::vector<std::vector<bool> > Quant(const std::vector<Eigen::MatrixXf> &hash_mats) {
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
            fout << n_bit << " " << n_dim << " " << n_table << std::endl;
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
            ss >> n_bit >> n_dim >> n_table;
            printf("n_bit: %d, n_dim: %d, n_table: %d\n", n_bit, n_dim, n_table);
            int table_cnt = 0;
            r_vec.clear();
            r_vec.reserve(n_table);
            while(table_cnt < n_table) {
                Eigen::MatrixXf r_tmp(n_dim, n_bit);
                int n_line = 0;
                while(n_line < n_dim) {
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
                assert(n_line == n_dim);
                r_vec.push_back(r_tmp);
                table_cnt ++;
            }
            std::cout << "Load R Mat Complete." << std::endl;
            std::cout << "RHP Model Load Complete." << std::endl;
            return true;
        }
        inline int GetBitNum() {
        	return n_bit;	
        }
        inline int GetTableNum() {
        	return n_table;
        }
    private:
        int n_bit;
        int n_dim;
        int n_table;
        std::vector<Eigen::MatrixXf> r_vec;
};
};

#endif /*RHPLSH_H*/
