#ifndef W2V_H
#define W2V_H

#include <tr1/unordered_map>
#include <tr1/unordered_set>
#include "mat.h"
#include <string>
#include <cstdio>
#include <vector>
#include <fstream>
#include <sstream>

class W2V {
    public:
        W2V() {}
        bool LoadModel(const char *vocab_file, const char *syn0_file, const char *rewrite_file) {
            bool result = false;
            result = (LoadVocab(vocab_file) && LoadW2VBin(syn0_file) && LoadRewrite(rewrite_file));
            return result;
        }
        ~ W2V () {
            delete [] syn0.dptr;
        }
        inline void SetDim(size_t _n_dim) {
            n_dim = _n_dim;
        }
        inline size_t GetDim() const {
            return this->n_dim;
        }
        inline std::vector<float> MaxPool(const std::vector<std::vector<float> > &features) {
            std::vector<float> result;
            if (features.empty())   return result;
            for(size_t i = 0; i < features[0].size(); ++ i) {
                float max = features[0][i];
                for(size_t j = 1; j < features.size(); ++ j) {
                    if(features[j][i] > max)    max = features[j][i];
                }
                result.push_back(max);
            }
            return result;
        }
        inline bool GetMat(const std::vector<std::string> &words, mat::Mat &words_mat, bool is_rewrite) {
            std::vector<std::string> words_new;
            if (is_rewrite) {
                words_new = GetRewrite(words);
            } else {
                words_new = words;
            }
            bool flag = false;
            std::ostringstream ss;
            for(size_t i = 0; i < words_new.size(); ++ i) {
                if (vocab_map.find(words_new[i]) == vocab_map.end())    continue;
                ss << words_new[i] << ", "; 
                mat::Mat word_vec(syn0[vocab_map[words_new[i]]], 1, n_dim);
                words_mat.SubMat(i, i+1).deepcopy(word_vec);
                flag = true;
                // std::cerr << "sub words_mat " << words_mat.ToString() << std::endl;
            }
            if (flag)   fprintf(stderr, "[cdssm] %s\n", ss.str().c_str());
            return flag;
        }
        inline std::vector<float> GetVec(const std::vector<std::string> &words, bool is_rewrite) {
            std::vector<std::string> words_new;
            if (is_rewrite) {
                words_new = GetRewrite(words);
            } else {
                words_new = words;
            }
            std::vector<std::vector<float> > features;
            for(size_t i = 0; i < words_new.size(); ++ i) {
                if (vocab_map.find(words_new[i]) == vocab_map.end())    continue;
                std::vector<float> tmp_vec(syn0[vocab_map[words_new[i]]], syn0[vocab_map[words_new[i]]] + n_dim);
                features.push_back(tmp_vec);         
            }
            return MaxPool(features);
        }
        // Rewrite Query Words
        // 暂时没必要AC自动机
        inline std::vector<std::string> GetRewrite(const std::vector<std::string> &words) {
            std::vector<std::string> result;
            for(size_t i = 0; i < words.size(); ++ i) {
                if(black_term_set.find(words[i]) != black_term_set.end()) {
                    result.clear();
                    break;
                }
                if(save_term_set.find(words[i]) != save_term_set.end()) {
                    result.clear();
                    result.push_back(words[i]);
                    break;
                }
                if (rewrite_map.find(words[i]) != rewrite_map.end()) {
                    result.insert(result.end(), rewrite_map[words[i]].begin(), rewrite_map[words[i]].end());
                } else {
                    result.push_back(words[i]);
                }
            }
            return result;
        }
    private:
        bool LoadVocab(const char *vocab_file) {
            std::ifstream fin(vocab_file);
            if(fin.fail()) {
                std::cerr << vocab_file << " open failed." << std::endl;
                return false;
            }
            std::string line;
            while(std::getline(fin, line)) {
                std::stringstream ss(line);
                std::string word;
                size_t idx;
                ss >> word >> idx;
                vocab_map[word] = idx;
            }
            std::cerr << "Total Vocab Size: " << vocab_map.size() << std::endl;
            return true;
        }
        bool LoadW2VBin(const char *syn0_file) {
            FILE *fp = fopen(syn0_file, "rb");
            float *buf = NULL;
            size_t n_row = 0, n_col = 0;
            if (fp) {
                // std::cout << sizeof(size_t) << std::endl;
                fread(&n_row, 1, sizeof(int), fp);
                fread(&n_col, 1, sizeof(int), fp);
                fprintf(stderr, "n_row: %lu, n_col: %lu\n", n_row, n_col);
                buf = new float[n_row * n_col];
                fread(buf, sizeof(float), n_row * n_col, fp);
                syn0.Set(buf, n_row, n_col);
                fclose(fp);
                return true;
            }
            return false;
        }
        // for Rewrite Query Words
        // 0. query pattern blacklist
        // 1. terms left
        // 2. terms rewrite
        bool LoadRewrite(const char *rewrite_file) {
            std::ifstream fin(rewrite_file);
            if(fin.fail()) {
                std::cerr << rewrite_file << " open failed." << std::endl;
                return false;
            }
            std::string line;
            while(std::getline(fin, line)) {
                std::stringstream ss(line);
                int flag;
                std::string word;
                std::string rewrite_word;
                std::vector<std::string> rewrite_words;
                ss >> flag;
                switch (flag) {
                    case 0: 
                        ss >> word;
                        black_term_set.insert(word);
                        break;
                    case 1:
                        ss >> word;
                        save_term_set.insert(word);
                        break;
                    case 2:
                        ss >> word;
                        while(!ss.eof()) {
                            if (!(ss >> rewrite_word))  break;
                            rewrite_words.push_back(rewrite_word);
                        }
                        rewrite_map[word] = rewrite_words;
                        break;
                    default:
                        break;
                }
            }
            std::cerr << "Total Rewrite Size: " << rewrite_map.size() << std::endl;
            std::cerr << "Black Term Cnt: " << black_term_set.size() << std::endl;
            std::cerr << "Save Term Cnt: " << save_term_set.size() << std::endl;
            // for(std::tr1::unordered_map<std::string, std::vector<std::string> >::iterator iter = rewrite_map.begin(); iter != rewrite_map.end(); ++ iter) {
            //     std::cerr << iter->first << std::endl;
            //     for(size_t i = 0; i < iter->second.size(); ++ i) {
            //         std::cerr << iter->second[i] << " ";
            //     }
            //     std::cerr << std::endl;
            // }
            return true;           
        }
        std::tr1::unordered_map<std::string, size_t> vocab_map;
        mat::Mat syn0;
        size_t n_dim;
        std::tr1::unordered_map<std::string, std::vector<std::string> > rewrite_map;
        std::tr1::unordered_set<std::string> black_term_set;
        std::tr1::unordered_set<std::string> save_term_set;

        W2V(const W2V &other);
        W2V& operator=(const W2V &other);
};

#endif /*W2V_H*/
