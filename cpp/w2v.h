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
        bool Init(size_t n_dim, const char *vocab_file, const char *syn0_file, const char *rewrite_file) {
            SetDim(n_dim);
            return LoadModel(vocab_file, syn0_file, rewrite_file);
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
        inline bool GetMat(const std::vector<std::string> &words, mat::Mat &words_mat, std::vector<bool> &is_in_w2v) {
            bool flag = false;
            std::ostringstream ss;
            size_t i = 0, j = 0;
            while(i < words.size()) {
                if (vocab_map.find(words[i]) == vocab_map.end()) {
                    is_in_w2v.push_back(false);
                } else {
                    is_in_w2v.push_back(true);
                    ss << words[i] << ", ";
                    mat::Mat word_vec(syn0[vocab_map[words[i]]], 1, n_dim);
                    words_mat.SubMat(j, j+1).deepcopy(word_vec);
                    ++ j;
                    if(words[i] != "#")     flag = true;
                }
                ++ i;
            }
            words_mat.SetRowNum(j);
            // std::cerr << "sub words_mat " << words_mat.ToString() << std::endl;
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
                fprintf(stderr, "%s open failed.\n", vocab_file);
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
            fprintf(stderr, "Total Vocab Size: %lu.\n", vocab_map.size());
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
                fprintf(stderr, "%s open failed.\n", rewrite_file);
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
            fprintf(stderr, "Total Rewrite Size: %lu.\n", rewrite_map.size());
            fprintf(stderr, "Black Term Cnt: %lu.\n", black_term_set.size());
            fprintf(stderr, "Save Term Cnt: %lu.\n", save_term_set.size());
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
