#ifndef LSH_H
#define LSH_H

#include <vector>

class LSH {
    public:
        inline static size_t hamming_dist(const std::vector<bool> &left, const std::vector<bool> &right) {
            size_t dist = 0;
            for(size_t i = 0; i < left.size(); ++ i) {
                if(left[i] != right[i]) dist ++;
            }
            return dist;
        }
};


#endif /*LSH_H*/
