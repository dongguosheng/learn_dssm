#ifndef LSH_H
#define LSH_H

#include <vector>
#include <cmath>
#include <cstdio>

namespace lsh {
class LSH {
    public:
        inline static size_t CalHamming(const std::vector<bool> &left, const std::vector<bool> &right) {
            size_t dist = 0;
            for(size_t i = 0; i < left.size(); ++ i) {
                if(left[i] != right[i]) dist ++;
            }
            return dist;
        }
        inline static float CalCosine(const std::vector<float> &left, const std::vector<float> &right) {
            float dot_product = 0.0f, a = 0.0f, b = 0.0f;
            for(size_t i = 0; i < left.size(); ++ i) {
                dot_product += left[i] * right[i];
                a += left[i] * left[i];
                b += right[i] * right[i];
            }
            float eps = 10e-5f;
            if ((-eps < a && a < eps) || (-eps < b && b < eps)) {
                fprintf(stderr, "a: %f, b: %f\n", a, b);
                return -1;
            }
            return dot_product / std::sqrt(a) / std::sqrt(b);
        }
};
}

#endif /*LSH_H*/
