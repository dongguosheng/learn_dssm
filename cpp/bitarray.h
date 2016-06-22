#ifndef BITARRAY_H
#define BITARRAY_H

#include <vector>
#include <iostream>
#include <cassert>

static unsigned char bit_table[] = {128, 64, 32, 16, 8, 4, 2, 1};
static unsigned char bit_count[] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

class BitArray {
    public:
    static inline std::vector<unsigned char> Bool2uchar(const std::vector<bool> &bits) {
        // if(bits.size() % 8 != 0)    std::cout << "bits size is " << bits.size() << std::endl;
        unsigned int len = bits.size() / 8 + (bits.size() % 8 != 0 ? 1 : 0);
        std::vector<unsigned char> result(len, 0);
        unsigned char e = 0;
        unsigned int cnt = 0;
        for(size_t i = 0; i < bits.size(); ++ i) {
            if(cnt == 8) {
                result[(i-1) / 8] = e;
                cnt = 0;
                e = 0;
            }
            e += (bits[i] ? bit_table[cnt] : 0);
            cnt ++;
        }
        if (cnt > 0)    result[len-1] = e;
        return result;
    }
    static inline std::vector<unsigned char> Bool2uchar(std::vector<bool>::iterator left, std::vector<bool>::iterator right) {
    	std::vector<bool> bits(left, right);
    	return Bool2uchar(bits);
    }
    static inline unsigned char GetBitCount(unsigned char a) {
        unsigned char cnt = 0;
        while(a) {
            a &= (a - 1);
            cnt ++;
        }
        return cnt;
    }
    static inline unsigned int CalHammingDist(const std::vector<unsigned char> &a, const std::vector<unsigned char> &b) {
        assert(a.size() == b.size());
        unsigned int dist = 0;
        for(size_t i = 0; i < a.size(); ++ i) {
            dist += bit_count[a[i] ^ b[i]];
        }
        return dist;
    }
    static inline unsigned int CalHammingDistArr(const unsigned char *a, const unsigned char *b, int size) {
        unsigned int dist = 0;
        for(int i = 0; i < size; ++ i) {
            dist += bit_count[a[i] ^ b[i]];
        }
        return dist;
    }
    static inline unsigned int CalHammingDist(const uint64_t *a, const uint64_t *b, int size) {
        unsigned int dist = 0;
        for(int i = 0; i < size; ++ i) {
            uint64_t tmp = a[i] ^ b[i];
            for(size_t j = 0; j < 8; ++ j) {
                dist += bit_count[(tmp >> (j*8)) & 0xff];
            }
        }
        return dist;
    }
};

#endif /*BITARRAY_H*/
