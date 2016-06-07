#ifndef MAT_TEST_H
#define MAT_TEST_H

#include <cstdio>
#include <sstream>
#include <string>
#include <cstring>
#include <cmath>
extern "C" {
#include <cblas.h>
};

namespace mat {
struct Mat {
    size_t n_row;
    size_t n_col;
    float *dptr;
    Mat() {}
    Mat(float *dptr, size_t n_row, size_t n_col) : n_row(n_row), n_col(n_col), dptr(dptr) {}
    Mat deepcopy(const Mat &mat) {
        Mat rs;
        memcpy(rs.dptr, mat.dptr, sizeof(float) * mat.n_row * mat.n_col);
        rs.n_row = mat.n_row;
        rs.n_col = mat.n_col;
        return rs;
    }
    Mat& operator=(const Mat &mat) {
        memcpy(dptr, mat.dptr, sizeof(float) * mat.n_row * mat.n_col);
        n_row = mat.n_row;
        n_col = mat.n_col;
        return *this;
    }
    float* operator[](size_t i) {
        return dptr + i*n_col;
    }
    inline void Set(float *_dptr, size_t _n_row, size_t _n_col) {
        dptr = _dptr;
        n_row = _n_row;
        n_col = _n_col;
    }
    inline void Reset() {
        memset(dptr, 0, sizeof(float) * n_row * n_col);
    }
    inline float* GetRow(size_t i) const {
        return dptr + i;
    }
    inline void Save(const char *filename) {
        FILE *fp = fopen(filename, "wb");
        if(fp) {
            fwrite(dptr, sizeof(float), n_row * n_col, fp);
        }
        fclose(fp);
    }
    inline std::string shape() {
        std::stringstream ss;
        ss << "(" << n_row << ", " << n_col << ")";
        return ss.str();
    }
    inline std::string to_string() {
        std::stringstream ss;
        for(size_t i = 0; i < n_row*n_col; ++ i) {
            ss << dptr[i] << " ";
            if((i + 1) % n_col == 0)  ss << "\n";
        }
        return ss.str();
    }
};

// c = a * b + c
static inline void sgemm(const Mat &a, const Mat &b, const Mat &c) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a.n_row, b.n_col, a.n_col, 1, a.dptr, a.n_col, b.dptr, b.n_col, 1, c.dptr, c.n_col); 
}
static inline float sigmoid(float a) {
    return 1.0f / (1.0 + exp(-a));
}
static inline void sigmoid(Mat &m) {
    for(size_t i = 0; i < m.n_row * m.n_col; ++ i) {
        m.dptr[i] = sigmoid(m.dptr[i]);
    }
}

}
#endif /*MAT_H*/
