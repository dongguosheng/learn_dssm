#ifndef MAT_H
#define MAT_H

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
    Mat() { dptr = NULL; }
    Mat(float *dptr, size_t n_row, size_t n_col) : n_row(n_row), n_col(n_col), dptr(dptr) {}
    inline void deepcopy(const Mat &mat) {
        memcpy(dptr, mat.dptr, sizeof(float) * mat.n_row * mat.n_col);
        n_row = mat.n_row;
        n_col = mat.n_col;
    }
    Mat& operator+=(const Mat &rhs) {
        size_t n_total = n_row * n_col;
        for(size_t i = 0; i < n_total; ++ i) {
            this->dptr[i] += rhs.dptr[i];
        }
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
        return dptr + i*n_col;
    }
    inline Mat SubMat(size_t start_row, size_t end_row) {
        Mat mat(GetRow(start_row), end_row - start_row, n_col);
        return mat;
    }
    inline Mat Reshape(size_t n_row_new, size_t n_col_new) {
        this->n_row = n_row_new;
        this->n_col = n_col_new;
        return *this;
    }
    inline void Save(const char *filename) {
        FILE *fp = fopen(filename, "wb");
        if(fp) {
            fwrite(dptr, sizeof(float), n_row * n_col, fp);
        }
        fclose(fp);
    }
    inline std::string Shape() const {
        std::stringstream ss;
        ss << "(" << n_row << ", " << n_col << ")";
        return ss.str();
    }
    inline std::string ToString() const {
        std::stringstream ss;
        for(size_t i = 0; i < n_row*n_col; ++ i) {
            ss << dptr[i] << " ";
            if((i + 1) % n_col == 0)  ss << "|\n";
        }
        return ss.str();
    }
};

// c = dot(a, b) + c
static inline void sgemm(const Mat &a, const Mat &b, const Mat &c) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a.n_row, b.n_col, a.n_col, 1, a.dptr, a.n_col, b.dptr, b.n_col, 1, c.dptr, c.n_col); 
}

}
#endif /*MAT_H*/
