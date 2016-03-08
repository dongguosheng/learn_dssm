#ifndef MAT_H
#define MAT_H

extern "C" {
#include <cblas.h>
};
#include <cstdlib>
#include <sstream>
#include <cstring>
#include <string>
#include <cmath>

template<typename SubType>
struct Exp {
    inline const SubType& self(void) const {
        return *static_cast<const SubType*>(this);
    }
};

template<typename TLhs, typename TRhs>
struct BinaryExp: public Exp<BinaryExp<TLhs, TRhs> > {
    const TLhs &lhs;
    const TRhs &rhs;
    BinaryExp(const TLhs &lhs, const TRhs &rhs) : lhs(lhs), rhs(rhs) {}
};

struct Mat: public Exp<Mat> {
    size_t n_row;   // when n_row is 1, Mat -> Vec
    size_t n_col;
    float *dptr;    // do not allocate and de-allocate memory
    Mat() {}
    Mat(float* dptr, size_t n_row, size_t n_col) : n_row(n_row), n_col(n_col), dptr(dptr) {}

    Mat(const Mat &mat) {
        memcpy(dptr, mat.dptr, sizeof(float) * mat.n_row * mat.n_col);
        n_row = mat.n_row;
        n_col = mat.n_col;
    }
    Mat& operator=(const Mat &mat) {
        memcpy(dptr, mat.dptr, sizeof(float) * mat.n_row * mat.n_col);
        n_row = mat.n_row;
        n_col = mat.n_col;
        return *this;
    }

    void Set(float *_dptr, size_t _n_row, size_t _n_col) {
        dptr = _dptr;
        n_row = _n_row;
        n_col = _n_col;
    }

    void Reset() {
        memset(dptr, 0, sizeof(float) * n_row * n_col);
    }

    template<typename EType>
    inline Mat &operator=(const Exp<EType> &src_) {
        const EType &src = src_.self();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, src.lhs.n_row, src.rhs.n_col, src.lhs.n_col, 1, src.lhs.dptr, src.lhs.n_col, src.rhs.dptr, src.rhs.n_col, 1, dptr, n_col);
        return *this;
    }
    
    inline float sigmoid(float a) {
        return 1.0f / (1.0 + exp(-a));
    }

    inline void sigmoid() {
        for(size_t i = 0; i < n_row * n_col; ++ i) {
            dptr[i] = sigmoid(dptr[i]);
        }
    }

    inline std::string to_string() {
        std::stringstream ss;
        for(size_t i = 0; i < n_row * n_col; ++ i) {
            ss << dptr[i] << " ";
            if((i + 1) % n_col == 0)  ss << "\n";
        }
        return ss.str();
    }

    inline std::string shape() {
        std::stringstream ss;
        ss << "(" << n_row << ", " << n_col << ")";
        return ss.str();
    }
};

template<typename TLhs, typename TRhs>
inline BinaryExp<TLhs, TRhs> gemm(const Exp<TLhs> &lhs, const Exp<TRhs> &rhs) {
    return BinaryExp<TLhs, TRhs>(lhs.self(), rhs.self());
}


#endif /*MAT_H*/
