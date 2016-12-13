#include <iostream>
#include "mat.h"

using namespace std;

int main() {
    const int m = 3;
    const int k = 3;
    const int n = 2;
    // test gemm
    float mat_a[m * k] = {1, 2, 4, 1, 2, 4, 1, 2, 4};
    float mat_b[k * n] = {1, 2, 3, 4, 5, 6};
    float mat_rs[m * n] = {3, 1, 3, 2, 3, 3};
    mat::Mat A(mat_a, m, k), B(mat_b, k, n), C(mat_rs, m, n);
    cout << "A: " << A.Shape() << "\n" << A.ToString()
         << "\nB: " << B.Shape() << "\n" << B.ToString()
         << "\nC: " << C.Shape() << "\n" << C.ToString() << endl;
    // C = gemm(A, B);  // dot(A, B) + C
    mat::sgemm(A, B, C);
    cout << "C = dot(A, B) + C " << C.Shape() << "\n" << C.ToString() << endl;
    return 0;
}
