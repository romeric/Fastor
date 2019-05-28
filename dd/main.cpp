#include <complex>
#include "Fastor.h"
using namespace Fastor;

enum {I,J,K,L,M,N};


int main() {

    constexpr size_t nn = 2;

    using T = std::complex<double>;
    // Tensor<T,5,5> a,b, c;
    Tensor<T,nn,nn,nn> a,b;
    a.iota(1);
    // a.arange();
    // a.zeros();

    // print(DEFAULT_ABI);
    // a += a;
    // a *= a;
    // a /= a;
    // a -= a;
    // print(a.sum());
    print(a);


    // print(sin(a)); // check
    // print(exp(a));

    // b = matmul(a,a);
    // b = transpose(a);
    // print(b);

    // Tensor<T,nn,nn,nn,nn> c = einsum<Index<I,J>,Index<K,L>>(a,a);
    Tensor<T,nn,nn> c = einsum<Index<I,J,K>,Index<I,K,L>>(a,a);
    // Tensor<T,5,5> d = einsum<Index<I,J,K,L>,Index<I,J,K,N>>(c,c);
    print(c);
    // print(sizeof(T));



    return 0;
}