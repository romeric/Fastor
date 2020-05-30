#include <Fastor/Fastor.h>
using namespace Fastor;

#if FASTOR_CXX_VERSION >= 2017

#define Tol 1e-12
#define BigTol 1e-5
#define HugeTol 1e-2


template<typename T>
void run() {

    using std::abs;
    enum {i,j,k,l,m,n};

    {
        constexpr size_t I=2, J=4, K=3, L=5, M=3;
        Tensor<T,K,L> a; a.iota(3);
        Tensor<T,M,I,L> b; b.iota(7);
        auto b1 = einsum<Index<k,l>,Index<m,i,l>>(a,b);
        // no permutation
        auto b2 = einsum<Index<k,l>,Index<m,i,l>,OIndex<k,m,i>>(a,b);

        for (size_t kk=0; kk<K; ++kk) {
            for (size_t mm=0; mm<M; ++mm) {
                for (size_t ii=0; ii<I; ++ii) {
                    FASTOR_EXIT_ASSERT( std::abs( b1(kk,mm,ii) - b2(kk,mm,ii) ) < Tol );
                }
            }
        }
    }

    {
        constexpr size_t I=2, J=4, K=3, L=5, M=3;
        Tensor<T,K,L> a; a.iota(3);
        Tensor<T,M,I,L> b; b.iota(7);
        auto b1 = einsum<Index<k,l>,Index<m,i,l>>(a,b);
        auto b2 = einsum<Index<k,l>,Index<m,i,l>,OIndex<k,i,m>>(a,b);

        for (size_t kk=0; kk<K; ++kk) {
            for (size_t mm=0; mm<M; ++mm) {
                for (size_t ii=0; ii<I; ++ii) {
                    FASTOR_EXIT_ASSERT( std::abs( b1(kk,mm,ii) - b2(kk,ii,mm) ) < Tol );
                }
            }
        }
    }

    {
        constexpr size_t I=2, J=4, K=3, L=5, M=3;
        Tensor<T,K,L> a; a.iota(3);
        Tensor<T,M,I,L> b; b.iota(7);
        auto b1 = einsum<Index<k,l>,Index<m,i,l>>(a,b);
        auto b2 = einsum<Index<k,l>,Index<m,i,l>,OIndex<i,k,m>>(a,b);

        for (size_t kk=0; kk<K; ++kk) {
            for (size_t mm=0; mm<M; ++mm) {
                for (size_t ii=0; ii<I; ++ii) {
                    FASTOR_EXIT_ASSERT( std::abs( b1(kk,mm,ii) - b2(ii,kk,mm) ) < Tol );
                }
            }
        }
    }

    {
        constexpr size_t I=2, J=4, K=3, L=5, M=3;
        Tensor<T,K,L> a; a.iota(3);
        Tensor<T,M,I,L> b; b.iota(7);
        auto b1 = einsum<Index<k,l>,Index<m,i,l>>(a,b);
        auto b2 = einsum<Index<k,l>,Index<m,i,l>,OIndex<i,m,k>>(a,b);

        for (size_t kk=0; kk<K; ++kk) {
            for (size_t mm=0; mm<M; ++mm) {
                for (size_t ii=0; ii<I; ++ii) {
                    FASTOR_EXIT_ASSERT( std::abs( b1(kk,mm,ii) - b2(ii,mm,kk) ) < Tol );
                }
            }
        }
    }

    {
        constexpr size_t I=2, J=4, K=3, L=5, M=3;
        Tensor<T,K,L> a; a.iota(3);
        Tensor<T,M,I,L> b; b.iota(7);
        auto b1 = einsum<Index<k,l>,Index<m,i,l>>(a,b);
        auto b2 = einsum<Index<k,l>,Index<m,i,l>,OIndex<m,i,k>>(a,b);

        for (size_t kk=0; kk<K; ++kk) {
            for (size_t mm=0; mm<M; ++mm) {
                for (size_t ii=0; ii<I; ++ii) {
                    FASTOR_EXIT_ASSERT( std::abs( b1(kk,mm,ii) - b2(mm,ii,kk) ) < Tol );
                }
            }
        }
    }

    {
        constexpr size_t I=2, J=4, K=3, L=5, M=3;
        Tensor<T,K,L> a; a.iota(3);
        Tensor<T,M,I,L> b; b.iota(7);
        auto b1 = einsum<Index<k,l>,Index<m,i,l>>(a,b);
        auto b2 = einsum<Index<k,l>,Index<m,i,l>,OIndex<m,k,i>>(a,b);

        for (size_t kk=0; kk<K; ++kk) {
            for (size_t mm=0; mm<M; ++mm) {
                for (size_t ii=0; ii<I; ++ii) {
                    FASTOR_EXIT_ASSERT( std::abs( b1(kk,mm,ii) - b2(mm,kk,ii) ) < Tol );
                }
            }
        }
    }


    // expressions
    {
        constexpr size_t I=2, J=4, K=3, L=5, M=3;
        Tensor<T,K,L> a; a.iota(3);
        Tensor<T,M,I,L> b; b.iota(7);
        auto b1 = einsum<Index<k,l>,Index<m,i,l>>(a,b);
        auto b2 = einsum<Index<k,l>,Index<m,i,l>,OIndex<m,k,i>>(a+0,b);

        for (size_t kk=0; kk<K; ++kk) {
            for (size_t mm=0; mm<M; ++mm) {
                for (size_t ii=0; ii<I; ++ii) {
                    FASTOR_EXIT_ASSERT( std::abs( b1(kk,mm,ii) - b2(mm,kk,ii) ) < Tol );
                }
            }
        }
    }

    {
        constexpr size_t I=2, J=4, K=3, L=5, M=3;
        Tensor<T,K,L> a; a.iota(3);
        Tensor<T,M,I,L> b; b.iota(7);
        auto b1 = einsum<Index<k,l>,Index<m,i,l>>(a,b);
        auto b2 = einsum<Index<k,l>,Index<m,i,l>,OIndex<m,k,i>>(a,1*b);

        for (size_t kk=0; kk<K; ++kk) {
            for (size_t mm=0; mm<M; ++mm) {
                for (size_t ii=0; ii<I; ++ii) {
                    FASTOR_EXIT_ASSERT( std::abs( b1(kk,mm,ii) - b2(mm,kk,ii) ) < Tol );
                }
            }
        }
    }

    {
        constexpr size_t I=2, J=4, K=3, L=5, M=3;
        Tensor<T,K,L> a; a.iota(3);
        Tensor<T,M,I,L> b; b.iota(7);
        auto b1 = einsum<Index<k,l>,Index<m,i,l>>(a,b);
        auto b2 = einsum<Index<k,l>,Index<m,i,l>,OIndex<m,k,i>>(a+0,b+1-1);

        for (size_t kk=0; kk<K; ++kk) {
            for (size_t mm=0; mm<M; ++mm) {
                for (size_t ii=0; ii<I; ++ii) {
                    FASTOR_EXIT_ASSERT( std::abs( b1(kk,mm,ii) - b2(mm,kk,ii) ) < Tol );
                }
            }
        }
    }


    print(FGRN(BOLD("All tests passed successfully")));
}

int main() {

    print(FBLU(BOLD("Testing single einsum: single precision")));
    run<float>();
    print(FBLU(BOLD("Testing single einsum: double precision")));
    run<double>();

    return 0;
}

#endif // CXX 2017

