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
        Tensor<T,3,5> a; a.iota(1);
        Tensor<T,4,2,5> b; b.iota(2);
        Tensor<T,4,6,3> c; c.iota(5);
        auto b1 = einsum<Index<k,l>,Index<m,i,l>,Index<m,j,k>>(a,b,c);
        // no permutation
        auto b2 = einsum<Index<k,l>,Index<m,i,l>,Index<m,j,k>,OIndex<i,j>>(a,b,c);

        for (size_t ii=0; ii<2; ++ii) {
            for (size_t jj=0; jj<6; ++jj) {
                FASTOR_EXIT_ASSERT( std::abs( b1(ii,jj) - b2(ii,jj) ) < Tol );
            }
        }
    }

    {
        Tensor<T,3,5> a; a.iota(1);
        Tensor<T,4,2,5> b; b.iota(2);
        Tensor<T,4,6,3> c; c.iota(5);
        auto b1 = einsum<Index<k,l>,Index<m,i,l>,Index<m,j,k>>(a,b,c);
        // transpose
        auto b2 = einsum<Index<k,l>,Index<m,i,l>,Index<m,j,k>,OIndex<j,i>>(a,b,c);

        for (size_t ii=0; ii<2; ++ii) {
            for (size_t jj=0; jj<6; ++jj) {
                FASTOR_EXIT_ASSERT( std::abs( b1(ii,jj) - b2(jj,ii) ) < Tol );
            }
        }

        FASTOR_EXIT_ASSERT( all_of(b1 == trans(b2)) );
    }

    {
        Tensor<T,3,5>   a; a.iota(1);
        Tensor<T,4,2,5> b; b.iota(2);
        Tensor<T,4,6,3> c; c.iota(5);
        auto b1 = einsum<Index<k,l>,Index<m,i,l>,Index<n,j,k>>(a,b,c);
        // no permutation
        auto b2 = einsum<Index<k,l>,Index<m,i,l>,Index<n,j,k>,OIndex<m,i,n,j>>(a,b,c);

        for (size_t mm=0; mm<4; ++mm) {
            for (size_t ii=0; ii<2; ++ii) {
                for (size_t nn=0; nn<4; ++nn) {
                    for (size_t jj=0; jj<6; ++jj) {
                        FASTOR_EXIT_ASSERT( std::abs( b1(mm,ii,nn,jj) - b2(mm,ii,nn,jj) ) < Tol );
                    }
                }
            }
        }
    }

    {
        Tensor<T,3,5>   a; a.iota(1);
        Tensor<T,4,2,5> b; b.iota(2);
        Tensor<T,4,6,3> c; c.iota(5);
        auto b1 = einsum<Index<k,l>,Index<m,i,l>,Index<n,j,k>>(a,b,c);
        auto b2 = einsum<Index<k,l>,Index<m,i,l>,Index<n,j,k>,OIndex<i,m,n,j>>(a,b,c);

        for (size_t mm=0; mm<4; ++mm) {
            for (size_t ii=0; ii<2; ++ii) {
                for (size_t nn=0; nn<4; ++nn) {
                    for (size_t jj=0; jj<6; ++jj) {
                        FASTOR_EXIT_ASSERT( std::abs( b1(mm,ii,nn,jj) - b2(ii,mm,nn,jj) ) < Tol );
                    }
                }
            }
        }
    }

    {
        Tensor<T,3,5>   a; a.iota(1);
        Tensor<T,4,2,5> b; b.iota(2);
        Tensor<T,4,6,3> c; c.iota(5);
        auto b1 = einsum<Index<k,l>,Index<m,i,l>,Index<n,j,k>>(a,b,c);
        auto b2 = einsum<Index<k,l>,Index<m,i,l>,Index<n,j,k>,OIndex<i,m,j,n>>(a,b,c);

        for (size_t mm=0; mm<4; ++mm) {
            for (size_t ii=0; ii<2; ++ii) {
                for (size_t nn=0; nn<4; ++nn) {
                    for (size_t jj=0; jj<6; ++jj) {
                        FASTOR_EXIT_ASSERT( std::abs( b1(mm,ii,nn,jj) - b2(ii,mm,jj,nn) ) < Tol );
                    }
                }
            }
        }
    }

    {
        Tensor<T,3,5>   a; a.iota(1);
        Tensor<T,4,2,5> b; b.iota(2);
        Tensor<T,4,6,3> c; c.iota(5);
        auto b1 = einsum<Index<k,l>,Index<m,i,l>,Index<n,j,k>>(a,b,c);
        auto b2 = einsum<Index<k,l>,Index<m,i,l>,Index<n,j,k>,OIndex<i,j,m,n>>(a,b,c);

        for (size_t mm=0; mm<4; ++mm) {
            for (size_t ii=0; ii<2; ++ii) {
                for (size_t nn=0; nn<4; ++nn) {
                    for (size_t jj=0; jj<6; ++jj) {
                        FASTOR_EXIT_ASSERT( std::abs( b1(mm,ii,nn,jj) - b2(ii,jj,mm,nn) ) < Tol );
                    }
                }
            }
        }
    }

    {
        Tensor<T,3,5>   a; a.iota(1);
        Tensor<T,4,2,5> b; b.iota(2);
        Tensor<T,4,6,3> c; c.iota(5);
        auto b1 = einsum<Index<k,l>,Index<m,i,l>,Index<n,j,k>>(a,b,c);
        auto b2 = einsum<Index<k,l>,Index<m,i,l>,Index<n,j,k>,OIndex<n,m,j,i>>(a,b,c);

        for (size_t mm=0; mm<4; ++mm) {
            for (size_t ii=0; ii<2; ++ii) {
                for (size_t nn=0; nn<4; ++nn) {
                    for (size_t jj=0; jj<6; ++jj) {
                        FASTOR_EXIT_ASSERT( std::abs( b1(mm,ii,nn,jj) - b2(nn,mm,jj,ii) ) < Tol );
                    }
                }
            }
        }
    }

    {
        Tensor<T,3,5>   a; a.iota(1);
        Tensor<T,4,2,5> b; b.iota(2);
        Tensor<T,4,6,3> c; c.iota(5);
        auto b1 = einsum<Index<k,l>,Index<m,i,l>,Index<n,j,k>>(a,b+0,c);
        auto b2 = einsum<Index<k,l>,Index<m,i,l>,Index<n,j,k>,OIndex<n,m,j,i>>(a,b,c);

        for (size_t mm=0; mm<4; ++mm) {
            for (size_t ii=0; ii<2; ++ii) {
                for (size_t nn=0; nn<4; ++nn) {
                    for (size_t jj=0; jj<6; ++jj) {
                        FASTOR_EXIT_ASSERT( std::abs( b1(mm,ii,nn,jj) - b2(nn,mm,jj,ii) ) < Tol );
                    }
                }
            }
        }
    }

    print(FGRN(BOLD("All tests passed successfully")));
}

int main() {

    print(FBLU(BOLD("Testing single einsum: double precision")));
    run<double>();

    return 0;
}

#endif // CXX 2017

