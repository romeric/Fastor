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

    // permuting 2 free indices (results in doing a transpose)
    {
        Tensor<T,2,2,2,2> a; a.iota(3);
        auto b1 = einsum<Index<i,k,j,k>>(a);
        // no permutation
        auto b2 = einsum<Index<i,k,j,k>,OIndex<i,j>>(a);

        FASTOR_EXIT_ASSERT( isequal(b1, b2, Tol) );
    }

    {
        Tensor<T,2,2,2,2> a; a.iota(3);
        auto b1 = einsum<Index<i,k,j,k>>(a);
        auto b2 = einsum<Index<i,k,j,k>,OIndex<j,i>>(a);

        FASTOR_EXIT_ASSERT( isequal(b1, transpose(b2), Tol) );
    }

    {
        Tensor<T,3,3,3,3> a; a.iota(3);
        auto b1 = einsum<Index<i,k,j,k>>(a);
        auto b2 = einsum<Index<i,k,j,k>,OIndex<j,i>>(a);

        FASTOR_EXIT_ASSERT( isequal(b1, transpose(b2), Tol) );
    }

    {
        Tensor<T,3,5,7,5> a; a.iota(3);
        auto b1 = einsum<Index<i,k,j,k>>(a);
        auto b2 = einsum<Index<i,k,j,k>,OIndex<j,i>>(a);

        FASTOR_EXIT_ASSERT( isequal(b1, transpose(b2), Tol) );
    }

    {
        Tensor<T,2,8,5,5> a; a.iota(3);
        auto b1 = einsum<Index<i,j,k,k>>(a);
        auto b2 = einsum<Index<i,j,k,k>,OIndex<j,i>>(a);

        FASTOR_EXIT_ASSERT( isequal(b1, transpose(b2), Tol) );
    }

    {
        Tensor<T,3,4,5,3> a; a.iota(3);
        auto b1 = einsum<Index<l,i,k,l>>(a);
        auto b2 = einsum<Index<l,i,k,l>,OIndex<k,i>>(a);

        FASTOR_EXIT_ASSERT( isequal(b1, transpose(b2), Tol) );
    }


    // permuting 3 free indices
    {
        constexpr size_t I=2, J=4, K=5, L=3;
        Tensor<T,L,I,K,L,J> a; a.iota(3);
        auto b1 = einsum<Index<l,i,k,l,j>>(a);
        // no permutation
        auto b2 = einsum<Index<l,i,k,l,j>,OIndex<i,k,j>>(a);

        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    FASTOR_EXIT_ASSERT( std::abs( b1(ii,kk,jj) - b2(ii,kk,jj) ) < Tol );
                }
            }
        }
    }

    {
        constexpr size_t I=2, J=4, K=5, L=3;
        Tensor<T,L,I,K,L,J> a; a.iota(3);
        auto b1 = einsum<Index<l,i,k,l,j>>(a);
        auto b2 = einsum<Index<l,i,k,l,j>,OIndex<i,j,k>>(a);

        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    FASTOR_EXIT_ASSERT( std::abs( b1(ii,kk,jj) - b2(ii,jj,kk) ) < Tol );
                }
            }
        }
    }

    {
        constexpr size_t I=2, J=4, K=5, L=3;
        Tensor<T,L,I,K,L,J> a; a.iota(3);
        auto b1 = einsum<Index<l,i,k,l,j>>(a);
        auto b2 = einsum<Index<l,i,k,l,j>,OIndex<k,i,j>>(a);

        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    FASTOR_EXIT_ASSERT( std::abs( b1(ii,kk,jj) - b2(kk,ii,jj) ) < Tol );
                }
            }
        }
    }

    {
        constexpr size_t I=2, J=4, K=5, L=3;
        Tensor<T,L,I,K,L,J> a; a.iota(3);
        auto b1 = einsum<Index<l,i,k,l,j>>(a);
        auto b2 = einsum<Index<l,i,k,l,j>,OIndex<k,j,i>>(a);

        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    FASTOR_EXIT_ASSERT( std::abs( b1(ii,kk,jj) - b2(kk,jj,ii) ) < Tol );
                }
            }
        }
    }

    {
        constexpr size_t I=2, J=4, K=5, L=3;
        Tensor<T,L,I,K,L,J> a; a.iota(3);
        auto b1 = einsum<Index<l,i,k,l,j>>(a);
        auto b2 = einsum<Index<l,i,k,l,j>,OIndex<j,i,k>>(a);

        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    FASTOR_EXIT_ASSERT( std::abs( b1(ii,kk,jj) - b2(jj,ii,kk) ) < Tol );
                }
            }
        }
    }

    {
        constexpr size_t I=2, J=4, K=5, L=3;
        Tensor<T,L,I,K,L,J> a; a.iota(3);
        auto b1 = einsum<Index<l,i,k,l,j>>(a);
        auto b2 = einsum<Index<l,i,k,l,j>,OIndex<j,k,i>>(a);

        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    FASTOR_EXIT_ASSERT( std::abs( b1(ii,kk,jj) - b2(jj,kk,ii) ) < Tol );
                }
            }
        }
    }

    // permuting 4 free indices
    {
        constexpr size_t I=2, J=4, K=5, L=3, M=7;
        Tensor<T,K,I,M,J,J,L> a; a.iota(3);
        auto b1 = einsum<Index<k,i,m,j,j,l>>(a);
        // no permutation
        auto b2 = einsum<Index<k,i,m,j,j,l>,OIndex<k,i,m,l>>(a);

        for (size_t kk=0; kk<K; ++kk) {
            for (size_t ii=0; ii<I; ++ii) {
                for (size_t ll=0; ll<L; ++ll) {
                    for (size_t mm=0; mm<M; ++mm) {
                        FASTOR_EXIT_ASSERT( std::abs( b1(kk,ii,mm,ll) - b2(kk,ii,mm,ll)  ) < Tol );
                    }
                }
            }
        }
    }

    {
        constexpr size_t I=2, J=4, K=5, L=3, M=7;
        Tensor<T,K,I,M,J,J,L> a; a.iota(3);
        auto b1 = einsum<Index<k,i,m,j,j,l>>(a);
        // no permutation
        auto b2 = einsum<Index<k,i,m,j,j,l>,OIndex<k,i,l,m>>(a);

        for (size_t kk=0; kk<K; ++kk) {
            for (size_t ii=0; ii<I; ++ii) {
                for (size_t ll=0; ll<L; ++ll) {
                    for (size_t mm=0; mm<M; ++mm) {
                        FASTOR_EXIT_ASSERT( std::abs( b1(kk,ii,mm,ll) - b2(kk,ii,ll,mm)  ) < Tol );
                    }
                }
            }
        }
    }

    {
        constexpr size_t I=2, J=4, K=5, L=3, M=7;
        Tensor<T,K,I,M,J,J,L> a; a.iota(3);
        auto b1 = einsum<Index<k,i,m,j,j,l>>(a);
        auto b2 = einsum<Index<k,i,m,j,j,l>,OIndex<k,m,l,i>>(a);

        for (size_t kk=0; kk<K; ++kk) {
            for (size_t ii=0; ii<I; ++ii) {
                for (size_t ll=0; ll<L; ++ll) {
                    for (size_t mm=0; mm<M; ++mm) {
                        FASTOR_EXIT_ASSERT( std::abs( b1(kk,ii,mm,ll) - b2(kk,mm,ll,ii)  ) < Tol );
                    }
                }
            }
        }
    }

    {
        constexpr size_t I=2, J=4, K=5, L=3, M=7;
        Tensor<T,K,I,M,J,J,L> a; a.iota(3);
        auto b1 = einsum<Index<k,i,m,j,j,l>>(a);
        auto b2 = einsum<Index<k,i,m,j,j,l>,OIndex<k,m,i,l>>(a);

        for (size_t kk=0; kk<K; ++kk) {
            for (size_t ii=0; ii<I; ++ii) {
                for (size_t ll=0; ll<L; ++ll) {
                    for (size_t mm=0; mm<M; ++mm) {
                        FASTOR_EXIT_ASSERT( std::abs( b1(kk,ii,mm,ll) - b2(kk,mm,ii,ll)  ) < Tol );
                    }
                }
            }
        }
    }

    {
        constexpr size_t I=2, J=4, K=5, L=3, M=7;
        Tensor<T,K,I,M,J,J,L> a; a.iota(3);
        auto b1 = einsum<Index<k,i,m,j,j,l>>(a);
        auto b2 = einsum<Index<k,i,m,j,j,l>,OIndex<m,i,l,k>>(a);

        for (size_t kk=0; kk<K; ++kk) {
            for (size_t ii=0; ii<I; ++ii) {
                for (size_t ll=0; ll<L; ++ll) {
                    for (size_t mm=0; mm<M; ++mm) {
                        FASTOR_EXIT_ASSERT( std::abs( b1(kk,ii,mm,ll) - b2(mm,ii,ll,kk)  ) < Tol );
                    }
                }
            }
        }
    }

    {
        constexpr size_t I=2, J=4, K=5, L=3, M=7;
        Tensor<T,K,I,M,J,J,L> a; a.iota(3);
        auto b1 = einsum<Index<k,i,m,j,j,l>>(a);
        auto b2 = einsum<Index<k,i,m,j,j,l>,OIndex<i,m,k,l>>(a);

        for (size_t kk=0; kk<K; ++kk) {
            for (size_t ii=0; ii<I; ++ii) {
                for (size_t ll=0; ll<L; ++ll) {
                    for (size_t mm=0; mm<M; ++mm) {
                        FASTOR_EXIT_ASSERT( std::abs( b1(kk,ii,mm,ll) - b2(ii,mm,kk,ll)  ) < Tol );
                    }
                }
            }
        }
    }

    {
        constexpr size_t I=2, J=4, K=5, L=3, M=7;
        Tensor<T,K,I,M,J,J,L> a; a.iota(3);
        auto b1 = einsum<Index<k,i,m,j,j,l>>(a);
        auto b2 = einsum<Index<k,i,m,j,j,l>,OIndex<i,k,l,m>>(a);

        for (size_t kk=0; kk<K; ++kk) {
            for (size_t ii=0; ii<I; ++ii) {
                for (size_t ll=0; ll<L; ++ll) {
                    for (size_t mm=0; mm<M; ++mm) {
                        FASTOR_EXIT_ASSERT( std::abs( b1(kk,ii,mm,ll) - b2(ii,kk,ll,mm)  ) < Tol );
                    }
                }
            }
        }
    }


    // this checks the no-permutation of the specialisations of permute_mapped_index_impl
    {
        constexpr size_t I=2, J=4, K=3, L=5, M=3;
        Tensor<T,K,L,M,I,L> a; a.iota(3);
        auto b1 = einsum<Index<k,l,m,i,l>>(a);
        // no permutation
        auto b2 = einsum<Index<k,l,m,i,l>,OIndex<k,m,i>>(a);

        for (size_t kk=0; kk<K; ++kk) {
            for (size_t mm=0; mm<M; ++mm) {
                for (size_t ii=0; ii<I; ++ii) {
                    FASTOR_EXIT_ASSERT( std::abs( b1(kk,mm,ii) - b2(kk,mm,ii) ) < Tol );
                }
            }
        }
    }



    // expressions
    {
        constexpr size_t I=2, J=4, K=5, L=3, M=7;
        Tensor<T,K,I,M,J,J,L> a; a.iota(3);
        auto b1 = einsum<Index<k,i,m,j,j,l>>(a);
        auto b2 = einsum<Index<k,i,m,j,j,l>,OIndex<i,k,l,m>>(a+0);

        for (size_t kk=0; kk<K; ++kk) {
            for (size_t ii=0; ii<I; ++ii) {
                for (size_t ll=0; ll<L; ++ll) {
                    for (size_t mm=0; mm<M; ++mm) {
                        FASTOR_EXIT_ASSERT( std::abs( b1(kk,ii,mm,ll) - b2(ii,kk,ll,mm)  ) < Tol );
                    }
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

#else

int main() {
    return 0;
}

#endif // CXX 2017

