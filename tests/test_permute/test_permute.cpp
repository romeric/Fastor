#include <Fastor/Fastor.h>
using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5
#define HugeTol 1e-2


template<typename T, size_t I, size_t J>
void run_permute_2d() {

    using std::abs;
    enum {i,j,k,l,m,n};

    // 2D tensor permute
    {
        Tensor<T,I,J> a; a.iota(41);

        // no permutation
        auto b0 = permute<Index<i,j>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                FASTOR_EXIT_ASSERT( std::abs( a(ii,jj) - b0(ii,jj) ) < Tol );
            }
        }

        // transpose
        auto b1 = permute<Index<j,i>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                FASTOR_EXIT_ASSERT( std::abs( a(ii,jj) - b1(jj,ii) ) < Tol );
            }
        }

        FASTOR_EXIT_ASSERT( std::abs( sum(b1 - transpose(a)) ) < Tol );
        FASTOR_EXIT_ASSERT( std::abs( sum(b1 - trans(a))     ) < Tol );
    }

    // 2D tensor expression permute
    {
        Tensor<T,I,J> a; a.iota(41);

        // no permutation
        auto b0 = permute<Index<i,j>>(a+1-1);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                FASTOR_EXIT_ASSERT( std::abs( a(ii,jj) - b0(ii,jj) ) < Tol );
            }
        }

        // transpose
        auto b1 = permute<Index<j,i>>(trans(trans(a)+0));
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                FASTOR_EXIT_ASSERT( std::abs( a(ii,jj) - b1(jj,ii) ) < Tol );
            }
        }

        FASTOR_EXIT_ASSERT( std::abs( sum(b1 - transpose(a)) ) < Tol );
        FASTOR_EXIT_ASSERT( std::abs( sum(b1 - trans(a))     ) < Tol );
    }

    print(FGRN(BOLD("All tests passed successfully")));
}


template<typename T, size_t I, size_t J, size_t K>
void run_permute_3d() {

    using std::abs;
    enum {i,j,k,l,m,n};

    // 3D tensor permute
    {
        Tensor<T,I,J,K> a; a.iota(13);

        // no permutation
        auto b0 = permute<Index<i,j,k>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk) - b0(ii,jj,kk) ) < Tol );
                }
            }
        }

        auto b1 = permute<Index<i,k,j>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk) - b1(ii,kk,jj) ) < Tol );
                }
            }
        }

        auto b2 = permute<Index<j,i,k>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk) - b2(jj,ii,kk) ) < Tol );
                }
            }
        }

        auto b3 = permute<Index<j,k,i>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk) - b3(jj,kk,ii) ) < Tol );
                }
            }
        }

        auto b4 = permute<Index<k,i,j>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk) - b4(kk,ii,jj) ) < Tol );
                }
            }
        }

        auto b5 = permute<Index<k,j,i>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk) - b5(kk,jj,ii) ) < Tol );
                }
            }
        }
    }

    // 3D tensor expression permute
    {
        Tensor<T,I,J,K> a; a.iota(13);

        // no permutation
        auto b0 = permute<Index<i,j,k>>(a+0);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk) - b0(ii,jj,kk) ) < Tol );
                }
            }
        }

        auto b1 = permute<Index<i,k,j>>(a+1-1);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk) - b1(ii,kk,jj) ) < Tol );
                }
            }
        }

        auto b2 = permute<Index<j,i,k>>(a*1);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk) - b2(jj,ii,kk) ) < Tol );
                }
            }
        }

        auto b3 = permute<Index<j,k,i>>(1+a-1);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk) - b3(jj,kk,ii) ) < Tol );
                }
            }
        }

        auto b4 = permute<Index<k,i,j>>(a+2-2);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk) - b4(kk,ii,jj) ) < Tol );
                }
            }
        }

        auto b5 = permute<Index<k,j,i>>(abs(a));
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk) - b5(kk,jj,ii) ) < Tol );
                }
            }
        }
    }

    print(FGRN(BOLD("All tests passed successfully")));
}


template<typename T, size_t I, size_t J, size_t K, size_t L>
void run_permute_4d() {

    using std::abs;
    enum {i,j,k,l,m,n};

    // 4D tensor permute
    {
        Tensor<T,I,J,K,L> a; a.iota(4);

        // no permutation
        auto b0 = permute<Index<i,j,k,l>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll) - b0(ii,jj,kk,ll) ) < Tol );
                    }
                }
            }
        }

        auto b00 = permute<Index<i,j,l,k>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll) - b00(ii,jj,ll,kk) ) < Tol );
                    }
                }
            }
        }

        auto b1 = permute<Index<i,k,j,l>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll) - b1(ii,kk,jj,ll) ) < Tol );
                    }
                }
            }
        }

        auto b2 = permute<Index<i,k,l,j>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll) - b2(ii,kk,ll,jj) ) < Tol );
                    }
                }
            }
        }

        auto b3 = permute<Index<i,l,k,j>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll) - b3(ii,ll,kk,jj) ) < Tol );
                    }
                }
            }
        }

        auto b4 = permute<Index<i,l,j,k>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll) - b4(ii,ll,jj,kk) ) < Tol );
                    }
                }
            }
        }

        auto b5 = permute<Index<j,l,i,k>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll) - b5(jj,ll,ii,kk) ) < Tol );
                    }
                }
            }
        }

        auto b6 = permute<Index<j,l,k,i>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll) - b6(jj,ll,kk,ii) ) < Tol );
                    }
                }
            }
        }

        auto b7 = permute<Index<k,l,i,j>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll) - b7(kk,ll,ii,jj) ) < Tol );
                    }
                }
            }
        }
    }

    // 4D tensor expression permute
    {
        Tensor<T,I,J,K,L> a; a.iota(4);

        // no permutation
        auto b0 = permute<Index<i,j,k,l>>(a+0);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll) - b0(ii,jj,kk,ll) ) < Tol );
                    }
                }
            }
        }

        auto b00 = permute<Index<i,j,l,k>>(a+1-1);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll) - b00(ii,jj,ll,kk) ) < Tol );
                    }
                }
            }
        }

        auto b1 = permute<Index<i,k,j,l>>(abs(a));
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll) - b1(ii,kk,jj,ll) ) < Tol );
                    }
                }
            }
        }

        auto b2 = permute<Index<i,k,l,j>>(3+a-3);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll) - b2(ii,kk,ll,jj) ) < Tol );
                    }
                }
            }
        }

        auto b3 = permute<Index<i,l,k,j>>(1+1*a-1);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll) - b3(ii,ll,kk,jj) ) < Tol );
                    }
                }
            }
        }

        auto b4 = permute<Index<i,l,j,k>>(a+0+0+0);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll) - b4(ii,ll,jj,kk) ) < Tol );
                    }
                }
            }
        }

        auto b5 = permute<Index<j,l,i,k>>(0+a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll) - b5(jj,ll,ii,kk) ) < Tol );
                    }
                }
            }
        }

        auto b6 = permute<Index<j,l,k,i>>(a-0);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll) - b6(jj,ll,kk,ii) ) < Tol );
                    }
                }
            }
        }

        auto b7 = permute<Index<k,l,i,j>>(1*a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll) - b7(kk,ll,ii,jj) ) < Tol );
                    }
                }
            }
        }
    }

    print(FGRN(BOLD("All tests passed successfully")));
}


template<typename T, size_t I, size_t J, size_t K, size_t L, size_t M>
void run_permute_5d() {

    using std::abs;
    enum {i,j,k,l,m,n};

    // 5D tensor permute
    {
        Tensor<T,I,J,K,L,M> a; a.iota(4);

        // no permutation
        auto b0 = permute<Index<i,j,k,l,m>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        for (size_t mm=0; mm<M; ++mm) {
                            FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll,mm) - b0(ii,jj,kk,ll,mm) ) < Tol );
                        }
                    }
                }
            }
        }

        auto b1 = permute<Index<i,j,m,l,k>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        for (size_t mm=0; mm<M; ++mm) {
                            FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll,mm) - b1(ii,jj,mm,ll,kk) ) < Tol );
                        }
                    }
                }
            }
        }

        auto b2 = permute<Index<l,m,j,k,i>>(a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        for (size_t mm=0; mm<M; ++mm) {
                            FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll,mm) - b2(ll,mm,jj,kk,ii) ) < Tol );
                        }
                    }
                }
            }
        }
    }


    // 5D tensor expression permute
    {
        Tensor<T,I,J,K,L,M> a; a.iota(4);

        // no permutation
        auto b0 = permute<Index<i,j,k,l,m>>(a+0);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        for (size_t mm=0; mm<M; ++mm) {
                            FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll,mm) - b0(ii,jj,kk,ll,mm) ) < Tol );
                        }
                    }
                }
            }
        }

        auto b1 = permute<Index<i,j,m,l,k>>(a+0);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        for (size_t mm=0; mm<M; ++mm) {
                            FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll,mm) - b1(ii,jj,mm,ll,kk) ) < Tol );
                        }
                    }
                }
            }
        }

        auto b2 = permute<Index<l,m,j,k,i>>(0+a);
        for (size_t ii=0; ii<I; ++ii) {
            for (size_t jj=0; jj<J; ++jj) {
                for (size_t kk=0; kk<K; ++kk) {
                    for (size_t ll=0; ll<L; ++ll) {
                        for (size_t mm=0; mm<M; ++mm) {
                            FASTOR_EXIT_ASSERT( std::abs( a(ii,jj,kk,ll,mm) - b2(ll,mm,jj,kk,ii) ) < Tol );
                        }
                    }
                }
            }
        }
    }

    print(FGRN(BOLD("All tests passed successfully")));
}

template<typename T>
void run() {

    // 2D
    // uniform
    run_permute_2d<T,2,2>();
    run_permute_2d<T,3,3>();
    run_permute_2d<T,4,4>();
    // non-uniform
    run_permute_2d<T,3,9>();

    // 3D
    // uniform
    run_permute_3d<T,2,2,2>();
    run_permute_3d<T,3,3,3>();
    run_permute_3d<T,4,4,4>();
    // non-uniform
    run_permute_3d<T,3,5,11>();

    // 4D
    // uniform
    run_permute_4d<T,2,2,2,2>();
    run_permute_4d<T,3,3,3,3>();
    run_permute_4d<T,4,4,4,4>();
    // non-uniform
    run_permute_4d<T,2,3,4,5>();

    // 5D
    // uniform
    run_permute_5d<T,2,2,2,2,2>();
    run_permute_5d<T,3,3,3,3,3>();
    // non-uniform
    run_permute_5d<T,2,3,4,5,7>();
}

int main() {

    print(FBLU(BOLD("Testing tesnor permute: single precision")));
    run<float>();
    print(FBLU(BOLD("Testing tesnor permute: double precision")));
    run<double>();

    return 0;
}
