#include <Fastor/Fastor.h>

using namespace Fastor;

#define Tol 1e-9
#define BigTol 1e-4
#define HugeTol 1e-2

template<typename T>
void test_factorisation() {


    // LU factorisation
    {
        // Simple test with a permuted identity matrix
        Tensor<T,3,3> A; A.fill(0);
        A(0,1) = 1;
        A(1,0) = 1;
        A(2,2) = 1;

        // BlockLU vector pivot
        {
            Tensor<T,3,3> L, U;
            Tensor<size_t,3> P;
            lu<LUCompType::BlockLUPiv>(A, L, U, P);
            auto reconA = reconstruct(L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconA)) < BigTol);

            for (size_t i=0; i<3; ++i)
                for (size_t j=0; j<3; ++j)
                    FASTOR_EXIT_ASSERT(std::abs(A(i,j) - reconA(i,j)) < BigTol);
        }

        // BlockLU matrix pivot
        {
            Tensor<T,3,3> L, U, P;
            lu<LUCompType::BlockLUPiv>(A, L, U, P);
            auto reconA = reconstruct(L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconA)) < BigTol);

            for (size_t i=0; i<3; ++i)
                for (size_t j=0; j<3; ++j)
                    FASTOR_EXIT_ASSERT(std::abs(A(i,j) - reconA(i,j)) < BigTol);
        }

        // SimpleLU vector pivot
        {
            Tensor<T,3,3> L, U;
            Tensor<size_t,3> P;
            lu<LUCompType::SimpleLUPiv>(A, L, U, P);
            auto reconA = reconstruct(L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconA)) < BigTol);

            for (size_t i=0; i<3; ++i)
                for (size_t j=0; j<3; ++j)
                    FASTOR_EXIT_ASSERT(std::abs(A(i,j) - reconA(i,j)) < BigTol);
        }

        // SimpleLU matrix pivot
        {
            Tensor<T,3,3> L, U, P;
            lu<LUCompType::SimpleLUPiv>(A, L, U, P);
            auto reconA = reconstruct(L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconA)) < BigTol);

            for (size_t i=0; i<3; ++i)
                for (size_t j=0; j<3; ++j)
                    FASTOR_EXIT_ASSERT(std::abs(A(i,j) - reconA(i,j)) < BigTol);
        }
    }

    {
        // LU 2x2
        {
            constexpr size_t M = 2;
            Tensor<size_t,M> p;
            Tensor<T,M,M> L, U, P;

            Tensor<T,M,M> A; A.iota();
            for (size_t i=0; i<M; ++i) {
                A(i,i) = T(100+i);
            }

            lu(A, L, U);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::BlockLU>(A, L, U);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::BlockLUPiv>(A, L, U, p);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::BlockLUPiv>(A, L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::SimpleLU>(A, L, U);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::SimpleLUPiv>(A, L, U, p);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::SimpleLUPiv>(A, L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);
        }

        // LU 3x3
        {
            constexpr size_t M = 3;
            Tensor<size_t,M> p;
            Tensor<T,M,M> L, U, P;

            Tensor<T,M,M> A; A.iota();
            for (size_t i=0; i<M; ++i) {
                A(i,i) = T(100+i);
            }

            lu(A, L, U);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::BlockLU>(A, L, U);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::BlockLUPiv>(A, L, U, p);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::BlockLUPiv>(A, L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::SimpleLU>(A, L, U);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::SimpleLUPiv>(A, L, U, p);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::SimpleLUPiv>(A, L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);
        }

        // LU 4x4
        {
            constexpr size_t M = 4;
            Tensor<size_t,M> p;
            Tensor<T,M,M> L, U, P;

            Tensor<T,M,M> A; A.iota();
            for (size_t i=0; i<M; ++i) {
                A(i,i) = T(100+i);
            }

            lu(A, L, U);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::BlockLU>(A, L, U);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::BlockLUPiv>(A, L, U, p);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::BlockLUPiv>(A, L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::SimpleLU>(A, L, U);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::SimpleLUPiv>(A, L, U, p);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::SimpleLUPiv>(A, L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);
        }

        // LU 5x5
        {
            constexpr size_t M = 5;
            Tensor<size_t,M> p;
            Tensor<T,M,M> L, U, P;

            Tensor<T,M,M> A; A.iota();
            for (size_t i=0; i<M; ++i) {
                A(i,i) = T(100+i);
            }

            lu(A, L, U);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::BlockLU>(A, L, U);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::BlockLUPiv>(A, L, U, p);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::BlockLUPiv>(A, L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::SimpleLU>(A, L, U);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::SimpleLUPiv>(A, L, U, p);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::SimpleLUPiv>(A, L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);
        }

        // LU 9x9
        {
            constexpr size_t M = 9;
            Tensor<size_t,M> p;
            Tensor<T,M,M> L, U, P;

            Tensor<T,M,M> A; A.iota();
            for (size_t i=0; i<M; ++i) {
                A(i,i) = T(100+i);
            }

            lu(A, L, U);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconstruct(L, U))) < BigTol);

            lu<LUCompType::BlockLU>(A, L, U);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconstruct(L, U))) < BigTol);

            lu<LUCompType::BlockLUPiv>(A, L, U, p);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconstruct(L, U, p))) < BigTol);

            lu<LUCompType::BlockLUPiv>(A, L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconstruct(L, U, P))) < BigTol);

            lu<LUCompType::SimpleLU>(A, L, U);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconstruct(L, U))) < BigTol);

            lu<LUCompType::SimpleLUPiv>(A, L, U, p);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconstruct(L, U, p))) < BigTol);

            lu<LUCompType::SimpleLUPiv>(A, L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconstruct(L, U, P))) < BigTol);
        }

        // LU 18x18
        {
            constexpr size_t M = 18;
            Tensor<size_t,M> p;
            Tensor<T,M,M> L, U, P;

            Tensor<T,M,M> A; A.iota();
            for (size_t i=0; i<M; ++i) {
                A(i,i) = T(100+i);
            }

            lu(A, L, U);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::BlockLU>(A, L, U);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::BlockLUPiv>(A, L, U, p);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::BlockLUPiv>(A, L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::SimpleLU>(A, L, U);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::SimpleLUPiv>(A, L, U, p);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::SimpleLUPiv>(A, L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);
        }

        // LU 33x33 - this is just to check for compilation as size 33
        // is when we switch over to block recursive algorithm
        {
            constexpr size_t M = 33;
            Tensor<size_t,M> p;
            Tensor<T,M,M> L, U, P;

            Tensor<T,M,M> A; A.iota();
            for (size_t i=0; i<M; ++i) {
                A(i,i) = T(100+i);
            }

            lu<LUCompType::BlockLUPiv>(A, L, U, p);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);

            lu<LUCompType::BlockLUPiv>(A, L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - L % U)) < BigTol);
        }

        // LU 34x34 - do not test higher sizes as it can be quite heavy
        // for the compiler
        {
            constexpr size_t M = 34;
            Tensor<size_t,M> p;
            Tensor<T,M,M> L, U, P;

            Tensor<T,M,M> A; A.iota();
            for (size_t i=0; i<M; ++i) {
                A(i,i) = T(100+i);
            }

            lu<LUCompType::BlockLUPiv>(A, L, U, p);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconstruct(L, U, p))) < BigTol);

            lu<LUCompType::BlockLUPiv>(A, L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconstruct(L, U, P))) < BigTol);

            lu<LUCompType::SimpleLUPiv>(A, L, U, p);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconstruct(L, U, p))) < BigTol);

            lu<LUCompType::SimpleLUPiv>(A, L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconstruct(L, U, P))) < BigTol);
        }

        // LU 3x3 - std::complex LU
        {
            using TT = std::complex<double>;
            constexpr size_t M = 3;
            Tensor<size_t,M> p;
            Tensor<TT,M,M> L, U, P;

            Tensor<TT,M,M> A;
            for (int i = 0; i < M; ++i)
            {
                for (int j = 0; j < M; ++j) {
                    A(i, j) = TT(i,j);
                    if (i==j) A(i, j) *= 1000;
                }
            }

            lu<LUCompType::BlockLUPiv>(A, L, U, p);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconstruct(L, U, p))) < BigTol);

            lu<LUCompType::BlockLUPiv>(A, L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconstruct(L, U, P))) < BigTol);

            lu<LUCompType::SimpleLUPiv>(A, L, U, p);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconstruct(L, U, p))) < BigTol);

            lu<LUCompType::SimpleLUPiv>(A, L, U, P);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - reconstruct(L, U, P))) < BigTol);
        }
    }

    print(FGRN(BOLD("All tests passed successfully")));

}

int main() {

    // We will test only double precision here
    print(FBLU(BOLD("Testing LU factorisation: double precision")));
    test_factorisation<double>();

    return 0;
}

