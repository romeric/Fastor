#include <Fastor/Fastor.h>

using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5
#define HugeTol 1e-2

template<typename T>
void test_solve() {

    // solve
    {
        constexpr size_t M = 2;
        Tensor<T,M,M> a; a.iota(1);
        for (size_t i=0; i<M; ++i) a(i,i) = T(100);
        Tensor<T,M> b; b.iota(100);
        Tensor<T,M,1> c; c.iota(100);
        Tensor<T,M,5> d; d.iota(100);

        Tensor<T,M> b1 = solve(a,b);
        Tensor<T,M,1> c1 = solve(a,c);
        Tensor<T,M,5> d1 = solve(a,d);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 1.960976585951571 ) < BigTol);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(c1) - 1.960976585951571 ) < BigTol);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(d1) - 10.196117670602362) < BigTol);
    }
    {
        constexpr size_t M = 3;
        Tensor<T,M,M> a; a.iota(1);
        for (size_t i=0; i<M; ++i) a(i,i) = T(100);
        Tensor<T,M> b; b.iota(100);
        Tensor<T,M,1> c; c.iota(100);
        Tensor<T,M,5> d; d.iota(100);

        Tensor<T,M> b1 = solve(a,b);
        Tensor<T,M,1> c1 = solve(a,c);
        Tensor<T,M,5> d1 = solve(a,d);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 2.753858012252136 ) < BigTol);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(c1) - 2.753858012252136 ) < BigTol);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(d1) - 14.591039617926809) < BigTol);
    }
    {
        constexpr size_t M = 4;
        Tensor<T,M,M> a; a.iota(1);
        for (size_t i=0; i<M; ++i) a(i,i) = T(100);
        Tensor<T,M> b; b.iota(100);
        Tensor<T,M,1> c; c.iota(100);
        Tensor<T,M,5> d; d.iota(100);

        Tensor<T,M> b1 = solve(a,b);
        Tensor<T,M,1> c1 = solve(a,c);
        Tensor<T,M,5> d1 = solve(a,d);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 3.2316174170320178) < BigTol);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(c1) - 3.2316174170320178) < BigTol);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(d1) - 17.44017784877636 ) < BigTol);
    }

    // solve expressions
    {
        constexpr size_t M = 2;
        Tensor<T,M,M> a; a.iota(1);
        for (size_t i=0; i<M; ++i) a(i,i) = T(100);
        Tensor<T,M> b; b.iota(100);
        Tensor<T,M,1> c; c.iota(100);
        Tensor<T,M,5> d; d.iota(100);

        FASTOR_DOES_CHECK_PASS(std::abs(sum(solve(a  ,b+0)) - 1.960976585951571 ) < BigTol);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(solve(a+0,b  )) - 1.960976585951571 ) < BigTol);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(solve(a+0,b+0)) - 1.960976585951571 ) < BigTol);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(solve(a  ,c+0)) - 1.960976585951571 ) < BigTol);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(solve(a+0,c  )) - 1.960976585951571 ) < BigTol);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(solve(a+0,c+0)) - 1.960976585951571 ) < BigTol);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(solve(a  ,d+0)) - 10.196117670602362) < BigTol);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(solve(a+0,d  )) - 10.196117670602362) < BigTol);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(solve(a+0,d+0)) - 10.196117670602362) < BigTol);
    }
    {
        constexpr size_t M = 3;
        Tensor<T,M,M> a; a.iota(1);
        for (size_t i=0; i<M; ++i) a(i,i) = T(100);
        Tensor<T,M> b; b.iota(100);

        Tensor<T,M> b1;
        b1.iota(1);
        b1 += solve(a,b+0);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 8.753858012252135) < BigTol);

        b1.iota(1);
        b1 += solve(a*1,b);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 8.753858012252135) < BigTol);

        b1.iota(1);
        b1 += solve(a*1,b+b-b);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 8.753858012252135) < BigTol);

        b1.iota(1);
        b1 -= solve(a,b+0);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 3.2461419877478646) < BigTol);

        b1.iota(1);
        b1 -= solve(a+1-1,b);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 3.2461419877478646) < BigTol);

        b1.iota(1);
        b1 -= solve(3*a-2*a,2*b-b);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 3.2461419877478646) < BigTol);

        b1.iota(1);
        b1 *= solve(a,b+0);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 5.432097372239239 ) < BigTol);

        b1.iota(1);
        b1 *= solve(1+a-1,b);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 5.432097372239239 ) < BigTol);

        b1.iota(1);
        b1 *= solve(a+a+a-a*2,1*b+0);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 5.432097372239239 ) < BigTol);

        b1.iota(1);
        b1 /= solve(a,b+0);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 6.6337042772104695) < BigTol);

        b1.iota(1);
        b1 /= solve(1+a-1,b);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 6.6337042772104695) < BigTol);

        b1.iota(1);
        b1 /= solve(a+a+a-a*2,1*b+0);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 6.6337042772104695) < BigTol);

        // mix with other expressions
        b1.iota(1);
        b1 += solve(a,b+0) + b1 - b1;
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 8.753858012252135) < BigTol);

        b1.iota(1);
        b1 += solve(a*1,b)*1;
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 8.753858012252135) < BigTol);

        b1.iota(1);
        b1 += solve(a*1,b+b-b) - 1 + 1;
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 8.753858012252135) < BigTol);
    }
    {
        constexpr size_t M = 3;
        Tensor<T,M,M> a; a.iota(1);
        for (size_t i=0; i<M; ++i) a(i,i) = T(100);
        Tensor<T,M,5> b; b.iota(100);

        Tensor<T,M,5> b1;
        b1.iota(1);
        b1 += solve(a,b+0);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 134.5910396179268) < BigTol);

        b1.iota(1);
        b1 += solve(a*1,b);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 134.5910396179268) < BigTol);

        b1.iota(1);
        b1 += solve(a*1,b+b-b);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 134.5910396179268) < BigTol);

        b1.iota(1);
        b1 -= solve(a,b+0);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 105.4089603820732) < BigTol);

        b1.iota(1);
        b1 -= solve(a+1-1,b);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 105.4089603820732) < BigTol);

        b1.iota(1);
        b1 -= solve(3*a-2*a,2*b-b);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 105.4089603820732) < BigTol);

        b1.iota(1);
        b1 *= solve(a,b+0);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 117.0727470578752 ) < BigTol);

        b1.iota(1);
        b1 *= solve(1+a-1,b);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 117.0727470578752 ) < BigTol);

        b1.iota(1);
        b1 *= solve(a+a+a-a*2,1*b+0);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 117.0727470578752 ) < BigTol);

        b1.iota(1);
        b1 /= solve(a,b+0);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 123.02004364710513) < BigTol);

        b1.iota(1);
        b1 /= solve(1+a-1,b);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 123.02004364710513) < BigTol);

        b1.iota(1);
        b1 /= solve(a+a+a-a*2,1*b+0);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 123.02004364710513) < BigTol);
    }

    // Solve by LU
    {
        constexpr size_t M = 4;
        Tensor<T,M,M> a; a.iota(1);
        for (size_t i=0; i<M; ++i) a(i,i) = T(100);
        Tensor<T,M> b; b.iota(100);
        Tensor<T,M,1> c; c.iota(100);
        Tensor<T,M,5> d; d.iota(100);

        {
            Tensor<T,M>   b1 = solve<SolveCompType::SimpleInvPiv>(a,b);
            Tensor<T,M,1> c1 = solve<SolveCompType::SimpleInvPiv>(a,c);
            Tensor<T,M,5> d1 = solve<SolveCompType::SimpleInvPiv>(a,d);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 3.2316174170320178) < BigTol);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(c1) - 3.2316174170320178) < BigTol);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(d1) - 17.44017784877636 ) < BigTol);
        }

        {
            Tensor<T,M>   b1 = solve<SolveCompType::SimpleLU>(a,b);
            Tensor<T,M,1> c1 = solve<SolveCompType::SimpleLU>(a,c);
            Tensor<T,M,5> d1 = solve<SolveCompType::SimpleLU>(a,d);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 3.2316174170320178) < BigTol);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(c1) - 3.2316174170320178) < BigTol);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(d1) - 17.44017784877636 ) < BigTol);
        }

        {
            Tensor<T,M>   b1 = solve<SolveCompType::SimpleLUPiv>(a,b);
            Tensor<T,M,1> c1 = solve<SolveCompType::SimpleLUPiv>(a,c);
            Tensor<T,M,5> d1 = solve<SolveCompType::SimpleLUPiv>(a,d);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 3.2316174170320178) < BigTol);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(c1) - 3.2316174170320178) < BigTol);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(d1) - 17.44017784877636 ) < BigTol);
        }

        {
            Tensor<T,M>   b1 = solve<SolveCompType::BlockLU>(a,b);
            Tensor<T,M,1> c1 = solve<SolveCompType::BlockLU>(a,c);
            Tensor<T,M,5> d1 = solve<SolveCompType::BlockLU>(a,d);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 3.2316174170320178) < BigTol);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(c1) - 3.2316174170320178) < BigTol);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(d1) - 17.44017784877636 ) < BigTol);
        }

        {
            Tensor<T,M>   b1 = solve<SolveCompType::BlockLUPiv>(a,b);
            Tensor<T,M,1> c1 = solve<SolveCompType::BlockLUPiv>(a,c);
            Tensor<T,M,5> d1 = solve<SolveCompType::BlockLUPiv>(a,d);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(b1) - 3.2316174170320178) < BigTol);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(c1) - 3.2316174170320178) < BigTol);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(d1) - 17.44017784877636 ) < BigTol);
        }
    }

    // Big matrices
    {
        constexpr size_t M = 35;
        constexpr size_t N = M;
        Tensor<T,M,M> A; A.arange(0);
        Tensor<T,M> b; b.iota(1);
        for (size_t i=0; i<M; ++i) A(i,i) = T(100 + i);

        Tensor<T,M> sol = solve(A,b);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(solve<SolveCompType::BlockLUPiv> (A,b) - sol)) < BigTol);
        FASTOR_DOES_CHECK_PASS(std::abs(sum(solve<SolveCompType::SimpleLUPiv>(A,b) - sol)) < BigTol);
    }

    // complex valued solve - issue 110
    {
        using TT = std::complex<double>;
        constexpr size_t M = 3;
        Tensor<size_t,M> p;
        Tensor<T,M,M> L, U, P;

        Tensor<TT,M,M> A;
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < M; ++j) {
                A(i, j) = TT(i,j);
                if (i==j) A(i, j) *= 1000;
            }
        }

        Tensor<TT,M> b(TT(0,0));
        b(0) = TT(1,0);

        Tensor<TT,M>   x0 = solve<SolveCompType::SimpleInv>(A,b);
        FASTOR_DOES_CHECK_PASS(std::abs(std::abs(sum(x0)) - 471.404756493233) < BigTol);

        Tensor<TT,M>   x1 = solve<SolveCompType::SimpleInvPiv>(A,b);
        FASTOR_DOES_CHECK_PASS(std::abs(std::abs(sum(x1)) - 471.404756493233) < BigTol);

        Tensor<TT,M>   x2 = solve<SolveCompType::SimpleLUPiv>(A,b);
        FASTOR_DOES_CHECK_PASS(std::abs(std::abs(sum(x2)) - 471.404756493233) < BigTol);

        Tensor<TT,M>   x3 = solve<SolveCompType::BlockLUPiv>(A,b);
        FASTOR_DOES_CHECK_PASS(std::abs(std::abs(sum(x3)) - 471.404756493233) < BigTol);
    }

    print(FGRN(BOLD("All tests passed successfully")));

}

int main() {

    print(FBLU(BOLD("Testing tensor solve: double/complex double precision")));
    test_solve<double>();

    return 0;
}

