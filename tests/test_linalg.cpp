#include <Fastor/Fastor.h>

using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5
#define HugeTol 1e-2

template<typename T, size_t M, size_t N>
Tensor<T,N,M> transpose_ref(const Tensor<T,M,N>& a) {
    Tensor<T,N,M> out;
    for (size_t i=0; i<M; ++i) {
        for (size_t j=0; j<N; ++j) {
            out(j,i) = a(i,j);
        }
    }
    return out;
}

template<typename T, size_t M, size_t N>
void TEST_TRANSPOSE(Tensor<T,M,N>& a) {

    Tensor<T,N,M> b1 = transpose_ref(a);
    Tensor<T,N,M> b2 = transpose(a);
    Tensor<T,N,M> b3 = trans(a);

    for (size_t i=0; i<N; ++i) {
        for (size_t j=0; j<M; ++j) {
            FASTOR_EXIT_ASSERT(std::abs(b1(i,j) - b2(i,j)) < Tol);
            FASTOR_EXIT_ASSERT(std::abs(b1(i,j) - b3(i,j)) < Tol);
        }
    }

    FASTOR_EXIT_ASSERT(std::abs(norm(transpose(a))-norm(a))< HugeTol);
    FASTOR_EXIT_ASSERT(std::abs(norm(trans(a))-norm(a))< HugeTol);
}


template<typename T>
void test_linalg() {

    // 2D
    {
        Tensor<T,2,2> t1; t1.iota(5);
        FASTOR_EXIT_ASSERT(std::abs(determinant(t1)+2.0) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(trace(t1)-13) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(norm(matmul(t1,t1)) - 173.4646938) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(matmul(t1-0,t1+t1-t1)) - 173.4646938) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(cofactor(t1)) - 13.1909059) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(transpose(adjoint(t1))) - 13.1909059) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(det(t1)+2.) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(trace(t1) - 13) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(norm(t1 % t1) - 173.4646938) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(cof(t1)) - 13.1909059)< BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(trans(adj(t1))) - 13.1909059)< BigTol);

        Tensor<T,2> t2; t2.iota(102);
        FASTOR_EXIT_ASSERT(std::abs(norm(outer(t2,t2)) - 21013.0) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(inner(t2,t2) - 21013.0) < Tol);
    }

    // 3D
    {
        Tensor<T,3,3> t1; t1.iota(0);
        FASTOR_EXIT_ASSERT(std::abs(determinant(t1)) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(trace(t1)-12) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(norm(matmul(t1,t1)) - 187.637949) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(matmul(2*t1-t1,t1/2*2)) - 187.637949) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(cofactor(t1)) - 18) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(transpose(adjoint(t1))) - 18) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(det(t1)) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(trace(t1) - 12) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(norm(t1 % t1) - 187.637949) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(cof(t1)) - 18)< BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(trans(adj(t1))) - 18) < BigTol);

        Tensor<T,3> t2; t2.iota(102);
        FASTOR_EXIT_ASSERT(std::abs(norm(outer(t2,t2)) - 31829.0) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(inner(t2,t2) - 31829.0) < Tol);
    }

    // cross product
    {
        // classic cross product
        Tensor<T,3> a = {1,2,3};
        Tensor<T,3> b = {4,5,17};
        Tensor<T,3> res = cross(a,b);
        FASTOR_EXIT_ASSERT(std::abs(res(0) - 19) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(res(1) + 5 ) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(res(2) + 3 ) < Tol);

        FASTOR_EXIT_ASSERT(std::abs(sum(cross(a  ,b+0)) - 11) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(sum(cross(a+0,b  )) - 11) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(sum(cross(a+0,b+0)) - 11) < Tol);
    }

    // cofactor and adjoint
    {
        Tensor<T,2,2> a0; a0.random();
        Tensor<T,3,3> a1; a1.random();
        Tensor<T,4,4> a2; a2.random();
        for (size_t i=0; i<2; ++i) a0(i,i) = 10;
        for (size_t i=0; i<3; ++i) a1(i,i) = 10;
        for (size_t i=0; i<4; ++i) a2(i,i) = 10;

        FASTOR_EXIT_ASSERT(std::abs(sum(transpose( adjoint(a0)) - cofactor(a0))) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(transpose(cofactor(a0)) -  adjoint(a0))) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(sum(transpose( adjoint(a1)) - cofactor(a1))) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(transpose(cofactor(a1)) -  adjoint(a1))) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(sum(transpose( adjoint(a2)) - cofactor(a2))) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(transpose(cofactor(a2)) -  adjoint(a2))) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(sum(trans(adj(a0)) - cof(a0))) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(trans(cof(a0)) - adj(a0))) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(sum(trans(adj(a1)) - cof(a1))) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(trans(cof(a1)) - adj(a1))) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(sum(trans(adj(a2)) - cof(a2))) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(trans(cof(a2)) - adj(a2))) < BigTol);

        // expressions
        FASTOR_EXIT_ASSERT(std::abs(sum(transpose( adjoint(a0-1)) - cofactor(a0-1))) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(transpose(cofactor(a0+2)) -  adjoint(a0+2))) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(sum(trans(adj(a1+0)) - cof(a1-a1+a1))) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(trans(cof(a1+0)) - adj(a1+a1-a1))) < BigTol);

        // if trans(cof/adj) dispatches to adj/cof then test
        FASTOR_EXIT_ASSERT(std::abs(sum(transpose(adj(a1+0)) - cof(a1-a1+a1))) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(transpose(cof(a1+0)) - adj(a1+a1-a1))) < BigTol);
    }

    {
        Tensor<T,2,2> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,3,3> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,4,4> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,5,5> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,7,7> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,8,8> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,9,9> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,10,10> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,12,12> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,16,16> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,17,17> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,20,20> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,24,24> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,40,40> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }

    // non-square
    {
        Tensor<T,2,3> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,3,4> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,4,5> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,5,6> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,6,7> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,17,29> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }

    // transpose expressions
    {
        Tensor<T,2,3> t1; t1.iota(5);
        FASTOR_EXIT_ASSERT(std::abs(sum(transpose(t1 + 0)) - sum(t1)) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(trans(t1 + 0))     - sum(t1)) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(sum(transpose(t1 + t1*2 - t1 - t1)) - sum(t1)) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(trans(t1 + t1*2 - t1 - t1))     - sum(t1)) < BigTol);
    }

    // Misc
    {
        Tensor<T,2,2> a0; a0.iota(3);
        a0 = a0 + transpose(a0);
        FASTOR_EXIT_ASSERT(a0.is_symmetric(Tol));

        Tensor<T,3,3> a1; a1.iota(5);
        a1 = a1 + transpose(a1);
        FASTOR_EXIT_ASSERT(a1.is_symmetric(Tol));

        Tensor<T,4,4> a2; a2.iota(55);
        a2 = 0.5*(a2 + transpose(a2))+1;
        FASTOR_EXIT_ASSERT(a2.is_symmetric(Tol));

        FASTOR_EXIT_ASSERT(a0.is_equal(a0));
        FASTOR_EXIT_ASSERT(a1.is_equal(a1,Tol));
        FASTOR_EXIT_ASSERT(a2.is_equal(a2,BigTol));
    }

    // inverse/inv + sum
    {
        Tensor<T,2,2> a0; a0.iota(1);
        for (size_t i=0; i<2; ++i) a0(i,i) = 100;
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse(a0)) - 0.01951170702421453) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(inv(a0))     - 0.01951170702421453) < BigTol);

        Tensor<T,3,3> a1; a1.iota(1);
        for (size_t i=0; i<3; ++i) a1(i,i) = 100;
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse(a1)) - 0.027264025471546025) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(inv(a1))     - 0.027264025471546025) < BigTol);

        Tensor<T,4,4> a2; a2.iota(1);
        for (size_t i=0; i<4; ++i) a2(i,i) = 100;
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse(a2)) - 0.031834300289961856) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(inv(a2))     - 0.031834300289961856) < BigTol);

        Tensor<T,5,5> a3; a3.iota(1);
        for (size_t i=0; i<5; ++i) a3(i,i) = 100;
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse(a3)) - 0.032799849456042335) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(inv(a3))     - 0.032799849456042335) < BigTol);

        Tensor<T,6,6> a4; a4.iota(1);
        for (size_t i=0; i<6; ++i) a4(i,i) = 100;
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse(a4)) - 0.03099899735295672) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(inv(a4))     - 0.03099899735295672) < BigTol);

        Tensor<T,7,7> a5; a5.iota(1);
        for (size_t i=0; i<7; ++i) a5(i,i) = 100;
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse(a5)) - 0.027748441684078293) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(inv(a5))     - 0.027748441684078293) < BigTol);

        Tensor<T,8,8> a6; a6.iota(1);
        for (size_t i=0; i<8; ++i) a6(i,i) = 100;
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse(a6)) - 0.02408691236995123) < HugeTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(inv(a6))     - 0.02408691236995123) < HugeTol);

        Tensor<T,15,15> a7; a7.iota(1);
        for (size_t i=0; i<15; ++i) a7(i,i) = 100;
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse(a7)) - 0.009215486449015996) < HugeTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(inv(a7))     - 0.009215486449015996) < HugeTol);
    }

    // inverse/inv of expressions
    {
        Tensor<T,2,2> a0; a0.iota(1);
        for (size_t i=0; i<2; ++i) a0(i,i) = 100;
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse(a0 + 0)) - 0.01951170702421453) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(inv(a0 + 0))     - 0.01951170702421453) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(sum(inverse(a0 + a0 - a0)) - 0.01951170702421453) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(inv(a0 + a0 - a0))     - 0.01951170702421453) < BigTol);
    }

    // solve
    {
        constexpr size_t M = 2;
        Tensor<T,M,M> a; a.iota(1);
        for (size_t i=0; i<M; ++i) a(i,i) = 100;
        Tensor<T,M> b; b.iota(100);
        Tensor<T,M,1> c; c.iota(100);
        Tensor<T,M,5> d; d.iota(100);

        Tensor<T,M> b1 = solve(a,b);
        Tensor<T,M,1> c1 = solve(a,c);
        Tensor<T,M,5> d1 = solve(a,d);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 1.960976585951571 ) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(c1) - 1.960976585951571 ) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(d1) - 10.196117670602362) < BigTol);
    }
    {
        constexpr size_t M = 3;
        Tensor<T,M,M> a; a.iota(1);
        for (size_t i=0; i<M; ++i) a(i,i) = 100;
        Tensor<T,M> b; b.iota(100);
        Tensor<T,M,1> c; c.iota(100);
        Tensor<T,M,5> d; d.iota(100);

        Tensor<T,M> b1 = solve(a,b);
        Tensor<T,M,1> c1 = solve(a,c);
        Tensor<T,M,5> d1 = solve(a,d);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 2.753858012252136 ) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(c1) - 2.753858012252136 ) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(d1) - 14.591039617926809) < BigTol);
    }
    {
        constexpr size_t M = 4;
        Tensor<T,M,M> a; a.iota(1);
        for (size_t i=0; i<M; ++i) a(i,i) = 100;
        Tensor<T,M> b; b.iota(100);
        Tensor<T,M,1> c; c.iota(100);
        Tensor<T,M,5> d; d.iota(100);

        Tensor<T,M> b1 = solve(a,b);
        Tensor<T,M,1> c1 = solve(a,c);
        Tensor<T,M,5> d1 = solve(a,d);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 3.2316174170320178) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(c1) - 3.2316174170320178) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(d1) - 17.44017784877636 ) < BigTol);
    }

    // solve expressions
    {
        constexpr size_t M = 2;
        Tensor<T,M,M> a; a.iota(1);
        for (size_t i=0; i<M; ++i) a(i,i) = 100;
        Tensor<T,M> b; b.iota(100);
        Tensor<T,M,1> c; c.iota(100);
        Tensor<T,M,5> d; d.iota(100);

        FASTOR_EXIT_ASSERT(std::abs(sum(solve(a  ,b+0)) - 1.960976585951571 ) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(solve(a+0,b  )) - 1.960976585951571 ) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(solve(a+0,b+0)) - 1.960976585951571 ) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(solve(a  ,c+0)) - 1.960976585951571 ) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(solve(a+0,c  )) - 1.960976585951571 ) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(solve(a+0,c+0)) - 1.960976585951571 ) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(solve(a  ,d+0)) - 10.196117670602362) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(solve(a+0,d  )) - 10.196117670602362) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(solve(a+0,d+0)) - 10.196117670602362) < BigTol);
    }
    {
        constexpr size_t M = 3;
        Tensor<T,M,M> a; a.iota(1);
        for (size_t i=0; i<M; ++i) a(i,i) = 100;
        Tensor<T,M> b; b.iota(100);

        Tensor<T,M> b1;
        b1.iota(1);
        b1 += solve(a,b+0);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 8.753858012252135) < BigTol);

        b1.iota(1);
        b1 += solve(a*1,b);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 8.753858012252135) < BigTol);

        b1.iota(1);
        b1 += solve(a*1,b+b-b);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 8.753858012252135) < BigTol);

        b1.iota(1);
        b1 -= solve(a,b+0);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 3.2461419877478646) < BigTol);

        b1.iota(1);
        b1 -= solve(a+1-1,b);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 3.2461419877478646) < BigTol);

        b1.iota(1);
        b1 -= solve(3*a-2*a,2*b-b);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 3.2461419877478646) < BigTol);

        b1.iota(1);
        b1 *= solve(a,b+0);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 5.432097372239239 ) < BigTol);

        b1.iota(1);
        b1 *= solve(1+a-1,b);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 5.432097372239239 ) < BigTol);

        b1.iota(1);
        b1 *= solve(a+a+a-a*2,1*b+0);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 5.432097372239239 ) < BigTol);

        b1.iota(1);
        b1 /= solve(a,b+0);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 6.6337042772104695) < BigTol);

        b1.iota(1);
        b1 /= solve(1+a-1,b);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 6.6337042772104695) < BigTol);

        b1.iota(1);
        b1 /= solve(a+a+a-a*2,1*b+0);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 6.6337042772104695) < BigTol);

        // mix with other expressions
        b1.iota(1);
        b1 += solve(a,b+0) + b1 - b1;
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 8.753858012252135) < BigTol);

        b1.iota(1);
        b1 += solve(a*1,b)*1;
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 8.753858012252135) < BigTol);

        b1.iota(1);
        b1 += solve(a*1,b+b-b) - 1 + 1;
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 8.753858012252135) < BigTol);
    }
    {
        constexpr size_t M = 3;
        Tensor<T,M,M> a; a.iota(1);
        for (size_t i=0; i<M; ++i) a(i,i) = 100;
        Tensor<T,M,5> b; b.iota(100);

        Tensor<T,M,5> b1;
        b1.iota(1);
        b1 += solve(a,b+0);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 134.5910396179268) < BigTol);

        b1.iota(1);
        b1 += solve(a*1,b);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 134.5910396179268) < BigTol);

        b1.iota(1);
        b1 += solve(a*1,b+b-b);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 134.5910396179268) < BigTol);

        b1.iota(1);
        b1 -= solve(a,b+0);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 105.4089603820732) < BigTol);

        b1.iota(1);
        b1 -= solve(a+1-1,b);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 105.4089603820732) < BigTol);

        b1.iota(1);
        b1 -= solve(3*a-2*a,2*b-b);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 105.4089603820732) < BigTol);

        b1.iota(1);
        b1 *= solve(a,b+0);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 117.0727470578752 ) < BigTol);

        b1.iota(1);
        b1 *= solve(1+a-1,b);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 117.0727470578752 ) < BigTol);

        b1.iota(1);
        b1 *= solve(a+a+a-a*2,1*b+0);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 117.0727470578752 ) < BigTol);

        b1.iota(1);
        b1 /= solve(a,b+0);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 123.02004364710513) < BigTol);

        b1.iota(1);
        b1 /= solve(1+a-1,b);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 123.02004364710513) < BigTol);

        b1.iota(1);
        b1 /= solve(a+a+a-a*2,1*b+0);
        FASTOR_EXIT_ASSERT(std::abs(sum(b1) - 123.02004364710513) < BigTol);
    }

    // QR
    {
        Tensor<T,4,4> A; A.arange();
        for (size_t i=0; i<4; ++i) A(i,i) = 10;

        {
            Tensor<T,4,4> Q, R;
            std::tie(Q,R) = qr(A);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - Q%R)) < BigTol);
        }
        {
            Tensor<T,4,4> Q, R;
            std::tie(Q,R) = qr(A+0);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - Q%R)) < BigTol);
        }
        {
            Tensor<T,4,4> Q, R;
            std::tie(Q,R) = qr<QRCompType::MGSR>(A+0);
            FASTOR_EXIT_ASSERT(std::abs(sum(A - Q%R)) < BigTol);
        }
        // absdet by QR
        {
            FASTOR_EXIT_ASSERT(std::abs(absdet<DetCompType::QR>(A) - std::abs(determinant(A))) < HugeTol);
        }
        // logdet by QR
        {
            FASTOR_EXIT_ASSERT(std::abs(logdet<DetCompType::QR>(A) - std::log(std::abs(determinant(A)))) < HugeTol);
        }
    }

    print(FGRN(BOLD("All tests passed successfully")));

}

int main() {

    print(FBLU(BOLD("Testing tensor algebra routines: single precision")));
    test_linalg<float>();
    print(FBLU(BOLD("Testing tensor algebra routines: double precision")));
    test_linalg<double>();

    return 0;
}

