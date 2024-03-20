#include <Fastor/Fastor.h>
using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5
#define HugeTol 1e-2



template<typename T>
void run() {

    {
        Tensor<T,4,5> a; a.iota(1);
        TensorMap<T,4,5> ma(a);
        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - ma.sum()) < Tol);

        ma +=1;
        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - ma.sum()) < Tol);

        ma(all,0) = 2;
        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - ma.sum()) < Tol);

        ma(fall,fall) = 2;
        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - ma.sum()) < Tol);

        ma = a - ma;
        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - ma.sum()) < Tol);

        a.iota(1);
        ma *= a/ma + 2;
        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - ma.sum()) < Tol);
    }

    {
        Tensor<T,2,4,5> a; a.iota(1);
        TensorMap<T,2,4,5> ma(a);
        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - ma.sum()) < Tol);

        ma +=1;
        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - ma.sum()) < Tol);

        ma(0,all,0) = 2;
        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - ma.sum()) < Tol);

        ma(fseq<0,1>{},fall,fall) = 3;
        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - ma.sum()) < Tol);

        ma = a - ma;
        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - ma.sum()) < Tol);

        a.iota(1);
        ma *= a/ma + 2;
        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - ma.sum()) < Tol);
    }

    // Compound assignment operators.
    {
        Tensor<T,4,5> src;
        src = 3;
        TensorMap<T,4,5> msrc(src);
        Tensor<T,4,5> dst;
        dst = 0;
        Tensor<T,4,5> check;

        // +=
        check = 3;
        dst += msrc;
        FASTOR_DOES_CHECK_PASS(all_of(dst == check));

        // *=
        check = 9;
        dst *= msrc;
        FASTOR_DOES_CHECK_PASS(all_of(dst == check));

        // -=
        check = 6;
        dst -= msrc;
        FASTOR_DOES_CHECK_PASS(all_of(dst == check));

        // /=
        check = 2;
        dst /= msrc;
        FASTOR_DOES_CHECK_PASS(all_of(dst == check));
    }

    // Map a const array and copy-assign it to a non-const tensor.
    {
        const T data[] = {1, 2, 3};
        TensorMap<const T, 3> mdata{data};
        Tensor<T, 3> tdata = mdata;
        Tensor<T, 3> check{1, 2, 3};

        FASTOR_DOES_CHECK_PASS(all_of(tdata == check));
    }

    // Bug 116
    {
        T data[4] = {1,2,3,4};
        auto A = TensorMap<T,2,2>(data);
        Tensor<T,2,1> b;
        b = {{1},{1}};
        A(fseq<0,2>(),fseq<1,2>()) = b;
        FASTOR_DOES_CHECK_PASS(std::abs(sum(A) - 6) < Tol);
    }

    // Bug 117 - standard tensor arithmetic with const tensormap
    {
        const T dataA[] = {1,2,3};
        const T dataB[] = {1,3,30};

        TensorMap<const T,3,1> a ( dataA );
        TensorMap<const T,3,1> b ( dataB );
        const auto res0 = evaluate( a + b );
        const auto res1 = evaluate( a - b );
        const auto res2 = evaluate( a * b );
        FASTOR_DOES_CHECK_PASS(std::abs(res1.sum() + 28 ) < Tol);
    }

    // Bug 132 - add 1D tensormaps, outer tensormaps
    {
        std::array<double,3> a = {1,1,1};
        TensorMap<double,3> at(a.data());

        std::array<double,3> b = {2,2,2};
        TensorMap<double,3> bt(b.data());

        Tensor<double,3> ct = bt + at;
        FASTOR_DOES_CHECK_PASS(std::abs(ct.sum() - 9 ) < Tol);

        Tensor<double,3,3> dt = outer(at, bt);
        FASTOR_DOES_CHECK_PASS(std::abs(dt.sum() - 18 ) < Tol);
    }

    print(FGRN(BOLD("All tests passed successfully")));
}


int main() {

    print(FBLU(BOLD("Testing tensor map: int 32")));
    run<int>();
    print(FBLU(BOLD("Testing tensor map: int 64")));
    run<Int64>();
    print(FBLU(BOLD("Testing tensor map: single precision")));
    run<float>();
    print(FBLU(BOLD("Testing tensor map: double precision")));
    run<double>();

    return 0;
}
