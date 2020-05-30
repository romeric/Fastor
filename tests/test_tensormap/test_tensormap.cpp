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
        FASTOR_EXIT_ASSERT(abs(a.sum() - ma.sum()) < Tol);

        ma +=1;
        FASTOR_EXIT_ASSERT(abs(a.sum() - ma.sum()) < Tol);

        ma(all,0) = 2;
        FASTOR_EXIT_ASSERT(abs(a.sum() - ma.sum()) < Tol);

        ma(fall,fall) = 2;
        FASTOR_EXIT_ASSERT(abs(a.sum() - ma.sum()) < Tol);

        ma = a - ma;
        FASTOR_EXIT_ASSERT(abs(a.sum() - ma.sum()) < Tol);

        a.iota(1);
        ma *= a/ma + 2;
        FASTOR_EXIT_ASSERT(abs(a.sum() - ma.sum()) < Tol);
    }

    {
        Tensor<T,2,4,5> a; a.iota(1);
        TensorMap<T,2,4,5> ma(a);
        FASTOR_EXIT_ASSERT(abs(a.sum() - ma.sum()) < Tol);

        ma +=1;
        FASTOR_EXIT_ASSERT(abs(a.sum() - ma.sum()) < Tol);

        ma(0,all,0) = 2;
        FASTOR_EXIT_ASSERT(abs(a.sum() - ma.sum()) < Tol);

        ma(fseq<0,1>{},fall,fall) = 3;
        FASTOR_EXIT_ASSERT(abs(a.sum() - ma.sum()) < Tol);

        ma = a - ma;
        FASTOR_EXIT_ASSERT(abs(a.sum() - ma.sum()) < Tol);

        a.iota(1);
        ma *= a/ma + 2;
        FASTOR_EXIT_ASSERT(abs(a.sum() - ma.sum()) < Tol);
    }

    // Map a const array and copy-assign it to a non-const tensor.
    {
        const T data[] = {1, 2, 3};
        TensorMap<const T, 3> mdata{data};
        Tensor<T, 3> tdata = mdata;
        Tensor<T, 3> check{1, 2, 3};

        FASTOR_EXIT_ASSERT(all_of(tdata == check));
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
