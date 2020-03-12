#include <Fastor/Fastor.h>
using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5

template<typename T>
void run() {

    using std::abs;
    {
        Tensor<T,9,10> a1; a1.iota(0);
        Tensor<int,5,5> it1;
        for (auto i=0; i<5; ++i)
            for (auto j=0; j<5; ++j)
                it1(i,j) = i*9 + j + 11 + i;

        // Check construction from views
        Tensor<T,5,5> a2 = a1(it1);
        FASTOR_EXIT_ASSERT(abs(a2.sum() - 825) < Tol);
        a2 += a1(it1);
        FASTOR_EXIT_ASSERT(abs(a2.sum() - 2*825) < Tol);
        a2 -= a1(it1);
        FASTOR_EXIT_ASSERT(abs(a2.sum() - 825) < Tol);
        a2 *= 2. + a1(it1);
        FASTOR_EXIT_ASSERT(abs(a2.sum() - 33925) < Tol);
        a2 /= a1(it1) + 5000;
        FASTOR_EXIT_ASSERT(abs(a2.sum() - 6.72701) < 1e-2);

        // Assigning to a view from numbers/tensors/views
        Tensor<T,9,10> a3; a3.iota(10);
        a3(it1) = 4;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 3930) < Tol);
        a3(it1) += 3;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 4005) < Tol);
        a3(it1) -= 3;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 3930) < Tol);
        a3(it1) *= 3;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 4130) < Tol);
        a3(it1) /= 3;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 3930) < Tol);

        Tensor<T,5,5> a4; a4.iota(1);
        a3.fill(1);
        a3(it1) = a4;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 390) < Tol);
        a3(it1) += 2*a4;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 1040) < Tol);
        a3(it1) -= a4+2;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 665) < Tol);
        a3(it1) *= -a4-100;
        FASTOR_EXIT_ASSERT(abs(a3.sum() + 70335) < Tol);
        a3(it1) /= 100000.+a4;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 64.2961) < 1e-2);

        a3.iota(10);
        a3(it1) = a3(it1);
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 4905) < Tol);
        a3(it1) += a3(it1);
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 5980) < Tol);
        a3(it1) -= a3(it1);
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 3830) < Tol);
        a3(it1) = 2;
        a3(it1) *= a3(it1);
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 3930) < Tol);
        a3(it1) /= a3(it1);
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 3855) < BigTol);

        // Check overlap
        a3.iota(10);
        Tensor<int,5,5> it2; it2.iota(0);
        it1.iota(2);
        a3(it1).noalias() = a3(it2);
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 4855) < BigTol);
        a3(it1).noalias() += a3(it2);
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 5359) < BigTol);
        a3(it1).noalias() -= a3(it2);
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 4414) < BigTol);
        a3(it1).noalias() *= a3(it2);
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 4888) < BigTol);
        a3(it1).noalias() /= a3(it2);
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 4348.14550) < 1e-2);
        
        a3.iota(10);
        a3(it1).noalias() = a3(it2);
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 4855) < BigTol);
        a3(it1).noalias() += a3(it2);
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 5359) < BigTol);
        a3(it1).noalias() -= a3(it2);
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 4414) < BigTol);
        a3(it1).noalias() *= a3(it2);
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 4888) < BigTol);
        a3(it1).noalias() /= a3(it2);
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 4348.14550) < 1e-2);

        print(FGRN(BOLD("All tests passed successfully")));
    }
}


int main() {

    print(FBLU(BOLD("Testing multi-dimensional tensor views: single precision")));
    run<float>();
    print(FBLU(BOLD("Testing multi-dimensional tensor views: double precision")));
    run<double>();

    return 0;
}
