#include <Fastor/Fastor.h>

using namespace Fastor;


#define Tol 1e-09
#define BigTol 1e-4
#define HugeTol 1e-2

template<typename T>
void test_inverse() {


    // inverse/inv
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

    // inverse using LU
    {
        Tensor<T,7,7> a0; a0.iota(1);
        for (size_t i=0; i<7; ++i) a0(i,i) = 100;
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse<InvCompType::SimpleInvPiv>(a0)) - 0.027748441684078293) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse<InvCompType::SimpleLU>(a0))     - 0.027748441684078293) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse<InvCompType::BlockLU>(a0))      - 0.027748441684078293) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse<InvCompType::SimpleLUPiv>(a0))  - 0.027748441684078293) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse<InvCompType::BlockLUPiv>(a0))   - 0.027748441684078293) < BigTol);
    }

    {
        Tensor<T,7,7> a0; a0.iota(1);
        for (size_t i=0; i<7; ++i) a0(i,i) = 100;
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse<InvCompType::SimpleInvPiv>(a0+0)) - 0.027748441684078293) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse<InvCompType::SimpleLU>(a0+0))     - 0.027748441684078293) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse<InvCompType::BlockLU>(a0+0))      - 0.027748441684078293) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse<InvCompType::SimpleLUPiv>(a0+0))  - 0.027748441684078293) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(inverse<InvCompType::BlockLUPiv>(a0+0))   - 0.027748441684078293) < BigTol);
    }

    print(FGRN(BOLD("All tests passed successfully")));

}

int main() {

    print(FBLU(BOLD("Testing tensor algebra routines: single precision")));
    test_inverse<float>();
    print(FBLU(BOLD("Testing tensor algebra routines: double precision")));
    test_inverse<double>();

    return 0;
}

