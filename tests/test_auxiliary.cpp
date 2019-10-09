#include <Fastor.h>

using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5



int main() {

    print(FBLU(BOLD("Testing auxiliary functions e.g. meta functions etc")));

    {
        {
            constexpr size_t ind0[2] = {0,1};
            constexpr size_t ind1[1] = {1};
            bool a1 = match_indices_from_end(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==true);
            bool a2 = match_indices_from_end(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==true);

            constexpr size_t ind2[2] = {0,1};
            constexpr size_t ind3[1] = {0};
            bool a3 = match_indices_from_end(ind2, ind3);
            FASTOR_EXIT_ASSERT(a3!=true);
            bool a4 = match_indices_from_end(ind3, ind2);
            FASTOR_EXIT_ASSERT(a4!=true);

            a4 = match_indices_from_end(ind3, ind3);
            FASTOR_EXIT_ASSERT(a4==true);
        }
    }

    {
        {
            constexpr size_t ind0[3] = {0,1,2};
            constexpr size_t ind1[2] = {1,2};
            bool a1 = match_indices_from_end(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==true);
            bool a2 = match_indices_from_end(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==true);

            constexpr size_t ind00[3] = {0,1,2};
            constexpr size_t ind11[1] = {2};
            a1 = match_indices_from_end(ind00, ind11);
            FASTOR_EXIT_ASSERT(a1==true);
            a2 = match_indices_from_end(ind11, ind00);
            FASTOR_EXIT_ASSERT(a2==true);

            constexpr size_t ind2[3] = {0,1,2};
            constexpr size_t ind3[1] = {1};
            bool a3 = match_indices_from_end(ind2, ind3);
            FASTOR_EXIT_ASSERT(a3!=true);
            bool a4 = match_indices_from_end(ind3, ind2);
            FASTOR_EXIT_ASSERT(a4!=true);

            constexpr size_t ind22[3] = {0,1,2};
            constexpr size_t ind33[1] = {0};
            a3 = match_indices_from_end(ind22, ind33);
            FASTOR_EXIT_ASSERT(a3!=true);
            a4 = match_indices_from_end(ind33, ind22);
            FASTOR_EXIT_ASSERT(a4!=true);
        }
    }


    {
        {
            constexpr size_t ind0[5] = {7,6,8,9,10};
            constexpr size_t ind1[3] = {8,9,10};
            bool a1 = match_indices_from_end(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==true);
            bool a2 = match_indices_from_end(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==true);

            constexpr size_t ind00[5] = {7,6,8,9,10};
            constexpr size_t ind11[3] = {8,8,10};
            bool a3 = match_indices_from_end(ind00, ind11);
            FASTOR_EXIT_ASSERT(a3!=true);
            a3 = match_indices_from_end(ind11, ind00);
            FASTOR_EXIT_ASSERT(a3!=true);
        }
    }


    {
        {
            constexpr size_t ind0[2] = {0,1};
            constexpr size_t ind1[1] = {1};
            size_t a1 = match_indices_from_end_index(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==0);
            size_t a2 = match_indices_from_end_index(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==0);
            // size_t a4 = match_indices_from_end_index(ind0, ind0);
            // FASTOR_EXIT_ASSERT(a4==0);
        }

        {
            constexpr size_t ind0[3] = {0,1,2};
            constexpr size_t ind1[1] = {2};
            size_t a1 = match_indices_from_end_index(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==1);
            size_t a2 = match_indices_from_end_index(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==1);
        }

        {
            constexpr size_t ind0[4] = {4,5,6,7};
            constexpr size_t ind1[2] = {6,7};
            size_t a1 = match_indices_from_end_index(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==1);
            size_t a2 = match_indices_from_end_index(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==1);
        }
    }


    {
        {
            constexpr size_t ind0[2] = {0,1};
            constexpr size_t ind1[1] = {0};
            bool a1 = match_indices_from_start(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==true);
            bool a2 = match_indices_from_start(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==true);

            constexpr size_t ind2[2] = {0,1};
            constexpr size_t ind3[1] = {1};
            bool a3 = match_indices_from_start(ind2, ind3);
            FASTOR_EXIT_ASSERT(a3!=true);
            bool a4 = match_indices_from_start(ind3, ind2);
            FASTOR_EXIT_ASSERT(a4!=true);

            a4 = match_indices_from_start(ind3, ind3);
            FASTOR_EXIT_ASSERT(a4==true);
        }
    }

    {
        {
            constexpr size_t ind0[5] = {7,6,8,9,10};
            constexpr size_t ind1[3] = {7,6,8};
            bool a1 = match_indices_from_start(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==true);
            bool a2 = match_indices_from_start(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==true);

            constexpr size_t ind00[5] = {7,6,8,9,10};
            constexpr size_t ind11[3] = {7,8,8};
            bool a3 = match_indices_from_start(ind00, ind11);
            FASTOR_EXIT_ASSERT(a3!=true);
            a3 = match_indices_from_start(ind11, ind00);
            FASTOR_EXIT_ASSERT(a3!=true);
        }
    }

    {
        {
            constexpr size_t ind0[2] = {0,1};
            constexpr size_t ind1[1] = {0};
            size_t a1 = match_indices_from_start_index(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==1);
            size_t a2 = match_indices_from_start_index(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==1);
        }

        {
            constexpr size_t ind0[3] = {0,1,2};
            constexpr size_t ind1[1] = {0};
            size_t a1 = match_indices_from_start_index(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==1);
            size_t a2 = match_indices_from_start_index(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==1);
        }

        {
            constexpr size_t ind0[4] = {4,5,6,7};
            constexpr size_t ind1[2] = {4,5};
            size_t a1 = match_indices_from_start_index(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==2);
            size_t a2 = match_indices_from_start_index(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==2);
        }
    }

    print(FGRN(BOLD("All tests passed successfully")));

    return 0;
}