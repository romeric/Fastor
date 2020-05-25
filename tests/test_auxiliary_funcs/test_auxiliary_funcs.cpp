#include <Fastor/Fastor.h>

using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5



int main() {

    print(FBLU(BOLD("Testing auxiliary functions e.g. meta functions etc")));

    {
        {
            constexpr size_t ind0[2] = {0,1};
            constexpr size_t ind1[1] = {1};
            bool a1 = internal::match_indices_from_end(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==true);
            bool a2 = internal::match_indices_from_end(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==true);

            constexpr size_t ind2[2] = {0,1};
            constexpr size_t ind3[1] = {0};
            bool a3 = internal::match_indices_from_end(ind2, ind3);
            FASTOR_EXIT_ASSERT(a3!=true);
            bool a4 = internal::match_indices_from_end(ind3, ind2);
            FASTOR_EXIT_ASSERT(a4!=true);

            a4 = internal::match_indices_from_end(ind3, ind3);
            FASTOR_EXIT_ASSERT(a4==true);
        }
    }

    {
        {
            constexpr size_t ind0[3] = {0,1,2};
            constexpr size_t ind1[2] = {1,2};
            bool a1 = internal::match_indices_from_end(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==true);
            bool a2 = internal::match_indices_from_end(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==true);

            constexpr size_t ind00[3] = {0,1,2};
            constexpr size_t ind11[1] = {2};
            a1 = internal::match_indices_from_end(ind00, ind11);
            FASTOR_EXIT_ASSERT(a1==true);
            a2 = internal::match_indices_from_end(ind11, ind00);
            FASTOR_EXIT_ASSERT(a2==true);

            constexpr size_t ind2[3] = {0,1,2};
            constexpr size_t ind3[1] = {1};
            bool a3 = internal::match_indices_from_end(ind2, ind3);
            FASTOR_EXIT_ASSERT(a3!=true);
            bool a4 = internal::match_indices_from_end(ind3, ind2);
            FASTOR_EXIT_ASSERT(a4!=true);

            constexpr size_t ind22[3] = {0,1,2};
            constexpr size_t ind33[1] = {0};
            a3 = internal::match_indices_from_end(ind22, ind33);
            FASTOR_EXIT_ASSERT(a3!=true);
            a4 = internal::match_indices_from_end(ind33, ind22);
            FASTOR_EXIT_ASSERT(a4!=true);
        }
    }


    {
        {
            constexpr size_t ind0[5] = {7,6,8,9,10};
            constexpr size_t ind1[3] = {8,9,10};
            bool a1 = internal::match_indices_from_end(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==true);
            bool a2 = internal::match_indices_from_end(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==true);

            constexpr size_t ind00[5] = {7,6,8,9,10};
            constexpr size_t ind11[3] = {8,8,10};
            bool a3 = internal::match_indices_from_end(ind00, ind11);
            FASTOR_EXIT_ASSERT(a3!=true);
            a3 = internal::match_indices_from_end(ind11, ind00);
            FASTOR_EXIT_ASSERT(a3!=true);
        }
    }


    {
        {
            constexpr size_t ind0[2] = {0,1};
            constexpr size_t ind1[1] = {1};
            size_t a1 = internal::match_indices_from_end_index(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==0);
            size_t a2 = internal::match_indices_from_end_index(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==0);
            // size_t a4 = match_indices_from_end_index(ind0, ind0);
            // FASTOR_EXIT_ASSERT(a4==0);
        }

        {
            constexpr size_t ind0[3] = {0,1,2};
            constexpr size_t ind1[1] = {2};
            size_t a1 = internal::match_indices_from_end_index(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==1);
            size_t a2 = internal::match_indices_from_end_index(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==1);
        }

        {
            constexpr size_t ind0[4] = {4,5,6,7};
            constexpr size_t ind1[2] = {6,7};
            size_t a1 = internal::match_indices_from_end_index(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==1);
            size_t a2 = internal::match_indices_from_end_index(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==1);
        }
    }


    {
        {
            constexpr size_t ind0[2] = {0,1};
            constexpr size_t ind1[1] = {0};
            bool a1 = internal::match_indices_from_start(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==true);
            bool a2 = internal::match_indices_from_start(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==true);

            constexpr size_t ind2[2] = {0,1};
            constexpr size_t ind3[1] = {1};
            bool a3 = internal::match_indices_from_start(ind2, ind3);
            FASTOR_EXIT_ASSERT(a3!=true);
            bool a4 = internal::match_indices_from_start(ind3, ind2);
            FASTOR_EXIT_ASSERT(a4!=true);

            a4 = internal::match_indices_from_start(ind3, ind3);
            FASTOR_EXIT_ASSERT(a4==true);
        }
    }

    {
        {
            constexpr size_t ind0[5] = {7,6,8,9,10};
            constexpr size_t ind1[3] = {7,6,8};
            bool a1 = internal::match_indices_from_start(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==true);
            bool a2 = internal::match_indices_from_start(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==true);

            constexpr size_t ind00[5] = {7,6,8,9,10};
            constexpr size_t ind11[3] = {7,8,8};
            bool a3 = internal::match_indices_from_start(ind00, ind11);
            FASTOR_EXIT_ASSERT(a3!=true);
            a3 = internal::match_indices_from_start(ind11, ind00);
            FASTOR_EXIT_ASSERT(a3!=true);
        }
    }

    {
        {
            constexpr size_t ind0[2] = {0,1};
            constexpr size_t ind1[1] = {0};
            size_t a1 = internal::match_indices_from_start_index(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==1);
            size_t a2 = internal::match_indices_from_start_index(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==1);
        }

        {
            constexpr size_t ind0[3] = {0,1,2};
            constexpr size_t ind1[1] = {0};
            size_t a1 = internal::match_indices_from_start_index(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==1);
            size_t a2 = internal::match_indices_from_start_index(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==1);
        }

        {
            constexpr size_t ind0[4] = {4,5,6,7};
            constexpr size_t ind1[2] = {4,5};
            size_t a1 = internal::match_indices_from_start_index(ind0, ind1);
            FASTOR_EXIT_ASSERT(a1==2);
            size_t a2 = internal::match_indices_from_start_index(ind1, ind0);
            FASTOR_EXIT_ASSERT(a2==2);
        }
    }




    {
        bool is_mat_vec = internal::is_generalised_matrix_vector<Index<0,1,2>,Index<1,2>>::value;
        bool is_vec_mat = internal::is_generalised_vector_matrix<Index<0,1,2>,Index<1,2>>::value;
        bool is_mat_mat = internal::is_generalised_matrix_matrix<Index<0,1,2>,Index<1,2>>::value;
        FASTOR_EXIT_ASSERT(is_mat_vec == true);
        FASTOR_EXIT_ASSERT(is_vec_mat == false);
        FASTOR_EXIT_ASSERT(is_mat_mat == false);
    }

    {
        bool is_mat_vec = internal::is_generalised_matrix_vector<Index<1,2>,Index<1,2,3>>::value;
        bool is_vec_mat = internal::is_generalised_vector_matrix<Index<1,2>,Index<1,2,3>>::value;
        bool is_mat_mat = internal::is_generalised_matrix_matrix<Index<1,2>,Index<1,2,3>>::value;
        FASTOR_EXIT_ASSERT(is_mat_vec == false);
        FASTOR_EXIT_ASSERT(is_vec_mat == true);
        FASTOR_EXIT_ASSERT(is_mat_mat == false);
    }

    {
        bool is_mat_vec = internal::is_generalised_matrix_vector<Index<0,1,2>,Index<1,2,3>>::value;
        bool is_vec_mat = internal::is_generalised_vector_matrix<Index<0,1,2>,Index<1,2,3>>::value;
        bool is_mat_mat = internal::is_generalised_matrix_matrix<Index<0,1,2>,Index<1,2,3>>::value;
        FASTOR_EXIT_ASSERT(is_mat_vec == false);
        FASTOR_EXIT_ASSERT(is_vec_mat == false);
        FASTOR_EXIT_ASSERT(is_mat_mat == true);
    }

    {
        Index<0,1,2> ii;
        Index<0,1,2> jj;
        bool is_mat_vec = internal::is_generalised_matrix_vector<decltype(ii),decltype(jj)>::value;
        bool is_vec_mat = internal::is_generalised_vector_matrix<decltype(ii),decltype(jj)>::value;
        bool is_mat_mat = internal::is_generalised_matrix_matrix<decltype(ii),decltype(jj)>::value;
        FASTOR_EXIT_ASSERT(is_mat_vec == false);
        FASTOR_EXIT_ASSERT(is_vec_mat == false);
        FASTOR_EXIT_ASSERT(is_mat_mat == false);
    }

    {
        Index<0,1,2> ii;
        Index<0,1,2,4> jj;
        bool is_mat_vec = internal::is_generalised_matrix_vector<decltype(ii),decltype(jj)>::value;
        bool is_vec_mat = internal::is_generalised_vector_matrix<decltype(ii),decltype(jj)>::value;
        bool is_mat_mat = internal::is_generalised_matrix_matrix<decltype(ii),decltype(jj)>::value;
        FASTOR_EXIT_ASSERT(is_mat_vec == false);
        FASTOR_EXIT_ASSERT(is_vec_mat == true);
        FASTOR_EXIT_ASSERT(is_mat_mat == false);
    }

    {
        Index<3,0,1,2> ii;
        Index<0,1,2,4> jj;
        bool is_mat_vec = internal::is_generalised_matrix_vector<decltype(ii),decltype(jj)>::value;
        bool is_vec_mat = internal::is_generalised_vector_matrix<decltype(ii),decltype(jj)>::value;
        bool is_mat_mat = internal::is_generalised_matrix_matrix<decltype(ii),decltype(jj)>::value;
        FASTOR_EXIT_ASSERT(is_mat_vec == false);
        FASTOR_EXIT_ASSERT(is_vec_mat == false);
        FASTOR_EXIT_ASSERT(is_mat_mat == true);
    }

    {
        Index<3,0,7,2> ii;
        Index<0,1,2,4> jj;
        bool is_mat_vec = internal::is_generalised_matrix_vector<decltype(ii),decltype(jj)>::value;
        bool is_vec_mat = internal::is_generalised_vector_matrix<decltype(ii),decltype(jj)>::value;
        bool is_mat_mat = internal::is_generalised_matrix_matrix<decltype(ii),decltype(jj)>::value;
        FASTOR_EXIT_ASSERT(is_mat_vec == false);
        FASTOR_EXIT_ASSERT(is_vec_mat == false);
        FASTOR_EXIT_ASSERT(is_mat_mat == false);
    }

    {
        Index<2,3> ii;
        Index<0,1,2> jj;
        bool is_mat_vec = internal::is_generalised_matrix_vector<decltype(ii),decltype(jj)>::value;
        bool is_vec_mat = internal::is_generalised_vector_matrix<decltype(ii),decltype(jj)>::value;
        bool is_mat_mat = internal::is_generalised_matrix_matrix<decltype(ii),decltype(jj)>::value;
        FASTOR_EXIT_ASSERT(is_mat_vec == false);
        FASTOR_EXIT_ASSERT(is_vec_mat == false);
        FASTOR_EXIT_ASSERT(is_mat_mat == false);
    }

    print(FGRN(BOLD("All tests passed successfully")));

    return 0;
}
