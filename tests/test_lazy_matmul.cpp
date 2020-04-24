#include <Fastor/Fastor.h>

using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5


// Generic matmul function for AbstractTensor types
// Works as long as the return tensor is compile time deducible
template<typename Derived0, size_t DIM0, typename Derived1, size_t DIM1,
    enable_if_t_<is_less_equal_v_<DIM0,2> && is_less_equal_v_<DIM1,2>,bool> = 0 >
FASTOR_INLINE
conditional_t_<Derived0::result_type::Dimension_t::value == 1,
    Tensor<typename scalar_type_finder<Derived0>::type,
        Derived1::result_type::Dimension_t::value == 2 ? get_tensor_dimensions<typename Derived1::result_type>::dims[1] : 1>,
    conditional_t_<Derived1::result_type::Dimension_t::value == 1,
        Tensor<typename scalar_type_finder<Derived0>::type,
            Derived0::result_type::Dimension_t::value == 2 ? get_tensor_dimensions<typename Derived0::result_type>::dims[0] : 1>,
        Tensor<typename scalar_type_finder<Derived0>::type,
            Derived0::result_type::Dimension_t::value == 2 ? get_tensor_dimensions<typename Derived0::result_type>::dims[0] : 1,
            Derived1::result_type::Dimension_t::value == 2 ? get_tensor_dimensions<typename Derived1::result_type>::dims[1] : 1>
    >
>
// auto
matmul(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b) {

    using lhs_type = typename Derived0::result_type;
    using rhs_type = typename Derived1::result_type;

    const Derived0 &a_src = a.self();
    const Derived1 &b_src = b.self();

    FASTOR_IF_CONSTEXPR(requires_evaluation_v<Derived0> || requires_evaluation_v<Derived1>) {
        const lhs_type tmp_a(a_src);
        const lhs_type tmp_b(b_src);
        matmul(a,b);
    }

    constexpr FASTOR_INDEX lhs_rank = lhs_type::Dimension_t::value;
    constexpr FASTOR_INDEX rhs_rank = rhs_type::Dimension_t::value;
    constexpr FASTOR_INDEX M = lhs_rank == 2 ? get_tensor_dimensions<lhs_type>::dims[0] : 1;
    constexpr FASTOR_INDEX K = lhs_rank == 2 ? get_tensor_dimensions<lhs_type>::dims[1] : 1;
    constexpr FASTOR_INDEX N = rhs_rank == 2 ? get_tensor_dimensions<rhs_type>::dims[1] : 1;

    using T = typename scalar_type_finder<Derived0>::type;

    // We cannot choose the best simd type because the simd types of an expression can't be mixed
    using V = SIMDVector<T,DEFAULT_ABI>;
    // using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;
    constexpr size_t SIZE_ = V::Size;
    int ROUND = ROUND_DOWN(N,(int)SIZE_);

    using result_type = conditional_t_<lhs_rank == 1,   // vector-matrix
                                            Tensor<T,N> ,
                                            conditional_t_<rhs_rank == 1, // matrix-vector
                                                Tensor<T,M>,
                                                Tensor<T,M,N>   // matrix-matrix
                                            >
                                        >;
    result_type out;
    T *out_data = out.data();

    for (size_t j=0; j<M; ++j) {
        size_t k=0;
        for (; k<(size_t)ROUND; k+=SIZE_) {
            V out_row, vec_a;
            for (size_t i=0; i<K; ++i) {
                // V brow; brow.load(&b[i*N+k],false);
                V brow = b_src.template eval<T>(i*N+k);
                vec_a.set(a_src.template eval_s<T>(j*K+i));
                out_row = fmadd(vec_a,brow,out_row);
            }
            out_row.store(out_data+k+N*j,false);
        }

        for (; k<N; k++) {
            T out_row = 0.;
            for (size_t i=0; i<K; ++i) {
                // out_row += a[j*K+i]*b[i*N+k];
                out_row += a_src.template eval_s<T>(j*K+i)*b_src.template eval_s<T>(i*N+k);
            }
            out_data[N*j+k] = out_row;
        }
    }

    return out;
}




template<typename T>
void run() {

    // the test assumes that matmul is tested already elsewhere
    {
        Tensor<T,3,4> a;
        Tensor<T,4,5> b;
        Tensor<T,3,5> ab(2);
        Tensor<T,3> c(0);
        Tensor<T,5> d(1);
        a.iota(1);
        b.iota(2);

        Tensor<T,3,5> e0 = matmul(a,b);
        Tensor<T,3,5> e1 = a%b;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        // test matmul expression assigns
        e0 += matmul(a,b);
        e1 += a%b;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 -= matmul(a,b);
        e1 -= a%b;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b);
        e1 *= a%b;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 /= matmul(a,b);
        e1 /= a%b;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);


        // test binary_add expression assigns when matmul is present
        e0 = matmul(a,b) + 2;
        e1 = a%b + 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 += matmul(a,b) + 2;
        e1 += a%b + 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 -= matmul(a,b) + 2;
        e1 -= a%b + 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) + 2;
        e1 *= a%b + 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 /= matmul(a,b) + 2;
        e1 /= a%b + 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);


        // test binary_sub expression assigns when matmul is present
        e0 = matmul(a,b) - 2;
        e1 = a%b - 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 += matmul(a,b) - 2;
        e1 += a%b - 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 -= matmul(a,b) - 2;
        e1 -= a%b - 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) - 2;
        e1 *= a%b - 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 /= matmul(a,b) - 2;
        e1 /= a%b - 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);


        // test binary_mul expression assigns when matmul is present
        e0 = matmul(a,b) * 2;
        e1 = a%b * 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 += matmul(a,b) * 2;
        e1 += a%b * 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 -= matmul(a,b) * 2;
        e1 -= a%b * 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) * 2;
        e1 *= a%b * 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 /= matmul(a,b) * 2;
        e1 /= a%b * 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);


        // test binary_div expression assigns when matmul is present
        e0 = matmul(a,b) / 2;
        e1 = a%b / 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 += matmul(a,b) / 2;
        e1 += a%b / 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 -= matmul(a,b) / 2;
        e1 -= a%b / 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) / 2;
        e1 *= a%b / 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 /= matmul(a,b) / 2;
        e1 /= a%b / 2;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);


       // test binary_add expression assigns when matmul and aliasing is present
        e0 = matmul(a,b) + e0;
        e1 = a%b + e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 += matmul(a,b) + e0;
        e1 += a%b + e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 -= e0 * matmul(a,b) + e0;
        e1 -= e1 * (a%b) + e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= e0 / matmul(a,b) + e0;
        e1 *= e1 / (a%b) + e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 /= e0 * matmul(a,b) + e0;
        e1 /= e1 * (a%b) + e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);


       // test binary_sub expression assigns when matmul and aliasing is present
        e0 = e0 - matmul(a,b) - e0;
        e1 = e1 - a%b - e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 += e0 - matmul(a,b) - e0;
        e1 += e1 - a%b - e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 -= e0 * matmul(a,b) - e0;
        e1 -= e1 * (a%b) - e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= e0 / matmul(a,b) - e0;
        e1 *= e1 / (a%b) - e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 /= matmul(a,b) - e0;
        e1 /= (a%b) - e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);


        // test binary_mul expression assigns when matmul and aliasing is present
        e0 = matmul(a,b) * e0;
        e1 = (a%b) * e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 += matmul(a,b) * e0;
        e1 += (a%b) * e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) * e0;
        e1 *= (a%b) * e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) * e0;
        e1 *= (a%b) * e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) * e0;
        e1 *= (a%b) * e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);


        // test binary_div expression assigns when matmul and aliasing is present
        e0 = matmul(a,b) / e0;
        e1 = (a%b) / e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 += matmul(a,b) / e0;
        e1 += (a%b) / e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) / e0;
        e1 *= (a%b) / e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) / e0;
        e1 *= (a%b) / e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) / e0;
        e1 *= (a%b) / e1;
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < Tol);


        // test unary ops
        e0 = sqrt(matmul(a,b));
        e1 = sqrt(a%b);
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < BigTol);

        e0 += sqrt(matmul(a,b));
        e1 += sqrt(a%b);
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < BigTol);

        e0 -= sqrt(matmul(a,b));
        e1 -= sqrt(a%b);
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < BigTol);

        e0 *= sqrt(matmul(a,b));
        e1 *= sqrt(a%b);
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < BigTol);

        e0 /= sqrt(matmul(a,b));
        e1 /= sqrt(a%b);
        FASTOR_EXIT_ASSERT(std::abs(e0.sum() - e1.sum()) < BigTol);


        // double matmul
        Tensor<T,3> e2 = matmul(matmul(a,b),d);
        Tensor<T,3> e3 = a % b % d;
        FASTOR_EXIT_ASSERT(std::abs(e3.sum() - e2.sum()) < BigTol);

        e2 += matmul(matmul(a,b),d);
        e3 += a % b % d;
        FASTOR_EXIT_ASSERT(std::abs(e3.sum() - e2.sum()) < BigTol);

        e2 -= matmul(matmul(a,b),d);
        e3 -= a % b % d;
        FASTOR_EXIT_ASSERT(std::abs(e3.sum() - e2.sum()) < BigTol);

        e2 *= matmul(matmul(a,b),d);
        e3 *= a % b % d;
        FASTOR_EXIT_ASSERT(std::abs(e3.sum() - e2.sum()) < BigTol);

        e2 /= matmul(matmul(a,b),d);
        e3 /= a % b % d;
        FASTOR_EXIT_ASSERT(std::abs(e3.sum() - e2.sum()) < BigTol);

        // double matmul expr
        e2 = 1 + matmul(matmul(a,b),d);
        e3 = 1 + a % b % d;
        FASTOR_EXIT_ASSERT(std::abs(e3.sum() - e2.sum()) < BigTol);

        e2 += matmul(matmul(a,b),d) - 3;
        e3 += a % b % d - 3;
        FASTOR_EXIT_ASSERT(std::abs(e3.sum() - e2.sum()) < BigTol);

        e2 -= matmul(matmul(a,b),d) * e2;
        e3 -= (a % b % d) * e3;
        FASTOR_EXIT_ASSERT(std::abs(e3.sum() - e2.sum()) < BigTol);

        e2 *= matmul(matmul(a,b),d) / e2;
        e3 *= (a % b % d) / e3;
        FASTOR_EXIT_ASSERT(std::abs(e3.sum() - e2.sum()) < BigTol);

        e2 /= 2*matmul(matmul(a,b),d) + 2*e2 + e2 + 1;
        e3 /= 2*(a % b % d) + 2*e3 + e3 + 1;
        FASTOR_EXIT_ASSERT(std::abs(e3.sum() - e2.sum()) < BigTol);
    }


    print(FGRN(BOLD("All tests passed successfully")));

}

int main() {

    print(FBLU(BOLD("Testing matmul expression with double precision")));
    run<double>();

    return 0;
}