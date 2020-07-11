#ifndef UNARY_INV_OP_H
#define UNARY_INV_OP_H

#include "Fastor/meta/meta.h"
#include "Fastor/backend/inverse.h"
#include "Fastor/backend/lut_inverse.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/Aliasing.h"
#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/Ranges.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/expressions/expression_traits.h"
#include "Fastor/expressions/linalg_ops/linalg_computation_types.h"
#include "Fastor/expressions/linalg_ops/unary_piv_op.h"


namespace Fastor {

template<typename Expr, size_t DIM0>
struct UnaryInvOp: public AbstractTensor<UnaryInvOp<Expr, DIM0>,DIM0> {
    using expr_type = expression_t<Expr>;
    using result_type = typename Expr::result_type;
    static constexpr FASTOR_INDEX M = get_tensor_dimension_v<0,result_type>;
    static constexpr FASTOR_INDEX N = get_tensor_dimension_v<1,result_type>;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    using scalar_type = typename scalar_type_finder<UnaryInvOp<Expr, DIM0>>::type;
    using simd_vector_type = typename Expr::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;

    FASTOR_INLINE UnaryInvOp(expr_type inexpr) : _expr(inexpr) {
        static_assert(M==N, "MATRIX MUST BE SQUARE");
    }

    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return M*N;}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX ) const {return M;}

    constexpr FASTOR_INLINE expr_type expr() const {return _expr;}

private:
    expr_type _expr;
};

template<typename Expr, size_t DIM0>
FASTOR_INLINE UnaryInvOp<Expr, DIM0>
inv(const AbstractTensor<Expr,DIM0> &src) {
  return UnaryInvOp<Expr, DIM0>(src.self());
}


// Upper triangular block matrix inversion
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
namespace internal {

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,0> && is_less_equal_v_<M,4>,bool> = false>
FASTOR_INLINE void ut_inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {
    _inverse<T,M>(in.data(),out.data());
}

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,4> && is_less_equal_v_<M,8>,bool> = false>
FASTOR_INLINE void ut_inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {

    constexpr size_t N = 4UL; // start size
    Tensor<T,N  ,N  > a = in(fseq<0,N>(),fseq<0,N>());
    Tensor<T,N  ,M-N> b = in(fseq<0,N>(),fseq<N,M>());
    // Tensor<T,M-N,  N> c = in(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> d = in(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> inv_a;
    ut_inverse_dispatcher(a, inv_a);

    Tensor<T,M-N,M-N> inv_d;
    ut_inverse_dispatcher(d, inv_d);

    Tensor<T,N,M-N> b_invd = matmul(b, inv_d);

    out(fseq<0,N>(),fseq<0,N>()) = inv_a;
    out(fseq<0,N>(),fseq<N,M>()) = -matmul(inv_a, b_invd);
    out(fseq<N,M>(),fseq<0,N>()) = 0;
    out(fseq<N,M>(),fseq<N,M>()) = inv_d;
}

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,8> && is_less_equal_v_<M,16>,bool> = false>
FASTOR_INLINE void ut_inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {

    constexpr size_t N = 8UL; // start size
    Tensor<T,N  ,N  > a = in(fseq<0,N>(),fseq<0,N>());
    Tensor<T,N  ,M-N> b = in(fseq<0,N>(),fseq<N,M>());
    // Tensor<T,M-N,  N> c = in(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> d = in(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> inv_a;
    ut_inverse_dispatcher(a, inv_a);

    Tensor<T,M-N,M-N> inv_d;
    ut_inverse_dispatcher(d, inv_d);

    Tensor<T,N,M-N> b_invd = matmul(b, inv_d);

    out(fseq<0,N>(),fseq<0,N>()) = inv_a;
    out(fseq<0,N>(),fseq<N,M>()) = -matmul(inv_a, b_invd);
    out(fseq<N,M>(),fseq<0,N>()) = 0;
    out(fseq<N,M>(),fseq<N,M>()) = inv_d;
}

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,16> && is_less_equal_v_<M,32>,bool> = false>
FASTOR_INLINE void ut_inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {

    // constexpr size_t N = 16UL; // start size
    constexpr size_t N = (M / 8UL * 8UL) / 2UL; // start size
    Tensor<T,N  ,N  > a = in(fseq<0,N>(),fseq<0,N>());
    Tensor<T,N  ,M-N> b = in(fseq<0,N>(),fseq<N,M>());
    // Tensor<T,M-N,  N> c = in(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> d = in(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> inv_a;
    ut_inverse_dispatcher(a, inv_a);

    Tensor<T,M-N,M-N> inv_d;
    ut_inverse_dispatcher(d, inv_d);

    Tensor<T,N,M-N> b_invd = matmul(b, inv_d);

    out(fseq<0,N>(),fseq<0,N>()) = inv_a;
    out(fseq<0,N>(),fseq<N,M>()) = -matmul(inv_a, b_invd);
    out(fseq<N,M>(),fseq<0,N>()) = 0;
    out(fseq<N,M>(),fseq<N,M>()) = inv_d;
}

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,32> && is_less_equal_v_<M,64>,bool> = false>
FASTOR_INLINE void ut_inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {

    // constexpr size_t N = 32UL; // start size
    constexpr size_t N = (M / 16UL * 16UL) / 2UL; // start size
    Tensor<T,N  ,N  > a = in(fseq<0,N>(),fseq<0,N>());
    Tensor<T,N  ,M-N> b = in(fseq<0,N>(),fseq<N,M>());
    // Tensor<T,M-N,  N> c = in(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> d = in(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> inv_a;
    ut_inverse_dispatcher(a, inv_a);

    Tensor<T,M-N,M-N> inv_d;
    ut_inverse_dispatcher(d, inv_d);

    Tensor<T,N,M-N> b_invd = tmatmul<UpLoType::General,UpLoType::Upper>(b, inv_d);

    out(fseq<0,N>(),fseq<0,N>()) = inv_a;
    out(fseq<0,N>(),fseq<N,M>()) = -tmatmul<UpLoType::Upper,UpLoType::General>(inv_a, b_invd);
    out(fseq<N,M>(),fseq<0,N>()) = 0;
    out(fseq<N,M>(),fseq<N,M>()) = inv_d;
}

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,64> && is_less_equal_v_<M,128>,bool> = false>
FASTOR_INLINE void ut_inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {

    // constexpr size_t N = 64UL; // start size
    constexpr size_t N = (M / 32UL * 32UL) / 2UL; // start size
    Tensor<T,N  ,N  > a = in(fseq<0,N>(),fseq<0,N>());
    Tensor<T,N  ,M-N> b = in(fseq<0,N>(),fseq<N,M>());
    // Tensor<T,M-N,  N> c = in(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> d = in(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> inv_a;
    ut_inverse_dispatcher(a, inv_a);

    Tensor<T,M-N,M-N> inv_d;
    ut_inverse_dispatcher(d, inv_d);

    Tensor<T,N,M-N> b_invd = tmatmul<UpLoType::General,UpLoType::Upper>(b, inv_d);

    out(fseq<0,N>(),fseq<0,N>()) = inv_a;
    out(fseq<0,N>(),fseq<N,M>()) = -tmatmul<UpLoType::Upper,UpLoType::General>(inv_a, b_invd);
    out(fseq<N,M>(),fseq<0,N>()) = 0;
    out(fseq<N,M>(),fseq<N,M>()) = inv_d;
}

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,128> && is_less_equal_v_<M,256>,bool> = false>
FASTOR_INLINE void ut_inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {

    // constexpr size_t N = 128UL; // start size
    constexpr size_t N = (M / 64UL * 64UL) / 2UL; // start size
    Tensor<T,N  ,N  > a = in(fseq<0,N>(),fseq<0,N>());
    Tensor<T,N  ,M-N> b = in(fseq<0,N>(),fseq<N,M>());
    // Tensor<T,M-N,  N> c = in(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> d = in(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> inv_a;
    ut_inverse_dispatcher(a, inv_a);

    Tensor<T,M-N,M-N> inv_d;
    ut_inverse_dispatcher(d, inv_d);

    Tensor<T,N,M-N> b_invd = tmatmul<UpLoType::General,UpLoType::Upper>(b, inv_d);

    out(fseq<0,N>(),fseq<0,N>()) = inv_a;
    out(fseq<0,N>(),fseq<N,M>()) = -tmatmul<UpLoType::Upper,UpLoType::General>(inv_a, b_invd);
    out(fseq<N,M>(),fseq<0,N>()) = 0;
    out(fseq<N,M>(),fseq<N,M>()) = inv_d;
}
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
} // internal




// Lower uni-triangular block matrix inversion
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
namespace internal {

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,0> && is_less_equal_v_<M,4>,bool> = false>
FASTOR_INLINE void lut_inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {
    // This restricts the recursive block to lower uni-triangualar matrices
    _lowunitri_inverse<T,M>(in.data(),out.data());
}

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,4> && is_less_equal_v_<M,8>,bool> = false>
FASTOR_INLINE void lut_inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {

    constexpr size_t N = 4UL; // start size
    Tensor<T,N  ,N  > a = in(fseq<0,N>(),fseq<0,N>());
    // Tensor<T,N  ,M-N> b = in(fseq<0,N>(),fseq<N,M>());
    Tensor<T,M-N,  N> c = in(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> d = in(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> inv_a;
    lut_inverse_dispatcher(a, inv_a);

    Tensor<T,M-N,N> c_inva = matmul(c, inv_a);

    Tensor<T,M-N,M-N> inv_d;
    lut_inverse_dispatcher(d, inv_d);

    out(fseq<0,N>(),fseq<0,N>()) = inv_a;
    out(fseq<0,N>(),fseq<N,M>()) = 0;
    out(fseq<N,M>(),fseq<0,N>()) = -matmul(inv_d, c_inva);
    out(fseq<N,M>(),fseq<N,M>()) = inv_d;
}

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,8> && is_less_equal_v_<M,16>,bool> = false>
FASTOR_INLINE void lut_inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {

    constexpr size_t N = 8UL; // start size
    Tensor<T,N  ,N  > a = in(fseq<0,N>(),fseq<0,N>());
    // Tensor<T,N  ,M-N> b = in(fseq<0,N>(),fseq<N,M>());
    Tensor<T,M-N,  N> c = in(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> d = in(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> inv_a;
    lut_inverse_dispatcher(a, inv_a);

    Tensor<T,M-N,N> c_inva = matmul(c, inv_a);

    Tensor<T,M-N,M-N> inv_d;
    lut_inverse_dispatcher(d, inv_d);

    out(fseq<0,N>(),fseq<0,N>()) = inv_a;
    out(fseq<0,N>(),fseq<N,M>()) = 0;
    out(fseq<N,M>(),fseq<0,N>()) = -matmul(inv_d, c_inva);
    out(fseq<N,M>(),fseq<N,M>()) = inv_d;
}

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,16> && is_less_equal_v_<M,32>,bool> = false>
FASTOR_INLINE void lut_inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {

    // constexpr size_t N = 16UL; // start size
    constexpr size_t N = (M / 8UL * 8UL) / 2UL; // start size
    Tensor<T,N  ,N  > a = in(fseq<0,N>(),fseq<0,N>());
    // Tensor<T,N  ,M-N> b = in(fseq<0,N>(),fseq<N,M>());
    Tensor<T,M-N,  N> c = in(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> d = in(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> inv_a;
    lut_inverse_dispatcher(a, inv_a);

    Tensor<T,M-N,N> c_inva = matmul(c, inv_a);

    Tensor<T,M-N,M-N> inv_d;
    lut_inverse_dispatcher(d, inv_d);

    out(fseq<0,N>(),fseq<0,N>()) = inv_a;
    out(fseq<0,N>(),fseq<N,M>()) = 0;
    out(fseq<N,M>(),fseq<0,N>()) = -matmul(inv_d, c_inva);
    out(fseq<N,M>(),fseq<N,M>()) = inv_d;
}

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,32> && is_less_equal_v_<M,64>,bool> = false>
FASTOR_INLINE void lut_inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {

    // constexpr size_t N = 32UL; // start size
    constexpr size_t N = (M / 16UL * 16UL) / 2UL; // start size
    Tensor<T,N  ,N  > a = in(fseq<0,N>(),fseq<0,N>());
    // Tensor<T,N  ,M-N> b = in(fseq<0,N>(),fseq<N,M>());
    Tensor<T,M-N,  N> c = in(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> d = in(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> inv_a;
    lut_inverse_dispatcher(a, inv_a);

    Tensor<T,M-N,N> c_inva = tmatmul<UpLoType::General,UpLoType::Lower>(c, inv_a);

    Tensor<T,M-N,M-N> inv_d;
    lut_inverse_dispatcher(d, inv_d);

    out(fseq<0,N>(),fseq<0,N>()) = inv_a;
    out(fseq<0,N>(),fseq<N,M>()) = 0;
    out(fseq<N,M>(),fseq<0,N>()) = -tmatmul<UpLoType::Lower,UpLoType::General>(inv_d, c_inva);
    out(fseq<N,M>(),fseq<N,M>()) = inv_d;
}

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,64> && is_less_equal_v_<M,128>,bool> = false>
FASTOR_INLINE void lut_inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {

    // constexpr size_t N = 64UL; // start size
    constexpr size_t N = (M / 32UL * 32UL) / 2UL; // start size
    Tensor<T,N  ,N  > a = in(fseq<0,N>(),fseq<0,N>());
    // Tensor<T,N  ,M-N> b = in(fseq<0,N>(),fseq<N,M>());
    Tensor<T,M-N,  N> c = in(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> d = in(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> inv_a;
    lut_inverse_dispatcher(a, inv_a);

    Tensor<T,M-N,N> c_inva = tmatmul<UpLoType::General,UpLoType::Lower>(c, inv_a);

    Tensor<T,M-N,M-N> inv_d;
    lut_inverse_dispatcher(d, inv_d);

    out(fseq<0,N>(),fseq<0,N>()) = inv_a;
    out(fseq<0,N>(),fseq<N,M>()) = 0;
    out(fseq<N,M>(),fseq<0,N>()) = -tmatmul<UpLoType::Lower,UpLoType::General>(inv_d, c_inva);
    out(fseq<N,M>(),fseq<N,M>()) = inv_d;
}

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,128> && is_less_equal_v_<M,256>,bool> = false>
FASTOR_INLINE void lut_inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {

    // constexpr size_t N = 128UL; // start size
    constexpr size_t N = (M / 64UL * 64UL) / 2UL; // start size
    Tensor<T,N  ,N  > a = in(fseq<0,N>(),fseq<0,N>());
    // Tensor<T,N  ,M-N> b = in(fseq<0,N>(),fseq<N,M>());
    Tensor<T,M-N,  N> c = in(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> d = in(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> inv_a;
    lut_inverse_dispatcher(a, inv_a);

    Tensor<T,M-N,N> c_inva = tmatmul<UpLoType::General,UpLoType::Lower>(c, inv_a);

    Tensor<T,M-N,M-N> inv_d;
    lut_inverse_dispatcher(d, inv_d);

    out(fseq<0,N>(),fseq<0,N>()) = inv_a;
    out(fseq<0,N>(),fseq<N,M>()) = 0;
    out(fseq<N,M>(),fseq<0,N>()) = -tmatmul<UpLoType::Lower,UpLoType::General>(inv_d, c_inva);
    out(fseq<N,M>(),fseq<N,M>()) = inv_d;
}
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
} // internal




// General recursive block matrix inversion
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
namespace internal {

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,0> && is_less_equal_v_<M,4>,bool> = false>
FASTOR_INLINE void inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {
    _inverse<T,M>(in.data(),out.data());
}

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,4> && is_less_equal_v_<M,8>,bool> = false>
FASTOR_INLINE void inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {

    constexpr size_t N = 4UL; // start size
    Tensor<T,N  ,N  > a = in(fseq<0,N>(),fseq<0,N>());
    Tensor<T,N  ,M-N> b = in(fseq<0,N>(),fseq<N,M>());
    Tensor<T,M-N,  N> c = in(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> d = in(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> inv_a = inverse(a);

    Tensor<T,M-N,N> c_inva = matmul(c, inv_a);

    Tensor<T,M-N,M-N> block_bb = inverse(static_cast<Tensor<T,M-N,M-N>>(d - matmul(c_inva, b)));

    Tensor<T,N,M-N> inva_b = matmul(inv_a, b);
    Tensor<T,M-N,N> bb_c_inva = matmul(block_bb, c_inva);

    Tensor<T,N  ,N  > block_aa = inv_a + matmul(inva_b, bb_c_inva);
    Tensor<T,N  ,M-N> block_ab = -matmul(inva_b, block_bb);
    Tensor<T,M-N,N  > block_ba = -bb_c_inva;

    out(fseq<0,N>(),fseq<0,N>()) = block_aa;
    out(fseq<0,N>(),fseq<N,M>()) = block_ab;
    out(fseq<N,M>(),fseq<0,N>()) = block_ba;
    out(fseq<N,M>(),fseq<N,M>()) = block_bb;
}

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,8> && is_less_equal_v_<M,16>,bool> = false>
FASTOR_INLINE void inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {

    constexpr size_t N = 8UL; // start size
    Tensor<T,N  ,N  > a = in(fseq<0,N>(),fseq<0,N>());
    Tensor<T,N  ,M-N> b = in(fseq<0,N>(),fseq<N,M>());
    Tensor<T,M-N,  N> c = in(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> d = in(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> inv_a;
    inverse_dispatcher(a, inv_a);

    Tensor<T,M-N,N> c_inva = matmul(c, inv_a);

    Tensor<T,M-N,M-N> block_bb;
    inverse_dispatcher(static_cast<Tensor<T,M-N,M-N>>(d - matmul(c_inva, b)),block_bb);

    Tensor<T,N,M-N> inva_b = matmul(inv_a, b);
    Tensor<T,M-N,N> bb_c_inva = matmul(block_bb, c_inva);

    Tensor<T,N  ,N  > block_aa = inv_a + matmul(inva_b, bb_c_inva);
    Tensor<T,N  ,M-N> block_ab = -matmul(inva_b, block_bb);
    Tensor<T,M-N,N  > block_ba = -bb_c_inva;

    out(fseq<0,N>(),fseq<0,N>()) = block_aa;
    out(fseq<0,N>(),fseq<N,M>()) = block_ab;
    out(fseq<N,M>(),fseq<0,N>()) = block_ba;
    out(fseq<N,M>(),fseq<N,M>()) = block_bb;
}

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,16> && is_less_equal_v_<M,32>,bool> = false>
FASTOR_INLINE void inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {

    // constexpr size_t N = 16UL; // start size
    constexpr size_t N = (M / 8UL * 8UL) / 2UL; // start size
    Tensor<T,N  ,N  > a = in(fseq<0,N>(),fseq<0,N>());
    Tensor<T,N  ,M-N> b = in(fseq<0,N>(),fseq<N,M>());
    Tensor<T,M-N,  N> c = in(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> d = in(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> inv_a;
    inverse_dispatcher(a, inv_a);

    Tensor<T,M-N,N> c_inva = matmul(c, inv_a);

    Tensor<T,M-N,M-N> block_bb;
    inverse_dispatcher(static_cast<Tensor<T,M-N,M-N>>(d - matmul(c_inva, b)),block_bb);

    Tensor<T,N,M-N> inva_b = matmul(inv_a, b);
    Tensor<T,M-N,N> bb_c_inva = matmul(block_bb, c_inva);

    Tensor<T,N  ,N  > block_aa = inv_a + matmul(inva_b, bb_c_inva);
    Tensor<T,N  ,M-N> block_ab = -matmul(inva_b, block_bb);
    Tensor<T,M-N,N  > block_ba = -bb_c_inva;

    out(fseq<0,N>(),fseq<0,N>()) = block_aa;
    out(fseq<0,N>(),fseq<N,M>()) = block_ab;
    out(fseq<N,M>(),fseq<0,N>()) = block_ba;
    out(fseq<N,M>(),fseq<N,M>()) = block_bb;
}

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,32> && is_less_equal_v_<M,64>,bool> = false>
FASTOR_INLINE void inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {

    // constexpr size_t N = 32UL; // start size
    constexpr size_t N = (M / 16UL * 16UL) / 2UL; // start size
    Tensor<T,N  ,N  > a = in(fseq<0,N>(),fseq<0,N>());
    Tensor<T,N  ,M-N> b = in(fseq<0,N>(),fseq<N,M>());
    Tensor<T,M-N,  N> c = in(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> d = in(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> inv_a;
    inverse_dispatcher(a, inv_a);

    Tensor<T,M-N,N> c_inva = matmul(c, inv_a);

    Tensor<T,M-N,M-N> block_bb;
    inverse_dispatcher(static_cast<Tensor<T,M-N,M-N>>(d - matmul(c_inva, b)),block_bb);

    Tensor<T,N,M-N> inva_b = matmul(inv_a, b);
    Tensor<T,M-N,N> bb_c_inva = matmul(block_bb, c_inva);

    Tensor<T,N  ,N  > block_aa = inv_a + matmul(inva_b, bb_c_inva);
    Tensor<T,N  ,M-N> block_ab = -matmul(inva_b, block_bb);
    Tensor<T,M-N,N  > block_ba = -bb_c_inva;

    out(fseq<0,N>(),fseq<0,N>()) = block_aa;
    out(fseq<0,N>(),fseq<N,M>()) = block_ab;
    out(fseq<N,M>(),fseq<0,N>()) = block_ba;
    out(fseq<N,M>(),fseq<N,M>()) = block_bb;
}

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,64> && is_less_equal_v_<M,128>,bool> = false>
FASTOR_INLINE void inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {

    // constexpr size_t N = 64UL; // start size
    constexpr size_t N = (M / 32UL * 32UL) / 2UL; // start size
    Tensor<T,N  ,N  > a = in(fseq<0,N>(),fseq<0,N>());
    Tensor<T,N  ,M-N> b = in(fseq<0,N>(),fseq<N,M>());
    Tensor<T,M-N,  N> c = in(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> d = in(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> inv_a;
    inverse_dispatcher(a, inv_a);

    Tensor<T,M-N,N> c_inva = matmul(c, inv_a);

    Tensor<T,M-N,M-N> block_bb;
    inverse_dispatcher(static_cast<Tensor<T,M-N,M-N>>(d - matmul(c_inva, b)),block_bb);

    Tensor<T,N,M-N> inva_b = matmul(inv_a, b);
    Tensor<T,M-N,N> bb_c_inva = matmul(block_bb, c_inva);

    Tensor<T,N  ,N  > block_aa = inv_a + matmul(inva_b, bb_c_inva);
    Tensor<T,N  ,M-N> block_ab = -matmul(inva_b, block_bb);
    Tensor<T,M-N,N  > block_ba = -bb_c_inva;

    out(fseq<0,N>(),fseq<0,N>()) = block_aa;
    out(fseq<0,N>(),fseq<N,M>()) = block_ab;
    out(fseq<N,M>(),fseq<0,N>()) = block_ba;
    out(fseq<N,M>(),fseq<N,M>()) = block_bb;
}

template<typename T, size_t M, enable_if_t_<is_greater_v_<M,128> && is_less_equal_v_<M,256>,bool> = false>
FASTOR_INLINE void inverse_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {

    // constexpr size_t N = 128UL; // start size
    constexpr size_t N = (M / 64UL * 64UL) / 2UL; // start size
    Tensor<T,N  ,N  > a = in(fseq<0,N>(),fseq<0,N>());
    Tensor<T,N  ,M-N> b = in(fseq<0,N>(),fseq<N,M>());
    Tensor<T,M-N,  N> c = in(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> d = in(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> inv_a;
    inverse_dispatcher(a, inv_a);

    Tensor<T,M-N,N> c_inva = matmul(c, inv_a);

    Tensor<T,M-N,M-N> block_bb;
    inverse_dispatcher(static_cast<Tensor<T,M-N,M-N>>(d - matmul(c_inva, b)),block_bb);

    Tensor<T,N,M-N> inva_b = matmul(inv_a, b);
    Tensor<T,M-N,N> bb_c_inva = matmul(block_bb, c_inva);

    Tensor<T,N  ,N  > block_aa = inv_a + matmul(inva_b, bb_c_inva);
    Tensor<T,N  ,M-N> block_ab = -matmul(inva_b, block_bb);
    Tensor<T,M-N,N  > block_ba = -bb_c_inva;

    out(fseq<0,N>(),fseq<0,N>()) = block_aa;
    out(fseq<0,N>(),fseq<N,M>()) = block_ab;
    out(fseq<N,M>(),fseq<0,N>()) = block_ba;
    out(fseq<N,M>(),fseq<N,M>()) = block_bb;
}
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
} // internal



// Inversion using LU decomposition is in the LU module [unary_lu_op]
// For tensors
// SimpleInv - no pivot
template<InvCompType InvType = InvCompType::SimpleInv,
    typename T, size_t M, enable_if_t_<is_less_equal_v_<M,4UL> && InvType == InvCompType::SimpleInv,bool> = false>
FASTOR_INLINE Tensor<T,M,M> inverse(const Tensor<T,M,M> &a) {
    Tensor<T,M,M> out;
    _inverse<T,M>(a.data(),out.data());
    return out;
}
template<InvCompType InvType = InvCompType::SimpleInv,
    typename T, size_t M, enable_if_t_<is_greater_v_<M,4UL> && InvType == InvCompType::SimpleInv,bool> = false>
FASTOR_INLINE Tensor<T,M,M> inverse(const Tensor<T,M,M> &in) {
    Tensor<T,M,M> out;
    internal::inverse_dispatcher(in,out);
    return out;
}
// SimpleInv pivot
template<InvCompType InvType = InvCompType::SimpleInv,
    typename T, size_t M, enable_if_t_<InvType == InvCompType::SimpleInvPiv,bool> = false>
FASTOR_INLINE Tensor<T,M,M> inverse(const Tensor<T,M,M> &in) {
    Tensor<T,M,M> out;
    // We need to post multiply - swap columns using row permutation vector
    Tensor<size_t,M> P;
    pivot_inplace(in,P);
    auto A(apply_pivot(in,P));
    internal::inverse_dispatcher(A,out);
    return reconstruct_colwise(out,P);

    // // matrix version
    // Tensor<T,M,M> P;
    // pivot_inplace(in,P);
    // auto A(apply_pivot(in,P));
    // internal::inverse_dispatcher(A,out);
    // return matmul(out,P);
}

// For high order tensors
template<InvCompType InvType = InvCompType::SimpleInv,
    typename T, size_t ... Rest, enable_if_t_<sizeof...(Rest)>=3 && InvType == InvCompType::SimpleInv,bool> = false >
FASTOR_INLINE Tensor<T,Rest...>
inverse(const Tensor<T,Rest...> &a) {

    constexpr size_t remaining_product = last_matrix_extracter<Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Rest)-2>::type>::remaining_product;

    constexpr size_t I = get_value<sizeof...(Rest)-1,Rest...>::value;
    constexpr size_t J = get_value<sizeof...(Rest),Rest...>::value;
    static_assert(I==J,"THE LAST TWO DIMENSIONS OF TENSOR MUST BE THE SAME");

    Tensor<T,Rest...> out;
    T *a_data = a.data();
    T *out_data = out.data();

    for (size_t i=0; i<remaining_product; ++i) {
        T det = _det<T,J,J>(static_cast<const T *>(a_data+i*J*J));
        _inverse<T,J>(a_data+i*J*J,out_data+i*J*J);
    }

    return out;
}

// Inverse for generic expressions is provided here
template<InvCompType InvType = InvCompType::SimpleInv,
    typename Derived, size_t DIM>
FASTOR_INLINE
typename Derived::result_type
inverse(const AbstractTensor<Derived,DIM> &src) {
    // If we are here Derived is already an expression
    using result_type = typename Derived::result_type;
    const result_type tmp(src.self());
    return inverse<InvType>(tmp);
}


// Inverse of lower uni-triangular matrices
template<InvCompType InvType = InvCompType::SimpleInv, typename ULType = UpLoType::General,
    typename T, size_t M, enable_if_t_<InvType == InvCompType::SimpleInv && is_same_v_<ULType,UpLoType::UniLower>,bool> = false>
FASTOR_INLINE Tensor<T,M,M> tinverse(const Tensor<T,M,M> &in) {
    Tensor<T,M,M> out;
    internal::lut_inverse_dispatcher(in,out);
    return out;
}

// Inverse of lower uni-triangular matrices
template<InvCompType InvType = InvCompType::SimpleInv, typename ULType = UpLoType::General,
    typename T, size_t M, enable_if_t_<InvType == InvCompType::SimpleInv && is_same_v_<ULType,UpLoType::Upper>,bool> = false>
FASTOR_INLINE Tensor<T,M,M> tinverse(const Tensor<T,M,M> &in) {
    Tensor<T,M,M> out;
    internal::ut_inverse_dispatcher(in,out);
    return out;
}
// Inverse for generic expressions of lower uni-triangular matrices is provided here
template<InvCompType InvType = InvCompType::SimpleInv, typename ULType = UpLoType::General,
    typename Derived, size_t DIM>
FASTOR_INLINE
typename Derived::result_type
tinverse(const AbstractTensor<Derived,DIM> &src) {
    // If we are here Derived is already an expression
    using result_type = typename Derived::result_type;
    const result_type tmp(src.self());
    return tinverse<InvType,ULType>(tmp);
}






// assignments
template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const UnaryInvOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    const result_type& tmp = evaluate(src.expr().self());
    internal::inverse_dispatcher(tmp,dst.self());
}

template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const UnaryInvOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    // no copies if expr is a tensor
    const result_type& tmp = evaluate(src.expr().self());
    // one copy for the inverse
    result_type tmp_inv;
    internal::inverse_dispatcher(tmp,tmp_inv);
    trivial_assign_add(dst.self(),tmp_inv);
}

template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const UnaryInvOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    // no copies if expr is a tensor
    const result_type& tmp = evaluate(src.expr().self());
    // one copy for the inverse
    result_type tmp_inv;
    internal::inverse_dispatcher(tmp,tmp_inv);
    trivial_assign_sub(dst.self(),tmp_inv);
}

template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign_mul(AbstractTensor<Derived,DIM> &dst, const UnaryInvOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    // no copies if expr is a tensor
    const result_type& tmp = evaluate(src.expr().self());
    // one copy for the inverse
    result_type tmp_inv;
    internal::inverse_dispatcher(tmp,tmp_inv);
    trivial_assign_mul(dst.self(),tmp_inv);
}

template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign_div(AbstractTensor<Derived,DIM> &dst, const UnaryInvOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    // no copies if expr is a tensor
    const result_type& tmp = evaluate(src.expr().self());
    // one copy for the inverse
    result_type tmp_inv;
    internal::inverse_dispatcher(tmp,tmp_inv);
    trivial_assign_div(dst.self(),tmp_inv);
}


} // end of namespace Fastor


#endif // UNARY_INV_OP_H
