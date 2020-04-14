#ifndef BINARY_MATMUL_OP_H
#define BINARY_MATMUL_OP_H

#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/expressions/expression_traits.h"


namespace Fastor {

template<typename TLhs, typename TRhs, size_t DIM0>
struct BinaryMatMulOp: public AbstractTensor<BinaryMatMulOp<TLhs, TRhs, DIM0>,DIM0> {
private:
    expression_t<TLhs> _lhs;
    expression_t<TRhs> _rhs;
public:

    using lhs_expr_type = expression_t<TLhs>;
    using rhs_expr_type = expression_t<TRhs>;
    // using lhs_type = typename tensor_type_finder<TLhs>::type;
    // using rhs_type = typename tensor_type_finder<TRhs>::type;
    using lhs_type = typename TLhs::result_type;
    using rhs_type = typename TRhs::result_type;
    static constexpr FASTOR_INDEX M = put_dims_in_Index<lhs_type>::type::_IndexHolder[0];
    static constexpr FASTOR_INDEX K = put_dims_in_Index<lhs_type>::type::_IndexHolder[1];
    static constexpr FASTOR_INDEX N = put_dims_in_Index<rhs_type>::type::_IndexHolder[1];
    static constexpr FASTOR_INDEX flop_count = M*N*K;

    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    using scalar_type = typename scalar_type_finder<BinaryMatMulOp<TLhs, TRhs, DIM0>>::type;
    using result_type = Tensor<scalar_type,M,N>;

    FASTOR_INLINE BinaryMatMulOp(lhs_expr_type inlhs, rhs_expr_type inrhs) : _lhs(inlhs), _rhs(inrhs) {}

    FASTOR_INLINE FASTOR_INDEX size() const {return M*N;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return i==0 ? M : N;}

    constexpr FASTOR_INLINE lhs_expr_type lhs() const {return _lhs;}
    constexpr FASTOR_INLINE rhs_expr_type rhs() const {return _rhs;}

#if 0
    // Generic version of eval
    template<typename U>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        return _lhs.template eval<U>(i) + _rhs.template eval<U>(i);
    }

    // scalar based
    template<typename U>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return _lhs.template eval_s<U>(i) + _rhs.template eval_s<U>(i);
    }

    // for 2D tensors
    template<typename U>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _lhs.template eval<U>(i,j) + _rhs.template eval<U>(i,j);
    }

    // scalar based (for 2D tensors)
    template<typename U>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _lhs.template eval_s<U>(i,j) + _rhs.template eval_s<U>(i,j);
    }

    // for nD tensors
    template<typename U>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> teval(const std::array<int,DIM0> &as) const {
        return _lhs.template teval<U>(as) + _rhs.template teval<U>(as);
    }

    // scalar based (for nD tensors)
    template<typename U>
    FASTOR_INLINE U teval_s(const std::array<int,DIM0> &as) const {
        return _lhs.template teval_s<U>(as) + _rhs.template teval_s<U>(as);
    }
#endif
};

// template<typename TLhs, typename TRhs, size_t DIM0,
//          typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
//                                  !std::is_arithmetic<TRhs>::value,bool>::type = 0 >
// FASTOR_INLINE BinaryMatMulOp<TLhs, TRhs, DIM0>
// gemm(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM0> &rhs) {
//   return BinaryMatMulOp<TLhs, TRhs, DIM0>(lhs.self(), rhs.self());
// }

// template<typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,
//          typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
//                                  !std::is_arithmetic<TRhs>::value &&
//                                  DIM0!=DIM1,bool>::type = 0 >
// FASTOR_INLINE BinaryMatMulOp<TLhs, TRhs, meta_min<DIM0,DIM1>::value>
// gemm(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM1> &rhs) {
//   return BinaryMatMulOp<TLhs, TRhs, meta_min<DIM0,DIM1>::value>(lhs.self(), rhs.self());
// }


template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
                                 !std::is_arithmetic<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinaryMatMulOp<TLhs, TRhs, DIM0>
operator %(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM0> &rhs) {
  return BinaryMatMulOp<TLhs, TRhs, DIM0>(lhs.self(), rhs.self());
}

template<typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
                                 !std::is_arithmetic<TRhs>::value &&
                                 DIM0!=DIM1,bool>::type = 0 >
FASTOR_INLINE BinaryMatMulOp<TLhs, TRhs, meta_min<DIM0,DIM1>::value>
operator %(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM1> &rhs) {
  return BinaryMatMulOp<TLhs, TRhs, meta_min<DIM0,DIM1>::value>(lhs.self(), rhs.self());
}


} // end of namespace Fastor


#endif // BINARY_MATMUL_OP_H