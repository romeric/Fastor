#ifndef BINARY_DIV_OP_H
#define BINARY_DIV_OP_H

#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/expressions/expression_traits.h"

namespace Fastor {

// Dispatch based on type of expressions not the tensor
#define FASTOR_BD_OP_EVAL_TYPE scalar_type

// Dispatch based on the type of tensor and not the expression
// #define FASTOR_BD_OP_EVAL_TYPE U


template<typename TLhs, typename TRhs, size_t DIM0>
struct BinaryDivOp: public AbstractTensor<BinaryDivOp<TLhs, TRhs, DIM0>,DIM0> {
private:
    expression_t<TLhs> _lhs;
    expression_t<TRhs> _rhs;
public:

    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    using scalar_type = typename scalar_type_finder<BinaryDivOp<TLhs, TRhs, DIM0>>::type;
    using simd_vector_type = binary_op_simd_vector_t<BinaryDivOp<TLhs, TRhs, DIM0> >;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using result_type = binary_arithmetic_result_t< BinaryDivOp<TLhs, TRhs, DIM0> >;

    FASTOR_INLINE BinaryDivOp(expression_t<TLhs> inlhs, expression_t<TRhs> inrhs) : _lhs((inlhs)), _rhs((inrhs)) {}

    FASTOR_INLINE FASTOR_INDEX size() const {return helper_size<TLhs,TRhs>();}
    template<class LExpr, class RExpr,
             typename std::enable_if<is_primitive_v_<LExpr>,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_size() const {return _rhs.size();}
    template<class LExpr, class RExpr,
             typename std::enable_if<is_primitive_v_<RExpr>,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_size() const {return _lhs.size();}
    template<class LExpr, class RExpr,
             typename std::enable_if<!is_primitive_v_<LExpr> &&
                                     !is_primitive_v_<RExpr>,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_size() const {
#ifndef NDEBUG
        FASTOR_ASSERT(_rhs.size()==_lhs.size(),"EXPRESSION SIZE MISMATCH");
#endif
        return _rhs.size();
    }

    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return helper_dimension<TLhs,TRhs>(i);}
    template<class LExpr, class RExpr,
             typename std::enable_if<is_primitive_v_<LExpr>,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {return _rhs.dimension(i);}
    template<class LExpr, class RExpr,
             typename std::enable_if<is_primitive_v_<RExpr>,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {return _lhs.dimension(i);}
    template<class LExpr, class RExpr,
             typename std::enable_if<!is_primitive_v_<LExpr> &&
                                     !is_primitive_v_<RExpr>,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {
#ifndef NDEBUG
        FASTOR_ASSERT(_rhs.dimension(i)==_lhs.dimension(i),"EXPRESSION SHAPE MISMATCH");
#endif
        return _rhs.dimension(i);
    }
    constexpr FASTOR_INLINE expression_t<TLhs> lhs() const {return _lhs;}
    constexpr FASTOR_INLINE expression_t<TRhs> rhs() const {return _rhs;}

    template<typename U>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,simd_abi_type> eval(FASTOR_INDEX i) const {
        return helper<TLhs,TRhs,U>(i);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!is_primitive_v_<LExpr> &&
                                   !is_primitive_v_<RExpr>,bool>::type = 0>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,simd_abi_type> helper(FASTOR_INDEX i) const {
#ifndef FASTOR_UNSAFE_MATH
        return _lhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i) / _rhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i);
#else
        return _lhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i) * rcp(_rhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i));
#endif
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<is_primitive_v_<LExpr> &&
                                   !is_primitive_v_<RExpr>,bool>::type = 0>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,simd_abi_type> helper(FASTOR_INDEX i) const {
#ifndef FASTOR_UNSAFE_MATH
        return (FASTOR_BD_OP_EVAL_TYPE)_lhs / _rhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i);
#else
        return (FASTOR_BD_OP_EVAL_TYPE)_lhs * rcp(_rhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i));
#endif
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!is_primitive_v_<LExpr> &&
                                   is_primitive_v_<RExpr>,bool>::type = 0>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,simd_abi_type> helper(FASTOR_INDEX i) const {
#ifndef FASTOR_UNSAFE_MATH
        return _lhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i) / (FASTOR_BD_OP_EVAL_TYPE)_rhs;
#else
        return _lhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i) * rcp(SIMDVector<FASTOR_BD_OP_EVAL_TYPE,simd_abi_type>(_rhs));
#endif
    }

    // scalar based
    template<typename U>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE eval_s(FASTOR_INDEX i) const {
        return helper_s<TLhs,TRhs,U>(i);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!is_primitive_v_<LExpr> &&
                                   !is_primitive_v_<RExpr>,bool>::type = 0>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE helper_s(FASTOR_INDEX i) const {
        return _lhs.template eval_s<FASTOR_BD_OP_EVAL_TYPE>(i) / _rhs.template eval_s<FASTOR_BD_OP_EVAL_TYPE>(i);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<is_primitive_v_<LExpr> &&
                                   !is_primitive_v_<RExpr>,bool>::type = 0>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE helper_s(FASTOR_INDEX i) const {
        return (FASTOR_BD_OP_EVAL_TYPE)_lhs / _rhs.template eval_s<FASTOR_BD_OP_EVAL_TYPE>(i);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!is_primitive_v_<LExpr> &&
                                   is_primitive_v_<RExpr>,bool>::type = 0>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE helper_s(FASTOR_INDEX i) const {
        return _lhs.template eval_s<FASTOR_BD_OP_EVAL_TYPE>(i) / (FASTOR_BD_OP_EVAL_TYPE)_rhs;
    }

    // for 2D tensors
    template<typename U>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return helper<TLhs,TRhs,U>(i,j);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!is_primitive_v_<LExpr> &&
                                   !is_primitive_v_<RExpr>,bool>::type = 0>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,simd_abi_type> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {
#ifndef FASTOR_UNSAFE_MATH
        return _lhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i,j) / _rhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i,j);
#else
        return _lhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i,j) * rcp(_rhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i,j));
#endif
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<is_primitive_v_<LExpr> &&
                                   !is_primitive_v_<RExpr>,bool>::type = 0>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,simd_abi_type> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {
#ifndef FASTOR_UNSAFE_MATH
        return (FASTOR_BD_OP_EVAL_TYPE)_lhs / _rhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i,j);
#else
        return (FASTOR_BD_OP_EVAL_TYPE)_lhs * rcp(_rhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i,j));
#endif
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!is_primitive_v_<LExpr> &&
                                   is_primitive_v_<RExpr>,bool>::type = 0>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,simd_abi_type> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {
#ifndef FASTOR_UNSAFE_MATH
        return _lhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i,j) / (FASTOR_BD_OP_EVAL_TYPE)_rhs;
#else
        return _lhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i,j) * rcp(SIMDVector<FASTOR_BD_OP_EVAL_TYPE,simd_abi_type>(_rhs));
#endif
    }

    // scalar based (for 2D tensors)
    template<typename U>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return helper_s<TLhs,TRhs,U>(i,j);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!is_primitive_v_<LExpr> &&
                                   !is_primitive_v_<RExpr>,bool>::type = 0>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _lhs.template eval_s<FASTOR_BD_OP_EVAL_TYPE>(i,j) / _rhs.template eval_s<FASTOR_BD_OP_EVAL_TYPE>(i,j);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<is_primitive_v_<LExpr> &&
                                   !is_primitive_v_<RExpr>,bool>::type = 0>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return (FASTOR_BD_OP_EVAL_TYPE)_lhs / _rhs.template eval_s<FASTOR_BD_OP_EVAL_TYPE>(i,j);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!is_primitive_v_<LExpr> &&
                                   is_primitive_v_<RExpr>,bool>::type = 0>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _lhs.template eval_s<FASTOR_BD_OP_EVAL_TYPE>(i,j) / (FASTOR_BD_OP_EVAL_TYPE)_rhs;
    }

    // for nD tensors
    template<typename U>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int,DIM0> &as) const {
        return thelper<TLhs,TRhs,U>(as);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!is_primitive_v_<LExpr> &&
                                   !is_primitive_v_<RExpr>,bool>::type = 0>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,simd_abi_type> thelper(const std::array<int,DIM0> &as) const {
        return _lhs.template teval<FASTOR_BD_OP_EVAL_TYPE>(as) / _rhs.template teval<FASTOR_BD_OP_EVAL_TYPE>(as);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<is_primitive_v_<LExpr> &&
                                   !is_primitive_v_<RExpr>,bool>::type = 0>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,simd_abi_type> thelper(const std::array<int,DIM0> &as) const {
        return (FASTOR_BD_OP_EVAL_TYPE)_lhs / _rhs.template teval<FASTOR_BD_OP_EVAL_TYPE>(as);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!is_primitive_v_<LExpr> &&
                                   is_primitive_v_<RExpr>,bool>::type = 0>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,simd_abi_type> thelper(const std::array<int,DIM0> &as) const {
        return _lhs.template teval<FASTOR_BD_OP_EVAL_TYPE>(as) / (FASTOR_BD_OP_EVAL_TYPE)_rhs;
    }

    // scalar based (for nD tensors)
    template<typename U>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE teval_s(const std::array<int,DIM0> &as) const {
        return thelper_s<TLhs,TRhs,U>(as);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!is_primitive_v_<LExpr> &&
                                   !is_primitive_v_<RExpr>,bool>::type = 0>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE thelper_s(const std::array<int,DIM0> &as) const {
        return _lhs.template teval_s<FASTOR_BD_OP_EVAL_TYPE>(as) / _rhs.template teval_s<FASTOR_BD_OP_EVAL_TYPE>(as);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<is_primitive_v_<LExpr> &&
                                   !is_primitive_v_<RExpr>,bool>::type = 0>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE thelper_s(const std::array<int,DIM0> &as) const {
        return (FASTOR_BD_OP_EVAL_TYPE)_lhs / _rhs.template teval_s<FASTOR_BD_OP_EVAL_TYPE>(as);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!is_primitive_v_<LExpr> &&
                                   is_primitive_v_<RExpr>,bool>::type = 0>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE thelper_s(const std::array<int,DIM0> &as) const {
        return _lhs.template teval_s<FASTOR_BD_OP_EVAL_TYPE>(as) / (FASTOR_BD_OP_EVAL_TYPE)_rhs;
    }
};

template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<!is_primitive_v_<TLhs> &&
                                 !is_primitive_v_<TRhs>,bool>::type = 0 >
FASTOR_INLINE BinaryDivOp<TLhs, TRhs, DIM0> operator/(const AbstractTensor<TLhs,DIM0> &_lhs, const AbstractTensor<TRhs,DIM0> &_rhs) {
  return BinaryDivOp<TLhs, TRhs, DIM0>(_lhs.self(), _rhs.self());
}

#ifndef FASTOR_DISPATCH_DIV_TO_MUL_EXPR
template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<!is_primitive_v_<TLhs> &&
                                 is_primitive_v_<TRhs>,bool>::type = 0 >
FASTOR_INLINE BinaryDivOp<TLhs, TRhs, DIM0> operator/(const AbstractTensor<TLhs,DIM0> &_lhs, TRhs bb) {
  return BinaryDivOp<TLhs, TRhs, DIM0>(_lhs.self(), bb);
}
template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<is_primitive_v_<TLhs> &&
                                 !is_primitive_v_<TRhs>,bool>::type = 0 >
FASTOR_INLINE BinaryDivOp<TLhs, TRhs, DIM0> operator/(TLhs bb, const AbstractTensor<TRhs,DIM0> &_rhs) {
  return BinaryDivOp<TLhs, TRhs, DIM0>(bb,_rhs.self());
}
#else
template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<!is_primitive_v_<TLhs> &&
                                 is_primitive_v_<TRhs> && !std::is_integral<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinaryMulOp<TLhs, TRhs, DIM0> operator/(const AbstractTensor<TLhs,DIM0> &_lhs, TRhs bb) {
  return BinaryMulOp<TLhs, TRhs, DIM0>(
    _lhs.self(),
    static_cast<typename scalar_type_finder<TLhs>::type>(1)/static_cast<typename scalar_type_finder<TLhs>::type>(bb));
}
template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<is_primitive_v_<TLhs> && !std::is_integral<TRhs>::value &&
                                 !is_primitive_v_<TRhs>,bool>::type = 0 >
FASTOR_INLINE BinaryMulOp<TLhs, TRhs, DIM0> operator/(TLhs bb, const AbstractTensor<TRhs,DIM0> &_rhs) {
  return BinaryMulOp<TLhs, TRhs, DIM0>(
    static_cast<typename scalar_type_finder<TLhs>::type>(1)/static_cast<typename scalar_type_finder<TLhs>::type>(bb),
    _rhs.self());
}
// Special case for integral types
template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<!is_primitive_v_<TLhs> &&
                                 is_primitive_v_<TRhs>  && std::is_integral<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinaryDivOp<TLhs, TRhs, DIM0> operator/(const AbstractTensor<TLhs,DIM0> &_lhs, TRhs bb) {
  return BinaryDivOp<TLhs, TRhs, DIM0>(_lhs.self(), bb);
}
template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<is_primitive_v_<TLhs> && std::is_integral<TRhs>::value &&
                                 !is_primitive_v_<TRhs>,bool>::type = 0 >
FASTOR_INLINE BinaryDivOp<TLhs, TRhs, DIM0> operator/(TLhs bb, const AbstractTensor<TRhs,DIM0> &_rhs) {
  return BinaryDivOp<TLhs, TRhs, DIM0>(bb,_rhs.self());
}
#endif

template<typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,
         enable_if_t_<!is_arithmetic_v_<TLhs> &&
                      !is_arithmetic_v_<TRhs> &&
                      DIM0!=DIM1,bool> = false >
FASTOR_INLINE BinaryDivOp<TLhs, TRhs, meta_min<DIM0,DIM1>::value>
operator/(const AbstractTensor<TLhs,DIM0> &_lhs, const AbstractTensor<TRhs,DIM1> &_rhs) {
  return BinaryDivOp<TLhs, TRhs, meta_min<DIM0,DIM1>::value>(_lhs.self(), _rhs.self());
}

}

#endif // BINARY_DIV_OP_H

