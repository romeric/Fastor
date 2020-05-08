#ifndef BINARY_SUB_OP_H
#define BINARY_SUB_OP_H

#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/expressions/expression_traits.h"



namespace Fastor {

template<typename TLhs, typename TRhs, size_t DIM0>
struct BinarySubOp: public AbstractTensor<BinarySubOp<TLhs, TRhs, DIM0>,DIM0> {
private:
    expression_t<TLhs> _lhs;
    expression_t<TRhs> _rhs;
public:
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    using scalar_type = typename scalar_type_finder<BinarySubOp<TLhs, TRhs, DIM0>>::type;
    using simd_vector_type = binary_op_simd_vector_t<BinarySubOp<TLhs, TRhs, DIM0> >;
    using simd_abi_type = typename simd_vector_type::abi_type;

    FASTOR_INLINE BinarySubOp(expression_t<TLhs> inlhs, expression_t<TRhs> inrhs) : _lhs((inlhs)), _rhs((inrhs)) {}

    FASTOR_INLINE FASTOR_INDEX size() const {return helper_size<TLhs,TRhs>();}
    template<class LExpr, class RExpr,
             typename std::enable_if<std::is_arithmetic<LExpr>::value,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_size() const {return _rhs.size();}
    template<class LExpr, class RExpr,
             typename std::enable_if<std::is_arithmetic<RExpr>::value,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_size() const {return _lhs.size();}
    template<class LExpr, class RExpr,
             typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                     !std::is_arithmetic<RExpr>::value,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_size() const {
#ifndef NDEBUG
        FASTOR_ASSERT(_rhs.size()==_lhs.size(),"EXPRESSION SIZE MISMATCH");
#endif
        return _rhs.size();
    }

    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return helper_dimension<TLhs,TRhs>(i);}
    template<class LExpr, class RExpr,
             typename std::enable_if<std::is_arithmetic<LExpr>::value,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {return _rhs.dimension(i);}
    template<class LExpr, class RExpr,
             typename std::enable_if<std::is_arithmetic<RExpr>::value,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {return _lhs.dimension(i);}
    template<class LExpr, class RExpr,
             typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                     !std::is_arithmetic<RExpr>::value,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {
#ifndef NDEBUG
        FASTOR_ASSERT(_rhs.dimension(i)==_lhs.dimension(i),"EXPRESSION SHAPE MISMATCH");
#endif
        return _rhs.dimension(i);
    }
    constexpr FASTOR_INLINE expression_t<TLhs> lhs() const {return _lhs;}
    constexpr FASTOR_INLINE expression_t<TRhs> rhs() const {return _rhs;}

    template<typename U>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i) const {
        return helper<TLhs,TRhs,U>(i);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> helper(FASTOR_INDEX i) const {
        return _lhs.template eval<U>(i) - _rhs.template eval<U>(i);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> helper(FASTOR_INDEX i) const {
        return (U)_lhs - _rhs.template eval<U>(i);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> helper(FASTOR_INDEX i) const {
        return _lhs.template eval<U>(i) - (U)_rhs;
    }


    // scalar based
    template<typename U>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return helper_s<TLhs,TRhs,U>(i);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE U helper_s(FASTOR_INDEX i) const {
        return _lhs.template eval_s<U>(i) - _rhs.template eval_s<U>(i);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE U helper_s(FASTOR_INDEX i) const {
        return (U)_lhs - _rhs.template eval_s<U>(i);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE U helper_s(FASTOR_INDEX i) const {
        return _lhs.template eval_s<U>(i) - (U)_rhs;
    }


    // for 2D tensors
    template<typename U>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return helper<TLhs,TRhs,U>(i,j);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _lhs.template eval<U>(i,j) - _rhs.template eval<U>(i,j);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return (U)_lhs - _rhs.template eval<U>(i,j);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _lhs.template eval<U>(i,j) - (U)_rhs;
    }

    // scalar based (for 2D tensors)
    template<typename U>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return helper_s<TLhs,TRhs,U>(i,j);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE U helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _lhs.template eval_s<U>(i,j) - _rhs.template eval_s<U>(i,j);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE U helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return (U)_lhs - _rhs.template eval_s<U>(i,j);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE U helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _lhs.template eval_s<U>(i,j) - (U)_rhs;
    }

    // for nD tensors
    template<typename U>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int,DIM0> &as) const {
        return thelper<TLhs,TRhs,U>(as);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> thelper(const std::array<int,DIM0> &as) const {
        return _lhs.template teval<U>(as) - _rhs.template teval<U>(as);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> thelper(const std::array<int,DIM0> &as) const {
        return (U)_lhs - _rhs.template teval<U>(as);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> thelper(const std::array<int,DIM0> &as) const {
        return _lhs.template teval<U>(as) - (U)_rhs;
    }

    // scalar based (for nD tensors)
    template<typename U>
    FASTOR_INLINE U teval_s(const std::array<int,DIM0> &as) const {
        return thelper_s<TLhs,TRhs,U>(as);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE U thelper_s(const std::array<int,DIM0> &as) const {
        return _lhs.template teval_s<U>(as) - _rhs.template teval_s<U>(as);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE U thelper_s(const std::array<int,DIM0> &as) const {
        return (U)_lhs - _rhs.template teval_s<U>(as);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE U thelper_s(const std::array<int,DIM0> &as) const {
        return _lhs.template teval_s<U>(as) - (U)_rhs;
    }
};

template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
                                 !std::is_arithmetic<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinarySubOp<TLhs, TRhs, DIM0> operator-(const AbstractTensor<TLhs,DIM0> &_lhs, const AbstractTensor<TRhs,DIM0> &_rhs) {
  return BinarySubOp<TLhs, TRhs, DIM0>(_lhs.self(), _rhs.self());
}
template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
                                 std::is_arithmetic<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinarySubOp<TLhs, TRhs, DIM0> operator-(const AbstractTensor<TLhs,DIM0> &_lhs, TRhs bb) {
  return BinarySubOp<TLhs, TRhs, DIM0>(_lhs.self(), bb);
}
template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<std::is_arithmetic<TLhs>::value &&
                                 !std::is_arithmetic<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinarySubOp<TLhs, TRhs, DIM0> operator-(TLhs bb, const AbstractTensor<TRhs,DIM0> &_rhs) {
  return BinarySubOp<TLhs, TRhs, DIM0>(bb,_rhs.self());
}

template<typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
                                 !std::is_arithmetic<TRhs>::value &&
                                 DIM0!=DIM1,bool>::type = 0 >
FASTOR_INLINE BinarySubOp<TLhs, TRhs, meta_min<DIM0,DIM1>::value>
operator-(const AbstractTensor<TLhs,DIM0> &_lhs, const AbstractTensor<TRhs,DIM1> &_rhs) {
  return BinarySubOp<TLhs, TRhs, meta_min<DIM0,DIM1>::value>(_lhs.self(), _rhs.self());
}

}



#endif // BINARY_SUB_OP_H

