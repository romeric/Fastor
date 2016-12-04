#ifndef BINARY_ADD_OP_H
#define BINARY_ADD_OP_H

#include "simd_vector/SIMDVector.h"
#include "tensor/Tensor.h"

namespace Fastor {


template<typename TLhs, typename TRhs, size_t DIM0>
struct BinaryAddOp: public AbstractTensor<BinaryAddOp<TLhs, TRhs, DIM0>,DIM0> {
//    static constexpr FASTOR_INDEX Size = TLhs::Size;

    BinaryAddOp(const TLhs& lhs, const TRhs& rhs) : lhs(lhs), rhs(rhs) {
//        print(lhs,rhs);
    }

    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return helper_size<TLhs,TRhs>();}
    template<class LExpr, class RExpr,
             typename std::enable_if<std::is_arithmetic<LExpr>::value,bool>::type =0 >
    static constexpr FASTOR_INLINE FASTOR_INDEX helper_size() {return RExpr::Size;}
    template<class LExpr, class RExpr,
             typename std::enable_if<std::is_arithmetic<RExpr>::value,bool>::type =0 >
    static constexpr FASTOR_INLINE FASTOR_INDEX helper_size() {return LExpr::Size;}
    template<class LExpr, class RExpr,
             typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                     !std::is_arithmetic<RExpr>::value,bool>::type =0 >
    static constexpr FASTOR_INLINE FASTOR_INDEX helper_size() {return RExpr::Size;}
    static constexpr FASTOR_INDEX Size = size();


    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return helper_dimension<TLhs,TRhs>(i);}
    template<class LExpr, class RExpr,
             typename std::enable_if<std::is_arithmetic<LExpr>::value,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {return rhs.dimension(i);}
    template<class LExpr, class RExpr,
             typename std::enable_if<std::is_arithmetic<RExpr>::value,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {return lhs.dimension(i);}
    template<class LExpr, class RExpr,
             typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                     !std::is_arithmetic<RExpr>::value,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {return rhs.dimension(i);}

    // The eval function evaluates the expression at position i
    template<typename U>
    FASTOR_INLINE SIMDVector<U> eval(U i) const {
        // Delay evaluation using a helper function to fully inform BinaryOp about lhs and rhs
        return helper<TLhs,TRhs>(i);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<U> helper(U i) const {
        SIMDVector<U> result;
        result = lhs.eval(static_cast<U>(i)) + rhs.eval(static_cast<U>(i));
        return result;
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<U> helper(U i) const {
        SIMDVector<U> result;
        result = lhs + rhs.eval(static_cast<U>(i));
        return result;
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<U> helper(U i) const {
        SIMDVector<U> result;
//        print(lhs.eval(static_cast<U>(i)) , rhs);
//        print(rhs);
        result = lhs.eval(static_cast<U>(i)) + (U)rhs;
//        result = lhs.eval(static_cast<U>(i)) + SIMDVector<U>(rhs);
        return result;
    }

//private:
    const TLhs &lhs;
    const TRhs &rhs;
};

template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
                                 !std::is_arithmetic<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinaryAddOp<TLhs, TRhs, DIM0> operator+(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM0> &rhs) {
  return BinaryAddOp<TLhs, TRhs, DIM0>(lhs.self(), rhs.self());
}
template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
                                 std::is_arithmetic<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinaryAddOp<TLhs, TRhs, DIM0> operator+(const AbstractTensor<TLhs,DIM0> &lhs, const TRhs &bb) {
  return BinaryAddOp<TLhs, TRhs, DIM0>(lhs.self(), bb);
}
template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<std::is_arithmetic<TLhs>::value &&
                                 !std::is_arithmetic<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinaryAddOp<TLhs, TRhs, DIM0> operator+(const TLhs &bb, const AbstractTensor<TRhs,DIM0> &rhs) {
  return BinaryAddOp<TLhs, TRhs, DIM0>(bb,rhs.self());
}

}


#endif // BINARY_ADD_OP_H
