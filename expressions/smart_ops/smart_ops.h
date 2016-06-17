#ifndef SMART_OPS_H
#define SMART_OPS_H

#include "backend/backend.h"
#include "meta/tensor_meta.h"

namespace Fastor {


//!--------------------------------------------------------------!//
template<typename TLhs, typename TRhs>
struct BinaryMatMulOp {

    BinaryMatMulOp(const TLhs& lhs, const TRhs& rhs) : lhs(lhs), rhs(rhs) {}

    // The eval function evaluates the expression at position i
    template<typename U>
    FASTOR_INLINE U eval(U i, U j) const {
        U result = 0;
        for (U k=0; k<lhs.dimension(1); k+=1) {
            result += lhs.eval(i,static_cast<U>(k))*rhs.eval(static_cast<U>(k),j);
        }
        return result;
    }

    FASTOR_INDEX dimension(FASTOR_INDEX i) const {return lhs.dimension(i);}

//private:
    const TLhs &lhs;
    const TRhs &rhs;
};
template<typename TLhs, typename TRhs,
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
                                 !std::is_arithmetic<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinaryMatMulOp<TLhs, TRhs> lmatmul(const TLhs &lhs, const TRhs &rhs) {
//  return BinaryMatMulOp<TLhs, TRhs>(lhs.self(), rhs.self());
  return BinaryMatMulOp<TLhs, TRhs>(lhs, rhs);
}
//!--------------------------------------------------------------!//


//!--------------------------------------------------------------!//
template<typename Expr>
struct UnaryTransposeOp {

    UnaryTransposeOp(const Expr& expr) : expr(expr) {}

    // The eval function evaluates the expression at position i
    template<typename U>
    FASTOR_INLINE U eval(U i, U j) const {
        return expr(static_cast<U>(j),static_cast<U>(i));
    }

//private:
    const Expr &expr;
};
template<typename Expr,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnaryTransposeOp<Expr> ltranspose(const Expr &expr) {
//  return UnaryTransposeOp<Expr>(expr.self());
  return UnaryTransposeOp<Expr>(expr);
}
//!--------------------------------------------------------------!//


//!--------------------------------------------------------------!//
template<typename Expr>
struct UnaryTraceOp {

    UnaryTraceOp(const Expr& expr) : expr(expr) {}

    // The eval function evaluates the expression at position i
    template<typename U>
    FASTOR_INLINE U eval(U i) const {
        U result = 0;
        for (U i=0; i<static_cast<U>(expr.dimension(0)); i+=1) {
            result += expr.eval(static_cast<U>(i),static_cast<U>(i));
        }
        return result;
    }

    const Expr &expr;
};
template<typename Expr,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnaryTraceOp<Expr> ltrace(const Expr &expr) {
//  return UnaryTraceOp<Expr>(expr.self());
  return UnaryTraceOp<Expr>(expr);
}
//!--------------------------------------------------------------!//


//!--------------------------------------------------------------!//
template<typename Expr>
struct UnaryDetOp {

    UnaryDetOp(const Expr& expr) : expr(expr) {}

    template<typename U>
    FASTOR_INLINE U eval(U i) const {
        U result = i-i;
        constexpr FASTOR_INDEX dimension = static_cast<FASTOR_INDEX>(
                    meta_sqrt<static_cast<int>(Expr::Size)>::ret);
        result = _det<U,dimension,dimension>(expr.data());
        return result;
    }


    const Expr &expr;
};
template<typename Expr,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnaryDetOp<Expr> ldeterminant(const Expr &expr) {
  return UnaryDetOp<Expr>(expr);
}
//!--------------------------------------------------------------!//


//!--------------------------------------------------------------!//
template<typename Expr>
struct UnaryAdjOp {

    UnaryAdjOp(const Expr& expr) : expr(expr) {}

    template<typename U>
    FASTOR_INLINE void eval(U *_data) const {
        constexpr FASTOR_INDEX dimension = static_cast<FASTOR_INDEX>(
                    meta_sqrt<static_cast<int>(Expr::Size)>::ret);
        _adjoint<U,dimension,dimension>(expr.data(),_data);
    }


    const Expr &expr;
};
template<typename Expr,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnaryAdjOp<Expr> ladjoint(const Expr &expr) {
  return UnaryAdjOp<Expr>(expr);
}
//!--------------------------------------------------------------!//


//!--------------------------------------------------------------!//
template<typename Expr>
struct UnaryCofOp {

    UnaryCofOp(const Expr& expr) : expr(expr) {}

    template<typename U>
    FASTOR_INLINE void eval(U *_data) const {
        constexpr FASTOR_INDEX dimension = static_cast<FASTOR_INDEX>(
                    meta_sqrt<static_cast<int>(Expr::Size)>::ret);
        _cofactor<U,dimension,dimension>(expr.data(),_data);
    }


    const Expr &expr;
};
template<typename Expr,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnaryCofOp<Expr> lcofactor(const Expr &expr) {
  return UnaryCofOp<Expr>(expr);
}
//!--------------------------------------------------------------!//


//!--------------------------------------------------------------!//
template<typename Expr>
struct UnaryInvOp {

    UnaryInvOp(const Expr& expr) : expr(expr) {}

    template<typename U>
    FASTOR_INLINE U eval(U *_data) const {
        constexpr FASTOR_INDEX dimension = static_cast<FASTOR_INDEX>(
                    meta_sqrt<static_cast<int>(Expr::Size)>::ret);
        _adjoint<U,dimension,dimension>(expr.data(),_data);
        U result = _det<U,dimension,dimension>(expr.data());
        return result;
    }


    const Expr &expr;
};
template<typename Expr,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnaryInvOp<Expr> linverse(const Expr &expr) {
  return UnaryInvOp<Expr>(expr);
}
//!--------------------------------------------------------------!//


}


#endif // SMART_OPS_H

