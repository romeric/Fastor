#ifndef UNARY_ADD_OP_H
#define UNARY_ADD_OP_H


#include "tensor/Tensor.h"

namespace Fastor {

// sqrt
template<typename Expr, size_t DIM0>
struct UnarySqrtOp: public AbstractTensor<UnarySqrtOp<Expr, DIM0>,DIM0> {
private:
    const Expr &expr;
public:
    static constexpr FASTOR_INDEX Size = Expr::Size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return Expr::Size;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnarySqrtOp(const Expr& expr) : expr(expr) {}

    template<typename U>
    FASTOR_INLINE SIMDVector<U> eval(U i) const {
    return sqrt(expr.eval(static_cast<U>(i)));
    }
};

template<typename Expr, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnarySqrtOp<Expr, DIM0> sqrt(const AbstractTensor<Expr,DIM0> &expr) {
  return UnarySqrtOp<Expr, DIM0>(expr.self());
}




// exp
template<typename Expr, size_t DIM0>
struct UnaryExpOp: public AbstractTensor<UnaryExpOp<Expr, DIM0>,DIM0> {
private:
    const Expr &expr;
public:
    static constexpr FASTOR_INDEX Size = Expr::Size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return Expr::Size;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryExpOp(const Expr& expr) : expr(expr) {}

    template<typename U>
    FASTOR_INLINE SIMDVector<U> eval(U i) const {
    return exp(expr.eval(static_cast<U>(i)));
    }
};

template<typename Expr, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnaryExpOp<Expr, DIM0> exp(const AbstractTensor<Expr,DIM0> &expr) {
  return UnaryExpOp<Expr, DIM0>(expr.self());
}

// log
template<typename Expr, size_t DIM0>
struct UnaryLogOp: public AbstractTensor<UnaryLogOp<Expr, DIM0>,DIM0> {
private:
    const Expr &expr;
public:
    static constexpr FASTOR_INDEX Size = Expr::Size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return Expr::Size;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryLogOp(const Expr& expr) : expr(expr) {}

    template<typename U>
    FASTOR_INLINE SIMDVector<U> eval(U i) const {
    return log(expr.eval(static_cast<U>(i)));
    }
};

template<typename Expr, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnaryLogOp<Expr, DIM0> log(const AbstractTensor<Expr,DIM0> &expr) {
  return UnaryLogOp<Expr, DIM0>(expr.self());
}



// sin
template<typename Expr, size_t DIM0>
struct UnarySinOp: public AbstractTensor<UnarySinOp<Expr, DIM0>,DIM0> {
private:
    const Expr &expr;
public:
    static constexpr FASTOR_INDEX Size = Expr::Size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return Expr::Size;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnarySinOp(const Expr& expr) : expr(expr) {}

    template<typename U>
    FASTOR_INLINE SIMDVector<U> eval(U i) const {
    return sin(expr.eval(static_cast<U>(i)));
    }
};

template<typename Expr, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnarySinOp<Expr, DIM0> sin(const AbstractTensor<Expr,DIM0> &expr) {
  return UnarySinOp<Expr, DIM0>(expr.self());
}



// cos
template<typename Expr, size_t DIM0>
struct UnaryCosOp: public AbstractTensor<UnaryCosOp<Expr, DIM0>,DIM0> {
private:
    const Expr &expr;
public:
    static constexpr FASTOR_INDEX Size = Expr::Size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return Expr::Size;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryCosOp(const Expr& expr) : expr(expr) {}

    template<typename U>
    FASTOR_INLINE SIMDVector<U> eval(U i) const {
    return cos(expr.eval(static_cast<U>(i)));
    }
};

template<typename Expr, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnaryCosOp<Expr, DIM0> cos(const AbstractTensor<Expr,DIM0> &expr) {
  return UnaryCosOp<Expr, DIM0>(expr.self());
}



// tan
template<typename Expr, size_t DIM0>
struct UnaryTanOp: public AbstractTensor<UnaryTanOp<Expr, DIM0>,DIM0> {
private:
    const Expr &expr;
public:
    static constexpr FASTOR_INDEX Size = Expr::Size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return Expr::Size;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryTanOp(const Expr& expr) : expr(expr) {}

    template<typename U>
    FASTOR_INLINE SIMDVector<U> eval(U i) const {
    return tan(expr.eval(static_cast<U>(i)));
    }
};

template<typename Expr, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnaryTanOp<Expr, DIM0> tan(const AbstractTensor<Expr,DIM0> &expr) {
  return UnaryTanOp<Expr, DIM0>(expr.self());
}



// asin
template<typename Expr, size_t DIM0>
struct UnaryAsinOp: public AbstractTensor<UnaryAsinOp<Expr, DIM0>,DIM0> {
private:
    const Expr &expr;
public:
    static constexpr FASTOR_INDEX Size = Expr::Size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return Expr::Size;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryAsinOp(const Expr& expr) : expr(expr) {}

    template<typename U>
    FASTOR_INLINE SIMDVector<U> eval(U i) const {
    return asin(expr.eval(static_cast<U>(i)));
    }
};

template<typename Expr, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnaryAsinOp<Expr, DIM0> asin(const AbstractTensor<Expr,DIM0> &expr) {
  return UnaryAsinOp<Expr, DIM0>(expr.self());
}



// acos
template<typename Expr, size_t DIM0>
struct UnaryAcosOp: public AbstractTensor<UnaryAcosOp<Expr, DIM0>,DIM0> {
private:
    const Expr &expr;
public:
    static constexpr FASTOR_INDEX Size = Expr::Size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return Expr::Size;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryAcosOp(const Expr& expr) : expr(expr) {}

    template<typename U>
    FASTOR_INLINE SIMDVector<U> eval(U i) const {
    return acos(expr.eval(static_cast<U>(i)));
    }
};

template<typename Expr, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnaryAcosOp<Expr, DIM0> acos(const AbstractTensor<Expr,DIM0> &expr) {
  return UnaryAcosOp<Expr, DIM0>(expr.self());
}



// atan
template<typename Expr, size_t DIM0>
struct UnaryAtanOp: public AbstractTensor<UnaryAtanOp<Expr, DIM0>,DIM0> {
private:
    const Expr &expr;
public:
    static constexpr FASTOR_INDEX Size = Expr::Size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return Expr::Size;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryAtanOp(const Expr& expr) : expr(expr) {}

    template<typename U>
    FASTOR_INLINE SIMDVector<U> eval(U i) const {
    return atan(expr.eval(static_cast<U>(i)));
    }
};

template<typename Expr, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnaryAtanOp<Expr, DIM0> atan(const AbstractTensor<Expr,DIM0> &expr) {
  return UnaryAtanOp<Expr, DIM0>(expr.self());
}



// sinh
template<typename Expr, size_t DIM0>
struct UnarySinhOp: public AbstractTensor<UnarySinhOp<Expr, DIM0>,DIM0> {
private:
    const Expr &expr;
public:
    static constexpr FASTOR_INDEX Size = Expr::Size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return Expr::Size;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnarySinhOp(const Expr& expr) : expr(expr) {}

    template<typename U>
    FASTOR_INLINE SIMDVector<U> eval(U i) const {
    return sinh(expr.eval(static_cast<U>(i)));
    }
};

template<typename Expr, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnarySinhOp<Expr, DIM0> sinh(const AbstractTensor<Expr,DIM0> &expr) {
  return UnarySinhOp<Expr, DIM0>(expr.self());
}



// cosh
template<typename Expr, size_t DIM0>
struct UnaryCoshOp: public AbstractTensor<UnaryCoshOp<Expr, DIM0>,DIM0> {
private:
    const Expr &expr;
public:
    static constexpr FASTOR_INDEX Size = Expr::Size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return Expr::Size;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryCoshOp(const Expr& expr) : expr(expr) {}

    template<typename U>
    FASTOR_INLINE SIMDVector<U> eval(U i) const {
    return cosh(expr.eval(static_cast<U>(i)));
    }
};

template<typename Expr, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnaryCoshOp<Expr, DIM0> cosh(const AbstractTensor<Expr,DIM0> &expr) {
  return UnaryCoshOp<Expr, DIM0>(expr.self());
}



// tanh
template<typename Expr, size_t DIM0>
struct UnaryTanhOp: public AbstractTensor<UnaryTanhOp<Expr, DIM0>,DIM0> {
private:
    const Expr &expr;
public:
    static constexpr FASTOR_INDEX Size = Expr::Size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return Expr::Size;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryTanhOp(const Expr& expr) : expr(expr) {}

    template<typename U>
    FASTOR_INLINE SIMDVector<U> eval(U i) const {
    return tanh(expr.eval(static_cast<U>(i)));
    }
};

template<typename Expr, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnaryTanhOp<Expr, DIM0> tanh(const AbstractTensor<Expr,DIM0> &expr) {
  return UnaryTanhOp<Expr, DIM0>(expr.self());
}

}


#endif // UNARY_ADD_OP_H

