#ifndef UNARY_ADD_OP_H
#define UNARY_ADD_OP_H


#include "tensor/Tensor.h"
#include "meta/tensor_post_meta.h"

namespace Fastor {

// addition
template<typename Expr, size_t DIM0>
struct UnaryAddOp: public AbstractTensor<UnaryAddOp<Expr, DIM0>,DIM0> {
private:
    const Expr &expr;
public:
    using scalar_type = typename scalar_type_finder<Expr>::type;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    FASTOR_INLINE FASTOR_INDEX size() const {return expr.size();}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryAddOp(const Expr& expr) : expr(expr) {}

    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        return expr.template eval<U>(i);
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return expr.template eval_s<U>(i);
    }
    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return expr.template eval<U>(i,j);
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return expr.template eval_s<U>(i,j);
    }
};

template<typename Expr, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnaryAddOp<Expr, DIM0> operator+(const AbstractTensor<Expr,DIM0> &expr) {
  return UnaryAddOp<Expr, DIM0>(expr.self());
}


// subtraction
template<typename Expr, size_t DIM0>
struct UnarySubOp: public AbstractTensor<UnarySubOp<Expr, DIM0>,DIM0> {
private:
    const Expr &expr;
public:
    using scalar_type = typename scalar_type_finder<Expr>::type;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    FASTOR_INLINE FASTOR_INDEX size() const {return expr.size();}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnarySubOp(const Expr& expr) : expr(expr) {}

    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        return -expr.template eval<U>(i);
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return -expr.template eval_s<U>(i);
    }
    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return -expr.template eval<U>(i,j);
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return -expr.template eval_s<U>(i,j);
    }
};

template<typename Expr, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnarySubOp<Expr, DIM0> operator-(const AbstractTensor<Expr,DIM0> &expr) {
  return UnarySubOp<Expr, DIM0>(expr.self());
}


// absolute value
template<typename Expr, size_t DIM0>
struct UnaryAbsOp: public AbstractTensor<UnaryAbsOp<Expr, DIM0>,DIM0> {
private:
    const Expr &expr;
public:
    using scalar_type = typename scalar_type_finder<Expr>::type;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    FASTOR_INLINE FASTOR_INDEX size() const {return expr.size();}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryAbsOp(const Expr& expr) : expr(expr) {}

    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        return abs(expr.template eval<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return std::abs(expr.template eval_s<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return abs(expr.template eval<U>(i,j));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return std::abs(expr.template eval_s<U>(i,j));
    }
};

template<typename Expr, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnaryAbsOp<Expr, DIM0> abs(const AbstractTensor<Expr,DIM0> &expr) {
  return UnaryAbsOp<Expr, DIM0>(expr.self());
}


// Mathematical functions

// sqrt
template<typename Expr, size_t DIM0>
struct UnarySqrtOp: public AbstractTensor<UnarySqrtOp<Expr, DIM0>,DIM0> {
private:
    const Expr &expr;
public:
    using scalar_type = typename scalar_type_finder<Expr>::type;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    FASTOR_INLINE FASTOR_INDEX size() const {return expr.size();}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnarySqrtOp(const Expr& _expr) : expr(_expr) {}

    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        return sqrt(expr.template eval<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return std::sqrt(expr.template eval_s<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return sqrt(expr.template eval<U>(i,j));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return sqrts(expr.template eval_s<U>(i,j));
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
    using scalar_type = typename scalar_type_finder<Expr>::type;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    FASTOR_INLINE FASTOR_INDEX size() const {return expr.size();}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryExpOp(const Expr& expr) : expr(expr) {}

    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        return exp(expr.template eval<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return std::exp(expr.template eval_s<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return exp(expr.template eval<U>(i,j));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return std::exp(expr.template eval_s<U>(i,j));
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
    using scalar_type = typename scalar_type_finder<Expr>::type;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    FASTOR_INLINE FASTOR_INDEX size() const {return expr.size();}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryLogOp(const Expr& expr) : expr(expr) {}

    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        return log(expr.template eval<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return std::log(expr.template eval_s<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return log(expr.template eval<U>(i,j));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return std::log(expr.template eval_s<U>(i,j));
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
    using scalar_type = typename scalar_type_finder<Expr>::type;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    FASTOR_INLINE FASTOR_INDEX size() const {return expr.size();}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnarySinOp(const Expr& expr) : expr(expr) {}

    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        return sin(expr.template eval<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return std::sin(expr.template eval_s<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return sin(expr.template eval<U>(i,j));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return std::sin(expr.template eval_s<U>(i,j));
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
    using scalar_type = typename scalar_type_finder<Expr>::type;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    FASTOR_INLINE FASTOR_INDEX size() const {return expr.size();}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryCosOp(const Expr& expr) : expr(expr) {}

    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        return cos(expr.template eval<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return std::cos(expr.template eval_s<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return cos(expr.template eval<U>(i,j));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return std::cos(expr.template eval_s<U>(i,j));
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
    using scalar_type = typename scalar_type_finder<Expr>::type;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    FASTOR_INLINE FASTOR_INDEX size() const {return expr.size();}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryTanOp(const Expr& expr) : expr(expr) {}

    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        return tan(expr.template eval<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return std::tan(expr.template eval_s<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return tan(expr.template eval<U>(i,j));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return std::tan(expr.template eval_s<U>(i,j));
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
    using scalar_type = typename scalar_type_finder<Expr>::type;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    FASTOR_INLINE FASTOR_INDEX size() const {return expr.size();}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryAsinOp(const Expr& expr) : expr(expr) {}

    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        return asin(expr.template eval<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return std::asin(expr.template eval_s<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return asin(expr.template eval<U>(i,j));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return std::asin(expr.template eval_s<U>(i,j));
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
    using scalar_type = typename scalar_type_finder<Expr>::type;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    FASTOR_INLINE FASTOR_INDEX size() const {return expr.size();}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryAcosOp(const Expr& expr) : expr(expr) {}

    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        return acos(expr.template eval<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return std::acos(expr.template eval_s<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return acos(expr.template eval<U>(i,j));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return std::acos(expr.template eval_s<U>(i,j));
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
    using scalar_type = typename scalar_type_finder<Expr>::type;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    FASTOR_INLINE FASTOR_INDEX size() const {return expr.size();}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryAtanOp(const Expr& expr) : expr(expr) {}

    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        return atan(expr.template eval<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return std::atan(expr.template eval_s<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return atan(expr.template eval<U>(i,j));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return std::atan(expr.template eval_s<U>(i,j));
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
    using scalar_type = typename scalar_type_finder<Expr>::type;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    FASTOR_INLINE FASTOR_INDEX size() const {return expr.size();}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnarySinhOp(const Expr& expr) : expr(expr) {}

    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        return sinh(expr.template eval<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return std::sinh(expr.template eval_s<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return sinh(expr.template eval<U>(i,j));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return std::sinh(expr.template eval_s<U>(i,j));
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
    using scalar_type = typename scalar_type_finder<Expr>::type;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    FASTOR_INLINE FASTOR_INDEX size() const {return expr.size();}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryCoshOp(const Expr& expr) : expr(expr) {}

    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        return cosh(expr.template eval<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return std::cosh(expr.template eval_s<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return cosh(expr.template eval<U>(i,j));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return std::cosh(expr.template eval_s<U>(i,j));
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
    using scalar_type = typename scalar_type_finder<Expr>::type;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    FASTOR_INLINE FASTOR_INDEX size() const {return expr.size();}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return expr.dimension(i);}

    UnaryTanhOp(const Expr& expr) : expr(expr) {}

    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        return tanh(expr.template eval<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return std::tanh(expr.template eval_s<U>(i));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return tanh(expr.template eval<U>(i,j));
    }
    template<typename U=scalar_type>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return std::tanh(expr.template eval_s<U>(i,j));
    }
};

template<typename Expr, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >
FASTOR_INLINE UnaryTanhOp<Expr, DIM0> tanh(const AbstractTensor<Expr,DIM0> &expr) {
  return UnaryTanhOp<Expr, DIM0>(expr.self());
}

}


#endif // UNARY_ADD_OP_H

