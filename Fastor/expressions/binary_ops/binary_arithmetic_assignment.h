#ifndef BINARY_ARITHMETIC_ASSIGNMENT_H
#define BINARY_ARITHMETIC_ASSIGNMENT_H


#include "Fastor/expressions/binary_ops/binary_arithmetic_ops.h"


namespace Fastor {

// Create assign, assign_add, assign_sub for BinaryAddOp and BinarySubOp
// Create assign for BinarymulOp and BinaryDivOp
#define FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_1(NAME, ASSIGN_TYPE, OP_ASSIGN_TYPE)\
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,\
    enable_if_t_<!is_arithmetic_v_<TLhs> && !is_arithmetic_v_<TRhs>, bool> = false >\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, const Binary ##NAME ## Op<TLhs, TRhs, OtherDIM> &src) {\
    FASTOR_IF_CONSTEXPR (!(requires_evaluation_v<TLhs> || requires_evaluation_v<TRhs>)) {\
        trivial_assign ##ASSIGN_TYPE (dst.self(), src.self());\
    }\
    else {\
        assign ##ASSIGN_TYPE (dst.self(), src.lhs().self());\
        assign ##OP_ASSIGN_TYPE (dst.self(), src.rhs().self());\
    }\
}\
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,\
    enable_if_t_<is_arithmetic_v_<TLhs> && !is_arithmetic_v_<TRhs>, bool> = false>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, const Binary ##NAME ## Op<TLhs, TRhs, OtherDIM> &src) {\
    FASTOR_IF_CONSTEXPR (!requires_evaluation_v<TRhs>) {\
        trivial_assign ##ASSIGN_TYPE (dst.self(), src.self());\
    }\
    else {\
        assign ##ASSIGN_TYPE (dst.self(), src.lhs());\
        assign  ##OP_ASSIGN_TYPE (dst.self(), src.rhs().self());\
    }\
}\
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,\
    enable_if_t_<!is_arithmetic_v_<TLhs> && is_arithmetic_v_<TRhs>, bool> = false>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, const Binary ##NAME ## Op<TLhs, TRhs, OtherDIM> &src) {\
    FASTOR_IF_CONSTEXPR (!requires_evaluation_v<TLhs>) {\
        trivial_assign ##ASSIGN_TYPE (dst.self(), src.self());\
    }\
    else {\
        assign ##ASSIGN_TYPE (dst.self(), src.lhs().self());\
        assign  ##OP_ASSIGN_TYPE (dst.self(), src.rhs());\
    }\
}\


// Create assign_add, assign_sub for BinaryMulOp and BinaryDivOp
// Create assign_mul, assign_div for all binary ops
#define FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_2(NAME, ASSIGN_TYPE)\
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, const Binary ##NAME ## Op<TLhs, TRhs, OtherDIM> &src) {\
    FASTOR_IF_CONSTEXPR (!(requires_evaluation_v<TLhs> || requires_evaluation_v<TRhs>)) {\
        trivial_assign ##ASSIGN_TYPE (dst.self(), src.self());\
    }\
    else {\
        using result_type = typename Binary ##NAME ## Op<TLhs, TRhs, OtherDIM>::result_type;\
        const result_type a(src.self());\
        trivial_assign ##ASSIGN_TYPE (dst.self(), a);\
    }\
}\

// assign
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_1(Add,     , _add)
// assign_add
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_1(Add, _add, _add)
// assign_sub
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_1(Add, _sub, _sub)
// assign_mul
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_2(Add, _mul)
// assign_div
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_2(Add, _div)

// assign
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_1(Sub,     , _sub)
// assign_add
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_1(Sub, _add, _sub)
// assign_sub
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_1(Sub, _sub, _add)
// assign_mul
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_2(Sub, _mul)
// assign_div
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_2(Sub, _div)


// assign
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_1(Mul,     , _mul)
// assign_add
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_2(Mul, _add)
// assign_sub
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_2(Mul, _sub)
// assign_mul
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_2(Mul, _mul)
// assign_div
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_2(Mul, _div)


// assign
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_1(Div,     , _div)
// assign_add
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_2(Div, _add)
// assign_sub
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_2(Div, _sub)
// assign_mul
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_2(Div, _mul)
// assign_div
FASTOR_MAKE_BINARY_ARITHMETIC_ASSIGNMENT_2(Div, _div)


} // end of namespace Fastor


#endif // BINARY_ARITHMETIC_ASSIGNMENT_H