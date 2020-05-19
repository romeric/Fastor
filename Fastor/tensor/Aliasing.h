#ifndef ALIASING_H_
#define ALIASING_H_

#include "Fastor/tensor/ForwardDeclare.h"
#include "Fastor/tensor/Tensor.h"

namespace Fastor {

// template<typename T, size_t ...Rest0, size_t ... Rest1>
// FASTOR_INLINE bool does_alias(const Tensor<T,Rest0...> &dst, const Tensor<T,Rest1...> &src) {
//     return dst.data() == src.data() ? true : false;
// }
template<typename Derived, size_t DIM, typename T, size_t ... Rest>
FASTOR_INLINE bool does_alias(const AbstractTensor<Derived,DIM> &dst, const Tensor<T,Rest...> &src) {
    return dst.self().data() == src.data() ? true : false;
}
// template<typename Derived, size_t DIM, typename OtherDerived, size_t OtherDIM>
// FASTOR_INLINE bool does_alias(const AbstractTensor<Derived,DIM> &dst, const AbstractTensor<OtherDerived,OtherDIM> &src) {
//     return does_alias(dst.self(),src.self());
// }


#define FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(NAME)\
template<typename Derived, size_t DIM, typename OtherDerived, size_t OtherDIM>\
FASTOR_INLINE bool does_alias(const AbstractTensor<Derived,DIM> &dst, const Unary ##NAME ## Op<OtherDerived,OtherDIM> &src) {\
    return does_alias(dst.self(),src.expr().self());\
}\

FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Add )
FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Sub )
FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Abs )
FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Sqrt)
FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Exp )
FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Log )
FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Sin )
FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Cos )
FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Tan )
FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Asin)
FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Acos)
FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Atan)
FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Sinh)
FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Cosh)
FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Tanh)


// FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Det  )
// FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Norm)
// FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Trace)
FASTOR_MAKE_ALIAS_FUNC_UNARY_OPS(Trans)


#define FASTOR_MAKE_ALIAS_FUNC_BINARY_OPS(NAME)\
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM>\
FASTOR_INLINE bool does_alias(const AbstractTensor<Derived,DIM> &dst, const Binary ##NAME ## Op<TLhs,TRhs,OtherDIM> &src) {\
    return does_alias(dst.self(),src.lhs().self()) || does_alias(dst.self(),src.rhs().self());\
}\

FASTOR_MAKE_ALIAS_FUNC_BINARY_OPS(Add)
FASTOR_MAKE_ALIAS_FUNC_BINARY_OPS(Sub)
FASTOR_MAKE_ALIAS_FUNC_BINARY_OPS(Mul)
FASTOR_MAKE_ALIAS_FUNC_BINARY_OPS(Div)
FASTOR_MAKE_ALIAS_FUNC_BINARY_OPS(MatMul)

}


#endif // ALIASING_H_
