#ifndef TENSOR_VIEWS_ASSIGNMENT_H
#define TENSOR_VIEWS_ASSIGNMENT_H


#include "Fastor/expressions/views/tensor_views.h"
#include "Fastor/expressions/views/tensor_fixed_views_1d.h"
#include "Fastor/expressions/views/tensor_fixed_views_2d.h"
#include "Fastor/expressions/views/tensor_random_views.h"


namespace Fastor {

#define FASTOR_MAKE_ALL_TENSOR_VIEWS_ASSIGNMENT(ASSIGN_TYPE)\
template<typename Derived, size_t DIM, typename TensorType, typename Seq>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, TensorConstFixedViewExpr1D<TensorType,Seq,1> src) {\
    trivial_assign ##ASSIGN_TYPE (dst.self(), src.self());\
}\
template<typename Derived, size_t DIM, typename TensorType, typename Seq>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, TensorFixedViewExpr1D<TensorType,Seq,1> src) {\
    trivial_assign ##ASSIGN_TYPE (dst.self(), src.self());\
}\
template<typename Derived, size_t DIM, typename TensorType, typename Seq0, typename Seq1>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, TensorFixedViewExpr2D<TensorType,Seq0,Seq1,2> src) {\
    trivial_assign ##ASSIGN_TYPE (dst.self(), src.self());\
}\
template<typename Derived, size_t DIM, typename TensorType, typename Seq0, typename Seq1>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, TensorConstFixedViewExpr2D<TensorType,Seq0,Seq1,2> src) {\
    trivial_assign ##ASSIGN_TYPE (dst.self(), src.self());\
}\
template<typename Derived, size_t DIM, typename TensorType, size_t OtherDIM>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, TensorConstViewExpr<TensorType,OtherDIM> src) {\
    trivial_assign ##ASSIGN_TYPE (dst.self(), src.self());\
}\
template<typename Derived, size_t DIM, typename TensorType, size_t OtherDIM>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, TensorViewExpr<TensorType,OtherDIM> src) {\
    trivial_assign ##ASSIGN_TYPE (dst.self(), src.self());\
}\
template<typename Derived, size_t DIM, typename TensorType, typename TensorIndexType>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, TensorConstRandomViewExpr<TensorType,TensorIndexType,DIM> src) {\
    trivial_assign ##ASSIGN_TYPE (dst.self(), src.self());\
}\
template<typename Derived, size_t DIM, typename TensorType, typename TensorIndexType>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, TensorRandomViewExpr<TensorType,TensorIndexType,DIM> src) {\
    trivial_assign ##ASSIGN_TYPE (dst.self(), src.self());\
}\

FASTOR_MAKE_ALL_TENSOR_VIEWS_ASSIGNMENT(    )
FASTOR_MAKE_ALL_TENSOR_VIEWS_ASSIGNMENT(_add)
FASTOR_MAKE_ALL_TENSOR_VIEWS_ASSIGNMENT(_sub)
FASTOR_MAKE_ALL_TENSOR_VIEWS_ASSIGNMENT(_mul)
FASTOR_MAKE_ALL_TENSOR_VIEWS_ASSIGNMENT(_div)

} // end of namespace Fastor


#endif // TENSOR_VIEWS_ASSIGNMENT_H