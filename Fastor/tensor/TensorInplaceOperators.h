#ifndef TENSOR_INPLACE_OPERATORS_H
#define TENSOR_INPLACE_OPERATORS_H

// CRTP Overloads for nth rank Tensors
//---------------------------------------------------------------------------------------------//
template<typename Derived, size_t DIMS>
FASTOR_INLINE void operator +=(const AbstractTensor<Derived,DIMS>& src_) {
    assign_add(*this, src_.self());
}

template<typename Derived, size_t DIMS>
FASTOR_INLINE void operator -=(const AbstractTensor<Derived,DIMS>& src_) {
    assign_sub(*this, src_.self());
}

template<typename Derived, size_t DIMS>
FASTOR_INLINE void operator *=(const AbstractTensor<Derived,DIMS>& src_) {
    assign_mul(*this, src_.self());
}

template<typename Derived, size_t DIMS>
FASTOR_INLINE void operator /=(const AbstractTensor<Derived,DIMS>& src_) {
    assign_div(*this, src_.self());
}
//---------------------------------------------------------------------------------------------//

// Scalar overloads for in-place operators
//---------------------------------------------------------------------------------------------//
template<typename U=T, enable_if_t_<is_arithmetic_v_<U>,bool> = 0 >
FASTOR_INLINE void operator +=(U num) {
    trivial_assign_add(*this, num);
}

template<typename U=T, enable_if_t_<is_arithmetic_v_<U>,bool> = 0 >
FASTOR_INLINE void operator -=(U num) {
    trivial_assign_sub(*this, num);
}

template<typename U=T, enable_if_t_<is_arithmetic_v_<U>,bool> = 0 >
FASTOR_INLINE void operator *=(U num) {
    trivial_assign_mul(*this, num);
}

template<typename U=T, enable_if_t_<is_arithmetic_v_<U>,bool> = 0 >
FASTOR_INLINE void operator /=(U num) {
    trivial_assign_div(*this, num);
}
//---------------------------------------------------------------------------------------------//

#endif // TENSOR_INPLACE_OPERATORS_H
