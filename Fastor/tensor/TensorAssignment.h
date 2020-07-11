#ifndef TENSOR_ASSIGNMENT_H
#define TENSOR_ASSIGNMENT_H

namespace Fastor {

//----------------------------------------------------------------------------------------------------------//
//----------------------------------------------------------------------------------------------------------//
template<typename Derived, size_t DIM, typename OtherDerived, size_t OtherDIM>
FASTOR_INLINE void trivial_assign(AbstractTensor<Derived,DIM> &dst, const AbstractTensor<OtherDerived,OtherDIM> &src_) {
    using T = typename Derived::scalar_type;
    using V = typename Derived::simd_vector_type;
    const OtherDerived &src = src_.self();
    FASTOR_ASSERT(src.size()==dst.self().size(), "TENSOR SIZE MISMATCH");
    T* _data = dst.self().data();

    FASTOR_IF_CONSTEXPR(!is_boolean_expression_v<OtherDerived>) {
        FASTOR_INDEX i = 0;
        for (; i <ROUND_DOWN(src.size(),V::Size); i+=V::Size) {
            src.template eval<T>(i).store(&_data[i], FASTOR_ALIGNED);
        }
        for (; i < src.size(); ++i) {
            _data[i] = src.template eval_s<T>(i);
        }
    }
    else {
        for (FASTOR_INDEX i = 0; i < src.size(); ++i) {
            _data[i] = src.template eval_s<T>(i);
        }
    }
}

template<typename Derived, size_t DIM, typename OtherDerived, size_t OtherDIM>
FASTOR_INLINE void trivial_assign_add(AbstractTensor<Derived,DIM> &dst, const AbstractTensor<OtherDerived,OtherDIM> &src_) {
    using T = typename Derived::scalar_type;
    using V = typename Derived::simd_vector_type;
    const OtherDerived &src = src_.self();
    FASTOR_ASSERT(src.size()==dst.self().size(), "TENSOR SIZE MISMATCH");
    T* _data = dst.self().data();

    FASTOR_IF_CONSTEXPR(!is_boolean_expression_v<OtherDerived>) {
        FASTOR_INDEX i = 0;
        for (; i <ROUND_DOWN(src.size(),V::Size); i+=V::Size) {
            V _vec = V(&_data[i], FASTOR_ALIGNED) + src.template eval<T>(i);
            _vec.store(&_data[i], FASTOR_ALIGNED);
        }
        for (; i < src.size(); ++i) {
            _data[i] += src.template eval_s<T>(i);
        }
    }
    else {
        for (FASTOR_INDEX i = 0; i < src.size(); ++i) {
            _data[i] += src.template eval_s<T>(i);
        }
    }
}

template<typename Derived, size_t DIM, typename OtherDerived, size_t OtherDIM>
FASTOR_INLINE void trivial_assign_sub(AbstractTensor<Derived,DIM> &dst, const AbstractTensor<OtherDerived,OtherDIM> &src_) {
    using T = typename Derived::scalar_type;
    using V = typename Derived::simd_vector_type;
    const OtherDerived &src = src_.self();
    FASTOR_ASSERT(src.size()==dst.self().size(), "TENSOR SIZE MISMATCH");
    T* _data = dst.self().data();

    FASTOR_IF_CONSTEXPR(!is_boolean_expression_v<OtherDerived>) {
        FASTOR_INDEX i = 0;
        for (; i <ROUND_DOWN(src.size(),V::Size); i+=V::Size) {
            V _vec = V(&_data[i], FASTOR_ALIGNED) - src.template eval<T>(i);
            _vec.store(&_data[i], FASTOR_ALIGNED);
        }
        for (; i < src.size(); ++i) {
            _data[i] -= src.template eval_s<T>(i);
        }
    }
    else {
        for (FASTOR_INDEX i = 0; i < src.size(); ++i) {
            _data[i] -= src.template eval_s<T>(i);
        }
    }
}

template<typename Derived, size_t DIM, typename OtherDerived, size_t OtherDIM>
FASTOR_INLINE void trivial_assign_mul(AbstractTensor<Derived,DIM> &dst, const AbstractTensor<OtherDerived,OtherDIM> &src_) {
    using T = typename Derived::scalar_type;
    using V = typename Derived::simd_vector_type;
    const OtherDerived &src = src_.self();
    FASTOR_ASSERT(src.size()==dst.self().size(), "TENSOR SIZE MISMATCH");
    T* _data = dst.self().data();

    FASTOR_IF_CONSTEXPR(!is_boolean_expression_v<OtherDerived>) {
        FASTOR_INDEX i = 0;
        for (; i <ROUND_DOWN(src.size(),V::Size); i+=V::Size) {
            V _vec = V(&_data[i], FASTOR_ALIGNED) * src.template eval<T>(i);
            _vec.store(&_data[i], FASTOR_ALIGNED);
        }
        for (; i < src.size(); ++i) {
            _data[i] *= src.template eval_s<T>(i);
        }
    }
    else {
        for (FASTOR_INDEX i = 0; i < src.size(); ++i) {
            _data[i] *= src.template eval_s<T>(i);
        }
    }
}

template<typename Derived, size_t DIM, typename OtherDerived, size_t OtherDIM>
FASTOR_INLINE void trivial_assign_div(AbstractTensor<Derived,DIM> &dst, const AbstractTensor<OtherDerived,OtherDIM> &src_) {
    using T = typename Derived::scalar_type;
    using V = typename Derived::simd_vector_type;
    const OtherDerived &src = src_.self();
    FASTOR_ASSERT(src.size()==dst.self().size(), "TENSOR SIZE MISMATCH");
    T* _data = dst.self().data();

    FASTOR_IF_CONSTEXPR(!is_boolean_expression_v<OtherDerived>) {
        FASTOR_INDEX i = 0;
        for (; i <ROUND_DOWN(src.size(),V::Size); i+=V::Size) {
            V _vec = V(&_data[i], FASTOR_ALIGNED) / src.template eval<T>(i);
            _vec.store(&_data[i], FASTOR_ALIGNED);
        }
        for (; i < src.size(); ++i) {
            _data[i] /= src.template eval_s<T>(i);
        }
    }
    else {
        for (FASTOR_INDEX i = 0; i < src.size(); ++i) {
            _data[i] /= src.template eval_s<T>(i);
        }
    }
}
//----------------------------------------------------------------------------------------------------------//

//----------------------------------------------------------------------------------------------------------//
template<typename Derived, size_t DIM, typename U,
    enable_if_t_<is_primitive_v_<U>, bool> = false>
FASTOR_INLINE void trivial_assign(AbstractTensor<Derived,DIM> &dst, U num) {
    using T = typename Derived::scalar_type;
    using V = typename Derived::simd_vector_type;
    T* _data = dst.self().data();
    T cnum = (T)num;
    V _vec(cnum);
    FASTOR_INDEX i = 0;
    for (; i< ROUND_DOWN(dst.self().size(),V::Size); i+=V::Size) {
        _vec.store(&_data[i], FASTOR_ALIGNED);
    }
    for (; i<dst.self().size(); ++i) {
        _data[i] = cnum;
    }
}

template<typename Derived, size_t DIM, typename U,
    enable_if_t_<is_primitive_v_<U>, bool> = false>
FASTOR_INLINE void trivial_assign_add(AbstractTensor<Derived,DIM> &dst, U num) {
    using T = typename Derived::scalar_type;
    using V = typename Derived::simd_vector_type;
    T* _data = dst.self().data();
    T cnum = (T)num;
    V _vec(cnum);
    FASTOR_INDEX i = 0;
    for (; i< ROUND_DOWN(dst.self().size(),V::Size); i+=V::Size) {
        V _vec_out(&_data[i], FASTOR_ALIGNED);
        _vec_out += _vec;
        _vec_out.store(&_data[i], FASTOR_ALIGNED);
    }
    for (; i<dst.self().size(); ++i) {
        _data[i] += cnum;
    }
}

template<typename Derived, size_t DIM, typename U,
    enable_if_t_<is_primitive_v_<U>, bool> = false>
FASTOR_INLINE void trivial_assign_sub(AbstractTensor<Derived,DIM> &dst, U num) {
    using T = typename Derived::scalar_type;
    using V = typename Derived::simd_vector_type;
    T* _data = dst.self().data();
    T cnum = (T)num;
    V _vec(cnum);
    FASTOR_INDEX i = 0;
    for (; i< ROUND_DOWN(dst.self().size(),V::Size); i+=V::Size) {
        V _vec_out(&_data[i], FASTOR_ALIGNED);
        _vec_out -= _vec;
        _vec_out.store(&_data[i], FASTOR_ALIGNED);
    }
    for (; i<dst.self().size(); ++i) {
        _data[i] -= cnum;
    }
}

template<typename Derived, size_t DIM, typename U,
    enable_if_t_<is_primitive_v_<U>, bool> = false>
FASTOR_INLINE void trivial_assign_mul(AbstractTensor<Derived,DIM> &dst, U num) {
    using T = typename Derived::scalar_type;
    using V = typename Derived::simd_vector_type;
    T* _data = dst.self().data();
    T cnum = (T)num;
    V _vec(cnum);
    FASTOR_INDEX i = 0;
    for (; i< ROUND_DOWN(dst.self().size(),V::Size); i+=V::Size) {
        V _vec_out(&_data[i], FASTOR_ALIGNED);
        _vec_out *= _vec;
        _vec_out.store(&_data[i], FASTOR_ALIGNED);
    }
    for (; i<dst.self().size(); ++i) {
        _data[i] *= cnum;
    }
}

template<typename Derived, size_t DIM, typename U,
    enable_if_t_<is_primitive_v_<U> && !is_integral_v_<U>, bool> = false>
FASTOR_INLINE void trivial_assign_div(AbstractTensor<Derived,DIM> &dst, U num) {
    using T = typename Derived::scalar_type;
    using V = typename Derived::simd_vector_type;
    T* _data = dst.self().data();
    T cnum = T(1) / (T)num;
    V _vec(cnum);
    FASTOR_INDEX i = 0;
    for (; i< ROUND_DOWN(dst.self().size(),V::Size); i+=V::Size) {
        V _vec_out(&_data[i], FASTOR_ALIGNED);
        _vec_out *= _vec;
        _vec_out.store(&_data[i], FASTOR_ALIGNED);
    }
    for (; i<dst.self().size(); ++i) {
        _data[i] *= cnum;
    }
}
template<typename Derived, size_t DIM, typename U,
    enable_if_t_<is_primitive_v_<U> && is_integral_v_<U>, bool> = false>
FASTOR_INLINE void trivial_assign_div(AbstractTensor<Derived,DIM> &dst, U num) {
    using T = typename Derived::scalar_type;
    using V = typename Derived::simd_vector_type;
    T* _data = dst.self().data();
    T cnum = (T)num;
    V _vec(cnum);
    FASTOR_INDEX i = 0;
    for (; i< ROUND_DOWN(dst.self().size(),V::Size); i+=V::Size) {
        V _vec_out(&_data[i], FASTOR_ALIGNED);
        _vec_out /= _vec;
        _vec_out.store(&_data[i], FASTOR_ALIGNED);
    }
    for (; i<dst.self().size(); ++i) {
        _data[i] /= cnum;
    }
}


//----------------------------------------------------------------------------------------------------------//
template<typename Derived, size_t DIM, typename T, size_t ...Rest>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const Tensor<T,Rest...> &src) {
    if (dst.self().data()==src.data()) return;
    trivial_assign(dst.self(),src);
}
template<typename Derived, size_t DIM, typename T, size_t ...Rest>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const Tensor<T,Rest...> &src) {
    trivial_assign_add(dst.self(),src);
}
template<typename Derived, size_t DIM, typename T, size_t ...Rest>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const Tensor<T,Rest...> &src) {
    trivial_assign_sub(dst.self(),src);
}
template<typename Derived, size_t DIM, typename T, size_t ...Rest>
FASTOR_INLINE void assign_mul(AbstractTensor<Derived,DIM> &dst, const Tensor<T,Rest...> &src) {
    trivial_assign_mul(dst.self(),src);
}
template<typename Derived, size_t DIM, typename T, size_t ...Rest>
FASTOR_INLINE void assign_div(AbstractTensor<Derived,DIM> &dst, const Tensor<T,Rest...> &src) {
    trivial_assign_div(dst.self(),src);
}

template<typename Derived, size_t DIM, typename U,
    enable_if_t_<is_primitive_v_<U>,bool> = false>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, U num) {
    trivial_assign(dst.self(),num);
}
template<typename Derived, size_t DIM, typename U,
    enable_if_t_<is_primitive_v_<U>,bool> = false>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, U num) {
    trivial_assign_add(dst.self(),num);
}
template<typename Derived, size_t DIM, typename U,
    enable_if_t_<is_primitive_v_<U>,bool> = false>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, U num) {
    trivial_assign_sub(dst.self(),num);
}
template<typename Derived, size_t DIM, typename U,
    enable_if_t_<is_primitive_v_<U>,bool> = false>
FASTOR_INLINE void assign_mul(AbstractTensor<Derived,DIM> &dst, U num) {
    trivial_assign_mul(dst.self(),num);
}
template<typename Derived, size_t DIM, typename U,
    enable_if_t_<is_primitive_v_<U>,bool> = false>
FASTOR_INLINE void assign_div(AbstractTensor<Derived,DIM> &dst, U num) {
    trivial_assign_div(dst.self(),num);
}
//----------------------------------------------------------------------------------------------------------//
//----------------------------------------------------------------------------------------------------------//

} // end of namespace Fastor

#endif // TENSOR_ASSIGNMENT_H
