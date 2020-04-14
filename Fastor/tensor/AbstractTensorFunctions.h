#ifndef ABSTRACT_TENSOR_FUNCTIONS_H
#define ABSTRACT_TENSOR_FUNCTIONS_H

#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/TensorTraits.h"

namespace Fastor {


template<class Derived, size_t DIMS>
FASTOR_INLINE typename tensor_type_finder<Derived>::type evaluate(const AbstractTensor<Derived,DIMS> &_src) {
    typename tensor_type_finder<Derived>::type out = _src;
    return out;
}


// These are the set of functions work on any expression without themselves being a
// Fastor expression. Note that all the mathematical functions (sin, cos etc) are
// Fastor expressions


template<class Derived, size_t DIMS>
FASTOR_INLINE typename Derived::scalar_type sum(const AbstractTensor<Derived,DIMS> &_src) {
    using T = typename Derived::scalar_type;
    using V = SIMDVector<T,DEFAULT_ABI>;
    const Derived &src = _src.self();
    FASTOR_INDEX i;
    T _scal=0; V _vec(_scal);
    for (i = 0; i < ROUND_DOWN(src.size(),V::Size); i+=V::Size) {
        _vec += src.template eval<T>(i);
    }
    for (; i < src.size(); ++i) {
        _scal += src.template eval_s<T>(i);
    }
    return _vec.sum() + _scal;
}

template<class Derived, size_t DIMS>
FASTOR_INLINE typename Derived::scalar_type product(const AbstractTensor<Derived,DIMS> &_src) {
    using T = typename Derived::scalar_type;
    using V = SIMDVector<T,DEFAULT_ABI>;
    const Derived &src = _src.self();
    FASTOR_INDEX i;
    T _scal=0; V _vec(_scal);
    for (i = 0; i < ROUND_DOWN(src.size(),V::Size); i+=V::Size) {
        _vec *= src.template eval<T>(i);
    }
    for (; i < src.size(); ++i) {
        _scal *= src.template eval_s<T>(i);
    }
    return _vec.product() * _scal;
}




template<class Derived, size_t DIMS>
FASTOR_INLINE typename Derived::scalar_type norm(const AbstractTensor<Derived,DIMS> &_src) {
    using T = typename Derived::scalar_type;
    using V = SIMDVector<T,DEFAULT_ABI>;
    const Derived &src = _src.self();
    FASTOR_INDEX i;
    T _scal=0; V _vec(_scal);
    for (i = 0; i < ROUND_DOWN(src.size(),V::Size); i+=V::Size) {
        // Evaluate the expression once
        auto eval_vec = src.template eval<T>(i);
        _vec = fmadd(eval_vec,eval_vec,_vec);
    }
    for (; i < src.size(); ++i) {
        // Evaluate the expression once
        auto eval_scal = src.template eval_s<T>(i);
        _scal += eval_scal*eval_scal;
    }
    return sqrts(_vec.sum() + _scal);
}


template<class Derived0, typename Derived1, size_t DIMS,
    typename std::enable_if<std::is_same<typename Derived0::scalar_type, typename Derived1::scalar_type>::value,bool>::type=0>
FASTOR_INLINE typename Derived0::scalar_type inner(const AbstractTensor<Derived0,DIMS> &_a, const AbstractTensor<Derived1,DIMS> &_b) {
    using T = typename Derived0::scalar_type;
    using V = SIMDVector<T,DEFAULT_ABI>;
    const Derived0 &srca = _a.self();
    const Derived1 &srcb = _b.self();
#ifdef NDEBUG
    FASTOR_ASSERT(srca.size()==srcb.size(), "EXPRESSION SIZE MISMATCH");
#endif
    FASTOR_INDEX i;
    T _scal=0; V _vec(_scal);
    for (i = 0; i < ROUND_DOWN(srca.size(),V::Size); i+=V::Size) {
        _vec = fmadd(srca.template eval<T>(i),srcb.template eval<T>(i),_vec);
    }
    for (; i < srca.size(); ++i) {
        // Evaluate the expression once
        _scal += srca.template eval_s<T>(i)*srcb.template eval_s<T>(i);
    }
    return _vec.sum() + _scal;
}



template<class Derived, size_t DIMS>
FASTOR_INLINE typename Derived::scalar_type trace(const AbstractTensor<Derived,DIMS> &_src) {
    using T = typename Derived::scalar_type;
    using tensor_type = typename tensor_type_finder<Derived>::type;
    constexpr std::array<size_t, DIMS> dims = LastMatrixExtracter<tensor_type,
        typename std_ext::make_index_sequence<DIMS>::type>::values;
    constexpr size_t M = dims[DIMS-2];
    constexpr size_t N = dims[DIMS-1];
    static_assert(DIMS==2,"TENSOR EXPRESSION SHOULD BE UNIFORM (SQUARE)");
    static_assert(M==N,"TENSOR EXPRESSION SHOULD BE TWO DIMENSIONAL");

    const Derived &src = _src.self();
    FASTOR_INDEX i;
    T _scal=0;
    for (i = 0; i < M; ++i) {
        _scal += src.template eval_s<T>(i*(N+1));
    }
    return _scal;
}


// Generic matmul function for AbstractTensor types
// Works as long as the return tensor is compile deducible
// Their applicability on dynamic views should be checked
template<typename Derived0, size_t DIM0, typename Derived1, size_t DIM1,
typename std::enable_if<DIM0==2 && DIM1==2,bool>::type=0>
FASTOR_INLINE
Tensor<typename scalar_type_finder<Derived0>::type,
    get_tensor_dimensions<typename tensor_type_finder<Derived0>::type>::dims[0],
    get_tensor_dimensions<typename tensor_type_finder<Derived1>::type>::dims[1]>
matmul(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b) {

    using T = typename scalar_type_finder<Derived0>::type;
    const Derived0 &a_src = a.self();
    const Derived1 &b_src = b.self();

    // size_t M = a_src.dimension(0);
    // size_t K = a_src.dimension(1);
    // size_t N = b_src.dimension(1);
    constexpr size_t M = get_tensor_dimensions<typename tensor_type_finder<Derived0>::type>::dims[0];
    constexpr size_t K = get_tensor_dimensions<typename tensor_type_finder<Derived0>::type>::dims[1];
    constexpr size_t N = get_tensor_dimensions<typename tensor_type_finder<Derived1>::type>::dims[1];

    // We cannot choose the best simd because we cannot mix simd types of an expression
    using V = SIMDVector<T,DEFAULT_ABI>;
    // using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;
    constexpr size_t SIZE_ = V::Size;
    int ROUND = ROUND_DOWN(N,(int)SIZE_);

    Tensor<typename scalar_type_finder<Derived0>::type,
    get_tensor_dimensions<typename tensor_type_finder<Derived0>::type>::dims[0],
    get_tensor_dimensions<typename tensor_type_finder<Derived1>::type>::dims[1]> out;
    T *out_data = out.data();

    for (size_t j=0; j<M; ++j) {
        size_t k=0;
        for (; k<(size_t)ROUND; k+=SIZE_) {
            V out_row, vec_a;
            for (size_t i=0; i<K; ++i) {
                // V brow; brow.load(&b[i*N+k],false);
                V brow = b_src.template eval<T>(i*N+k);
                vec_a.set(a_src.template eval_s<T>(j*K+i));
                out_row = fmadd(vec_a,brow,out_row);
            }
            out_row.store(out_data+k+N*j,false);
        }

        for (; k<N; k++) {
            T out_row = 0.;
            for (size_t i=0; i<K; ++i) {
                // out_row += a[j*K+i]*b[i*N+k];
                out_row += a_src.template eval_s<T>(j*K+i)*b_src.template eval_s<T>(i*N+k);
            }
            out_data[N*j+k] = out_row;
        }
    }

    return out;
}




template<class Derived, size_t DIMS>
FASTOR_INLINE bool all_of(const AbstractTensor<Derived,DIMS> &_src) {
    static_assert(internal::is_binary_cmp_op<Derived>::value,"INPUT SHOULD BE A BOOLEAN EXPRESSION");
    const Derived &src = _src.self();
    bool val = true;
    for (FASTOR_INDEX i = 0; i < src.size(); ++i) {
        if (src.template eval_s<bool>(i) == false) {
            val = false;
            break;
        }
    }
    return val;
}

template<class Derived, size_t DIMS>
FASTOR_INLINE bool any_of(const AbstractTensor<Derived,DIMS> &_src) {
    static_assert(internal::is_binary_cmp_op<Derived>::value,"INPUT SHOULD BE A BOOLEAN EXPRESSION");
    const Derived &src = _src.self();
    bool val = false;
    for (FASTOR_INDEX i = 0; i < src.size(); ++i) {
        if (src.template eval_s<bool>(i) == true) {
            val = true;
            break;
        }
    }
    return val;
}

template<class Derived, size_t DIMS>
FASTOR_INLINE bool none_of(const AbstractTensor<Derived,DIMS> &_src) {
    static_assert(internal::is_binary_cmp_op<Derived>::value,"INPUT SHOULD BE A BOOLEAN EXPRESSION");
    const Derived &src = _src.self();
    bool val = false;
    for (FASTOR_INDEX i = 0; i < src.size(); ++i) {
        if (src.template eval_s<bool>(i) == true) {
            val = true;
            break;
        }
    }
    return val;
}

}


#endif // #ifndef TENSOR_FUNCTIONS_H
