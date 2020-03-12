#ifndef REDUCTION_H
#define REDUCTION_H

#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor_algebra/indicial.h"
#include "Fastor/tensor_algebra/contraction.h"

namespace Fastor {


// reduction
//----------------------------------------------------------------------------------------------
template<typename T, size_t ... Rest>
T reduction(const Tensor<T,Rest...> &a) {
    //! Reduces a multi-dimensional tensor to a scalar
    //!
    //! If a is scalar/Tensor<T> returns the value itself
    //! If a is a vector Tensor<T,N> returns the sum of values
    //! If a is a second order tensor Tensor<T,N,N> returns the trace
    //! If a is a third order tensor Tensor<T,N,N,N> returns a_iii
    //! ...
    //!
    //! The size of the tensor in all dimensions should be equal (uniform)

    static_assert(no_of_unique<Rest...>::value<=1, "REDUCTION IS ONLY POSSIBLE ON UNIFORM TENSORS");
    constexpr int ndim = sizeof...(Rest);

    T *a_data = a.data();
    if (ndim==0) {
        return a_data[0];
    }
    else if (ndim==1) {
        return a.sum();
    }
    else {
        constexpr std::array<size_t,ndim> products = nprods<Index<Rest...>,
                typename std_ext::make_index_sequence<ndim>::type>::values;

        T reductor = static_cast<T>(0);
        for (size_t i=0; i<a.dimension(0); ++i) {
            size_t index_a = i;
            for(size_t it = 0; it< ndim; it++) {
                index_a += products[it]*i;
            }
            reductor += a_data[index_a];
        }
        return reductor;
    }
}



template<typename T, size_t ... Rest>
FASTOR_INLINE T inner(const Tensor<T,Rest...> &a) {
    //! Reduces a multi-dimensional tensor to a scalar
    //!
    //! If a is scalar/Tensor<T> returns the value itself
    //! If a is a vector Tensor<T,N> returns the sum of values
    //! If a is a second order tensor Tensor<T,N,N> returns the trace
    //! If a is a third order tensor Tensor<T,N,N,N> returns a_iii
    //! ...
    //!
    //! The size of the tensor in all dimensions should be equal (uniform)
    return reduction(a);
}


template<typename T, size_t ... Rest>
FASTOR_INLINE T inner(const Tensor<T,Rest...> &a, const Tensor<T,Rest...> &b) {
    //! Reduction of a tensor pair to a scalar, for instance A_ijklm * B_ijklm
    //! If a and b are scalars/vectors, returns dot product
    //! If a and b are matrices, returns double contraction
    //! For third order tensors returns a_ijk*b_ijk
    //! ...

    const T *a_data = a.data();
    const T *b_data = b.data();

    constexpr int ndim = sizeof...(Rest);
    if (ndim>0) {
        return _doublecontract<T,prod<Rest...>::value,1>(a_data,b_data);
    }
    else {
        return a_data[0]*b_data[0];
    }
}

#ifdef __SSE4_2__
// Specialisation for inner product of small vectors.
// This is similar to _doublecontract, but _doublecontract
// is specialised for 2nd order tensors
template<>
FASTOR_INLINE float inner<float,2>(const Tensor<float,2> &a, const Tensor<float,2> &b) {
    // 9 OPS
    const float *a_data = a.data();
    const float *b_data = b.data();
    __m128 va = {a_data[0],a_data[1],0,0};
    __m128 vb = {b_data[0],b_data[1],0,0};
    __m128 vc = _mm_mul_ps(va,vb);
    return _mm_cvtss_f32(_mm_add_ss(vc,_mm_shuffle_ps(vc,vc,0x1)));
}
template<>
FASTOR_INLINE float inner<float,3>(const Tensor<float,3> &a, const Tensor<float,3> &b) {
    // 13 OPS
    const float *a_data = a.data();
    const float *b_data = b.data();
    __m128 va = {a_data[0],a_data[1],a_data[2],0};
    __m128 vb = {b_data[0],b_data[1],b_data[2],0};
    __m128 vc = _mm_mul_ps(va,vb);
    return _mm_sum_ps(vc);
}
template<>
FASTOR_INLINE float inner<float,4>(const Tensor<float,4> &a, const Tensor<float,4> &b) {
    // 13 OPS
    return _mm_sum_ps(_mm_mul_ps(_mm_load_ps(a.data()),_mm_load_ps(b.data())));
}
template<>
FASTOR_INLINE double inner<double,2>(const Tensor<double,2> &a, const Tensor<double,2> &b) {
    // 9 OPS
    return _mm_sum_pd(_mm_mul_pd(_mm_load_pd(a.data()),_mm_load_pd(b.data())));
}
#endif
#ifdef __AVX__
template<>
FASTOR_INLINE double inner<double,3>(const Tensor<double,3> &a, const Tensor<double,3> &b) {
    // IVY 13 OPS - HW - 15 OPS
    return _mm256_sum_pd(_mm256_mul_pd(_mm256_loadl3_pd(a.data()),_mm256_loadl3_pd(b.data())));
}
template<>
FASTOR_INLINE double inner<double,4>(const Tensor<double,4> &a, const Tensor<double,4> &b) {
    // IVY 13 OPS - HW - 15 OPS
    return _mm256_sum_pd(_mm256_mul_pd(_mm256_load_pd(a.data()),_mm256_load_pd(b.data())));
}
#endif


template<typename T, size_t M, size_t N>
FASTOR_INLINE T inner(const Tensor<T,M,N> &a, const Tensor<T,M,N> &b) {
    //! Reduction of a tensor pair to a scalar (double contraction specilisation)
    return _doublecontract<T,M,N>(a.data(),b.data());
}


template<typename T, size_t ... Rest>
T inner(const Tensor<T,Rest...> &a, const Tensor<T,Rest...> &b, const Tensor<T,Rest...> &c) {
    //! No Einstein summation in here as indices get repeated more than twice.
    //! This is just reducing three tensors to a scalar like a[i]*b[i]*c[i]

    using V = SIMDVector<T,DEFAULT_ABI>;

    const T *a_data = a.data();
    const T *b_data = b.data();
    const T *c_data = c.data();

    constexpr int size = prod<Rest...>::value;
    constexpr int unroll_upto = V::unroll_size(size);
    constexpr int stride = V::Size;
    int i = 0;

    V vec_a=static_cast<T>(0), vec_b=static_cast<T>(0), vec_c=static_cast<T>(0), vec_out=static_cast<T>(0);
    for (; i< unroll_upto; i+=stride) {
        vec_a.load(a_data+i);
        vec_b.load(b_data+i);
        vec_c.load(c_data+i);
        vec_out += vec_a*vec_b*vec_c;
    }
    T scalar = static_cast<T>(0);
    for (int j=i; j< size; j++) {
        scalar += a_data[j]*b_data[j]*c_data[j];
    }
    return vec_out.sum() + scalar;
}


template<typename T, size_t ... Rest>
T inner(const Tensor<T,Rest...> &a, const Tensor<T,Rest...> &b, const Tensor<T,Rest...> &c, const Tensor<T,Rest...> &d) {
    //! No Einstein summation in here as indices get repeated more than twice.
    //! This is just reducing four tensors to a scalar like a[i]*b[i]*c[i]*d[i]

    using V = SIMDVector<T,DEFAULT_ABI>;

    const T *a_data = a.data();
    const T *b_data = b.data();
    const T *c_data = c.data();
    const T *d_data = d.data();

    constexpr int size = prod<Rest...>::value;
    constexpr int unroll_upto = V::unroll_size(size);
    constexpr int stride = V::Size;
    int i = 0;

    V vec_a=static_cast<T>(0), vec_b=static_cast<T>(0);
    V vec_c=static_cast<T>(0), vec_d=static_cast<T>(0), vec_out=static_cast<T>(0);
    for (; i< unroll_upto; i+=stride) {
        vec_a.load(a_data+i);
        vec_b.load(b_data+i);
        vec_c.load(c_data+i);
        vec_d.load(d_data+i);
        vec_out += vec_a*vec_b*vec_c*vec_d;
    }
    T scalar = static_cast<T>(0);
    for (int j=i; j< size; j++) {
        scalar += a_data[j]*b_data[j]*c_data[j]*d_data[j];
    }
    return vec_out.sum() + scalar;
}


}

#endif // REDUCTION_H

