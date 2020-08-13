#ifndef TENSOR_METHODS_NONCONST_H
#define TENSOR_METHODS_NONCONST_H

FASTOR_INLINE void fill(T num0) {
    FASTOR_INDEX i = 0UL;
    using V = simd_vector_type;
    V _vec(num0);
    for (; i<ROUND_DOWN(size(),V::Size); i+=V::Size) {
        _vec.store(&_data[i],false);
    }
    for (; i<size(); ++i) _data[i] = num0;
}

FASTOR_INLINE void iota(T num0=0) {
    iota_impl(_data, &_data[size()], num0);
}

FASTOR_INLINE void arange(T num0=0) {
    iota_impl(_data, &_data[size()], num0);
    // T num = static_cast<T>(num0);
    // using V = SIMDVector<T,simd_abi_type>;
    // V _vec;
    // FASTOR_INDEX i=0;
    // for (; i<ROUND_DOWN(size(),V::Size); i+=V::Size) {
    //     _vec.set_sequential(T(i)+num);
    //     _vec.store(&_data[i],false);
    // }
    // for (; i<size(); ++i) _data[i] = T(i)+num;
}

FASTOR_INLINE void zeros() {
    using V = simd_vector_type;
    V _zeros;
    FASTOR_INDEX i=0;
    for (; i<ROUND_DOWN(size(),V::Size); i+=V::Size) {
        _zeros.store(&_data[i],false);
    }
    for (; i<size(); ++i) _data[i] = 0;
}

FASTOR_INLINE void ones() {
    this->fill(static_cast<T>(1));
}

FASTOR_INLINE void eye2() {
    // Second order identity tensor (identity matrices)
    static_assert(sizeof...(Rest)==2, "CANNOT BUILD AN IDENTITY TENSOR");
    static_assert(no_of_unique<Rest...>::value==1, "TENSOR MUST BE UNIFORM");
    constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
    zeros();
    for (FASTOR_INDEX i=0; i<N; ++i) {
        _data[i*N+i] = (T)1;
    }
}

FASTOR_INLINE void eye() {
    // Arbitrary order identity tensor
    static_assert(sizeof...(Rest)>=2, "CANNOT BUILD AN IDENTITY TENSOR");
    static_assert(no_of_unique<Rest...>::value==1, "TENSOR MUST BE UNIFORM");
    zeros();

    constexpr int ndim = sizeof...(Rest);
    constexpr std::array<int,ndim> maxes_a = {Rest...};
    std::array<int,ndim> products;
    std::fill(products.begin(),products.end(),0);

    for (int j=ndim-1; j>0; --j) {
        int num = maxes_a[ndim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_a[ndim-1-k-1];
        }
        products[j] = num;
    }
    std::reverse(products.begin(),products.end());

    for (FASTOR_INDEX i=0; i<dimension(0); ++i) {
        int index_a = i;
        for(int it = 0; it< ndim; it++) {
            index_a += products[it]*i;
        }
        _data[index_a] = static_cast<T>(1);
    }
}

FASTOR_INLINE void random() {
    //! Populate tensor with random FP numbers
    for (FASTOR_INDEX i=0; i<size(); ++i) {
        _data[get_mem_index(i)] = (T)rand()/RAND_MAX;
    }
}

FASTOR_INLINE void randint() {
    //! Populate tensor with random integer numbers
    for (FASTOR_INDEX i=0; i<size(); ++i) {
        _data[get_mem_index(i)] = (T)rand();
    }
}

FASTOR_INLINE void reverse() {
    // in-place reverse
    if ((size()==0) || (size()==1)) return;
    // std::reverse(_data,_data+Size); return;

    // This requires copying the data to avoid aliasing
    // Despite that this method seems to be faster than
    // std::reverse for big _data both on GCC and Clang
    FASTOR_ARCH_ALIGN T tmp[size()];
    std::copy(_data,_data+size(),tmp);

    // Although SSE register reversing is faster
    // The AVX one outperforms it
    using V = SIMDVector<T,simd_abi_type>;
    V vec;
    FASTOR_INDEX i = 0;
    for (; i< ROUND_DOWN(size(),V::Size); i+=V::Size) {
        vec.load(&tmp[size() - i - V::Size],false);
        vec.reverse().store(&_data[i],false);
    }
    for (; i< size(); ++i) {
        _data[i] = tmp[size()-i-1];
    }
}

#endif // TENSOR_METHODS_NONCONST_H


#ifndef TENSOR_METHODS_CONST_H
#define TENSOR_METHODS_CONST_H

FASTOR_INLINE T sum() const {

    if ((size()==0) || (size()==1)) return _data[0];
    using V = SIMDVector<T,simd_abi_type>;
    V vec = static_cast<T>(0);
    V _vec_in;
    FASTOR_INDEX i = 0;
    for (; i<ROUND_DOWN(size(),V::Size); i+=V::Size) {
        _vec_in.load(&_data[i],false);
        vec += _vec_in;
    }
    T scalar = static_cast<T>(0);
    for (; i< size(); ++i) {
        scalar += _data[i];
    }
    return vec.sum() + scalar;
}

FASTOR_INLINE T product() const {

    if ((size()==0) || (size()==1)) return _data[0];

    using V = SIMDVector<T,simd_abi_type>;
    FASTOR_INDEX i = 0;

    V vec = static_cast<T>(1);
    for (; i< ROUND_DOWN(size(),V::Size); i+=V::Size) {
        vec *= V(&_data[i],false);
    }
    T scalar = static_cast<T>(1);
    for (; i< size(); ++i) {
        scalar *= _data[i];
    }

    return vec.product()*scalar;
}

#endif // TENSOR_METHODS_CONST_H
