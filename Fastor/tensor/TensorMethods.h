#ifndef TENSOR_METHODS_NONCONST_H
#define TENSOR_METHODS_NONCONST_H

template<typename U=T>
FASTOR_INLINE void fill(U num0) {
    T num = static_cast<T>(num0);
    FASTOR_INDEX i=0;
    for (; i<ROUND_DOWN(Size,Stride); i+=Stride) {
        SIMDVector<T,DEFAULT_ABI> _vec = num;
        _vec.store(&_data[i]);
    }
    for (; i<Size; ++i) _data[i] = num0;
}

template<typename U=T>
FASTOR_INLINE void iota(U num0=static_cast<U>(0)) {
    std::iota(_data, _data+prod<Rest...>::value, T(num0));
}

template<typename U=T>
FASTOR_INLINE void arange(U num0=0) {
    T num = static_cast<T>(num0);
    SIMDVector<T,DEFAULT_ABI> _vec;
    FASTOR_INDEX i=0;
    for (; i<ROUND_DOWN(Size,Stride); i+=Stride) {
        _vec.set_sequential(T(i)+T(num));
        _vec.store(&_data[i]);
    }
    for (; i<Size; ++i) _data[i] = T(i)+T(num0);
}

FASTOR_INLINE void zeros() {
    SIMDVector<T,DEFAULT_ABI> _zeros;
    FASTOR_INDEX i=0;
    for (; i<ROUND_DOWN(Size,Stride); i+=Stride) {
        _zeros.store(&_data[i]);
    }
    for (; i<Size; ++i) _data[i] = 0;
}

FASTOR_INLINE void ones() {
    this->fill(static_cast<T>(1));
}

FASTOR_INLINE void eye2() {
    // Second order identity tensor (identity matrices)
    static_assert(sizeof...(Rest)==2, "CANNOT BUILD AN IDENTITY TENSOR");
    static_assert(no_of_unique<Rest...>::value==1, "TENSOR MUST BE UNIFORM");
    constexpr int N = get_value<2,Rest...>::value;
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
    for (FASTOR_INDEX i=0; i<this->Size; ++i) {
        _data[get_mem_index(i)] = (T)rand()/RAND_MAX;
    }
}

FASTOR_INLINE void randint() {
    //! Populate tensor with random integer numbers
    for (FASTOR_INDEX i=0; i<this->Size; ++i) {
        _data[get_mem_index(i)] = (T)rand();
    }
}

FASTOR_INLINE void reverse() {
    // in-place reverse
    if ((Size==0) || (Size==1)) return;
    // std::reverse(_data,_data+Size); return;

    // This requires copying the data to avoid aliasing
    // Despite that this method seems to be faster than
    // std::reverse for big _data both on GCC and Clang
    T FASTOR_ALIGN tmp[Size];
    std::copy(_data,_data+Size,tmp);

    // Although SSE register reversing is faster
    // The AVX one outperforms it
    using V = SIMDVector<T,DEFAULT_ABI>;
    constexpr int unroll_upto = V::unroll_size(Size);
    constexpr int stride = V::Size;
    int i = 0;

    V vec;
    for (; i< unroll_upto; i+=stride) {
        vec.load(&tmp[Size - i - stride]);
        vec.reverse().store(&_data[i]);
    }
    for (; i< Size; ++i) {
        _data[i] = tmp[Size-i-1];
    }
}

#endif // TENSOR_METHODS_NONCONST_H


#ifndef TENSOR_METHODS_CONST_H
#define TENSOR_METHODS_CONST_H

FASTOR_INLINE T sum() const {

    if ((Size==0) || (Size==1)) return _data[0];
    using V = SIMDVector<T,DEFAULT_ABI>;

    V vec =static_cast<T>(0);
    V _vec_in;
    FASTOR_INDEX i = 0;
    for (; i<ROUND_DOWN(Size,V::Size); i+=V::Size) {
        _vec_in.load(&_data[i],false);
        vec += _vec_in;
    }
    T scalar = static_cast<T>(0);
    for (; i< Size; ++i) {
        scalar += _data[i];
    }
    return vec.sum() + scalar;
}

FASTOR_INLINE T product() const {

    if ((Size==0) || (Size==1)) return _data[0];

    using V = SIMDVector<T,DEFAULT_ABI>;
    constexpr int unroll_upto = V::unroll_size(Size);
    constexpr int stride = V::Size;
    int i = 0;

    V vec = static_cast<T>(1);
    for (; i< unroll_upto; i+=stride) {
        vec *= V(&_data[i],false);
    }
    T scalar = static_cast<T>(1);
    for (; i< Size; ++i) {
        scalar *= _data[i];
    }

    return vec.product()*scalar;
}

#endif // TENSOR_METHODS_CONST_H