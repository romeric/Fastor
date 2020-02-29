#ifndef SPECIALISED_CONSTRUCTORS_H
#define SPECIALISED_CONSTRUCTORS_H

template<typename U, size_t M1, size_t N1>
FASTOR_INLINE Tensor(const TensorViewExpr<Tensor<U,M1,N1>,2>& src) {
// FASTOR_INLINE Tensor(const AbstractTensor<TensorViewExpr<Tensor<U,M1,N1>,2>,2>& src_) {
    // const TensorViewExpr<Tensor<U,M1,N1>,2> &src = src_.self();
    verify_dimensions(src);
    constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
    constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
    for (FASTOR_INDEX i = 0; i <M; ++i) {
        FASTOR_INDEX j;
        for (j = 0; j <ROUND_DOWN(N,Stride); j+=Stride) {
            src.template eval<T>(i,j).store(_data+i*N+j, false);
        }
        for (; j < N; ++j) {
            _data[i*N+j] = src.template eval_s<T>(i,j);
        }
    }
}

template<typename U, size_t M1, size_t N1, typename Seq0, typename Seq1>
FASTOR_INLINE Tensor(const TensorFixedViewExpr2D<Tensor<U,M1,N1>,Seq0,Seq1,2>& src) {
    verify_dimensions(src);
    constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
    constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
    for (FASTOR_INDEX i = 0; i <M; ++i) {
        FASTOR_INDEX j;
        for (j = 0; j <ROUND_DOWN(N,Stride); j+=Stride) {
            src.template eval<T>(i,j).store(_data+i*N+j, false);
        }
        for (; j < N; ++j) {
            _data[i*N+j] = src.template eval_s<T>(i,j);
        }
    }
}


#endif // SPECIALISED_CONSTRUCTORS_H