#ifndef SPECIALISED_CONSTRUCTORS_H
#define SPECIALISED_CONSTRUCTORS_H

template<typename U, size_t M1, size_t N1>
FASTOR_INLINE Tensor(const TensorViewExpr<Tensor<U,M1,N1>,2>& src) {
// FASTOR_INLINE Tensor(const AbstractTensor<TensorViewExpr<Tensor<U,M1,N1>,2>,2>& src_) {
    // const TensorViewExpr<Tensor<U,M1,N1>,2> &src = src_.self();
    // verify_dimensions(src);
#ifndef NDEBUG
        FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
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
    // verify_dimensions(src);
#ifndef NDEBUG
        FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
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


template<typename Derived, size_t DIMS, typename std::enable_if<has_tensor_view<Derived>::value && DIMS==sizeof...(Rest),bool>::type=0>
FASTOR_INLINE Tensor(const AbstractTensor<Derived,DIMS>& src_) {
    // verify_dimensions(src_);
    const Derived &src = src_.self();
#ifndef NDEBUG
        FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif

    constexpr int DimensionHolder[Dimension] = {Rest...};
    std::array<int,Dimension> as = {};
    int jt, counter=0;

    while(counter < Size)
    {
        _data[counter] = src.template teval_s<T>(as);

        counter++;
        for(jt = Dimension-1; jt>=0; jt--)
        {
            as[jt] +=1;
            if(as[jt]<DimensionHolder[jt])
                break;
            else
                as[jt]=0;
        }
        if(jt<0)
            break;
    }

    // Generic vectorised version that takes care of the remainder scalar ops
    // using V=SIMDVector<T,DEFAULT_ABI>;
    // while(counter < Size)
    // {
    //     const FASTOR_INDEX remainder = DimensionHolder[Dimension-1] - as[Dimension-1];
    //     if (remainder % V::Size == 0) {
    //         V _vec = src.template eval<T>(counter);
    //         // V _vec = src.template teval<T>(as);
    //         _vec.store(&_data[counter],false);
    //         counter+=V::Size;
    //     }
    //     else {
    //         // print(22);
    //         _data[counter] = src.template eval_s<T>(counter);
    //         // _data[counter] = src.template teval_s<T>(as);
    //         counter++;
    //     }
    //     // println(as);

    //     for(jt = Dimension-1; jt>=0; jt--)
    //     {
    //         if (jt == Dimension-1) as[jt]+=V::Size;
    //         else as[jt] +=1;
    //         if(as[jt]<DimensionHolder[jt])
    //             break;
    //         else
    //             as[jt]=0;
    //     }
    //     if(jt<0)
    //         break;
    // }
}

template<size_t ...Rest1, typename std::enable_if<is_greater<sizeof...(Rest1),2>::value,bool>::type=0>
FASTOR_INLINE Tensor(const TensorViewExpr<Tensor<T,Rest1...>,sizeof...(Rest)>& src) {
    // verify_dimensions(src);
#ifndef NDEBUG
        FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
    constexpr int DimensionHolder[Dimension] = {Rest...};
    std::array<int,Dimension> as = {};
    int jt, counter=0;

    if (src.is_vectorisable() || src.is_strided_vectorisable())
    {
        using V = SIMDVector<T,DEFAULT_ABI>;
        V _vec;
        while(counter < Size)
        {
            _vec = src.template teval<T>(as);
            _vec.store(&_data[counter],false);

            counter+=V::Size;
            for(jt = Dimension-1; jt>=0; jt--)
            {
                if (jt == Dimension-1) as[jt]+=V::Size;
                else as[jt] +=1;
                if(as[jt]<DimensionHolder[jt])
                    break;
                else
                    as[jt]=0;
            }
            if(jt<0)
                break;
        }
    }
    else {
        while(counter < Size)
        {
            _data[counter] = src.template teval_s<T>(as);

            counter++;
            for(jt = Dimension-1; jt>=0; jt--)
            {
                as[jt] +=1;
                if(as[jt]<DimensionHolder[jt])
                    break;
                else
                    as[jt]=0;
            }
            if(jt<0)
                break;
        }
    }
}


#endif // SPECIALISED_CONSTRUCTORS_H