#ifndef SPECIALISED_CONSTRUCTORS_H
#define SPECIALISED_CONSTRUCTORS_H


//----------------------------------------------------------------------------------------------------------//
template<size_t ...Rest1, typename Seq0, typename Seq1,
    typename std::enable_if<sizeof...(Rest)==sizeof...(Rest1),bool>::type=0>
FASTOR_INLINE Tensor(const TensorFixedViewExpr2D<Tensor<T,Rest1...>,Seq0,Seq1,2>& src) {
    using scalar_type_ = T;
    constexpr FASTOR_INDEX Stride_ = stride_finder<scalar_type_>::value;
#ifndef NDEBUG
    FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
    for (FASTOR_INDEX i = 0; i<sizeof...(Rest); ++i) {
        FASTOR_ASSERT(src.dimension(i)==this->dimension(i), "TENSOR SHAPE MISMATCH");
    }
#endif
    constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
    constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
    for (FASTOR_INDEX i = 0; i <M; ++i) {
        FASTOR_INDEX j;
        for (j = 0; j <ROUND_DOWN(N,Stride_); j+=Stride_) {
            src.template eval<scalar_type_>(i,j).store(&_data[i*N+j], false);
        }
        for (; j < N; ++j) {
            _data[i*N+j] = src.template eval_s<scalar_type_>(i,j);
        }
    }
}

#ifndef FASTOR_DISABLE_SPECIALISED_CTR

template<typename Derived, size_t DIMS,
    typename std::enable_if<!internal::has_tensor_view<Derived>::value &&
    internal::has_tensor_fixed_view_2d<Derived>::value && DIMS==sizeof...(Rest),bool>::type=0>
FASTOR_INLINE Tensor(const AbstractTensor<Derived,DIMS>& src_) {
    using scalar_type_ = typename scalar_type_finder<Derived>::type;
    constexpr FASTOR_INDEX Stride_ = stride_finder<scalar_type_>::value;
    const Derived &src = src_.self();
#ifndef NDEBUG
    FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
    for (FASTOR_INDEX i = 0; i<sizeof...(Rest); ++i) {
        FASTOR_ASSERT(src.dimension(i)==this->dimension(i), "TENSOR SHAPE MISMATCH");
    }
#endif
    constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
    constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
    FASTOR_IF_CONSTEXPR(!internal::is_binary_cmp_op<Derived>::value) {
        for (FASTOR_INDEX i = 0; i <M; ++i) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(N,Stride_); j+=Stride_) {
                src.template eval<T>(i,j).store(&_data[i*N+j], false);
            }
            for (; j <N; ++j) {
                _data[i*N+j] = src.template eval_s<T>(i,j);
            }
        }
    }
    else {
        for (FASTOR_INDEX i = 0; i <M; ++i) {
            for (FASTOR_INDEX j = 0; j <N; ++j) {
                _data[i*N+j] = src.template eval_s<T>(i,j);
            }
        }
    }
}

#endif // FASTOR_DISABLE_SPECIALISED_CTR
//----------------------------------------------------------------------------------------------------------//


//----------------------------------------------------------------------------------------------------------//
template<size_t ...Rest1, typename std::enable_if<sizeof...(Rest)==sizeof...(Rest1),bool>::type=0>
FASTOR_INLINE Tensor(const TensorViewExpr<Tensor<T,Rest1...>,2>& src) {
    using scalar_type_ = T;
    constexpr FASTOR_INDEX Stride_ = stride_finder<scalar_type_>::value;
#ifndef NDEBUG
    FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
    for (FASTOR_INDEX i = 0; i<sizeof...(Rest); ++i) {
        FASTOR_ASSERT(src.dimension(i)==this->dimension(i), "TENSOR SHAPE MISMATCH");
    }
#endif
    constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
    constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
    for (FASTOR_INDEX i = 0; i <M; ++i) {
        FASTOR_INDEX j;
        for (j = 0; j <ROUND_DOWN(N,Stride_); j+=Stride_) {
            src.template eval<scalar_type_>(i,j).store(&_data[i*N+j], false);
        }
        for (; j < N; ++j) {
            _data[i*N+j] = src.template eval_s<scalar_type_>(i,j);
        }
    }
}


template<size_t ...Rest1, typename std::enable_if<is_greater<sizeof...(Rest1),2>::value,bool>::type=0>
FASTOR_INLINE Tensor(const TensorViewExpr<Tensor<T,Rest1...>,sizeof...(Rest)>& src) {
#ifndef NDEBUG
    FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
    for (FASTOR_INDEX i = 0; i<sizeof...(Rest); ++i) {
        FASTOR_ASSERT(src.dimension(i)==this->dimension(i), "TENSOR SHAPE MISMATCH");
    }
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

#ifndef FASTOR_DISABLE_SPECIALISED_CTR

template<typename Derived, size_t DIMS, typename std::enable_if<
    internal::has_tensor_view<Derived>::value &&
    DIMS==2 &&
    DIMS==sizeof...(Rest),bool>::type=0>
FASTOR_INLINE Tensor(const AbstractTensor<Derived,DIMS>& src_) {
    using scalar_type_ = typename scalar_type_finder<Derived>::type;
    constexpr FASTOR_INDEX Stride_ = stride_finder<scalar_type_>::value;
    const Derived &src = src_.self();
#ifndef NDEBUG
    FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
    for (FASTOR_INDEX i = 0; i<sizeof...(Rest); ++i) {
        FASTOR_ASSERT(src.dimension(i)==this->dimension(i), "TENSOR SHAPE MISMATCH");
    }
#endif
    constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
    constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
    FASTOR_IF_CONSTEXPR(!internal::is_binary_cmp_op<Derived>::value) {
        for (FASTOR_INDEX i = 0; i <M; ++i) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(N,Stride_); j+=Stride_) {
                src.template eval<T>(i,j).store(&_data[i*N+j], false);
            }
            for (; j < N; ++j) {
                _data[i*N+j] = src.template eval_s<T>(i,j);
            }
        }
    }
    else {
        for (FASTOR_INDEX i = 0; i <M; ++i) {
            for (FASTOR_INDEX j = 0; j < N; ++j) {
                _data[i*N+j] = src.template eval_s<T>(i,j);
            }
        }
    }
}


template<typename Derived, size_t DIMS, typename std::enable_if<
    internal::has_tensor_view<Derived>::value &&
    DIMS!=2 &&
    DIMS==sizeof...(Rest),bool>::type=0>
FASTOR_INLINE Tensor(const AbstractTensor<Derived,DIMS>& src_) {
    // using scalar_type_ = typename scalar_type_finder<Derived>::type;
    // constexpr FASTOR_INDEX Stride_ = stride_finder<scalar_type_>::value;
    const Derived &src = src_.self();
#ifndef NDEBUG
    FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
    for (FASTOR_INDEX i = 0; i<sizeof...(Rest); ++i) {
        FASTOR_ASSERT(src.dimension(i)==this->dimension(i), "TENSOR SHAPE MISMATCH");
    }
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

    // // Generic vectorised version that takes care of the remainder scalar ops
    // using V=SIMDVector<T,DEFAULT_ABI>;
    // while(counter < Size)
    // {
    //     const FASTOR_INDEX remainder = DimensionHolder[Dimension-1] - as[Dimension-1];
    //     if (remainder > V::Size) {
    //         // V _vec = src.template eval<T>(counter);
    //         V _vec = src.template teval<T>(as);
    //         _vec.store(&_data[counter],false);
    //         counter+=V::Size;
    //     }
    //     else {
    //         // _data[counter] = src.template eval_s<T>(counter);
    //         _data[counter] = src.template teval_s<T>(as);
    //         counter++;
    //     }

    //     for(jt = Dimension-1; jt>=0; jt--)
    //     {
    //         if (jt == Dimension-1 && remainder > V::Size) as[jt]+=V::Size;
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

#endif // FASTOR_DISABLE_SPECIALISED_CTR
//----------------------------------------------------------------------------------------------------------//


#endif // SPECIALISED_CONSTRUCTORS_H
