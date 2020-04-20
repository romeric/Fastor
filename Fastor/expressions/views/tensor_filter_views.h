#ifndef TENSOR_FILTER_VIEWS_H
#define TENSOR_FILTER_VIEWS_H


#include "Fastor/tensor/Tensor.h"


namespace Fastor {

template<typename T, size_t ... Rest, size_t DIMS>
struct TensorFilterViewExpr<Tensor<T,Rest...>,Tensor<bool,Rest...>,DIMS>:
    public AbstractTensor<TensorFilterViewExpr<Tensor<T,Rest...>,Tensor<bool,Rest...>,DIMS>,DIMS> {
private:
    Tensor<T,Rest...> &expr;
    const Tensor<bool,Rest...> &fl_expr;
    bool does_alias = false;

    constexpr FASTOR_INLINE Tensor<T,Rest...> get_tensor() const {return expr;}
public:
    using scalar_type = T;
    using result_type = Tensor<T,Rest...>;
    static constexpr FASTOR_INDEX Dimension = DIMS;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    static constexpr FASTOR_INDEX rank() {return DIMS;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return prod<Rest...>::value;}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return fl_expr.dimension(i);}

    FASTOR_INLINE TensorFilterViewExpr<Tensor<T,Rest...>,Tensor<bool,Rest...>,DIMS>& noalias() {
        does_alias = true;
        return *this;
    }

    constexpr FASTOR_INLINE TensorFilterViewExpr(Tensor<T,Rest...> &_ex, const Tensor<bool,Rest...> &_it) : expr(_ex), fl_expr(_it) {
        static_assert(sizeof...(Rest)==DIMS, "INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    }

    // In place operators
    //------------------------------------------------------------------------------------//
    void operator=(const TensorFilterViewExpr<Tensor<T,Rest...>,Tensor<bool,Rest...>,DIMS> &src) {
#ifndef NDEBUG
        FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _data[i] = src.template eval_s<T>(i);
            }
        }
    }


    template<typename Derived, size_t OTHER_DIMS>
    void operator=(const AbstractTensor<Derived,OTHER_DIMS> &src) {
#ifndef NDEBUG
        FASTOR_ASSERT(src.self().size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(src.self().dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _data[i] = src.self().template eval_s<T>(i);
            }
        }
    }

    template<typename Derived, size_t OTHER_DIMS>
    void operator +=(const AbstractTensor<Derived,OTHER_DIMS> &src) {
#ifndef NDEBUG
        FASTOR_ASSERT(src.self().size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(src.self().dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _data[i] += src.self().template eval_s<T>(i);
            }
        }
    }

    template<typename Derived, size_t OTHER_DIMS>
    void operator -=(const AbstractTensor<Derived,OTHER_DIMS> &src) {
#ifndef NDEBUG
        FASTOR_ASSERT(src.self().size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(src.self().dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _data[i] -= src.self().template eval_s<T>(i);
            }
        }
    }

    template<typename Derived, size_t OTHER_DIMS>
    void operator *=(const AbstractTensor<Derived,OTHER_DIMS> &src) {
#ifndef NDEBUG
        FASTOR_ASSERT(src.self().size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(src.self().dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _data[i] *= src.self().template eval_s<T>(i);
            }
        }
    }

    template<typename Derived, size_t OTHER_DIMS>
    void operator /=(const AbstractTensor<Derived,OTHER_DIMS> &src) {
#ifndef NDEBUG
        FASTOR_ASSERT(src.self().size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(src.self().dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _data[i] /= src.self().template eval_s<T>(i);
            }
        }
    }

    //------------------------------------------------------------------------------------//
    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator=(U num) {
        T tnum = (T)num;
        T *_data = expr.data();
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _data[i] = tnum;
            }
        }
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator+=(U num) {
        T tnum = (T)num;
        T *_data = expr.data();
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _data[i] += tnum;
            }
        }
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator-=(U num) {
        T tnum = (T)num;
        T *_data = expr.data();
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _data[i] -= tnum;
            }
        }
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator*=(U num) {
        T tnum = (T)num;
        T *_data = expr.data();
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _data[i] *= tnum;
            }
        }
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator/=(U num) {
        T tnum = T(1)/(T)num;
        T *_data = expr.data();
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _data[i] *= tnum;
            }
        }
    }
    //------------------------------------------------------------------------------------//


    //------------------------------------------------------------------------------------//
    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        constexpr size_t _Stride = SIMDVector<T,DEFAULT_ABI>::Size;
        SIMDVector<U,DEFAULT_ABI> _vec;
        U inds[_Stride];
        for (FASTOR_INDEX j=0; j<_Stride; ++j)
            inds[j] = fl_expr.eval_s(i+j) ? expr.eval_s(i+j) : 0;
        _vec.load(inds,false);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return fl_expr.eval_s(i) ? expr.eval_s(i) : 0;
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX k) const {
        constexpr size_t _Stride = SIMDVector<T,DEFAULT_ABI>::Size;
        SIMDVector<U,DEFAULT_ABI> _vec;
        U inds[_Stride];
        for (FASTOR_INDEX j=0; j<_Stride; ++j)
            inds[j] = fl_expr.eval_s(i,k+j) ? expr.eval_s(i,k+j) : 0;
        _vec.load(inds,false);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX k) const {
        return fl_expr.eval_s(i,k) ? expr.eval_s(i,k) : 0;
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> teval(const std::array<int,DIMS>& as) const {
        constexpr size_t _Stride = SIMDVector<T,DEFAULT_ABI>::Size;
        SIMDVector<U,DEFAULT_ABI> _vec;
        U inds[_Stride];
        for (FASTOR_INDEX j=0; j<_Stride; ++j)
            inds[j] = fl_expr.teval_s(as) ? expr.teval_s(as) : 0;
        _vec.load(inds,false);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U teval_s(const std::array<int,DIMS>& as) const {
        return fl_expr.teval_s(as) ? expr.teval_s(as) : 0;
    }
    //------------------------------------------------------------------------------------//

};

} // end of namespace Fastor


#endif // TENSOR_FILTER_VIEWS_H