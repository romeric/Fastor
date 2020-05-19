#ifndef TENSOR_FILTER_VIEWS_H
#define TENSOR_FILTER_VIEWS_H


#include "Fastor/tensor/Tensor.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"


namespace Fastor {

template<typename T, size_t ... Rest, size_t DIMS>
struct TensorFilterViewExpr<Tensor<T,Rest...>,Tensor<bool,Rest...>,DIMS>:
    public AbstractTensor<TensorFilterViewExpr<Tensor<T,Rest...>,Tensor<bool,Rest...>,DIMS>,DIMS> {
private:
    Tensor<T,Rest...> &_expr;
    const Tensor<bool,Rest...> &fl_expr;
    bool _does_alias = false;

    constexpr FASTOR_INLINE Tensor<T,Rest...> get_tensor() const {return _expr;}
public:
    using scalar_type = T;
    using simd_vector_type = typename Tensor<T,Rest...>::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using result_type = Tensor<T,Rest...>;
    static constexpr FASTOR_INDEX Dimension = DIMS;
    static constexpr FASTOR_INDEX Stride = simd_vector_type::Size;
    static constexpr FASTOR_INDEX rank() {return DIMS;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return pack_prod<Rest...>::value;}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return fl_expr.dimension(i);}
    constexpr const Tensor<T,Rest...>& expr() const {return _expr;}

    FASTOR_INLINE TensorFilterViewExpr<Tensor<T,Rest...>,Tensor<bool,Rest...>,DIMS>& noalias() {
        _does_alias = true;
        return *this;
    }

    constexpr FASTOR_INLINE TensorFilterViewExpr(Tensor<T,Rest...> &_ex, const Tensor<bool,Rest...> &_it) : _expr(_ex), fl_expr(_it) {
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
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _expr.data()[i] = src.template eval_s<T>(i);
            }
        }
    }


    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
    void operator=(const AbstractTensor<Derived,OTHER_DIMS> &src) {
        const typename Derived::result_type& tmp = evaluate(src.self());
        this->operator=(tmp);
    }
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    void operator=(const AbstractTensor<Derived,OTHER_DIMS> &src) {
#ifndef NDEBUG
        FASTOR_ASSERT(src.self().size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(src.self().dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _expr.data()[i] = src.self().template eval_s<T>(i);
            }
        }
    }

    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
    void operator+=(const AbstractTensor<Derived,OTHER_DIMS> &src) {
        const typename Derived::result_type& tmp = evaluate(src.self());
        this->operator+=(tmp);
    }
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    void operator+=(const AbstractTensor<Derived,OTHER_DIMS> &src) {
#ifndef NDEBUG
        FASTOR_ASSERT(src.self().size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(src.self().dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _expr.data()[i] += src.self().template eval_s<T>(i);
            }
        }
    }

    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
    void operator-=(const AbstractTensor<Derived,OTHER_DIMS> &src) {
        const typename Derived::result_type& tmp = evaluate(src.self());
        this->operator-=(tmp);
    }
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    void operator-=(const AbstractTensor<Derived,OTHER_DIMS> &src) {
#ifndef NDEBUG
        FASTOR_ASSERT(src.self().size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(src.self().dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _expr.data()[i] -= src.self().template eval_s<T>(i);
            }
        }
    }

    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
    void operator*=(const AbstractTensor<Derived,OTHER_DIMS> &src) {
        const typename Derived::result_type& tmp = evaluate(src.self());
        this->operator*=(tmp);
    }
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    void operator*=(const AbstractTensor<Derived,OTHER_DIMS> &src) {
#ifndef NDEBUG
        FASTOR_ASSERT(src.self().size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(src.self().dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _expr.data()[i] *= src.self().template eval_s<T>(i);
            }
        }
    }

    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
    void operator/=(const AbstractTensor<Derived,OTHER_DIMS> &src) {
        const typename Derived::result_type& tmp = evaluate(src.self());
        this->operator/=(tmp);
    }
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    void operator/=(const AbstractTensor<Derived,OTHER_DIMS> &src) {
#ifndef NDEBUG
        FASTOR_ASSERT(src.self().size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(src.self().dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _expr.data()[i] /= src.self().template eval_s<T>(i);
            }
        }
    }

    //------------------------------------------------------------------------------------//
    template<typename U=T, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    void operator=(U num) {
        T tnum = (T)num;
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _expr.data()[i] = tnum;
            }
        }
    }

    template<typename U=T, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    void operator+=(U num) {
        T tnum = (T)num;
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _expr.data()[i] += tnum;
            }
        }
    }

    template<typename U=T, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    void operator-=(U num) {
        T tnum = (T)num;
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _expr.data()[i] -= tnum;
            }
        }
    }

    template<typename U=T, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    void operator*=(U num) {
        T tnum = (T)num;
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _expr.data()[i] *= tnum;
            }
        }
    }

    template<typename U=T, enable_if_t_<is_arithmetic_v_<U> && !is_integral_v_<U>,bool> = false>
    void operator/=(U num) {
        T tnum = T(1)/(T)num;
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _expr.data()[i] *= tnum;
            }
        }
    }
    template<typename U=T, enable_if_t_<is_arithmetic_v_<U> && is_integral_v_<U>,bool> = false>
    void operator/=(U num) {
        T tnum = (T)num;
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            if (fl_expr.eval_s(i)) {
                _expr.data()[i] /= tnum;
            }
        }
    }
    //------------------------------------------------------------------------------------//


    //------------------------------------------------------------------------------------//
    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i) const {
        constexpr size_t _Stride = SIMDVector<T,simd_abi_type>::Size;
        SIMDVector<U,simd_abi_type> _vec;
        U inds[_Stride];
        for (FASTOR_INDEX j=0; j<_Stride; ++j)
            inds[j] = fl_expr.eval_s(i+j) ? _expr.eval_s(i+j) : 0;
        _vec.load(inds,false);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return fl_expr.eval_s(i) ? _expr.eval_s(i) : 0;
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX k) const {
        constexpr size_t _Stride = SIMDVector<T,simd_abi_type>::Size;
        SIMDVector<U,simd_abi_type> _vec;
        U inds[_Stride];
        for (FASTOR_INDEX j=0; j<_Stride; ++j)
            inds[j] = fl_expr.eval_s(i,k+j) ? _expr.eval_s(i,k+j) : 0;
        _vec.load(inds,false);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX k) const {
        return fl_expr.eval_s(i,k) ? _expr.eval_s(i,k) : 0;
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int,DIMS>& as) const {
        constexpr size_t _Stride = SIMDVector<T,simd_abi_type>::Size;
        SIMDVector<U,simd_abi_type> _vec;
        U inds[_Stride];
        for (FASTOR_INDEX j=0; j<_Stride; ++j)
            inds[j] = fl_expr.teval_s(as) ? _expr.teval_s(as) : 0;
        _vec.load(inds,false);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U teval_s(const std::array<int,DIMS>& as) const {
        return fl_expr.teval_s(as) ? _expr.teval_s(as) : 0;
    }
    //------------------------------------------------------------------------------------//

};

} // end of namespace Fastor


#endif // TENSOR_FILTER_VIEWS_H
