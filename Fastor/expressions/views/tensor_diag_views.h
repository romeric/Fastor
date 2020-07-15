#ifndef TENSOR_DIAG_VIEWS_H
#define TENSOR_DIAG_VIEWS_H

#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/TensorMap.h"
#include "Fastor/tensor/Ranges.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"

namespace Fastor {

template<template<typename,size_t...> class TensorType, typename T, size_t M, size_t N, size_t DIM>
struct TensorDiagViewExpr<TensorType<T,M,N>,DIM> : public AbstractTensor<TensorDiagViewExpr<TensorType<T,M,N>,DIM>,DIM> {
private:
    TensorType<T,M,N>& _expr;
    bool _does_alias = false;
    constexpr FASTOR_INLINE Tensor<T,M,N> get_tensor() const {return _expr;};
public:
    using scalar_type = T;
    using simd_vector_type = typename Tensor<T,M,N>::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using result_type = Tensor<T,M>;
    using V = SIMDVector<T,simd_abi_type>;
    static constexpr FASTOR_INDEX Dimension = 1;
    static constexpr FASTOR_INDEX Stride = simd_vector_type::Size;
    static constexpr FASTOR_INDEX rank() {return 1;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return M;}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX ) const {return M;}
    constexpr const TensorType<T,M,N>& expr() const {return _expr;}

    FASTOR_INLINE TensorViewExpr<TensorType<T,M,N>,2>& noalias() {
        _does_alias = true;
        return *this;
    }

    FASTOR_INLINE TensorDiagViewExpr(Tensor<T,M,N> &_ex) : _expr(_ex) {
        static_assert(M==N, "MATRIX MUST BE SQUARE FOR DIAGONAL VIEW");
    }

    // Copy assignment
    //---------------------------------------------------------------------------------//
    FASTOR_INLINE void operator=(const TensorDiagViewExpr<TensorType<T,M,N>,DIM>& other_src) {
        // Diagonal views don't alias
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            // Get around recursive inlining issue
            const result_type tmp(other_src);
            this->operator=(tmp);
            return;
        }
#endif
#ifndef NDEBUG
        FASTOR_ASSERT(other_src.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other_src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif

        for (FASTOR_INDEX i = 0; i < M; ++i) {
            _expr(i,i) = other_src.template eval_s<T>(i,i);
        }
    }

    // AbstractTensor binders
    //---------------------------------------------------------------------------------//
    template<typename Derived, size_t DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator=(const AbstractTensor<Derived,DIMS> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator=(tmp);
    }
    template<typename Derived, size_t DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator=(const AbstractTensor<Derived,DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorDiagViewExpr<TensorType<T,M,N>,2>(tmp_this_tensor);
            // Assign other to temporary
            tmp = other;
            // assign temporary to this
            this->operator=(tmp);
            return;
        }
#endif
        const Derived& other_src = other.self();
#ifndef NDEBUG
        FASTOR_ASSERT(other_src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
        for (FASTOR_INDEX i = 0; i < M; ++i) {
            _expr(i,i) = other_src.template eval_s<T>(i);
        }
    }

    template<typename Derived, size_t DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator+=(const AbstractTensor<Derived,DIMS> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator+=(tmp);
    }
    template<typename Derived, size_t DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator+=(const AbstractTensor<Derived,DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorDiagViewExpr<TensorType<T,M,N>,2>(tmp_this_tensor);
            // Assign other to temporary
            tmp = other;
            // assign temporary to this
            this->operator+=(tmp);
            return;
        }
#endif
        const Derived& other_src = other.self();
#ifndef NDEBUG
        FASTOR_ASSERT(other_src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
        for (FASTOR_INDEX i = 0; i < M; ++i) {
            _expr(i,i) += other_src.template eval_s<T>(i);
        }
    }

    template<typename Derived, size_t DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator-=(const AbstractTensor<Derived,DIMS> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator-=(tmp);
    }
    template<typename Derived, size_t DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator-=(const AbstractTensor<Derived,DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorDiagViewExpr<TensorType<T,M,N>,2>(tmp_this_tensor);
            // Assign other to temporary
            tmp = other;
            // assign temporary to this
            this->operator-=(tmp);
            return;
        }
#endif
        const Derived& other_src = other.self();
#ifndef NDEBUG
        FASTOR_ASSERT(other_src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
        for (FASTOR_INDEX i = 0; i < M; ++i) {
            _expr(i,i) -= other_src.template eval_s<T>(i);
        }
    }

    template<typename Derived, size_t DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator*=(const AbstractTensor<Derived,DIMS> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator*=(tmp);
    }
    template<typename Derived, size_t DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator*=(const AbstractTensor<Derived,DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorDiagViewExpr<TensorType<T,M,N>,2>(tmp_this_tensor);
            // Assign other to temporary
            tmp = other;
            // assign temporary to this
            this->operator*=(tmp);
            return;
        }
#endif
        const Derived& other_src = other.self();
#ifndef NDEBUG
        FASTOR_ASSERT(other_src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
        for (FASTOR_INDEX i = 0; i < M; ++i) {
            _expr(i,i) *= other_src.template eval_s<T>(i);
        }
    }

    template<typename Derived, size_t DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(const AbstractTensor<Derived,DIMS> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator/=(tmp);
    }
    template<typename Derived, size_t DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(const AbstractTensor<Derived,DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorDiagViewExpr<TensorType<T,M,N>,2>(tmp_this_tensor);
            // Assign other to temporary
            tmp = other;
            // assign temporary to this
            this->operator/=(tmp);
            return;
        }
#endif
        const Derived& other_src = other.self();
#ifndef NDEBUG
        FASTOR_ASSERT(other_src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
        for (FASTOR_INDEX i = 0; i < M; ++i) {
            _expr(i,i) /= other_src.template eval_s<T>(i);
        }
    }

    // Scalar binders
    //---------------------------------------------------------------------------------//
    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator=(U num) {
        for (FASTOR_INDEX i = 0; i < M; ++i) {
            _expr(i,i) = num;
        }
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator+=(U num) {
        for (FASTOR_INDEX i = 0; i < M; ++i) {
            _expr(i,i) += num;
        }
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator-=(U num) {
        for (FASTOR_INDEX i = 0; i < M; ++i) {
            _expr(i,i) -= num;
        }
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator*=(U num) {
        for (FASTOR_INDEX i = 0; i < M; ++i) {
            _expr(i,i) *= num;
        }
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U> && !is_integral_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(U num) {
        T inum = T(1) / T(num);
        for (FASTOR_INDEX i = 0; i < M; ++i) {
            _expr(i,i) *= inum;
        }
    }
    template<typename U, enable_if_t_<is_arithmetic_v_<U> && is_integral_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(U num) {
        for (FASTOR_INDEX i = 0; i < M; ++i) {
            _expr(i,i) /= num;
        }
    }

    // Evaluators
    //---------------------------------------------------------------------------------//
    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX idx) const {
        V _vec;
        std::array<int,V::Size> inds;
        for (auto j=0; j<V::Size; ++j) {
            const size_t it = idx+j;
            inds[j] = it*N+it;
        }
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx) const {
        return _expr(idx,idx);
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        V _vec;
        vector_setter(_vec,_expr.data(),i*N+j);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _expr(i,j);
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int,2>& as) const {
        V _vec;
        vector_setter(_vec,_expr.data(),as[0]*N+as[1]);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,2>& as) const {
        return _expr.eval_s(as);
    }
};



// Actual definition of diag function
// Note that like other views the input can only be an evaluated tensor
template<typename T, size_t M, size_t N>
TensorDiagViewExpr<Tensor<T,M,N>,2> diag(Tensor<T,M,N> &a) {
    return TensorDiagViewExpr<Tensor<T,M,N>,2>(a);
}

template<typename T, size_t M, size_t N>
TensorDiagViewExpr<TensorMap<T,M,N>,2> diag(TensorMap<T,M,N> &a) {
    return TensorDiagViewExpr<TensorMap<T,M,N>,2>(a);
}


} // end of namespace Fastor


#endif
