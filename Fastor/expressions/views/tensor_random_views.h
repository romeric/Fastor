#ifndef TENSOR_RANDOM_VIEWS_H
#define TENSOR_RANDOM_VIEWS_H

#include "Fastor/tensor/Tensor.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"


namespace Fastor {


// Const versions
//----------------------------------------------------------------------------------//
template<typename T, size_t N, typename Int, size_t IterSize>
struct TensorConstRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>:
    public AbstractTensor<TensorConstRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>,1> {
private:
    const Tensor<T,N> &_expr;
    const Tensor<Int,IterSize> &it_expr;
public:
    using scalar_type = T;
    using simd_vector_type = typename Tensor<T,N>::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using result_type = Tensor<T,IterSize>;
    static constexpr FASTOR_INDEX Dimension = 1;
    static constexpr FASTOR_INDEX Stride = simd_vector_type::Size;
    static constexpr FASTOR_INDEX rank() {return 1;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return IterSize;}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX ) const {return IterSize;}
    constexpr const Tensor<T,N>& expr() const {return _expr;}

    constexpr FASTOR_INLINE TensorConstRandomViewExpr(const Tensor<T,N> &_ex, const Tensor<Int,IterSize> &_it) : _expr(_ex), it_expr(_it) {}

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i) const {
        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,SIMDVector<T,simd_abi_type>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j)
            inds[j] = it_expr.data()[i+j];
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return _expr.data()[it_expr.data()[i]];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX k) const {
        i += k;
        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,SIMDVector<T,simd_abi_type>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j)
            inds[j] = it_expr.data()[i+j];
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _expr(it_expr(i+j));
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int,1>& as) const {
        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,SIMDVector<T,simd_abi_type>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j)
            inds[j] = it_expr.data()[as[0]+j];
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,1>& as) const {
        return _expr.data()[it_expr.data()[as[0]]];
    }
};


//----------------------------------------------------------------------------------//
template<typename T, size_t ... Rest, typename Int, size_t ... IterSizes, size_t DIMS>
struct TensorConstRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>:
    public AbstractTensor<TensorConstRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>,DIMS> {
private:
    const Tensor<T,Rest...> &_expr;
    // The index tensor needs to be copied for the purpose of indexing a
    // multidimensional tensor with multiple 1D or any other lower other
    // tensors. This issue could be solved by providing another type parameter/flag
    // such as CopyIndex to the class which would use std::conditional to choose
    // between "const Tensor<Int,IterSizes...> &" and "Tensor<Int,IterSizes...>"

    // const Tensor<Int,IterSizes...> &it_expr;
    Tensor<Int,IterSizes...> it_expr;
public:
    using scalar_type = T;
    using simd_vector_type = typename Tensor<T,Rest...>::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using result_type = Tensor<T,IterSizes...>;
    static constexpr FASTOR_INDEX Dimension = DIMS;
    static constexpr FASTOR_INDEX Stride = simd_vector_type::Size;
    static constexpr FASTOR_INDEX rank() {return DIMS;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return pack_prod<IterSizes...>::value;}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return it_expr.dimension(i);}
    constexpr const Tensor<T,Rest...>& expr() const {return _expr;}

    constexpr FASTOR_INLINE TensorConstRandomViewExpr(const Tensor<T,Rest...> &_ex,
        const Tensor<Int,IterSizes...> &_it) : _expr(_ex), it_expr(_it) {
        static_assert(sizeof...(Rest)==DIMS && sizeof...(IterSizes)==DIMS, "INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i) const {
        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,SIMDVector<T,simd_abi_type>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j)
            inds[j] = it_expr.data()[i+j];
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return _expr.data()[it_expr.data()[i]];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX k) const {
        i += k;
        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,SIMDVector<T,simd_abi_type>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j)
            inds[j] = it_expr.data()[i+j];
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _expr.data()[it_expr.data()[i+j]];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int,DIMS>& as) const {
        int i = std::accumulate(as.begin(), as.end(), 0);
        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,SIMDVector<T,simd_abi_type>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j)
            inds[j] = it_expr.data()[i+j];
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U teval_s(const std::array<int,DIMS>& as) const {
        int i = std::accumulate(as.begin(), as.end(), 0);
        return _expr.data()[it_expr.data()[i]];
    }

};

//----------------------------------------------------------------------------------//





template<typename T, size_t N, typename Int, size_t IterSize>
struct TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>:
    public AbstractTensor<TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>,1> {
private:
    Tensor<T,N> &_expr;
    const Tensor<Int,IterSize> &it_expr;
    bool _does_alias = false;

    constexpr FASTOR_INLINE Tensor<T,N> get_tensor() const {return _expr;};
public:
    using scalar_type = T;
    using simd_vector_type = typename Tensor<T,N>::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using result_type = Tensor<T,IterSize>;
    static constexpr FASTOR_INDEX Dimension = 1;
    static constexpr FASTOR_INDEX Stride = simd_vector_type::Size;
    static constexpr FASTOR_INDEX rank() {return 1;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return IterSize;}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX ) const {return IterSize;}
    constexpr const Tensor<T,N>& expr() const {return _expr;}

    FASTOR_INLINE TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>& noalias() {
        _does_alias = true;
        return *this;
    }

    constexpr FASTOR_INLINE TensorRandomViewExpr(Tensor<T,N> &_ex, const Tensor<Int,IterSize> &_it) : _expr(_ex), it_expr(_it) {}


    // View evalution operators
    // Copy assignment operators
    //----------------------------------------------------------------------------------//
    FASTOR_HINT_INLINE void operator=(const TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1> &other_src) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>(tmp_this_tensor,it_expr);
            // Assign other to temporary
            tmp = other_src;
            // assign temporary to this
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
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _expr(it_expr(i+j)) = _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _expr(it_expr(i)) = other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _expr(it_expr(i)) = other_src.template eval_s<T>(i);
        }
#endif
    }


    // AbstractTensor binders
    //----------------------------------------------------------------------------------//
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
            auto tmp = TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>(tmp_this_tensor,it_expr);
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
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_IF_CONSTEXPR(is_boolean_expression_v<Derived>) {
            for (FASTOR_INDEX i = 0; i <size(); i++) {
                _expr(it_expr(i)) = other_src.template eval_s<T>(i);
            }
            return;
        }
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _expr(it_expr(i+j)) = _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _expr(it_expr(i)) = other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _expr(it_expr(i)) = other_src.template eval_s<T>(i);
        }
#endif
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
            auto tmp = TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>(tmp_this_tensor,it_expr);
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
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _expr(it_expr(i+j)) += _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _expr(it_expr(i)) += other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _expr(it_expr(i)) += other_src.template eval_s<T>(i);
        }
#endif
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
            auto tmp = TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>(tmp_this_tensor,it_expr);
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
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _expr(it_expr(i+j)) -= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _expr(it_expr(i)) -= other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _expr(it_expr(i)) -= other_src.template eval_s<T>(i);
        }
#endif
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
            auto tmp = TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>(tmp_this_tensor,it_expr);
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
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _expr(it_expr(i+j)) *= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _expr(it_expr(i)) *= other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _expr(it_expr(i)) *= other_src.template eval_s<T>(i);
        }
#endif
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
            auto tmp = TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>(tmp_this_tensor,it_expr);
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
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _expr(it_expr(i+j)) /= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _expr(it_expr(i)) /= other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _expr(it_expr(i)) /= other_src.template eval_s<T>(i);
        }
#endif
    }
    //----------------------------------------------------------------------------------//


    // scalar binders
    //----------------------------------------------------------------------------------//
    template<typename U=T, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator=(U num) {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,simd_abi_type> _vec_other(static_cast<T>(num));
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _expr(it_expr(i+j)) = _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _expr(it_expr(i)) = num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _expr(it_expr(i)) = num;
        }
#endif
    }

    template<typename U=T, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator+=(U num) {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,simd_abi_type> _vec_other(static_cast<T>(num));
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _expr(it_expr(i+j)) += _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _expr(it_expr(i)) += num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _expr(it_expr(i)) += num;
        }
#endif
    }

    template<typename U=T, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator-=(U num) {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,simd_abi_type> _vec_other(static_cast<T>(num));
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _expr(it_expr(i+j)) -= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _expr(it_expr(i)) -= num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _expr(it_expr(i)) -= num;
        }
#endif
    }

    template<typename U=T, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator*=(U num) {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,simd_abi_type> _vec_other(static_cast<T>(num));
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _expr(it_expr(i+j)) *= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _expr(it_expr(i)) *= num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _expr(it_expr(i)) *= num;
        }
#endif
    }

    template<typename U=T, enable_if_t_<is_arithmetic_v_<U> && !is_integral_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(U num) {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        T inum = T(1)/(T)num;
        SIMDVector<T,simd_abi_type> _vec_other(inum);
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _expr(it_expr(i+j)) *= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _expr(it_expr(i)) *= inum;
        }
#else
        T inum = T(1.)/num;
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _expr(it_expr(i)) *= inum;
        }
#endif
    }
    template<typename U=T, enable_if_t_<is_arithmetic_v_<U> && is_integral_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(U num) {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,simd_abi_type> _vec_other(static_cast<T>(num));
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _expr(it_expr(i+j)) /= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _expr(it_expr(i)) /= num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _expr(it_expr(i)) /= num;
        }
#endif
    }
    //----------------------------------------------------------------------------------//

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i) const {
        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,SIMDVector<T,simd_abi_type>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j)
            inds[j] = it_expr(i+j);
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return _expr(it_expr(i));
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX k) const {
        i += k;
        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,SIMDVector<T,simd_abi_type>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j)
            inds[j] = it_expr(i+j);
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _expr(it_expr(i+j));
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int,1>& as) const {
        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,SIMDVector<T,simd_abi_type>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j)
            inds[j] = it_expr.data()[as[0]+j];
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,1>& as) const {
        return _expr.data()[it_expr.data()[as[0]]];
    }

};



























































template<typename T, size_t ... Rest, typename Int, size_t ... IterSizes, size_t DIMS>
struct TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>:
    public AbstractTensor<TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>,DIMS> {
private:
    Tensor<T,Rest...> &_expr;
    // The index tensor needs to be copied for the purpose of indexing a
    // multidimensional tensor with multiple 1D or any other lower other
    // tensors.

    // const Tensor<Int,IterSizes...> &it_expr;
    Tensor<Int,IterSizes...> it_expr;
    bool _does_alias = false;

    constexpr FASTOR_INLINE Tensor<T,Rest...> get_tensor() const {return _expr;}
public:
    using scalar_type = T;
    using simd_vector_type = typename Tensor<T,Rest...>::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using result_type = Tensor<T,IterSizes...>;
    static constexpr FASTOR_INDEX Dimension = DIMS;
    static constexpr FASTOR_INDEX Stride = simd_vector_type::Size;
    static constexpr FASTOR_INDEX rank() {return DIMS;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return pack_prod<IterSizes...>::value;}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return it_expr.dimension(i);}
    constexpr const Tensor<T,Rest...>& expr() const {return _expr;};

    FASTOR_INLINE TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>& noalias() {
        _does_alias = true;
        return *this;
    }

    constexpr FASTOR_INLINE TensorRandomViewExpr(Tensor<T,Rest...> &_ex, const Tensor<Int,IterSizes...> &_it) : _expr(_ex), it_expr(_it) {
        static_assert(sizeof...(Rest)==DIMS && sizeof...(IterSizes)==DIMS, "INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    }

    // Copy assignment
    //------------------------------------------------------------------------------------//
    FASTOR_HINT_INLINE void operator=(const TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS> &other_src) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>(tmp_this_tensor,it_expr);
            // Assign other to temporary
            tmp = other_src;
            // assign temporary to this
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
        T *_data = _expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _data[it_expr.data()[i+j]] = _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _data[it_expr.data()[i]] = other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _data[it_expr.data()[i]] = other_src.template eval_s<T>(i);
        }
#endif
    }

    // AbstractTensor overloads
    //------------------------------------------------------------------------------------//
    template<typename Derived, size_t OTHER_DIMS>
    FASTOR_HINT_INLINE void operator=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>(tmp_this_tensor,it_expr);
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
        T *_data = _expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_IF_CONSTEXPR(is_boolean_expression_v<Derived>) {
            for (FASTOR_INDEX i = 0; i <size(); i++) {
                _data[it_expr.data()[i]] = other_src.template eval_s<T>(i);
            }
            return;
        }
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _data[it_expr.data()[i+j]] = _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _data[it_expr.data()[i]] = other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _data[it_expr.data()[i]] = other_src.template eval_s<T>(i);
        }
#endif
    }

    template<typename Derived, size_t OTHER_DIMS>
    FASTOR_HINT_INLINE void operator+=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>(tmp_this_tensor,it_expr);
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
        T *_data = _expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _data[it_expr.data()[i+j]] += _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _data[it_expr.data()[i]] += other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _data[it_expr.data()[i]] += other_src.template eval_s<T>(i);
        }
#endif
    }

    template<typename Derived, size_t OTHER_DIMS>
    FASTOR_HINT_INLINE void operator-=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>(tmp_this_tensor,it_expr);
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
        T *_data = _expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _data[it_expr.data()[i+j]] -= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _data[it_expr.data()[i]] -= other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _data[it_expr.data()[i]] -= other_src.template eval_s<T>(i);
        }
#endif
    }

    template<typename Derived, size_t OTHER_DIMS>
    FASTOR_HINT_INLINE void operator*=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>(tmp_this_tensor,it_expr);
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
        T *_data = _expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _data[it_expr.data()[i+j]] *= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _data[it_expr.data()[i]] *= other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _data[it_expr.data()[i]] *= other_src.template eval_s<T>(i);
        }
#endif
    }

    template<typename Derived, size_t OTHER_DIMS>
    FASTOR_HINT_INLINE void operator/=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>(tmp_this_tensor,it_expr);
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
        T *_data = _expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _data[it_expr.data()[i+j]] /= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _data[it_expr.data()[i]] /= other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _data[it_expr.data()[i]] /= other_src.template eval_s<T>(i);
        }
#endif
    }

    // Scalar binders
    //------------------------------------------------------------------------------------//
    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator=(U num) {
        T *_data = _expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,simd_abi_type> _vec_other(num);
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _data[it_expr.data()[i+j]] = _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _data[it_expr.data()[i]] = num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _data[it_expr.data()[i]] = num;
        }
#endif
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator+=(U num) {
        T *_data = _expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,simd_abi_type> _vec_other(num);
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _data[it_expr.data()[i+j]] += _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _data[it_expr.data()[i]] += num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _data[it_expr.data()[i]] += num;
        }
#endif
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator-=(U num) {
        T *_data = _expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,simd_abi_type> _vec_other(num);
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _data[it_expr.data()[i+j]] -= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _data[it_expr.data()[i]] -= num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _data[it_expr.data()[i]] -= num;
        }
#endif
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator*=(U num) {
        T *_data = _expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,simd_abi_type> _vec_other(num);
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _data[it_expr.data()[i+j]] *= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _data[it_expr.data()[i]] *= num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _data[it_expr.data()[i]] *= num;
        }
#endif
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U> && !is_integral_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(U num) {
        T *_data = _expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        T inum = T(1.)/(T)num;
        SIMDVector<T,simd_abi_type> _vec_other(inum);
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _data[it_expr.data()[i+j]] *= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _data[it_expr.data()[i]] *= inum;
        }
#else
        T inum = T(1.)/num;
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _data[it_expr.data()[i]] *= inum;
        }
#endif
    }
    template<typename U=T, enable_if_t_<is_primitive_v_<U> && is_integral_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(U num) {
        T *_data = _expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,simd_abi_type> _vec_other((T)num);
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                _data[it_expr.data()[i+j]] /= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _data[it_expr.data()[i]] /= num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _data[it_expr.data()[i]] /= num;
        }
#endif
    }
    //------------------------------------------------------------------------------------//

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i) const {
        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,SIMDVector<T,simd_abi_type>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j)
            inds[j] = it_expr.data()[i+j];
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return _expr.data()[it_expr.data()[i]];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX k) const {
        i += k;
        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,SIMDVector<T,simd_abi_type>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j)
            inds[j] = it_expr.data()[i+j];
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _expr.data()[it_expr.data()[i+j]];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int,DIMS>& as) const {
        int i = std::accumulate(as.begin(), as.end(), 0);
        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,SIMDVector<T,simd_abi_type>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j)
            inds[j] = it_expr.data()[i+j];
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U teval_s(const std::array<int,DIMS>& as) const {
        int i = std::accumulate(as.begin(), as.end(), 0);
        return _expr.data()[it_expr.data()[i]];
    }
};




}




#endif // TENSOR_RANDOM_VIEWS_H
