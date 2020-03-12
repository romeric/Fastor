#ifndef TENSOR_RANDOM_VIEWS_H
#define TENSOR_RANDOM_VIEWS_H

#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/Ranges.h"


namespace Fastor {


// Const versions
//----------------------------------------------------------------------------------//
template<typename T, size_t N, typename Int, size_t IterSize>
struct TensorConstRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>:
    public AbstractTensor<TensorConstRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>,1> {
private:
    const Tensor<T,N> &expr;
    const Tensor<Int,IterSize> &it_expr;
public:
    using scalar_type = T;
    static constexpr FASTOR_INDEX Dimension = 1;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    static constexpr FASTOR_INDEX rank() {return 1;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return IterSize;}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX ) const {return IterSize;}

    constexpr FASTOR_INLINE TensorConstRandomViewExpr(const Tensor<T,N> &_ex, const Tensor<Int,IterSize> &_it) : expr(_ex), it_expr(_it) {}

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,SIMDVector<T,DEFAULT_ABI>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j)
            inds[j] = it_expr.data()[i+j];
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return expr.data()[it_expr.data()[i]];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX k) const {
        i += k;
        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,SIMDVector<T,DEFAULT_ABI>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j)
            inds[j] = it_expr.data()[i+j];
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return expr(it_expr(i+j));
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> teval(const std::array<int,1>& as) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,SIMDVector<T,DEFAULT_ABI>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j)
            inds[j] = it_expr.data()[as[0]+j];
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,1>& as) const {
        return expr.data()[it_expr.data()[as[0]]];
    }
};


//----------------------------------------------------------------------------------//
template<typename T, size_t ... Rest, typename Int, size_t ... IterSizes, size_t DIMS>
struct TensorConstRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>:
    public AbstractTensor<TensorConstRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>,DIMS> {
private:
    const Tensor<T,Rest...> &expr;
    // The index tensor needs to be copied for the purpose of indexing a
    // multidimensional tensor with multiple 1D or any other lower other
    // tensors. This issue could be solved by providing another type parameter/flag
    // such as CopyIndex to the class which would use std::conditional to choose
    // between "const Tensor<Int,IterSizes...> &" and "Tensor<Int,IterSizes...>"

    // const Tensor<Int,IterSizes...> &it_expr;
    Tensor<Int,IterSizes...> it_expr;
public:
    using scalar_type = T;
    static constexpr FASTOR_INDEX Dimension = DIMS;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    static constexpr FASTOR_INDEX rank() {return DIMS;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return prod<IterSizes...>::value;}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return it_expr.dimension(i);}

    constexpr FASTOR_INLINE TensorConstRandomViewExpr(const Tensor<T,Rest...> &_ex,
        const Tensor<Int,IterSizes...> &_it) : expr(_ex), it_expr(_it) {
        static_assert(sizeof...(Rest)==DIMS && sizeof...(IterSizes)==DIMS, "INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,SIMDVector<T,DEFAULT_ABI>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j)
            inds[j] = it_expr.data()[i+j];
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return expr.data()[it_expr.data()[i]];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX k) const {
        i += k;
        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,SIMDVector<T,DEFAULT_ABI>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j)
            inds[j] = it_expr.data()[i+j];
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return expr.data()[it_expr.data()[i+j]];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> teval(const std::array<int,DIMS>& as) const {
        int i = std::accumulate(as.begin(), as.end(), 0);
        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,SIMDVector<T,DEFAULT_ABI>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j)
            inds[j] = it_expr.data()[i+j];
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U teval_s(const std::array<int,DIMS>& as) const {
        int i = std::accumulate(as.begin(), as.end(), 0);
        return expr.data()[it_expr.data()[i]];
    }

};

//----------------------------------------------------------------------------------//





template<typename T, size_t N, typename Int, size_t IterSize>
struct TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>:
    public AbstractTensor<TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>,1> {
private:
    Tensor<T,N> &expr;
    const Tensor<Int,IterSize> &it_expr;
    bool does_alias = false;

    constexpr FASTOR_INLINE Tensor<T,N> get_tensor() const {return expr;};
public:
    using scalar_type = T;
    static constexpr FASTOR_INDEX Dimension = 1;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    static constexpr FASTOR_INDEX rank() {return 1;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return IterSize;}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX ) const {return IterSize;}

    FASTOR_INLINE TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>& noalias() {
        does_alias = true;
        return *this;
    }

    constexpr FASTOR_INLINE TensorRandomViewExpr(Tensor<T,N> &_ex, const Tensor<Int,IterSize> &_it) : expr(_ex), it_expr(_it) {}


    // View evalution operators
    // Copy assignment operators
    //----------------------------------------------------------------------------------//
    void operator=(const TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1> &other_src) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
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
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                expr(it_expr(i+j)) = _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            expr(it_expr(i)) = other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            expr(it_expr(i)) = other_src.template eval_s<T>(i);
        }
#endif
    }

    void operator+=(const TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1> &other_src) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>(tmp_this_tensor,it_expr);
            // Assign other to temporary
            tmp = other_src;
            // assign temporary to this
            this->operator+=(tmp);
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
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                expr(it_expr(i+j)) += _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            expr(it_expr(i)) += other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            expr(it_expr(i)) += other_src.template eval_s<T>(i);
        }
#endif
    }

    void operator-=(const TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1> &other_src) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>(tmp_this_tensor,it_expr);
            // Assign other to temporary
            tmp = other_src;
            // assign temporary to this
            this->operator-=(tmp);
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
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                expr(it_expr(i+j)) -= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            expr(it_expr(i)) -= other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            expr(it_expr(i)) -= other_src.template eval_s<T>(i);
        }
#endif
    }

    void operator*=(const TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1> &other_src) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>(tmp_this_tensor,it_expr);
            // Assign other to temporary
            tmp = other_src;
            // assign temporary to this
            this->operator*=(tmp);
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
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                expr(it_expr(i+j)) *= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            expr(it_expr(i)) *= other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            expr(it_expr(i)) *= other_src.template eval_s<T>(i);
        }
#endif
    }

    void operator/=(const TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1> &other_src) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorRandomViewExpr<Tensor<T,N>,Tensor<Int,IterSize>,1>(tmp_this_tensor,it_expr);
            // Assign other to temporary
            tmp = other_src;
            // assign temporary to this
            this->operator/=(tmp);
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
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                expr(it_expr(i+j)) /= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            expr(it_expr(i)) /= other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            expr(it_expr(i)) /= other_src.template eval_s<T>(i);
        }
#endif
    }

    // AbstractTensor binders
    //----------------------------------------------------------------------------------//
    template<typename Derived, size_t DIMS>
    void operator=(const AbstractTensor<Derived,DIMS> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
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
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                expr(it_expr(i+j)) = _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            expr(it_expr(i)) = other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            expr(it_expr(i)) = other_src.template eval_s<T>(i);
        }
#endif
    }

    template<typename Derived, size_t DIMS>
    void operator+=(const AbstractTensor<Derived,DIMS> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
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
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                expr(it_expr(i+j)) += _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            expr(it_expr(i)) += other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            expr(it_expr(i)) += other_src.template eval_s<T>(i);
        }
#endif
    }

    template<typename Derived, size_t DIMS>
    void operator-=(const AbstractTensor<Derived,DIMS> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
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
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                expr(it_expr(i+j)) -= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            expr(it_expr(i)) -= other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            expr(it_expr(i)) -= other_src.template eval_s<T>(i);
        }
#endif
    }

    template<typename Derived, size_t DIMS>
    void operator*=(const AbstractTensor<Derived,DIMS> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
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
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                expr(it_expr(i+j)) *= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            expr(it_expr(i)) *= other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            expr(it_expr(i)) *= other_src.template eval_s<T>(i);
        }
#endif
    }

    template<typename Derived, size_t DIMS>
    void operator/=(const AbstractTensor<Derived,DIMS> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
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
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                expr(it_expr(i+j)) /= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            expr(it_expr(i)) /= other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            expr(it_expr(i)) /= other_src.template eval_s<T>(i);
        }
#endif
    }
    //----------------------------------------------------------------------------------//


    // scalar binders
    //----------------------------------------------------------------------------------//
    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator=(U num) {

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                expr(it_expr(i+j)) = _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            expr(it_expr(i)) = num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            expr(it_expr(i)) = num;
        }
#endif
    }


    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator+=(U num) {

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                expr(it_expr(i+j)) += _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            expr(it_expr(i)) += num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            expr(it_expr(i)) += num;
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator-=(U num) {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                expr(it_expr(i+j)) -= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            expr(it_expr(i)) -= num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            expr(it_expr(i)) -= num;
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator*=(U num) {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                expr(it_expr(i+j)) *= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            expr(it_expr(i)) *= num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            expr(it_expr(i)) *= num;
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator/=(U num) {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        T inum = T(1.)/(T)num;
        SIMDVector<T,DEFAULT_ABI> _vec_other(inum);
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                expr(it_expr(i+j)) *= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            expr(it_expr(i)) *= inum;
        }
#else
        T inum = T(1.)/num;
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            expr(it_expr(i)) *= inum;
        }
#endif
    }
    //----------------------------------------------------------------------------------//

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,SIMDVector<T,DEFAULT_ABI>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j)
            inds[j] = it_expr(i+j);
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return expr(it_expr(i));
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX k) const {
        i += k;
        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,SIMDVector<T,DEFAULT_ABI>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j)
            inds[j] = it_expr(i+j);
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return expr(it_expr(i+j));
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> teval(const std::array<int,1>& as) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,SIMDVector<T,DEFAULT_ABI>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j)
            inds[j] = it_expr.data()[as[0]+j];
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,1>& as) const {
        return expr.data()[it_expr.data()[as[0]]];
    }

};



























































template<typename T, size_t ... Rest, typename Int, size_t ... IterSizes, size_t DIMS>
struct TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>:
    public AbstractTensor<TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>,DIMS> {
private:
    Tensor<T,Rest...> &expr;
    // The index tensor needs to be copied for the purpose of indexing a
    // multidimensional tensor with multiple 1D or any other lower other
    // tensors. This issue could be solved by providing another type parameter/flag
    // such as CopyIndex to the class which would use std::conditional to choose
    // between "const Tensor<Int,IterSizes...> &" and "Tensor<Int,IterSizes...>"

    // const Tensor<Int,IterSizes...> &it_expr;
    Tensor<Int,IterSizes...> it_expr;
    bool does_alias = false;

    constexpr FASTOR_INLINE Tensor<T,Rest...> get_tensor() const {return expr;}
public:
    using scalar_type = T;
    static constexpr FASTOR_INDEX Dimension = DIMS;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    static constexpr FASTOR_INDEX rank() {return DIMS;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return prod<IterSizes...>::value;}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return it_expr.dimension(i);}

    FASTOR_INLINE TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>& noalias() {
        does_alias = true;
        return *this;
    }

    constexpr FASTOR_INLINE TensorRandomViewExpr(Tensor<T,Rest...> &_ex, const Tensor<Int,IterSizes...> &_it) : expr(_ex), it_expr(_it) {
        static_assert(sizeof...(Rest)==DIMS && sizeof...(IterSizes)==DIMS, "INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    }

    // Copy assignment
    //------------------------------------------------------------------------------------//
    void operator=(const TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS> &other_src) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
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
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
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

    void operator+=(const TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS> &other_src) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>(tmp_this_tensor,it_expr);
            // Assign other to temporary
            tmp = other_src;
            // assign temporary to this
            this->operator+=(tmp);
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
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
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

    void operator-=(const TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS> &other_src) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>(tmp_this_tensor,it_expr);
            // Assign other to temporary
            tmp = other_src;
            // assign temporary to this
            this->operator-=(tmp);
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
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
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

    void operator*=(const TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS> &other_src) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>(tmp_this_tensor,it_expr);
            // Assign other to temporary
            tmp = other_src;
            // assign temporary to this
            this->operator*=(tmp);
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
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
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

    void operator/=(const TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS> &other_src) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,DIMS>(tmp_this_tensor,it_expr);
            // Assign other to temporary
            tmp = other_src;
            // assign temporary to this
            this->operator/=(tmp);
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
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
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

    // AbstractTensor overloads
    //------------------------------------------------------------------------------------//
    template<typename Derived, size_t OTHER_DIMS>
    void operator=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
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
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
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
    void operator+=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
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
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
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
    void operator-=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
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
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
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
    void operator*=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
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
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
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
    void operator/=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
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
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
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
    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator=(U num) {
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(num);
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                _data[it_expr.data()[i+j]] = _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _data[it_expr.data()[i]] = num;
        }
#else
        T inum = T(1.)/num;
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _data[it_expr.data()[i]] = num;
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator+=(U num) {
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(num);
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                _data[it_expr.data()[i+j]] += _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _data[it_expr.data()[i]] += num;
        }
#else
        T inum = T(1.)/num;
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _data[it_expr.data()[i]] += num;
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator-=(U num) {
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(num);
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                _data[it_expr.data()[i+j]] -= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _data[it_expr.data()[i]] -= num;
        }
#else
        T inum = T(1.)/num;
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _data[it_expr.data()[i]] -= num;
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator*=(U num) {
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(num);
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                _data[it_expr.data()[i+j]] *= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            _data[it_expr.data()[i]] *= num;
        }
#else
        T inum = T(1.)/num;
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            _data[it_expr.data()[i]] *= num;
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator/=(U num) {
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        T inum = T(1.)/(T)num;
        SIMDVector<T,DEFAULT_ABI> _vec_other(inum);
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
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
    //------------------------------------------------------------------------------------//

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,SIMDVector<T,DEFAULT_ABI>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j)
            inds[j] = it_expr.data()[i+j];
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return expr.data()[it_expr.data()[i]];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX k) const {
        i += k;
        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,SIMDVector<T,DEFAULT_ABI>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j)
            inds[j] = it_expr.data()[i+j];
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return expr.data()[it_expr.data()[i+j]];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> teval(const std::array<int,DIMS>& as) const {
        int i = std::accumulate(as.begin(), as.end(), 0);
        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,SIMDVector<T,DEFAULT_ABI>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j)
            inds[j] = it_expr.data()[i+j];
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U teval_s(const std::array<int,DIMS>& as) const {
        int i = std::accumulate(as.begin(), as.end(), 0);
        return expr.data()[it_expr.data()[i]];
    }
};




}




#endif // TENSOR_RANDOM_VIEWS_H