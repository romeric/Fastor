#ifndef TENSOR_VIEWS_1D_H
#define TENSOR_VIEWS_1D_H


#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/Ranges.h"

namespace Fastor {



// 1D const views
//-------------------------------------------------------------------------------------//
template<typename T, size_t N>
struct TensorConstViewExpr<Tensor<T,N>,1>: public AbstractTensor<TensorConstViewExpr<Tensor<T,N>,1>,1> {
private:
    const Tensor<T,N> &expr;
    seq _seq;
public:
    using scalar_type = T;
    static constexpr FASTOR_INDEX Dimension = 1;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    static constexpr FASTOR_INDEX rank() {return 1;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return _seq.size();}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX ) const {return _seq.size();}

    FASTOR_INLINE TensorConstViewExpr(const Tensor<T,N> &_ex, const seq &_s) : expr(_ex), _seq(_s) {
        if (_seq._last < 0) _seq._last += N + /*including the end point*/ 1;
        if (_seq._first < 0) _seq._first += N + /*including the end point*/ 1;
    }

    template<typename U>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        vector_setter(_vec,expr.data(),i*_seq._step+_seq._first,_seq._step);
        return _vec;
    }

    template<typename U>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return expr.data()[i*_seq._step+_seq._first];
    }

    template<typename U>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        vector_setter(_vec,expr.data(),(i+j)*_seq._step+_seq._first,_seq._step);
        return _vec;
    }

    template<typename U>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return expr.data()[(i+j)*_seq._step+_seq._first];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> teval(const std::array<int,1>& as) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        vector_setter(_vec,expr.data(),as[0]*_seq._step+_seq._first,_seq._step);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,1>& as) const {
        return expr.data()[as[0]*_seq._step+_seq._first];
    }
};















// 1D non-const views
//-------------------------------------------------------------------------------------//
template<typename T, size_t N>
struct TensorViewExpr<Tensor<T,N>,1>: public AbstractTensor<TensorViewExpr<Tensor<T,N>,1>,1> {
private:
    Tensor<T,N> &expr;
    seq _seq;
    bool does_alias = false;
    constexpr FASTOR_INLINE Tensor<T,N> get_tensor() const {return expr;}
public:
    using scalar_type = T;
    static constexpr FASTOR_INDEX Dimension = 1;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    static constexpr FASTOR_INDEX rank() {return 1;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return _seq.size();}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX ) const {return _seq.size();}

    FASTOR_INLINE TensorViewExpr<Tensor<T,N>,1>& noalias() {
        does_alias = true;
        return *this;
    }

    FASTOR_INLINE TensorViewExpr(Tensor<T,N> &_ex, const seq &_s) : expr(_ex), _seq(_s) {
        if (_seq._last < 0) _seq._last += N + /*including the end point*/ 1;
        if (_seq._first < 0) _seq._first += N + /*including the end point*/ 1;
    }

    // View evalution operators
    // Copy assignment operators [Needed in addition to generic AbstractTensor overload]
    //----------------------------------------------------------------------------------//
    void operator=(const TensorViewExpr<Tensor<T,N>,1> &other) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,N>,1>(tmp_this_tensor,_seq);
            // Assign other to temporary
            tmp = other;
            // assign temporary to this
            this->operator=(tmp);
            return;
        }
#endif
#ifndef NDEBUG
        FASTOR_ASSERT(other.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match - for this 1D case for loop unnecessary
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                auto idx = (i+j)*_seq._step+_seq._first;
                _data[idx] = _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] = other.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] = other.template eval_s<T>(i);
        }
#endif
    }


    void operator+=(const TensorViewExpr<Tensor<T,N>,1> &other) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,N>,1>(tmp_this_tensor,_seq);
            // Assign other to temporary
            tmp = other;
            // assign temporary to this
            this->operator+=(tmp);
            return;
        }
#endif
#ifndef NDEBUG
        FASTOR_ASSERT(other.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                auto idx = (i+j)*_seq._step+_seq._first;
                _data[idx] += _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] += other.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] += other.template eval_s<T>(i);
        }
#endif
    }


    void operator-=(const TensorViewExpr<Tensor<T,N>,1> &other) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,N>,1>(tmp_this_tensor,_seq);
            // Assign other to temporary
            tmp = other;
            // assign temporary to this
            this->operator-=(tmp);
            return;
        }
#endif
#ifndef NDEBUG
        FASTOR_ASSERT(other.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                auto idx = (i+j)*_seq._step+_seq._first;
                _data[idx] -= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] -= other.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] -= other.template eval_s<T>(i);
        }
#endif
    }


    void operator*=(const TensorViewExpr<Tensor<T,N>,1> &other) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,N>,1>(tmp_this_tensor,_seq);
            // Assign other to temporary
            tmp = other;
            // assign temporary to this
            this->operator*=(tmp);
            return;
        }
#endif
#ifndef NDEBUG
        FASTOR_ASSERT(other.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                auto idx = (i+j)*_seq._step+_seq._first;
                _data[idx] *= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] *= other.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] *= other.template eval_s<T>(i);
        }
#endif
    }


    void operator/=(const TensorViewExpr<Tensor<T,N>,1> &other) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,N>,1>(tmp_this_tensor,_seq);
            // Assign other to temporary
            tmp = other;
            // assign temporary to this
            this->operator/=(tmp);
            return;
        }
#endif
#ifndef NDEBUG
        FASTOR_ASSERT(other.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            auto _vec_other = other.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                auto idx = (i+j)*_seq._step+_seq._first;
                _data[idx] /= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] /= other.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] /= other.template eval_s<T>(i);
        }
#endif
    }



    // AbstractTensor binders
    //----------------------------------------------------------------------------------//
    template<typename Derived, size_t DIMS>
    void operator=(const AbstractTensor<Derived,DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,N>,1>(tmp_this_tensor,_seq);
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
                auto idx = (i+j)*_seq._step+_seq._first;
                _data[idx] = _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] = other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] = other_src.template eval_s<T>(i);
        }
#endif
    }

    template<typename Derived, size_t DIMS>
    void operator+=(const AbstractTensor<Derived,DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,N>,1>(tmp_this_tensor,_seq);
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
                auto idx = (i+j)*_seq._step+_seq._first;
                _data[idx] += _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] += other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] += other_src.template eval_s<T>(i);
        }
#endif
    }

    template<typename Derived, size_t DIMS>
    void operator-=(const AbstractTensor<Derived,DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,N>,1>(tmp_this_tensor,_seq);
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
                auto idx = (i+j)*_seq._step+_seq._first;
                _data[idx] -= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] -= other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] -= other_src.template eval_s<T>(i);
        }
#endif
    }

    template<typename Derived, size_t DIMS>
    void operator*=(const AbstractTensor<Derived,DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,N>,1>(tmp_this_tensor,_seq);
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
                auto idx = (i+j)*_seq._step+_seq._first;
                _data[idx] *= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] *= other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] *= other_src.template eval_s<T>(i);
        }
#endif
    }

    template<typename Derived, size_t DIMS>
    void operator/=(const AbstractTensor<Derived,DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,N>,1>(tmp_this_tensor,_seq);
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
                auto idx = (i+j)*_seq._step+_seq._first;
                _data[idx] /= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] /= other_src.template eval_s<T>(i);
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] /= other_src.template eval_s<T>(i);
        }
#endif
    }
    //----------------------------------------------------------------------------------//

    // scalar binders
    //----------------------------------------------------------------------------------//
    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator=(U num) {
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                auto idx = (i+j)*_seq._step+_seq._first;
                _data[idx] = _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] = num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] = num;
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator+=(U num) {
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                auto idx = (i+j)*_seq._step+_seq._first;
                _data[idx] += _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] += num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] += num;
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator-=(U num) {
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                auto idx = (i+j)*_seq._step+_seq._first;
                _data[idx] -= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] -= num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] -= num;
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator*=(U num) {
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                auto idx = (i+j)*_seq._step+_seq._first;
                _data[idx] *= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] *= num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] *= num;
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator/=(U num) {
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                auto idx = (i+j)*_seq._step+_seq._first;
                _data[idx] /= _vec_other[j];
            }
        }
        for (; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] /= num;
        }
#else
        for (FASTOR_INDEX i = 0; i <size(); i++) {
            auto idx = i*_seq._step+_seq._first;
            _data[idx] /= num;
        }
#endif
    }
    //----------------------------------------------------------------------------------//

    template<typename U>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        vector_setter(_vec,expr.data(),i*_seq._step+_seq._first,_seq._step);
        return _vec;
    }

    template<typename U>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return expr.data()[i*_seq._step+_seq._first];
    }

    template<typename U>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        vector_setter(_vec,expr.data(),(i+j)*_seq._step+_seq._first,_seq._step);
        return _vec;
    }

    template<typename U>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return expr.data()[(i+j)*_seq._step+_seq._first];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> teval(const std::array<int,1>& as) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        vector_setter(_vec,expr.data(),as[0]*_seq._step+_seq._first,_seq._step);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,1>& as) const {
        return expr.data()[as[0]*_seq._step+_seq._first];
    }
};


} // end of namespace Fastor


#endif // TENSOR_VIEWS_1D_H
