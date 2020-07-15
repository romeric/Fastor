#ifndef TENSOR_VIEWS_1D_H
#define TENSOR_VIEWS_1D_H


#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/Ranges.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"

namespace Fastor {



// 1D const views
//-------------------------------------------------------------------------------------//
template<typename T, size_t N>
struct TensorConstViewExpr<Tensor<T,N>,1>: public AbstractTensor<TensorConstViewExpr<Tensor<T,N>,1>,1> {
private:
    const Tensor<T,N> &_expr;
    seq _seq;
public:
    using scalar_type = T;
    using simd_vector_type = typename Tensor<T,N>::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using V = simd_vector_type;
    using result_type = Tensor<T,N>;
    static constexpr FASTOR_INDEX Dimension = 1;
    static constexpr FASTOR_INDEX Stride = simd_vector_type::Size;
    static constexpr FASTOR_INDEX rank() {return 1;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return _seq.size();}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX ) const {return _seq.size();}
    constexpr const Tensor<T,N>& expr() const {return _expr;}

    FASTOR_INLINE TensorConstViewExpr(const Tensor<T,N> &_ex, const seq &_s) : _expr(_ex), _seq(_s) {
        if (_seq._last < 0) _seq._last += N + /*including the end point*/ 1;
        if (_seq._first < 0) _seq._first += N + /*including the end point*/ 1;
    }

    template<typename U>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i) const {
        SIMDVector<U,simd_abi_type> _vec;
        vector_setter(_vec,_expr.data(),i*_seq._step+_seq._first,_seq._step);
        return _vec;
    }

    template<typename U>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return _expr.data()[i*_seq._step+_seq._first];
    }

    template<typename U>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        SIMDVector<U,simd_abi_type> _vec;
        vector_setter(_vec,_expr.data(),(i+j)*_seq._step+_seq._first,_seq._step);
        return _vec;
    }

    template<typename U>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _expr.data()[(i+j)*_seq._step+_seq._first];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int,1>& as) const {
        SIMDVector<U,simd_abi_type> _vec;
        vector_setter(_vec,_expr.data(),as[0]*_seq._step+_seq._first,_seq._step);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,1>& as) const {
        return _expr.data()[as[0]*_seq._step+_seq._first];
    }
};















// 1D non-const views
//-------------------------------------------------------------------------------------//
template<typename T, size_t N>
struct TensorViewExpr<Tensor<T,N>,1>: public AbstractTensor<TensorViewExpr<Tensor<T,N>,1>,1> {
private:
    Tensor<T,N> &_expr;
    seq _seq;
    bool _does_alias = false;
    constexpr FASTOR_INLINE Tensor<T,N> get_tensor() const {return _expr;}
public:
    using scalar_type = T;
    using simd_vector_type = typename Tensor<T,N>::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using V = simd_vector_type;
    using result_type = Tensor<T,N>;
    static constexpr FASTOR_INDEX Dimension = 1;
    static constexpr FASTOR_INDEX Stride = simd_vector_type::Size;
    static constexpr FASTOR_INDEX rank() {return 1;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return _seq.size();}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX ) const {return _seq.size();}
    constexpr const Tensor<T,N>& expr() const {return _expr;}

    FASTOR_INLINE TensorViewExpr<Tensor<T,N>,1>& noalias() {
        _does_alias = true;
        return *this;
    }

    FASTOR_INLINE TensorViewExpr(Tensor<T,N> &_ex, const seq &_s) : _expr(_ex), _seq(_s) {
        if (_seq._last < 0) _seq._last += N + /*including the end point*/ 1;
        if (_seq._first < 0) _seq._first += N + /*including the end point*/ 1;
    }

    // View evalution operators
    // Copy assignment operators [Needed in addition to generic AbstractTensor overload]
    //----------------------------------------------------------------------------------//
    FASTOR_HINT_INLINE void operator=(const TensorViewExpr<Tensor<T,N>,1> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
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
        T *FASTOR_RESTRICT _data = _expr.data();
        if (_seq._step == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = other.template eval<T>(i);
                _vec.store(&_data[i+_seq._first],false);
            }
            for (; i <size(); i++) {
                _data[i+_seq._first] = other.template eval_s<T>(i);
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec_other = other.template eval<T>(i);
                for (auto j=0; j<simd_vector_type::Size; ++j) {
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
    }
    //----------------------------------------------------------------------------------//

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
        T *FASTOR_RESTRICT _data = _expr.data();
        FASTOR_IF_CONSTEXPR(is_boolean_expression_v<Derived>) {
            for (FASTOR_INDEX i = 0; i <size(); i++) {
                auto idx = i*_seq._step+_seq._first;
                _data[idx] = other_src.template eval_s<T>(i);
            }
            return;
        }
        if (_seq._step == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = other_src.template eval<T>(i);
                _vec.store(&_data[i+_seq._first],false);
            }
            for (; i <size(); i++) {
                _data[i+_seq._first] = other_src.template eval_s<T>(i);
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec_other = other_src.template eval<T>(i);
                for (auto j=0; j<simd_vector_type::Size; ++j) {
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
        T *FASTOR_RESTRICT _data = _expr.data();
        if (_seq._step == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = this->template eval<T>(i) + other_src.template eval<T>(i);
                _vec.store(&_data[i+_seq._first],false);
            }
            for (; i <size(); i++) {
                _data[i+_seq._first] += other_src.template eval_s<T>(i);
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec_other = other_src.template eval<T>(i);
                for (auto j=0; j<simd_vector_type::Size; ++j) {
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
        T *FASTOR_RESTRICT _data = _expr.data();
        if (_seq._step == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = this->template eval<T>(i) - other_src.template eval<T>(i);
                _vec.store(&_data[i+_seq._first],false);
            }
            for (; i <size(); i++) {
                _data[i+_seq._first] -= other_src.template eval_s<T>(i);
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec_other = other_src.template eval<T>(i);
                for (auto j=0; j<simd_vector_type::Size; ++j) {
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
        T *FASTOR_RESTRICT _data = _expr.data();
        if (_seq._step == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = this->template eval<T>(i) * other_src.template eval<T>(i);
                _vec.store(&_data[i+_seq._first],false);
            }
            for (; i <size(); i++) {
                _data[i+_seq._first] *= other_src.template eval_s<T>(i);
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec_other = other_src.template eval<T>(i);
                for (auto j=0; j<simd_vector_type::Size; ++j) {
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
        T *FASTOR_RESTRICT _data = _expr.data();
        if (_seq._step == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = this->template eval<T>(i) / other_src.template eval<T>(i);
                _vec.store(&_data[i+_seq._first],false);
            }
            for (; i <size(); i++) {
                _data[i+_seq._first] /= other_src.template eval_s<T>(i);
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec_other = other_src.template eval<T>(i);
                for (auto j=0; j<simd_vector_type::Size; ++j) {
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
    }
    //----------------------------------------------------------------------------------//

    // scalar binders
    //----------------------------------------------------------------------------------//
    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator=(U num) {
        simd_vector_type _vec_other(static_cast<T>(num));
        T *FASTOR_RESTRICT _data = _expr.data();
        if (_seq._step == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                _vec_other.store(&_data[i+_seq._first],false);
            }
            for (; i <size(); i++) {
                _data[i+_seq._first] = num;
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                for (auto j=0; j<simd_vector_type::Size; ++j) {
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
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator+=(U num) {
        simd_vector_type _vec_other(static_cast<T>(num));
        T *FASTOR_RESTRICT _data = _expr.data();
        if (_seq._step == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = this->template eval<T>(i) + _vec_other;
                _vec.store(&_data[i+_seq._first],false);
            }
            for (; i <size(); i++) {
                _data[i+_seq._first] += num;
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                for (auto j=0; j<simd_vector_type::Size; ++j) {
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
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator-=(U num) {
        simd_vector_type _vec_other(static_cast<T>(num));
        T *FASTOR_RESTRICT _data = _expr.data();
        if (_seq._step == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = this->template eval<T>(i) - _vec_other;
                _vec.store(&_data[i+_seq._first],false);
            }
            for (; i <size(); i++) {
                _data[i+_seq._first] -= num;
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                for (auto j=0; j<simd_vector_type::Size; ++j) {
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
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator*=(U num) {
        simd_vector_type _vec_other(static_cast<T>(num));
        T *FASTOR_RESTRICT _data = _expr.data();
        if (_seq._step == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = this->template eval<T>(i) * _vec_other;
                _vec.store(&_data[i+_seq._first],false);
            }
            for (; i <size(); i++) {
                _data[i+_seq._first] *= num;
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                for (auto j=0; j<simd_vector_type::Size; ++j) {
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
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U> && !is_integral_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(U num) {
        T inum = T(1) / static_cast<T>(num);
        simd_vector_type _vec_other(inum);
        T *FASTOR_RESTRICT _data = _expr.data();
        if (_seq._step == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = this->template eval<T>(i) * _vec_other;
                _vec.store(&_data[i+_seq._first],false);
            }
            for (; i <size(); i++) {
                _data[i+_seq._first] *= inum;
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                for (auto j=0; j<simd_vector_type::Size; ++j) {
                    auto idx = (i+j)*_seq._step+_seq._first;
                    _data[idx] *= _vec_other[j];
                }
            }
            for (; i <size(); i++) {
                auto idx = i*_seq._step+_seq._first;
                _data[idx] *= inum;
            }
#else
            for (FASTOR_INDEX i = 0; i <size(); i++) {
                auto idx = i*_seq._step+_seq._first;
                _data[idx] *= inum;
            }
#endif
        }
    }
    template<typename U=T, enable_if_t_<is_primitive_v_<U> && is_integral_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(U num) {
        simd_vector_type _vec_other(static_cast<T>(num));
        T *FASTOR_RESTRICT _data = _expr.data();
        if (_seq._step == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = this->template eval<T>(i) / _vec_other;
                _vec.store(&_data[i+_seq._first],false);
            }
            for (; i <size(); i++) {
                _data[i+_seq._first] /= num;
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                for (auto j=0; j<simd_vector_type::Size; ++j) {
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
    }
    //----------------------------------------------------------------------------------//

    template<typename U>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i) const {
        SIMDVector<U,simd_abi_type> _vec;
        vector_setter(_vec,_expr.data(),i*_seq._step+_seq._first,_seq._step);
        return _vec;
    }

    template<typename U>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return _expr.data()[i*_seq._step+_seq._first];
    }

    template<typename U>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        SIMDVector<U,simd_abi_type> _vec;
        vector_setter(_vec,_expr.data(),(i+j)*_seq._step+_seq._first,_seq._step);
        return _vec;
    }

    template<typename U>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _expr.data()[(i+j)*_seq._step+_seq._first];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int,1>& as) const {
        SIMDVector<U,simd_abi_type> _vec;
        vector_setter(_vec,_expr.data(),as[0]*_seq._step+_seq._first,_seq._step);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,1>& as) const {
        return _expr.data()[as[0]*_seq._step+_seq._first];
    }
};


} // end of namespace Fastor


#endif // TENSOR_VIEWS_1D_H
