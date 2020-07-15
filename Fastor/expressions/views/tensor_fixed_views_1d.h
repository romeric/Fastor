#ifndef TENSOR_FIXED_VIEWS_1D_H
#define TENSOR_FIXED_VIEWS_1D_H


#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/Ranges.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"


namespace Fastor {


// 1D const fixed views
//----------------------------------------------------------------------------------------------//
template<typename T, size_t N, int F0, int L0, int S0>
struct TensorConstFixedViewExpr1D<Tensor<T,N>,fseq<F0,L0,S0>,1> :
    public AbstractTensor<TensorConstFixedViewExpr1D<Tensor<T,N>,fseq<F0,L0,S0>,1>,1> {
private:
    const Tensor<T,N> &_expr;
public:
    using scalar_type = T;
    using simd_vector_type = typename Tensor<T,N>::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using result_type = Tensor<T,range_detector<F0,L0,S0>::value>;
    static constexpr FASTOR_INDEX Dimension = 1;
    static constexpr FASTOR_INDEX Stride = simd_vector_type::Size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() { return range_detector<F0,L0,S0>::value;}
    static constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX ) {return range_detector<F0,L0,S0>::value;}
    constexpr const Tensor<T,N>& expr() const {return _expr;}

    constexpr FASTOR_INLINE TensorConstFixedViewExpr1D(const Tensor<T,N> &_ex) : _expr(_ex) {}

    template<typename U>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i) const {
        SIMDVector<U,simd_abi_type> _vec;
        vector_setter(_vec,_expr.data(),S0*i+F0,S0);
        return _vec;
    }

    template<typename U>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return _expr.data()[S0*i+F0];
    }

    template<typename U>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        SIMDVector<U,simd_abi_type> _vec;
        vector_setter(_vec,_expr.data(),S0*(i+j)+F0,S0);
        return _vec;
    }

    template<typename U>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _expr.data()[S0*(i+j)+F0];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int,1>& as) const {
        SIMDVector<U,simd_abi_type> _vec;
        vector_setter(_vec,_expr.data(),S0*as[0]+F0,S0);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,1>& as) const {
        return _expr.data()[S0*as[0]+F0];
    }
};





// 1D non-const fixed views
//----------------------------------------------------------------------------------------------//
template<typename T, size_t N, int F0, int L0, int S0>
struct TensorFixedViewExpr1D<Tensor<T,N>,fseq<F0,L0,S0>,1> :
    public AbstractTensor<TensorFixedViewExpr1D<Tensor<T,N>,fseq<F0,L0,S0>,1>,1> {
private:
    Tensor<T,N> &_expr;
    bool _does_alias = false;
    constexpr FASTOR_INLINE Tensor<T,N> get_tensor() const {return _expr;}
public:
    using scalar_type = T;
    using simd_vector_type = typename Tensor<T,N>::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using result_type = Tensor<T,range_detector<F0,L0,S0>::value>;
    static constexpr FASTOR_INDEX Dimension = 1;
    static constexpr FASTOR_INDEX Stride = simd_vector_type::Size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return range_detector<F0,L0,S0>::value;}
    static constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) {return range_detector<F0,L0,S0>::value;}
    constexpr const Tensor<T,N>& expr() const {return _expr;}

    FASTOR_INLINE TensorFixedViewExpr1D<Tensor<T,N>,fseq<F0,L0,S0>,1>& noalias() {
        _does_alias = true;
        return *this;
    }

    constexpr FASTOR_INLINE TensorFixedViewExpr1D(Tensor<T,N> &_ex) : _expr(_ex) {}


    //----------------------------------------------------------------------------------//
    FASTOR_HINT_INLINE void operator=(const TensorFixedViewExpr1D<Tensor<T,N>,fseq<F0,L0,S0>,1> &other_src) {
#ifndef FASTOR_NO_ALIAS
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorFixedViewExpr1D<Tensor<T,N>,fseq<F0,L0,S0>,1>(tmp_this_tensor);
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
        T *FASTOR_RESTRICT _data = _expr.data();
        FASTOR_IF_CONSTEXPR (S0 == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec_other = other_src.template eval<T>(i);
                _vec_other.store(&_data[i+F0],false);
            }
            for (; i <size(); i++) {
                _data[i+F0] = other_src.template eval_s<T>(i);
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec_other = other_src.template eval<T>(i);
                for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                    _data[S0*(i+j) + F0] = _vec_other[j];
                }
            }
            for (; i <size(); i++) {
                _data[S0*i+F0] = other_src.template eval_s<T>(i);
            }
#else
            for (FASTOR_INDEX i = 0; i <size(); i++) {
                _data[S0*i+F0] = other_src.template eval_s<T>(i);
            }
#endif
        }
    }
    //----------------------------------------------------------------------------------//

    //----------------------------------------------------------------------------------//
    template<typename Derived, size_t DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator=(const AbstractTensor<Derived,DIMS> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator=(tmp);
    }
    template<typename Derived, size_t DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator=(const AbstractTensor<Derived,DIMS> &other) {
#ifndef FASTOR_NO_ALIAS
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorFixedViewExpr1D<Tensor<T,N>,fseq<F0,L0,S0>,1>(tmp_this_tensor);
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
                _data[S0*i+F0] = other_src.template eval_s<T>(i);
            }
            return;
        }
        FASTOR_IF_CONSTEXPR (S0 == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec_other = other_src.template eval<T>(i);
                _vec_other.store(&_data[i+F0],false);
            }
            for (; i <size(); i++) {
                _data[i+F0] = other_src.template eval_s<T>(i);
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec_other = other_src.template eval<T>(i);
                for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                    _data[S0*(i+j) + F0] = _vec_other[j];
                }
            }
            for (; i <size(); i++) {
                _data[S0*i+F0] = other_src.template eval_s<T>(i);
            }
#else
            for (FASTOR_INDEX i = 0; i <size(); i++) {
                _data[S0*i+F0] = other_src.template eval_s<T>(i);
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
#ifndef FASTOR_NO_ALIAS
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorFixedViewExpr1D<Tensor<T,N>,fseq<F0,L0,S0>,1>(tmp_this_tensor);
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
        FASTOR_IF_CONSTEXPR (S0 == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = this->template eval<T>(i) + other_src.template eval<T>(i);
                _vec.store(&_data[i+F0],false);
            }
            for (; i <size(); i++) {
                _data[i+F0] += other_src.template eval_s<T>(i);
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec_other = other_src.template eval<T>(i);
                for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                    _data[S0*(i+j) + F0] += _vec_other[j];
                }
            }
            for (; i <size(); i++) {
                _data[S0*i+F0] += other_src.template eval_s<T>(i);
            }
#else
            for (FASTOR_INDEX i = 0; i <size(); i++) {
                _data[S0*i+F0] += other_src.template eval_s<T>(i);
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
#ifndef FASTOR_NO_ALIAS
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorFixedViewExpr1D<Tensor<T,N>,fseq<F0,L0,S0>,1>(tmp_this_tensor);
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
        FASTOR_IF_CONSTEXPR (S0 == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = this->template eval<T>(i) - other_src.template eval<T>(i);
                _vec.store(&_data[i+F0],false);
            }
            for (; i <size(); i++) {
                _data[i+F0] -= other_src.template eval_s<T>(i);
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec_other = other_src.template eval<T>(i);
                for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                    _data[S0*(i+j) + F0] -= _vec_other[j];
                }
            }
            for (; i <size(); i++) {
                _data[S0*i+F0] -= other_src.template eval_s<T>(i);
            }
#else
            for (FASTOR_INDEX i = 0; i <size(); i++) {
                _data[S0*i+F0] -= other_src.template eval_s<T>(i);
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
#ifndef FASTOR_NO_ALIAS
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorFixedViewExpr1D<Tensor<T,N>,fseq<F0,L0,S0>,1>(tmp_this_tensor);
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
        FASTOR_IF_CONSTEXPR (S0 == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = this->template eval<T>(i) * other_src.template eval<T>(i);
                _vec.store(&_data[i+F0],false);
            }
            for (; i <size(); i++) {
                _data[i+F0] *= other_src.template eval_s<T>(i);
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec_other = other_src.template eval<T>(i);
                for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                    _data[S0*(i+j) + F0] *= _vec_other[j];
                }
            }
            for (; i <size(); i++) {
                _data[S0*i+F0] *= other_src.template eval_s<T>(i);
            }
#else
            for (FASTOR_INDEX i = 0; i <size(); i++) {
                _data[S0*i+F0] *= other_src.template eval_s<T>(i);
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
#ifndef FASTOR_NO_ALIAS
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorFixedViewExpr1D<Tensor<T,N>,fseq<F0,L0,S0>,1>(tmp_this_tensor);
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
        FASTOR_IF_CONSTEXPR (S0 == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = this->template eval<T>(i) / other_src.template eval<T>(i);
                _vec.store(&_data[i+F0],false);
            }
            for (; i <size(); i++) {
                _data[i+F0] /= other_src.template eval_s<T>(i);
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec_other = other_src.template eval<T>(i);
                for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                    _data[S0*(i+j) + F0] /= _vec_other[j];
                }
            }
            for (; i <size(); i++) {
                _data[S0*i+F0] /= other_src.template eval_s<T>(i);
            }
#else
            for (FASTOR_INDEX i = 0; i <size(); i++) {
                _data[S0*i+F0] /= other_src.template eval_s<T>(i);
            }
#endif
        }
    }
    //----------------------------------------------------------------------------------//


    // Scalar binders
    //----------------------------------------------------------------------------------//
    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator=(U num) {
        simd_vector_type _vec_other(static_cast<T>(num));
        T *FASTOR_RESTRICT _data = _expr.data();
        FASTOR_IF_CONSTEXPR (S0 == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                _vec_other.store(&_data[i+F0],false);
            }
            for (; i <size(); i++) {
                _data[i+F0] = num;
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                    _data[S0*(i+j) + F0] = _vec_other[j];
                }
            }
            for (; i <size(); i++) {
                _data[S0*i+F0] = num;
            }
#else
            for (FASTOR_INDEX i = 0; i <size(); i++) {
                _data[S0*i+F0] = num;
            }
#endif
        }
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator+=(U num) {
        simd_vector_type _vec_other(static_cast<T>(num));
        T *FASTOR_RESTRICT _data = _expr.data();
        FASTOR_IF_CONSTEXPR (S0 == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = this->template eval<T>(i) + _vec_other;
                _vec.store(&_data[i+F0],false);
            }
            for (; i <size(); i++) {
                _data[i+F0] += num;
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                    _data[S0*(i+j) + F0] += _vec_other[j];
                }
            }
            for (; i <size(); i++) {
                _data[S0*i+F0] += num;
            }
#else
            for (FASTOR_INDEX i = 0; i <size(); i++) {
                _data[S0*i+F0] += num;
            }
#endif
        }
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator-=(U num) {
        simd_vector_type _vec_other(static_cast<T>(num));
        T *FASTOR_RESTRICT _data = _expr.data();
        FASTOR_IF_CONSTEXPR (S0 == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = this->template eval<T>(i) - _vec_other;
                _vec.store(&_data[i+F0],false);
            }
            for (; i <size(); i++) {
                _data[i+F0] -= num;
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                    _data[S0*(i+j) + F0] -= _vec_other[j];
                }
            }
            for (; i <size(); i++) {
                _data[S0*i+F0] -= num;
            }
#else
            for (FASTOR_INDEX i = 0; i <size(); i++) {
                _data[S0*i+F0] -= num;
            }
#endif
        }
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator*=(U num) {
        simd_vector_type _vec_other(static_cast<T>(num));
        T *FASTOR_RESTRICT _data = _expr.data();
        FASTOR_IF_CONSTEXPR (S0 == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = this->template eval<T>(i) * _vec_other;
                _vec.store(&_data[i+F0],false);
            }
            for (; i <size(); i++) {
                _data[i+F0] *= num;
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                    _data[S0*(i+j) + F0] *= _vec_other[j];
                }
            }
            for (; i <size(); i++) {
                _data[S0*i+F0] *= num;
            }
#else
            for (FASTOR_INDEX i = 0; i <size(); i++) {
                _data[S0*i+F0] *= num;
            }
#endif
        }
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U> && !is_integral_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(U num) {
        T inum = T(1) / static_cast<T>(num);
        simd_vector_type _vec_other(inum);
        T *FASTOR_RESTRICT _data = _expr.data();
        FASTOR_IF_CONSTEXPR (S0 == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = this->template eval<T>(i) * _vec_other;
                _vec.store(&_data[i+F0],false);
            }
            for (; i <size(); i++) {
                _data[i+F0] *= inum;
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                    _data[S0*(i+j) + F0] *= _vec_other[j];
                }
            }
            for (; i <size(); i++) {
                _data[S0*i+F0] *= inum;
            }
#else
            for (FASTOR_INDEX i = 0; i <size(); i++) {
                _data[S0*i+F0] *= inum;
            }
#endif
        }
    }
    template<typename U=T, enable_if_t_<is_primitive_v_<U> && is_integral_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(U num) {
        simd_vector_type _vec_other(static_cast<T>(num));
        T *FASTOR_RESTRICT _data = _expr.data();
        FASTOR_IF_CONSTEXPR (S0 == 1) {
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                auto _vec = this->template eval<T>(i) / _vec_other;
                _vec.store(&_data[i+F0],false);
            }
            for (; i <size(); i++) {
                _data[i+F0] /= num;
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            FASTOR_INDEX i;
            for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
                for (auto j=0; j<SIMDVector<T,simd_abi_type>::Size; ++j) {
                    _data[S0*(i+j) + F0] /= _vec_other[j];
                }
            }
            for (; i <size(); i++) {
                _data[S0*i+F0] /= num;
            }
#else
            for (FASTOR_INDEX i = 0; i <size(); i++) {
                _data[S0*i+F0] /= num;
            }
#endif
        }
    }
    //----------------------------------------------------------------------------------//

    template<typename U>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i) const {
        SIMDVector<U,simd_abi_type> _vec;
        vector_setter(_vec,_expr.data(),S0*i+F0,S0);
        return _vec;
    }

    template<typename U>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return _expr.data()[S0*i+F0];
    }

    template<typename U>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        SIMDVector<U,simd_abi_type> _vec;
        vector_setter(_vec,_expr.data(),S0*(i+j)+F0,S0);
        return _vec;
    }

    template<typename U>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _expr.data()[S0*(i+j)+F0];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int,1>& as) const {
        SIMDVector<U,simd_abi_type> _vec;
        vector_setter(_vec,_expr.data(),S0*as[0]+F0,S0);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,1>& as) const {
        return _expr.data()[S0*as[0]+F0];
    }
};


}


#endif // TENSOR_FIXED_VIEWS_1D_H
