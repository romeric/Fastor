#ifndef TENSOR_FIXED_VIEWS_2D_H
#define TENSOR_FIXED_VIEWS_2D_H


#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/Ranges.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"

namespace Fastor {


// 2D const fixed views
//----------------------------------------------------------------------------------------------//
template<typename T, size_t M, size_t N, int F0, int L0, int S0, int F1, int L1, int S1>
struct TensorConstFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2> :
    public AbstractTensor<TensorConstFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2>,2> {
private:
    const Tensor<T,M,N> &_expr;
public:
    using scalar_type = T;
    using simd_vector_type = typename Tensor<T,M,N>::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using result_type = Tensor<T, range_detector<F0,L0,S0>::value, range_detector<F1,L1,S1>::value>;
    static constexpr FASTOR_INDEX Dimension = 2;
    static constexpr FASTOR_INDEX Stride = simd_vector_type::Size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {
        return range_detector<F0,L0,S0>::value*range_detector<F1,L1,S1>::value;
    }
    static constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) {
        return i==0 ? range_detector<F0,L0,S0>::value : range_detector<F1,L1,S1>::value;
    }
    constexpr const Tensor<T,M,N>& expr() const {return _expr;}
    static constexpr FASTOR_INDEX Padding = F0*N+F1;

    constexpr FASTOR_INLINE TensorConstFixedViewExpr2D(const Tensor<T,M,N> &_ex) : _expr(_ex) {}

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX idx) const {
        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,SIMDVector<U,simd_abi_type>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<U,simd_abi_type>::Size; ++j) {
            auto it = (idx+j) / range_detector<F1,L1,S1>::value, jt = (idx+j) % range_detector<F1,L1,S1>::value;
            inds[j] = S0*it*N+S1*jt + Padding;
        }
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx) const {
        auto it = idx / range_detector<F1,L1,S1>::value, jt = idx % range_detector<F1,L1,S1>::value;
        auto ind = S0*it*N+S1*jt + Padding;
        return _expr.data()[ind];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        SIMDVector<U,simd_abi_type> _vec;
        FASTOR_IF_CONSTEXPR (S1==1) _vec.load(_expr.data()+S0*i*N+j + Padding, false);
        else vector_setter(_vec,_expr.data(),S0*i*N+S1*j + Padding,S1);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _expr.data()[S0*i*N+S1*j + Padding];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int,2>& as) const {
        SIMDVector<U,simd_abi_type> _vec;
        FASTOR_IF_CONSTEXPR (S1==1) _vec.load(_expr.data()+S0*as[0]*N+as[1] + Padding, false);
        else vector_setter(_vec,_expr.data(),S0*as[0]*N+S1*as[1] + Padding,S1);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,2>& as) const {
        return _expr.data()[S0*as[0]*N+S1*as[1] + Padding];
    }
};





// 2D non-const fixed views
//----------------------------------------------------------------------------------------------//
template<typename T, size_t M, size_t N, int F0, int L0, int S0, int F1, int L1, int S1>
struct TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2> :
    public AbstractTensor<TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2>,2> {

private:
    Tensor<T,M,N> &_expr;
    bool _does_alias = false;
    constexpr FASTOR_INLINE Tensor<T,M,N> get_tensor() const {return _expr;};
public:
    using scalar_type = T;
    using simd_vector_type = typename Tensor<T,M,N>::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using result_type = Tensor<T, range_detector<F0,L0,S0>::value, range_detector<F1,L1,S1>::value>;
    static constexpr FASTOR_INDEX Dimension = 2;
    static constexpr FASTOR_INDEX Stride = simd_vector_type::Size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {
        return range_detector<F0,L0,S0>::value*range_detector<F1,L1,S1>::value;
    }
    static constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) {
        return i==0 ? range_detector<F0,L0,S0>::value : range_detector<F1,L1,S1>::value;
    }
    constexpr const Tensor<T,M,N>& expr() const {return _expr;}
    static constexpr FASTOR_INDEX Padding = F0*N+F1;

    FASTOR_INLINE TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2>& noalias() {
        _does_alias = true;
        return *this;
    }

    constexpr FASTOR_INLINE TensorFixedViewExpr2D(Tensor<T,M,N> &_ex) : _expr(_ex) {}

    //----------------------------------------------------------------------------------//
    FASTOR_HINT_INLINE void operator=(const TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2> &other_src) {
#ifndef FASTOR_NO_ALIAS
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2>(tmp_this_tensor);
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
        FASTOR_IF_CONSTEXPR (S1==1) {
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = other_src.template eval<T>(i,j);
                    _vec.store(&_data[S0*i*N+j+Padding],false);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,j+F1) = other_src.template eval_s<T>(i,j);
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = other_src.template eval<T>(i,j);
                    data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,S1*j+F1) = other_src.template eval_s<T>(i,j);
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) = other_src.template eval_s<T>(i,j);
                }
            }
#endif
        }
    }

    // AbstractTensor binders - [equal order]
    //----------------------------------------------------------------------------------//
    template<typename Derived, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator=(const AbstractTensor<Derived,2> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator=(tmp);
    }
    template<typename Derived, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator=(const AbstractTensor<Derived,2> &other) {
#ifndef FASTOR_NO_ALIAS
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2>(tmp_this_tensor);
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
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other_src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *FASTOR_RESTRICT _data = _expr.data();
        FASTOR_IF_CONSTEXPR(is_boolean_expression_v<Derived>) {
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) = other_src.template eval_s<T>(i,j);
                }
            }
            return;
        }
        FASTOR_IF_CONSTEXPR (S1==1) {
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = other_src.template eval<T>(i,j);
                    _vec.store(&_data[S0*i*N+j+Padding],false);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,j+F1) = other_src.template eval_s<T>(i,j);
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = other_src.template eval<T>(i,j);
                    data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,S1*j+F1) = other_src.template eval_s<T>(i,j);
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) = other_src.template eval_s<T>(i,j);
                }
            }
#endif
        }
    }

    template<typename Derived, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator+=(const AbstractTensor<Derived,2> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator+=(tmp);
    }
    template<typename Derived, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator+=(const AbstractTensor<Derived,2> &other) {
#ifndef FASTOR_NO_ALIAS
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2>(tmp_this_tensor);
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
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other_src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *FASTOR_RESTRICT _data = _expr.data();
        FASTOR_IF_CONSTEXPR (S1==1) {
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) + other_src.template eval<T>(i,j);
                    _vec.store(&_data[S0*i*N+j+Padding],false);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,j+F1) += other_src.template eval_s<T>(i,j);
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) + other_src.template eval<T>(i,j);
                    data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,S1*j+F1) += other_src.template eval_s<T>(i,j);
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) += other_src.template eval_s<T>(i,j);
                }
            }
#endif
        }
    }

    template<typename Derived, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator-=(const AbstractTensor<Derived,2> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator-=(tmp);
    }
    template<typename Derived, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator-=(const AbstractTensor<Derived,2> &other) {
#ifndef FASTOR_NO_ALIAS
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2>(tmp_this_tensor);
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
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other_src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *FASTOR_RESTRICT _data = _expr.data();
        FASTOR_IF_CONSTEXPR (S1==1) {
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) - other_src.template eval<T>(i,j);
                    _vec.store(&_data[S0*i*N+j+Padding],false);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,j+F1) -= other_src.template eval_s<T>(i,j);
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) - other_src.template eval<T>(i,j);
                    data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,S1*j+F1) -= other_src.template eval_s<T>(i,j);
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) -= other_src.template eval_s<T>(i,j);
                }
            }
#endif
        }
    }

    template<typename Derived, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator*=(const AbstractTensor<Derived,2> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator*=(tmp);
    }
    template<typename Derived, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator*=(const AbstractTensor<Derived,2> &other) {
#ifndef FASTOR_NO_ALIAS
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2>(tmp_this_tensor);
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
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other_src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *FASTOR_RESTRICT _data = _expr.data();
        FASTOR_IF_CONSTEXPR (S1==1) {
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) * other_src.template eval<T>(i,j);
                    _vec.store(&_data[S0*i*N+j+Padding],false);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,j+F1) *= other_src.template eval_s<T>(i,j);
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) * other_src.template eval<T>(i,j);
                    data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,S1*j+F1) *= other_src.template eval_s<T>(i,j);
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) *= other_src.template eval_s<T>(i,j);
                }
            }
#endif
        }
    }

    template<typename Derived, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(const AbstractTensor<Derived,2> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator/=(tmp);
    }
    template<typename Derived, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(const AbstractTensor<Derived,2> &other) {
#ifndef FASTOR_NO_ALIAS
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2>(tmp_this_tensor);
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
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other_src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *FASTOR_RESTRICT _data = _expr.data();
        FASTOR_IF_CONSTEXPR (S1==1) {
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) / other_src.template eval<T>(i,j);
                    _vec.store(&_data[S0*i*N+j+Padding],false);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,j+F1) /= other_src.template eval_s<T>(i,j);
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) / other_src.template eval<T>(i,j);
                    data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,S1*j+F1) /= other_src.template eval_s<T>(i,j);
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) /= other_src.template eval_s<T>(i,j);
                }
            }
#endif
        }
    }
    //----------------------------------------------------------------------------------//


    // AbstractTensor binders for other nth rank tensors
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
            auto tmp = TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2>(tmp_this_tensor);
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
        FASTOR_INDEX counter = 0;
        FASTOR_IF_CONSTEXPR(is_boolean_expression_v<Derived>) {
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) = other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
            return;
        }
        FASTOR_IF_CONSTEXPR (S1==1) {
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = other_src.template eval<T>(counter);
                    _vec.store(&_data[S0*i*N+j+Padding],false);
                    counter += Stride;
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,j+F1) = other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = other_src.template eval<T>(counter);
                    data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                    counter += Stride;
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,S1*j+F1) = other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) = other_src.template eval_s<T>(counter);
                    counter++;
                }
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
            auto tmp = TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2>(tmp_this_tensor);
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
        FASTOR_INDEX counter = 0;
        FASTOR_IF_CONSTEXPR (S1==1) {
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) + other_src.template eval<T>(counter);
                    _vec.store(&_data[S0*i*N+j+Padding],false);
                    counter += Stride;
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,j+F1) += other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) + other_src.template eval<T>(counter);
                    data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                    counter += Stride;
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,S1*j+F1) += other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) += other_src.template eval_s<T>(counter);
                    counter++;
                }
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
            auto tmp = TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2>(tmp_this_tensor);
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
        FASTOR_INDEX counter = 0;
        FASTOR_IF_CONSTEXPR (S1==1) {
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) - other_src.template eval<T>(counter);
                    _vec.store(&_data[S0*i*N+j+Padding],false);
                    counter += Stride;
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,j+F1) -= other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) - other_src.template eval<T>(counter);
                    data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                    counter += Stride;
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,S1*j+F1) -= other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) -= other_src.template eval_s<T>(counter);
                    counter++;
                }
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
            auto tmp = TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2>(tmp_this_tensor);
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
        FASTOR_INDEX counter = 0;
        FASTOR_IF_CONSTEXPR (S1==1) {
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) * other_src.template eval<T>(counter);
                    _vec.store(&_data[S0*i*N+j+Padding],false);
                    counter += Stride;
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,j+F1) *= other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) * other_src.template eval<T>(counter);
                    data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                    counter += Stride;
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,S1*j+F1) *= other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) *= other_src.template eval_s<T>(counter);
                    counter++;
                }
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
            auto tmp = TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2>(tmp_this_tensor);
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
        FASTOR_INDEX counter = 0;
        FASTOR_IF_CONSTEXPR (S1==1) {
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) / other_src.template eval<T>(counter);
                    _vec.store(&_data[S0*i*N+j+Padding],false);
                    counter += Stride;
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,j+F1) /= other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) / other_src.template eval<T>(counter);
                    data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                    counter += Stride;
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,S1*j+F1) /= other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) /= other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
#endif
        }
    }
    //----------------------------------------------------------------------------------//


    // Scalar binders
    //----------------------------------------------------------------------------------//
    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator=(U num) {
        T *FASTOR_RESTRICT _data = _expr.data();
        SIMDVector<T,simd_abi_type> _vec_other(static_cast<T>(num));
        FASTOR_IF_CONSTEXPR (S1==1) {
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    _vec_other.store(&_data[S0*i*N+j+Padding],false);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,j+F1) = num;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    data_setter(_data,_vec_other,S0*i*N+S1*j+Padding,S1);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,S1*j+F1) = num;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) = num;
                }
            }
#endif
        }
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator+=(U num) {
        T *FASTOR_RESTRICT _data = _expr.data();
        SIMDVector<T,simd_abi_type> _vec_other(static_cast<T>(num));
        FASTOR_IF_CONSTEXPR (S1==1) {
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) + _vec_other;
                    _vec.store(&_data[S0*i*N+j+Padding],false);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,j+F1) += num;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) + _vec_other;
                    data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,S1*j+F1) += num;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) += num;
                }
            }
#endif
        }
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator-=(U num) {
        T *FASTOR_RESTRICT _data = _expr.data();
        SIMDVector<T,simd_abi_type> _vec_other(static_cast<T>(num));
        FASTOR_IF_CONSTEXPR (S1==1) {
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) - _vec_other;
                    _vec.store(&_data[S0*i*N+j+Padding],false);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,j+F1) -= num;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) - _vec_other;
                    data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,S1*j+F1) -= num;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) -= num;
                }
            }
#endif
        }
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator*=(U num) {
        T *FASTOR_RESTRICT _data = _expr.data();
        SIMDVector<T,simd_abi_type> _vec_other(static_cast<T>(num));
        FASTOR_IF_CONSTEXPR (S1==1) {
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) * _vec_other;
                    _vec.store(&_data[S0*i*N+j+Padding],false);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,j+F1) *= num;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) * _vec_other;
                    data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,S1*j+F1) *= num;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) *= num;
                }
            }
#endif
        }
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U> && !is_integral_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(U num) {
        T *FASTOR_RESTRICT _data = _expr.data();
        T inum = T(1)/num;
        SIMDVector<T,simd_abi_type> _vec_other(static_cast<T>(inum));
        FASTOR_IF_CONSTEXPR (S1==1) {
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) * _vec_other;
                    _vec.store(&_data[S0*i*N+j+Padding],false);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,j+F1) *= inum;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) * _vec_other;
                    data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,S1*j+F1) *= inum;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) *= inum;
                }
            }
#endif
        }
    }
    template<typename U=T, enable_if_t_<is_primitive_v_<U> && is_integral_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(U num) {
        T *FASTOR_RESTRICT _data = _expr.data();
        SIMDVector<T,simd_abi_type> _vec_other(static_cast<T>(num));
        FASTOR_IF_CONSTEXPR (S1==1) {
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) / _vec_other;
                    _vec.store(&_data[S0*i*N+j+Padding],false);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,j+F1) /= num;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) / _vec_other;
                    data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                }
                for (; j <dimension(1); ++j) {
                    _expr(S0*i+F0,S1*j+F1) /= num;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
                for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                    _expr(S0*i+F0,S1*j+F1) /= num;
                }
            }
#endif
        }
    }
    //----------------------------------------------------------------------------------//

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX idx) const {
        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,SIMDVector<U,simd_abi_type>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<U,simd_abi_type>::Size; ++j) {
            auto it = (idx+j) / range_detector<F1,L1,S1>::value, jt = (idx+j) % range_detector<F1,L1,S1>::value;
            inds[j] = S0*it*N+S1*jt + Padding;
        }
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx) const {
        auto it = idx / range_detector<F1,L1,S1>::value, jt = idx % range_detector<F1,L1,S1>::value;
        auto ind = S0*it*N+S1*jt + Padding;
        return _expr.data()[ind];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        SIMDVector<U,simd_abi_type> _vec;
        FASTOR_IF_CONSTEXPR (S1==1) _vec.load(_expr.data()+S0*i*N+j + Padding, false);
        else vector_setter(_vec,_expr.data(),S0*i*N+S1*j + Padding,S1);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _expr.data()[S0*i*N+S1*j + Padding];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int,2>& as) const {
        SIMDVector<U,simd_abi_type> _vec;
        FASTOR_IF_CONSTEXPR (S1==1) _vec.load(_expr.data()+S0*as[0]*N+as[1] + Padding, false);
        else vector_setter(_vec,_expr.data(),S0*as[0]*N+S1*as[1] + Padding,S1);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,2>& as) const {
        return _expr.data()[S0*as[0]*N+S1*as[1] + Padding];
    }
};


}


#endif // TENSOR_FIXED_VIEWS_2D_H
