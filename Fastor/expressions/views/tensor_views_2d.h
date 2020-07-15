#ifndef TENSOR_VIEWS_2D_H
#define TENSOR_VIEWS_2D_H


#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/Ranges.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"

namespace Fastor {


// 2D non-const views
//-------------------------------------------------------------------------------------//
template<typename T, size_t M, size_t N>
struct TensorConstViewExpr<Tensor<T,M,N>,2>: public AbstractTensor<TensorConstViewExpr<Tensor<T,M,N>,2>,2> {
private:
    const Tensor<T,M,N>& _expr;
    seq _seq0;
    seq _seq1;
public:
    using scalar_type = T;
    using simd_vector_type = typename Tensor<T,M,N>::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using result_type = Tensor<T,M,N>;
    static constexpr FASTOR_INDEX Dimension = 2;
    static constexpr FASTOR_INDEX Stride = simd_vector_type::Size;
    static constexpr FASTOR_INDEX rank() {return 2;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return _seq0.size()*_seq1.size();}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return i==0 ? _seq0.size() : _seq1.size();}
    constexpr const Tensor<T,M,N>& expr() const {return _expr;}

    FASTOR_INLINE TensorConstViewExpr(const Tensor<T,M,N> &_ex, seq _s0, seq _s1) : _expr(_ex), _seq0(std::move(_s0)), _seq1(std::move(_s1)) {
        if (_seq0._last < 0 && _seq0._first >= 0) {_seq0._last += M + 1;}
        else if (_seq0._last==0 && _seq0._first==-1) {_seq0._first=M-1; _seq0._last=M;}
        else if (_seq0._last < 0 && _seq0._first < 0) {_seq0._first += M +1; _seq0._last += M+1;}
        if (_seq1._last < 0 && _seq1._first >= 0) {_seq1._last += N + 1;}
        else if (_seq1._last==0 && _seq1._first==-1) {_seq1._first=N-1; _seq1._last=N;}
        else if (_seq1._last < 0 && _seq1._first < 0) {_seq1._first += N +1; _seq1._last += N+1;}
#ifndef NDEBUG
        FASTOR_ASSERT(_seq0._last <= M && _seq0._first<M,"INDEX OUT OF BOUNDS");
        FASTOR_ASSERT(_seq1._last <= N && _seq1._first<N,"INDEX OUT OF BOUNDS");
#endif
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX idx) const {
        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,SIMDVector<U,simd_abi_type>::Size> inds;
        for (auto j=0; j<SIMDVector<U,simd_abi_type>::Size; ++j) {
            auto it = (idx+j) / _seq1.size(), jt = (idx+j) % _seq1.size();
            inds[j] = _seq0._step*it*N+_seq1._step*jt + _seq0._first*N + _seq1._first;
        }
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx) const {
        auto it = idx / _seq1.size(), jt = idx % _seq1.size();
        auto ind = _seq0._step*it*N+_seq1._step*jt + _seq0._first*N + _seq1._first;
        return _expr.data()[ind];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        SIMDVector<U,simd_abi_type> _vec;
        if (_seq1._step==1) _vec.load(_expr.data()+_seq0._step*i*N+j + _seq0._first*N + _seq1._first,false);
        else vector_setter(_vec,_expr.data(),_seq0._step*i*N+_seq1._step*j + _seq0._first*N + _seq1._first,_seq1._step);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first);
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int,2>& as) const {
        SIMDVector<U,simd_abi_type> _vec;
        if (_seq1._step==1) _vec.load(_expr.data()+_seq0._step*as[0]*N+as[1] + _seq0._first*N + _seq1._first,false);
        else vector_setter(_vec,_expr.data(),_seq0._step*as[0]*N+_seq1._step*as[1] + _seq0._first*N + _seq1._first,_seq1._step);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,2>& as) const {
        return _expr(_seq0._step*as[0]+_seq0._first,_seq1._step*as[1]+_seq1._first);
    }
};













// 2D non-const views
//-------------------------------------------------------------------------------------//
template<typename T, size_t M, size_t N>
struct TensorViewExpr<Tensor<T,M,N>,2>: public AbstractTensor<TensorViewExpr<Tensor<T,M,N>,2>,2> {
private:
    Tensor<T,M,N>& _expr;
    seq _seq0;
    seq _seq1;
    bool _does_alias = false;
    // std::array<FASTOR_INDEX,2> _dims;
    constexpr FASTOR_INLINE Tensor<T,M,N> get_tensor() const {return _expr;};
public:
    using scalar_type = T;
    using simd_vector_type = typename Tensor<T,M,N>::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using result_type = Tensor<T,M,N>;
    static constexpr FASTOR_INDEX Dimension = 2;
    static constexpr FASTOR_INDEX Stride = simd_vector_type::Size;
    static constexpr FASTOR_INDEX rank() {return 2;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return _seq0.size()*_seq1.size();}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return i==0 ? _seq0.size() : _seq1.size();}
    constexpr const Tensor<T,M,N>& expr() const {return _expr;}

    FASTOR_INLINE TensorViewExpr<Tensor<T,M,N>,2>& noalias() {
        _does_alias = true;
        return *this;
    }

    FASTOR_INLINE TensorViewExpr(Tensor<T,M,N> &_ex, seq _s0, seq _s1) :
        _expr(_ex), _seq0(std::move(_s0)), _seq1(std::move(_s1)) {

        if (_seq0._last < 0 && _seq0._first >= 0) {_seq0._last += M + 1;}
        else if (_seq0._last==0 && _seq0._first==-1) {_seq0._first=M-1; _seq0._last=M;}
        else if (_seq0._last < 0 && _seq0._first < 0) {_seq0._first += M +1; _seq0._last += M+1;}
        if (_seq1._last < 0 && _seq1._first >= 0) {_seq1._last += N + 1;}
        else if (_seq1._last==0 && _seq1._first==-1) {_seq1._first=N-1; _seq1._last=N;}
        else if (_seq1._last < 0 && _seq1._first < 0) {_seq1._first += N +1; _seq1._last += N+1;}
#ifndef NDEBUG
        FASTOR_ASSERT(_seq0._last <= M && _seq0._first<M,"INDEX OUT OF BOUNDS");
        FASTOR_ASSERT(_seq1._last <= N && _seq1._first<N,"INDEX OUT OF BOUNDS");
#endif
        // for (FASTOR_INDEX i=0; i<2; ++i) _dims[i] = dimension(i);
    }

    // View evalution operators
    // Copy assignment operators
    //----------------------------------------------------------------------------------//
    FASTOR_HINT_INLINE void operator=(const TensorViewExpr<Tensor<T,M,N>,2> &other_src) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,M,N>,2>(tmp_this_tensor,_seq0,_seq1);
            // Assign other to temporary
            tmp = other_src;
            // assign temporary to this
            this->operator=(tmp);
            return;
            // Alternatively one could do, but slower
            // _does_alias = false;
            // // Evaluate this into a temporary
            // auto tmp_this_tensor = get_tensor();
            // tmp_this_tensor(_seq0,_seq1) = other_src;
            // // assign temporary to this
            // this->operator=(tmp_this_tensor(_seq0,_seq1));
            // return;
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
        if (_seq1._step == 1) {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec = other_src.template eval<T>(i,j);
                    _vec.store(&_data[(_seq0._step*i+_seq0._first)*N+j+_seq1._first],false);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,j+_seq1._first) = other_src.template eval_s<T>(i,j);
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec = other_src.template eval<T>(i,j);
                    data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) = other_src.template eval_s<T>(i,j);
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) = other_src.template eval_s<T>(i,j);
                }
            }
#endif
        }
    }
    //----------------------------------------------------------------------------------//



    // AbstractTensor binders - equal order
    //----------------------------------------------------------------------------------//
    template<typename Derived, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator=(const AbstractTensor<Derived,2> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator=(tmp);
    }
    template<typename Derived, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator=(const AbstractTensor<Derived,2> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,M,N>,2>(tmp_this_tensor,_seq0,_seq1);
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
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) = other_src.template eval_s<T>(i,j);
                }
            }
            return;
        }
        if (_seq1._step == 1) {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec = other_src.template eval<T>(i,j);
                    _vec.store(&_data[(_seq0._step*i+_seq0._first)*N+j+_seq1._first],false);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,j+_seq1._first) = other_src.template eval_s<T>(i,j);
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec = other_src.template eval<T>(i,j);
                    data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) = other_src.template eval_s<T>(i,j);
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) = other_src.template eval_s<T>(i,j);
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
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,M,N>,2>(tmp_this_tensor,_seq0,_seq1);
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
        T *_data = _expr.data();
        if (_seq1._step == 1) {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) + other_src.template eval<T>(i,j);
                    _vec.store(&_data[(_seq0._step*i+_seq0._first)*N+j+_seq1._first],false);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,j+_seq1._first) += other_src.template eval_s<T>(i,j);
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) + other_src.template eval<T>(i,j);
                    data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) += other_src.template eval_s<T>(i,j);
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) += other_src.template eval_s<T>(i,j);
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
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,M,N>,2>(tmp_this_tensor,_seq0,_seq1);
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
        T *_data = _expr.data();
        if (_seq1._step == 1) {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) - other_src.template eval<T>(i,j);
                    _vec.store(&_data[(_seq0._step*i+_seq0._first)*N+j+_seq1._first],false);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,j+_seq1._first) -= other_src.template eval_s<T>(i,j);
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) - other_src.template eval<T>(i,j);
                    data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) -= other_src.template eval_s<T>(i,j);
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) -= other_src.template eval_s<T>(i,j);
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
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,M,N>,2>(tmp_this_tensor,_seq0,_seq1);
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
        T *_data = _expr.data();
        if (_seq1._step == 1) {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) * other_src.template eval<T>(i,j);
                    _vec.store(&_data[(_seq0._step*i+_seq0._first)*N+j+_seq1._first],false);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,j+_seq1._first) *= other_src.template eval_s<T>(i,j);
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) * other_src.template eval<T>(i,j);
                    data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) *= other_src.template eval_s<T>(i,j);
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) *= other_src.template eval_s<T>(i,j);
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
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,M,N>,2>(tmp_this_tensor,_seq0,_seq1);
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
        T *_data = _expr.data();
        if (_seq1._step == 1) {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) / other_src.template eval<T>(i,j);
                    _vec.store(&_data[(_seq0._step*i+_seq0._first)*N+j+_seq1._first],false);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,j+_seq1._first) /= other_src.template eval_s<T>(i,j);
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) / other_src.template eval<T>(i,j);
                    data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) /= other_src.template eval_s<T>(i,j);
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) /= other_src.template eval_s<T>(i,j);
                }
            }
#endif
        }
    }
    //----------------------------------------------------------------------------------//


    // AbstractTensor binders [non-equal order]
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
            auto tmp = TensorViewExpr<Tensor<T,M,N>,2>(tmp_this_tensor,_seq0,_seq1);
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
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) = other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
            return;
        }
        if (_seq1._step == 1) {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec = other_src.template eval<T>(counter);
                    _vec.store(&_data[(_seq0._step*i+_seq0._first)*N+j+_seq1._first],false);
                    counter+=Stride;
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,j+_seq1._first) = other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec = other_src.template eval<T>(counter);
                    data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                    counter+=Stride;
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) = other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) = other_src.template eval_s<T>(counter);
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
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,M,N>,2>(tmp_this_tensor,_seq0,_seq1);
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
        if (_seq1._step == 1) {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) + other_src.template eval<T>(counter);
                    _vec.store(&_data[(_seq0._step*i+_seq0._first)*N+j+_seq1._first],false);
                    counter+=Stride;
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,j+_seq1._first) += other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) + other_src.template eval<T>(counter);
                    data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                    counter+=Stride;
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) += other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) += other_src.template eval_s<T>(counter);
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
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,M,N>,2>(tmp_this_tensor,_seq0,_seq1);
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
        if (_seq1._step == 1) {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) - other_src.template eval<T>(counter);
                    _vec.store(&_data[(_seq0._step*i+_seq0._first)*N+j+_seq1._first],false);
                    counter+=Stride;
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,j+_seq1._first) -= other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) - other_src.template eval<T>(counter);
                    data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                    counter+=Stride;
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) -= other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) -= other_src.template eval_s<T>(counter);
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
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,M,N>,2>(tmp_this_tensor,_seq0,_seq1);
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
        if (_seq1._step == 1) {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) * other_src.template eval<T>(counter);
                    _vec.store(&_data[(_seq0._step*i+_seq0._first)*N+j+_seq1._first],false);
                    counter+=Stride;
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,j+_seq1._first) *= other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) * other_src.template eval<T>(counter);
                    data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                    counter+=Stride;
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) *= other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) *= other_src.template eval_s<T>(counter);
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
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,M,N>,2>(tmp_this_tensor,_seq0,_seq1);
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
        if (_seq1._step == 1) {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) / other_src.template eval<T>(counter);
                    _vec.store(&_data[(_seq0._step*i+_seq0._first)*N+j+_seq1._first],false);
                    counter+=Stride;
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,j+_seq1._first) /= other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) / other_src.template eval<T>(counter);
                    data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                    counter+=Stride;
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) /= other_src.template eval_s<T>(counter);
                    counter++;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) /= other_src.template eval_s<T>(counter);
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
    FASTOR_INLINE void operator=(U num) {
        T *_data = _expr.data();
        SIMDVector<T,simd_abi_type> _vec_other(static_cast<T>(num));
        if (_seq1._step == 1) {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    _vec_other.store(&_data[(_seq0._step*i+_seq0._first)*N+j+_seq1._first],false);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,j+_seq1._first) = num;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    data_setter(_data,_vec_other,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) = num;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) = num;
                }
            }
#endif
        }
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_INLINE void operator+=(U num) {
        T *_data = _expr.data();
        SIMDVector<T,simd_abi_type> _vec_other(static_cast<T>(num));
        if (_seq1._step == 1) {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) + _vec_other;
                    _vec.store(&_data[(_seq0._step*i+_seq0._first)*N+j+_seq1._first],false);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,j+_seq1._first) += num;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) + _vec_other;
                    data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) += num;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) += num;
                }
            }
#endif
        }
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_INLINE void operator-=(U num) {
        T *_data = _expr.data();
        SIMDVector<T,simd_abi_type> _vec_other(static_cast<T>(num));
        if (_seq1._step == 1) {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) - _vec_other;
                    _vec.store(&_data[(_seq0._step*i+_seq0._first)*N+j+_seq1._first],false);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,j+_seq1._first) -= num;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) - _vec_other;
                    data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) -= num;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) -= num;
                }
            }
#endif
        }
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_INLINE void operator*=(U num) {
        T *_data = _expr.data();
        SIMDVector<T,simd_abi_type> _vec_other(static_cast<T>(num));
        if (_seq1._step == 1) {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) * _vec_other;
                    _vec.store(&_data[(_seq0._step*i+_seq0._first)*N+j+_seq1._first],false);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,j+_seq1._first) *= num;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) * _vec_other;
                    data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) *= num;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) *= num;
                }
            }
#endif
        }
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U> && !is_integral_v_<U>,bool> = false>
    FASTOR_INLINE void operator/=(U num) {
        T *_data = _expr.data();
        T inum = T(1.0)/T(num);
        SIMDVector<T,simd_abi_type> _vec_other((inum));
        if (_seq1._step == 1) {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) * _vec_other;
                    _vec.store(&_data[(_seq0._step*i+_seq0._first)*N+j+_seq1._first],false);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,j+_seq1._first) *= inum;
                }
            }
        }
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        else {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) * _vec_other;
                    data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) *= inum;
                }
            }
        }
#else
        else {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) *= inum;
                }
            }
        }
#endif
    }
    template<typename U=T, enable_if_t_<is_primitive_v_<U> && is_integral_v_<U>,bool> = false>
    FASTOR_INLINE void operator/=(U num) {
        T *_data = _expr.data();
        SIMDVector<T,simd_abi_type> _vec_other(static_cast<T>(num));
        if (_seq1._step == 1) {
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec =  this->template eval<T>(i,j) / _vec_other;
                    _vec.store(&_data[(_seq0._step*i+_seq0._first)*N+j+_seq1._first],false);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,j+_seq1._first) /= num;
                }
            }
        }
        else {
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                FASTOR_INDEX j;
                for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                    auto _vec = this->template eval<T>(i,j) / _vec_other;
                    data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                }
                for (; j <_seq1.size(); ++j) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) /= num;
                }
            }
#else
            for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
                for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                    _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) /= num;
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
        for (auto j=0; j<SIMDVector<U,simd_abi_type>::Size; ++j) {
            auto it = (idx+j) / _seq1.size(), jt = (idx+j) % _seq1.size();
            inds[j] = _seq0._step*it*N+_seq1._step*jt + _seq0._first*N + _seq1._first;
        }
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx) const {
        auto it = idx / _seq1.size(), jt = idx % _seq1.size();
        auto ind = _seq0._step*it*N+_seq1._step*jt + _seq0._first*N + _seq1._first;
        return _expr.data()[ind];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        SIMDVector<U,simd_abi_type> _vec;
        if (_seq1._step==1) _vec.load(_expr.data()+_seq0._step*i*N+j + _seq0._first*N + _seq1._first,false);
        else vector_setter(_vec,_expr.data(),_seq0._step*i*N+_seq1._step*j + _seq0._first*N + _seq1._first,_seq1._step);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return _expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first);
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int,2>& as) const {
        SIMDVector<U,simd_abi_type> _vec;
        if (_seq1._step==1) _vec.load(_expr.data()+_seq0._step*as[0]*N+as[1] + _seq0._first*N + _seq1._first,false);
        else vector_setter(_vec,_expr.data(),_seq0._step*as[0]*N+_seq1._step*as[1] + _seq0._first*N + _seq1._first,_seq1._step);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,2>& as) const {
        return _expr(_seq0._step*as[0]+_seq0._first,_seq1._step*as[1]+_seq1._first);
    }
};


} // end of namespace Fastor


#endif // TENSOR_VIEWS_2D_H
