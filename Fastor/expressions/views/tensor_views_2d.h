#ifndef TENSOR_VIEWS_2D_H
#define TENSOR_VIEWS_2D_H


#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/Ranges.h"

namespace Fastor {


// 2D non-const views
//-------------------------------------------------------------------------------------//
template<typename T, size_t M, size_t N>
struct TensorConstViewExpr<Tensor<T,M,N>,2>: public AbstractTensor<TensorConstViewExpr<Tensor<T,M,N>,2>,2> {
private:
    const Tensor<T,M,N>& expr;
    seq _seq0;
    seq _seq1;
public:
    using scalar_type = T;
    static constexpr FASTOR_INDEX Dimension = 2;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    static constexpr FASTOR_INDEX rank() {return 2;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return _seq0.size()*_seq1.size();}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return i==0 ? _seq0.size() : _seq1.size();}

    FASTOR_INLINE TensorConstViewExpr(const Tensor<T,M,N> &_ex, seq _s0, seq _s1) : expr(_ex), _seq0(std::move(_s0)), _seq1(std::move(_s1)) {
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
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX idx) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,SIMDVector<U,DEFAULT_ABI>::Size> inds;
        for (auto j=0; j<SIMDVector<U,DEFAULT_ABI>::Size; ++j) {
            auto it = (idx+j) / _seq1.size(), jt = (idx+j) % _seq1.size();
            inds[j] = _seq0._step*it*N+_seq1._step*jt + _seq0._first*N + _seq1._first;
        }

        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx) const {
        auto it = idx / _seq1.size(), jt = idx % _seq1.size();
        auto ind = _seq0._step*it*N+_seq1._step*jt + _seq0._first*N + _seq1._first;
        return expr.data()[ind];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        vector_setter(_vec,expr.data(),_seq0._step*i*N+_seq1._step*j + _seq0._first*N + _seq1._first,_seq1._step);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first);
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> teval(const std::array<int,2>& as) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        if (_seq1._step==1) _vec.load(expr.data()+_seq0._step*as[0]*N+as[1] + _seq0._first*N + _seq1._first,false);
        else vector_setter(_vec,expr.data(),_seq0._step*as[0]*N+_seq1._step*as[1] + _seq0._first*N + _seq1._first,_seq1._step);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,2>& as) const {
        return expr(_seq0._step*as[0]+_seq0._first,_seq1._step*as[1]+_seq1._first);
    }
};













// 2D non-const views
//-------------------------------------------------------------------------------------//
template<typename T, size_t M, size_t N>
struct TensorViewExpr<Tensor<T,M,N>,2>: public AbstractTensor<TensorViewExpr<Tensor<T,M,N>,2>,2> {
private:
    Tensor<T,M,N>& expr;
    seq _seq0;
    seq _seq1;
    bool does_alias = false;
    // std::array<FASTOR_INDEX,2> _dims;
    constexpr FASTOR_INLINE Tensor<T,M,N> get_tensor() const {return expr;};
public:
    using scalar_type = T;
    static constexpr FASTOR_INDEX Dimension = 2;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    static constexpr FASTOR_INDEX rank() {return 2;}
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return _seq0.size()*_seq1.size();}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return i==0 ? _seq0.size() : _seq1.size();}

    FASTOR_INLINE TensorViewExpr<Tensor<T,M,N>,2>& noalias() {
        does_alias = true;
        return *this;
    }

    FASTOR_INLINE TensorViewExpr(Tensor<T,M,N> &_ex, seq _s0, seq _s1) :
        expr(_ex), _seq0(std::move(_s0)), _seq1(std::move(_s1)) {

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
    void operator=(const TensorViewExpr<Tensor<T,M,N>,2> &other_src) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,M,N>,2>(tmp_this_tensor,_seq0,_seq1);
            // Assign other to temporary
            tmp = other_src;
            // assign temporary to this
            this->operator=(tmp);
            return;
            // Alternatively one could do, but slower
            // does_alias = false;
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
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        // // std::array<int,SIMDVector<T,DEFAULT_ABI>::Size> inds;
        // FASTOR_INDEX i;
        // for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
        //     auto _vec_other = other_src.template eval<T>(i);
        //     for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
        //         auto it = (i+j) / _seq1.size(), jt = (i+j) % _seq1.size();
        //         // inds[j] = _seq0._step*it*N+_seq1._step*jt + _seq0._first*N + _seq1._first;
        //         auto idx = _seq0._step*it*N+_seq1._step*jt + _seq0._first*N + _seq1._first;
        //         _data[idx] = _vec_other[j];
        //     }
        // }
        // for (; i <size(); i++) {
        //     auto it = i / _seq1.size(), jt = i % _seq1.size();
        //     auto idx = _seq0._step*it*N+_seq1._step*jt + _seq0._first*N + _seq1._first;
        //     _data[idx] = other_src.template eval_s<T>(i);
        // }
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec =  other_src.template eval<T>(i,j);
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) = other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) = other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }

    void operator+=(const TensorViewExpr<Tensor<T,M,N>,2> &other_src) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,M,N>,2>(tmp_this_tensor,_seq0,_seq1);
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
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec =  this->template eval<T>(i,j) + other_src.template eval<T>(i,j);
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) += other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) += other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }

    void operator-=(const TensorViewExpr<Tensor<T,M,N>,2> &other_src) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,M,N>,2>(tmp_this_tensor,_seq0,_seq1);
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
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec =  this->template eval<T>(i,j) - other_src.template eval<T>(i,j);
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) -= other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) -= other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }

    void operator*=(const TensorViewExpr<Tensor<T,M,N>,2> &other_src) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,M,N>,2>(tmp_this_tensor,_seq0,_seq1);
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
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec =  this->template eval<T>(i,j) * other_src.template eval<T>(i,j);
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) *= other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) *= other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }

    void operator/=(const TensorViewExpr<Tensor<T,M,N>,2> &other_src) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,M,N>,2>(tmp_this_tensor,_seq0,_seq1);
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
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec =  this->template eval<T>(i,j) / other_src.template eval<T>(i,j);
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) /= other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) /= other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }
    //----------------------------------------------------------------------------------//



    // AbstractTensor binders - this is a special case for assigning another 2D expressions
    //----------------------------------------------------------------------------------//
    template<typename Derived>
    void operator=(const AbstractTensor<Derived,2> &other) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
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
        T *FASTOR_RESTRICT _data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec = other_src.template eval<T>(i,j);
                // _vec.store(&_data[(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first],false);
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) = other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) = other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }

    template<typename Derived>
    void operator+=(const AbstractTensor<Derived,2> &other) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
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
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec =  this->template eval<T>(i,j) + other_src.template eval<T>(i,j);
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) += other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) += other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }

    template<typename Derived>
    void operator-=(const AbstractTensor<Derived,2> &other) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
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
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec =  this->template eval<T>(i,j) - other_src.template eval<T>(i,j);
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) -= other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) -= other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }

    template<typename Derived>
    void operator*=(const AbstractTensor<Derived,2> &other) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
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
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec =  this->template eval<T>(i,j) * other_src.template eval<T>(i,j);
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) *= other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) *= other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }

    template<typename Derived>
    void operator/=(const AbstractTensor<Derived,2> &other) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
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
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec =  this->template eval<T>(i,j) / other_src.template eval<T>(i,j);
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) /= other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) /= other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }
    //----------------------------------------------------------------------------------//


    // AbstractTensor binders for other nth rank tensors
    //----------------------------------------------------------------------------------//
    template<typename Derived, size_t DIMS>
    void operator=(const AbstractTensor<Derived,DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
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
        T *FASTOR_RESTRICT _data = expr.data();
        FASTOR_INDEX counter = 0;
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec = other_src.template eval<T>(counter);
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                counter+=Stride;
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) = other_src.template eval_s<T>(counter);
                counter++;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) = other_src.template eval_s<T>(counter);
                counter++;
            }
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
        T *FASTOR_RESTRICT _data = expr.data();
        FASTOR_INDEX counter = 0;
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec =  this->template eval<T>(counter) + other_src.template eval<T>(counter);
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                counter+=Stride;
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) += other_src.template eval_s<T>(counter);
                counter++;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) += other_src.template eval_s<T>(counter);
                counter++;
            }
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
        T *FASTOR_RESTRICT _data = expr.data();
        FASTOR_INDEX counter = 0;
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec =  this->template eval<T>(counter) - other_src.template eval<T>(counter);
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                counter+=Stride;
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) -= other_src.template eval_s<T>(counter);
                counter++;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) -= other_src.template eval_s<T>(counter);
                counter++;
            }
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
        T *FASTOR_RESTRICT _data = expr.data();
        FASTOR_INDEX counter = 0;
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec =  this->template eval<T>(counter) * other_src.template eval<T>(counter);
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                counter+=Stride;
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) *= other_src.template eval_s<T>(counter);
                counter++;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) *= other_src.template eval_s<T>(counter);
                counter++;
            }
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
        T *FASTOR_RESTRICT _data = expr.data();
        FASTOR_INDEX counter = 0;
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec =  this->template eval<T>(counter) / other_src.template eval<T>(counter);
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
                counter+=Stride;
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) /= other_src.template eval_s<T>(counter);
                counter++;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) /= other_src.template eval_s<T>(counter);
                counter++;
            }
        }
#endif
    }
    //----------------------------------------------------------------------------------//


    // scalar binders
    //----------------------------------------------------------------------------------//
    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    FASTOR_INLINE void operator=(U num) {
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                data_setter(_data,_vec_other,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) = num;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) = num;
            }
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    FASTOR_INLINE void operator+=(U num) {
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(i,j) + _vec_other;
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) += num;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) += num;
            }
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    FASTOR_INLINE void operator-=(U num) {
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(i,j) - _vec_other;
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) -= num;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) -= num;
            }
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    FASTOR_INLINE void operator*=(U num) {
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(i,j) * _vec_other;
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) *= num;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) *= num;
            }
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    FASTOR_INLINE void operator/=(U num) {
        T *_data = expr.data();
        T inum = T(1.0)/T(num);
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(inum));
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(_seq1.size(),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(i,j) * _vec_other;
                data_setter(_data,_vec,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
            }
            for (; j <_seq1.size(); ++j) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) *= inum;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <_seq0.size(); i++) {
            for (FASTOR_INDEX j = 0; j <_seq1.size(); j++) {
                expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first) *= inum;
            }
        }
#endif
    }
    //----------------------------------------------------------------------------------//

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX idx) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,SIMDVector<U,DEFAULT_ABI>::Size> inds;
        for (auto j=0; j<SIMDVector<U,DEFAULT_ABI>::Size; ++j) {
            auto it = (idx+j) / _seq1.size(), jt = (idx+j) % _seq1.size();
            // auto it = (idx+j) / _dims[1], jt = (idx+j) % _dims[0];
            inds[j] = _seq0._step*it*N+_seq1._step*jt + _seq0._first*N + _seq1._first;
        }
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        if (_seq1._step==1) _vec.load(expr.data()+_seq0._step*i*N+j + _seq0._first*N + _seq1._first,false);
        // if (_seq1._step==1) _vec.load(expr.data()+_seq0._step*i*N+_seq1._step*j + _seq0._first*N + _seq1._first,false);
        else vector_setter(_vec,expr.data(),_seq0._step*i*N+_seq1._step*j + _seq0._first*N + _seq1._first,_seq1._step);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx) const {
        auto it = idx / _seq1.size(), jt = idx % _seq1.size();
        auto ind = _seq0._step*it*N+_seq1._step*jt + _seq0._first*N + _seq1._first;
        return expr.data()[ind];
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return expr(_seq0._step*i+_seq0._first,_seq1._step*j+_seq1._first);
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> teval(const std::array<int,2>& as) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        if (_seq1._step==1) _vec.load(&(expr.data()[_seq0._step*as[0]*N+as[1] + _seq0._first*N + _seq1._first]),false);
        else vector_setter(_vec,expr.data(),_seq0._step*as[0]*N+_seq1._step*as[1] + _seq0._first*N + _seq1._first,_seq1._step);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,2>& as) const {
        // return expr(_seq0._step*as[0]+_seq0._first,_seq1._step*as[1]+_seq1._first);
        return expr.data()[_seq0._step*as[0]*N+_seq1._step*as[1] + _seq0._first*N + _seq1._first];
    }

};


} // end of namespace Fastor


#endif // TENSOR_VIEWS_2D_H
