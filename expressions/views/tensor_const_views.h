#ifndef TENSOR_CONST_VIEWS_H
#define TENSOR_CONST_VIEWS_H

#include "tensor/Tensor.h"
#include "tensor/ranges.h"

namespace Fastor {

// Generic tensor views based on sequences/slices
//----------------------------------------------------------------------------------------------//
template<typename T, size_t DIMS, size_t ... Rest>
struct TensorConstViewExpr<Tensor<T,Rest...>,DIMS>: public AbstractTensor<TensorConstViewExpr<Tensor<T,Rest...>,DIMS>,DIMS> {
private:
    const Tensor<T,Rest...> &expr;
    std::array<seq,sizeof...(Rest)> _seqs;
public: 
    using scalar_type = T;
    static constexpr FASTOR_INDEX Dimension = DIMS;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    FASTOR_INLINE FASTOR_INDEX size() const {
        int sizer = 1;
        for (auto &_seq: _seqs) sizer *= _seq.size(); 
        return sizer; 
    }
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return _seqs[i].size();}

    FASTOR_INLINE TensorConstViewExpr(const Tensor<T,Rest...> &_ex, std::array<seq,sizeof...(Rest)> _s) : expr(_ex), _seqs(std::move(_s)) {
        static_assert(DIMS==sizeof...(Rest),"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
        auto counter = 0;
        for (auto &_seq: _seqs) {
            if (_seq._last < 0 && _seq._first>=0) {
                _seq._last += expr.dimension(counter) + 1;
            }
            // take care of scalar indexing with -1
            else if (_seq._last == 0 && _seq._first==-1) {
                auto dim = expr.dimension(counter);
                _seq._first = dim-1;
                _seq._last = dim;
            }
            else if (_seq._last < 0 && _seq._first < 0) {
                auto dim = expr.dimension(counter);
                _seq._first += dim + 1;
                _seq._last += dim + 1;
            }
#ifndef NDEBUG
            FASTOR_ASSERT(_seq._last <= expr.dimension(counter) && _seq._first<expr.dimension(counter),"INDEX OUT OF BOUNDS");
#endif            
            counter++;
        }
    }


    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX idx) const {
        SIMDVector<U,DEFAULT_ABI> _vec; 

        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>, 
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);

        std::array<int,SIMDVector<U,DEFAULT_ABI>::Size> inds;
        for (auto j=0; j<SIMDVector<U,DEFAULT_ABI>::Size; ++j) {
            int remaining = size();
            for (int n = 0; n < DIMS; ++n) {
                remaining /= dimension(n);
                as[n] = ( (idx+j) / remaining ) % dimension(n);
            }
            inds[j] = 0;
            for(int it = 0; it< DIMS; it++) {
                inds[j] += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }
        }
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }


    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx) const {
        
        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>,
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);

        int remaining = size();
        for (int n = 0; n < DIMS; ++n) {
            remaining /= dimension(n);
            as[n] = ( idx / remaining ) % dimension(n);
        }
        int ind = 0;
        for(int it = 0; it< DIMS; it++) {
            ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
        }

        return expr.data()[ind];
    }
    

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX idx, FASTOR_INDEX j) const {

        idx += j;
        SIMDVector<U,DEFAULT_ABI> _vec; 
        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>, 
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);

        std::array<int,SIMDVector<U,DEFAULT_ABI>::Size> inds;
        for (auto j=0; j<SIMDVector<U,DEFAULT_ABI>::Size; ++j) {
            int remaining = size();
            for (int n = 0; n < DIMS; ++n) {
                remaining /= dimension(n);
                as[n] = ( (idx+j) / remaining ) % dimension(n);
            }
            inds[j] = 0;
            for(int it = 0; it< DIMS; it++) {
                inds[j] += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }
        }
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }


    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx, FASTOR_INDEX j) const {
        idx += j;
        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>,
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);

        int remaining = size();
        for (int n = 0; n < DIMS; ++n) {
            remaining /= dimension(n);
            as[n] = ( idx / remaining ) % dimension(n);
        }
        int ind = 0;
        for(int it = 0; it< DIMS; it++) {
            ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
        }

        return expr.data()[ind];
    }
};













// 1D
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
};



// 2D Views
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
};





}

#endif  // TENSOR_CONST_VIEWS_H