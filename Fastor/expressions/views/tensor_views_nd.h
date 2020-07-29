#ifndef TENSOR_VIEWS_ND_H
#define TENSOR_VIEWS_ND_H


#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/Ranges.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"

namespace Fastor {



// Generic const tensor views based on sequences/slices
//----------------------------------------------------------------------------------------------//
template<template<typename,size_t...> class TensorType, typename T, size_t DIMS, size_t ... Rest>
struct TensorConstViewExpr<TensorType<T,Rest...>,DIMS>: public AbstractTensor<TensorConstViewExpr<TensorType<T,Rest...>,DIMS>,DIMS> {
private:
    const TensorType<T,Rest...> &_expr;
    std::array<seq,sizeof...(Rest)> _seqs;
    std::array<int,DIMS> _dims;
    bool _is_vectorisable;
    bool _is_strided_vectorisable;
public:
    using scalar_type = T;
    using simd_vector_type = typename TensorType<T,Rest...>::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using result_type = TensorType<T,Rest...>;
    static constexpr FASTOR_INDEX Dimension = DIMS;
    static constexpr FASTOR_INDEX Stride = simd_vector_type::Size;
    static constexpr std::array<size_t,DIMS> products_ = nprods_views<Index<Rest...>,
        typename std_ext::make_index_sequence<DIMS>::type>::values;

    FASTOR_INLINE bool is_vectorisable() const {return _is_vectorisable;}
    FASTOR_INLINE bool is_strided_vectorisable() const {return _is_strided_vectorisable;}
    FASTOR_INLINE FASTOR_INDEX size() const {
        int sizer = 1;
        for (auto &_seq: _seqs) sizer *= _seq.size();
        return sizer;
    }
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return _seqs[i].size();}
    constexpr const TensorType<T,Rest...>& expr() const {return _expr;}

    FASTOR_INLINE TensorConstViewExpr(const TensorType<T,Rest...> &_ex, std::array<seq,sizeof...(Rest)> _s) : _expr(_ex), _seqs(std::move(_s)) {
        static_assert(DIMS==sizeof...(Rest),"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
        auto counter = 0;
        for (auto &_seq: _seqs) {
            if (_seq._last < 0 && _seq._first>=0) {
                _seq._last += _expr.dimension(counter) + 1;
            }
            // take care of scalar indexing with -1
            else if (_seq._last == 0 && _seq._first==-1) {
                auto dim = _expr.dimension(counter);
                _seq._first = dim-1;
                _seq._last = dim;
            }
            else if (_seq._last < 0 && _seq._first < 0) {
                auto dim = _expr.dimension(counter);
                _seq._first += dim + 1;
                _seq._last += dim + 1;
            }
#ifndef NDEBUG
            FASTOR_ASSERT(_seq._last <= _expr.dimension(counter) && _seq._first<_expr.dimension(counter),"INDEX OUT OF BOUNDS");
#endif
            counter++;
        }

        for (FASTOR_INDEX i=0; i<DIMS; ++i) _dims[i] = dimension(i);
        _is_vectorisable = !is_same_v_<T,bool> && _seqs[DIMS-1].size() % SIMDVector<T,simd_abi_type>::Size == 0 && (_seqs[DIMS-1]._step==1) ? true : false;
        _is_strided_vectorisable = !is_same_v_<T,bool> && _seqs[DIMS-1].size() % SIMDVector<T,simd_abi_type>::Size == 0 && (_seqs[DIMS-1]._step!=1) ? true : false;
    }


    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX idx) const {

        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,DIMS> as = {};
        std::array<int,SIMDVector<U,simd_abi_type>::Size> inds;
        for (auto j=0; j<SIMDVector<U,simd_abi_type>::Size; ++j) {
            int remaining = size();
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( int(idx+j) / remaining ) % _dims[n];
            }
            inds[j] = 0;
            for(int it = 0; it< DIMS; it++) {
                inds[j] += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }
        }
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }


    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx) const {

        std::array<int,DIMS> as = {};
        int remaining = size();
        for (int n = 0; n < DIMS; ++n) {
            remaining /= _dims[n];
            as[n] = ( (int)idx / remaining ) % _dims[n];
        }
        int ind = 0;
        for(int it = 0; it< DIMS; it++) {
            ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
        }

        return _expr.data()[ind];
    }


    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX idx, FASTOR_INDEX j) const {

        idx += j;
        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,DIMS> as = {};

        std::array<int,SIMDVector<U,simd_abi_type>::Size> inds;
        for (auto j=0; j<SIMDVector<U,simd_abi_type>::Size; ++j) {
            int remaining = size();
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( int(idx+j) / remaining ) % _dims[n];
            }
            inds[j] = 0;
            for(int it = 0; it< DIMS; it++) {
                inds[j] += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }
        }
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }


    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx, FASTOR_INDEX j) const {
        idx += j;
        std::array<int,DIMS> as = {};

        int remaining = size();
        for (int n = 0; n < DIMS; ++n) {
            remaining /= _dims[n];
            as[n] = ( (int)idx / remaining ) % _dims[n];
        }
        int ind = 0;
        for(int it = 0; it< DIMS; it++) {
            ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
        }

        return _expr.data()[ind];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int,DIMS>& as) const {
        int ind = 0;
        for(int it = 0; it< DIMS; it++) {
            ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
        }
        if (_is_vectorisable) return SIMDVector<T,simd_abi_type>(&_expr.data()[ind],false);
        else if (_is_strided_vectorisable) {
            SIMDVector<U,simd_abi_type> _vec;
            vector_setter(_vec,_expr.data(),ind,_seqs[DIMS-1]._step);
            return _vec;
        }
        else {
            // return eval(ind);

            SIMDVector<U,simd_abi_type> _vec;
            std::array<int,SIMDVector<U,simd_abi_type>::Size> inds;
            std::array<int,DIMS> as_ = as;
            for (auto j=0; j<SIMDVector<U,simd_abi_type>::Size; ++j) {
                int _sum = 0;
                for(int it = 0; it< DIMS; it++) {
                    _sum += products_[it]*as_[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
                }
                inds[j] = _sum;

                for(int jt = (int)DIMS-1; jt>=0; jt--)
                {
                  as_[jt] +=1;
                  if(as_[jt]<_dims[jt])
                      break;
                  else
                      as_[jt]=0;
                }
            }

            vector_setter(_vec,_expr.data(),inds);
            return _vec;
        }
    }

    template<typename U=T>
    FASTOR_INLINE U teval_s(const std::array<int,DIMS>& as) const {
        int ind = 0;
        for(int it = 0; it< DIMS; it++) {
            ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
        }
        return _expr.data()[ind];
    }
};
//----------------------------------------------------------------------------------------------//






// Generic non-const tensor views based on sequences/slices
//----------------------------------------------------------------------------------------------//
template<template<typename,size_t...> class TensorType, typename T, size_t DIMS, size_t ... Rest>
struct TensorViewExpr<TensorType<T,Rest...>,DIMS>: public AbstractTensor<TensorViewExpr<TensorType<T,Rest...>,DIMS>,DIMS> {
private:
    TensorType<T,Rest...> &_expr;
    std::array<seq,sizeof...(Rest)> _seqs;
    bool _does_alias = false;
    std::array<int,DIMS> _dims;
    bool _is_vectorisable;
    bool _is_strided_vectorisable;

    constexpr FASTOR_INLINE Tensor<T,Rest...> get_tensor() const {return _expr;};
    constexpr FASTOR_INLINE std::array<seq,sizeof...(Rest)> get_sequences() const {return _seqs;}

public:
    using scalar_type = T;
    using simd_vector_type = typename TensorType<T,Rest...>::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using result_type = TensorType<T,Rest...>;
    static constexpr FASTOR_INDEX Dimension = DIMS;
    static constexpr FASTOR_INDEX Stride = simd_vector_type::Size;
    static constexpr std::array<size_t,DIMS> products_ = nprods_views<Index<Rest...>,
        typename std_ext::make_index_sequence<DIMS>::type>::values;

    FASTOR_INLINE bool is_vectorisable() const {return _is_vectorisable;}
    FASTOR_INLINE bool is_strided_vectorisable() const {return _is_strided_vectorisable;}
    FASTOR_INLINE FASTOR_INDEX size() const {
        int sizer = 1;
        for (auto &_seq: _seqs) sizer *= _seq.size();
        return sizer;
    }
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return _seqs[i].size();}
    constexpr const TensorType<T,Rest...>& expr() const {return _expr;}

    FASTOR_INLINE TensorViewExpr<TensorType<T,Rest...>,DIMS>& noalias() {
        _does_alias = true;
        return *this;
    }

    TensorViewExpr(TensorType<T,Rest...> &_ex, std::array<seq,sizeof...(Rest)> _s) : _expr(_ex), _seqs(std::move(_s)) {
        static_assert(DIMS==sizeof...(Rest),"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
        auto counter = 0;
        for (auto &_seq: _seqs) {
            if (_seq._last < 0 && _seq._first>=0) {
                _seq._last += _expr.dimension(counter) + 1;
            }
            // take care of scalar indexing with -1
            else if (_seq._last == 0 && _seq._first==-1) {
                auto dim = _expr.dimension(counter);
                _seq._first = dim-1;
                _seq._last = dim;
            }
            else if (_seq._last < 0 && _seq._first < 0) {
                auto dim = _expr.dimension(counter);
                _seq._first += dim + 1;
                _seq._last += dim + 1;
            }
#ifndef NDEBUG
            FASTOR_ASSERT(_seq._last <= _expr.dimension(counter) && _seq._first<_expr.dimension(counter),"INDEX OUT OF BOUNDS");
#endif
            counter++;
        }

        for (FASTOR_INDEX i=0; i<DIMS; ++i) _dims[i] = dimension(i);
        _is_vectorisable = !is_same_v_<T,bool> && _seqs[DIMS-1].size() % SIMDVector<T,simd_abi_type>::Size == 0 && (_seqs[DIMS-1]._step==1) ? true : false;
        _is_strided_vectorisable = !is_same_v_<T,bool> && _seqs[DIMS-1].size() % SIMDVector<T,simd_abi_type>::Size == 0 && (_seqs[DIMS-1]._step!=1) ? true : false;
    }

    // View evalution operators
    // Copy assignment operators [Needed in addition to generic AbstractTensor overload]
    //----------------------------------------------------------------------------------//
    FASTOR_HINT_INLINE void operator=(const TensorViewExpr<TensorType<T,Rest...>,DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<TensorType<T,Rest...>,DIMS>(tmp_this_tensor,get_sequences());
            // Assign other to temporary
            tmp = other;
            // assign temporary to this
            this->operator=(tmp);
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

        T *_data = _expr.data();
        std::array<int,DIMS> as = {};
        int total = size();
        int jt, counter = 0;

        if (_is_vectorisable) {
            using V=SIMDVector<T,simd_abi_type>;
            // constexpr FASTOR_INDEX stride = V::Size;
            V _vec;
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                _vec = other.template teval<T>(as);
                _vec.store(&_data[ind],false);

                counter+=V::Size;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    if (jt == _dims.size()-1) as[jt]+=V::Size;
                    else as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        else {
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                _data[ind] = other.template teval_s<T>(as);
                // print(as);

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    as[jt] += 1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
    }
    //----------------------------------------------------------------------------------//

    // AbstractTensor binders [equal order]
    //----------------------------------------------------------------------------------//
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS==DIMS && requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator=(tmp);
    }
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS==DIMS && !requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,Rest...>,DIMS>(tmp_this_tensor,get_sequences());
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
        std::array<int,DIMS> as = {};
        int total = size();
        int jt, counter = 0;

        if (_is_vectorisable) {
            using V=SIMDVector<T,simd_abi_type>;
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                // V _vec = other_src.template eval<T>(counter);
                V _vec = other_src.template teval<T>(as);
                _vec.store(&_data[ind],false);

                counter+=V::Size;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    if (jt == _dims.size()-1) as[jt]+=V::Size;
                    else as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        else
        {
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                // _data[ind] = other_src.template eval_s<T>(counter);
                _data[ind] = other_src.template teval_s<T>(as);

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }

            // // Generic vectorised version that takes care of the remainder scalar ops
            // using V=SIMDVector<T,simd_abi_type>;
            // while(counter < total)
            // {
            //     int ind = 0;
            //     for(int it = 0; it< DIMS; it++) {
            //         ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
            //     }
            //     if (_dims[DIMS-1] - as[DIMS-1] % V::Size == 0) {
            //         // V _vec = other_src.template eval<T>(counter);
            //         V _vec = other_src.template teval<T>(as);
            //         _vec.store(&_data[ind],false);
            //         counter+=V::Size;
            //     }
            //     else {
            //         // _data[ind] = other_src.template eval_s<T>(counter);
            //         _data[ind] = other_src.template teval_s<T>(as);
            //         counter++;
            //     }

            //     for(jt = DIMS-1; jt>=0; jt--)
            //     {
            //         if (jt == _dims.size()-1) as[jt]+=V::Size;
            //         else as[jt] +=1;
            //         if(as[jt]<_dims[jt])
            //             break;
            //         else
            //             as[jt]=0;
            //     }
            //     if(jt<0)
            //         break;
            // }
        }
    }

    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS==DIMS && requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator+=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator+=(tmp);
    }
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS==DIMS && !requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator+=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,Rest...>,DIMS>(tmp_this_tensor,get_sequences());
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
        std::array<int,DIMS> as = {};
        int total = size();
        int jt, counter = 0;

        if (_is_vectorisable) {
            using V=SIMDVector<T,simd_abi_type>;
            V _vec_out;
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                // V _vec = other_src.template eval<T>(counter);
                V _vec = other_src.template teval<T>(as);
                _vec_out.load(&_data[ind],false);
                _vec_out += _vec;
                _vec_out.store(&_data[ind],false);

                counter+=V::Size;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    if (jt == _dims.size()-1) as[jt]+=V::Size;
                    else as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        else
        {
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                // _data[ind] += other_src.template eval_s<T>(counter);
                _data[ind] += other_src.template teval_s<T>(as);

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
    }


    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS==DIMS && requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator-=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator-=(tmp);
    }
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS==DIMS && !requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator-=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,Rest...>,DIMS>(tmp_this_tensor,get_sequences());
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
        std::array<int,DIMS> as = {};
        int total = size();
        int jt, counter = 0;

        if (_is_vectorisable) {
            using V=SIMDVector<T,simd_abi_type>;
            V _vec_out;
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                // V _vec = other_src.template eval<T>(counter);
                V _vec = other_src.template teval<T>(as);
                _vec_out.load(&_data[ind],false);
                _vec_out -= _vec;
                _vec_out.store(&_data[ind],false);

                counter+=V::Size;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    if (jt == _dims.size()-1) as[jt]+=V::Size;
                    else as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        else
        {
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                // _data[ind] -= other_src.template eval_s<T>(counter);
                _data[ind] -= other_src.template teval_s<T>(as);

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
    }

    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS==DIMS && requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator*=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator*=(tmp);
    }
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS==DIMS && !requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator*=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,Rest...>,DIMS>(tmp_this_tensor,get_sequences());
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
        std::array<int,DIMS> as = {};
        int total = size();
        int jt, counter = 0;

        if (_is_vectorisable) {
            using V=SIMDVector<T,simd_abi_type>;
            V _vec_out;
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                // V _vec = other_src.template eval<T>(counter);
                V _vec = other_src.template teval<T>(as);
                _vec_out.load(&_data[ind],false);
                _vec_out *= _vec;
                _vec_out.store(&_data[ind],false);

                counter+=V::Size;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    if (jt == _dims.size()-1) as[jt]+=V::Size;
                    else as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        else
        {
             while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                // _data[ind] *= other_src.template eval_s<T>(counter);
                _data[ind] *= other_src.template teval_s<T>(as);

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
    }

    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS==DIMS && requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator/=(tmp);
    }
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS==DIMS && !requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,Rest...>,DIMS>(tmp_this_tensor,get_sequences());
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
        std::array<int,DIMS> as = {};
        int total = size();
        int jt, counter = 0;

        if (_is_vectorisable) {
            using V=SIMDVector<T,simd_abi_type>;
            V _vec_out;
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                // V _vec = other_src.template eval<T>(counter);
                V _vec = other_src.template teval<T>(as);
                _vec_out.load(&_data[ind],false);
                _vec_out /= _vec;
                _vec_out.store(&_data[ind],false);

                counter+=V::Size;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    if (jt == _dims.size()-1) as[jt]+=V::Size;
                    else as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        else
        {
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                // _data[ind] /= other_src.template eval_s<T>(counter);
                _data[ind] /= other_src.template teval_s<T>(as);

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
    }
    //----------------------------------------------------------------------------------//

    // AbstractTensor binders [non-equal orders]
    //----------------------------------------------------------------------------------//
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS!=DIMS && requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator=(tmp);
    }
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS!=DIMS && !requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,Rest...>,DIMS>(tmp_this_tensor,get_sequences());
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
        std::array<int,DIMS> as = {};
        int total = size();
        int jt, counter = 0;

        if (_is_vectorisable) {
            using V=SIMDVector<T,simd_abi_type>;
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                V _vec = other_src.template eval<T>(counter);
                // V _vec = other_src.template teval<T>(as);
                _vec.store(&_data[ind],false);

                counter+=V::Size;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    if (jt == _dims.size()-1) as[jt]+=V::Size;
                    else as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        else
        {
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                _data[ind] = other_src.template eval_s<T>(counter);
                // _data[ind] = other_src.template teval_s<T>(as);

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
    }

    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS!=DIMS && requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator+=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator+=(tmp);
    }
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS!=DIMS && !requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator+=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,Rest...>,DIMS>(tmp_this_tensor,get_sequences());
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
        std::array<int,DIMS> as = {};
        int total = size();
        int jt, counter = 0;

        if (_is_vectorisable) {
            using V=SIMDVector<T,simd_abi_type>;
            constexpr FASTOR_INDEX stride = V::Size;
            V _vec_out;
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                V _vec = other_src.template eval<T>(counter);
                // V _vec = other_src.template teval<T>(as);
                _vec_out.load(&_data[ind],false);
                _vec_out += _vec;
                _vec_out.store(&_data[ind],false);

                counter+=V::Size;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    if (jt == _dims.size()-1) as[jt]+=stride;
                    else as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        else {
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                _data[ind] += other_src.template eval_s<T>(counter);
                // _data[ind] += other_src.template teval_s<T>(as);

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
    }

    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS!=DIMS && requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator-=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator-=(tmp);
    }
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS!=DIMS && !requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator-=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,Rest...>,DIMS>(tmp_this_tensor,get_sequences());
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
        std::array<int,DIMS> as = {};
        int total = size();
        int jt, counter = 0;

        if (_is_vectorisable) {
            using V=SIMDVector<T,simd_abi_type>;
            V _vec_out;
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                V _vec = other_src.template eval<T>(counter);
                // V _vec = other_src.template teval<T>(as);
                _vec_out.load(&_data[ind],false);
                _vec_out -= _vec;
                _vec_out.store(&_data[ind],false);

                counter+=V::Size;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    if (jt == _dims.size()-1) as[jt]+=V::Size;
                    else as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        else {
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                _data[ind] -= other_src.template eval_s<T>(counter);
                // _data[ind] -= other_src.template teval_s<T>(as);

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
    }

    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS!=DIMS && requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator*=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator*=(tmp);
    }
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS!=DIMS && !requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator*=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,Rest...>,DIMS>(tmp_this_tensor,get_sequences());
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
        std::array<int,DIMS> as = {};
        int total = size();
        int jt, counter = 0;

        if (_is_vectorisable) {
            using V=SIMDVector<T,simd_abi_type>;
            constexpr FASTOR_INDEX stride = V::Size;
            V _vec_out;
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                V _vec = other_src.template eval<T>(counter);
                // V _vec = other_src.template teval<T>(as);
                _vec_out.load(&_data[ind],false);
                _vec_out *= _vec;
                _vec_out.store(&_data[ind],false);

                counter+=V::Size;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    if (jt == _dims.size()-1) as[jt]+=stride;
                    else as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        else {
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                _data[ind] *= other_src.template eval_s<T>(counter);
                // _data[ind] *= other_src.template teval_s<T>(as);

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
    }

    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS!=DIMS && requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator/=(tmp);
    }
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS!=DIMS && !requires_evaluation_v<Derived>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (_does_alias) {
            _does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorViewExpr<Tensor<T,Rest...>,DIMS>(tmp_this_tensor,get_sequences());
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
        std::array<int,DIMS> as = {};
        int total = size();
        int jt, counter = 0;

        if (_is_vectorisable) {
            using V=SIMDVector<T,simd_abi_type>;
            constexpr FASTOR_INDEX stride = V::Size;
            V _vec_out;
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                V _vec = other_src.template eval<T>(counter);
                // V _vec = other_src.template teval<T>(as);
                _vec_out.load(&_data[ind],false);
                _vec_out /= _vec;
                _vec_out.store(&_data[ind],false);

                counter+=V::Size;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    if (jt == _dims.size()-1) as[jt]+=stride;
                    else as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        else {
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                _data[ind] /= other_src.template eval_s<T>(counter);
                // _data[ind] /= other_src.template teval_s<T>(as);

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
    }

    // scalar binders
    //----------------------------------------------------------------------------------//
    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator=(U num) {

        T *_data = _expr.data();
        std::array<int,DIMS> as = {};
        int total = size();
        int jt, counter = 0;

        if (_is_vectorisable) {
            using V=SIMDVector<T,simd_abi_type>;
            constexpr FASTOR_INDEX stride = V::Size;
            V _vec = (T)num;
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                _vec.store(&_data[ind],false);

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    if (jt == _dims.size()-1) as[jt]+=stride;
                    else as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        else {
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                _data[ind] = (T)num;

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator+=(U num) {

        T *_data = _expr.data();
        std::array<int,DIMS> as = {};
        int total = size();
        int jt, counter = 0;

        if (_is_vectorisable) {
            using V=SIMDVector<T,simd_abi_type>;
            constexpr FASTOR_INDEX stride = V::Size;
            V _vec = (T)num;
            V _vec_out;
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                _vec_out.load(&_data[ind],false);
                _vec_out += _vec;
                _vec_out.store(&_data[ind],false);

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    if (jt == _dims.size()-1) as[jt]+=stride;
                    else as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        else {
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                _data[ind] += (T)num;

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator-=(U num) {

        T *_data = _expr.data();
        std::array<int,DIMS> as = {};
        int total = size();
        int jt, counter = 0;

        if (_is_vectorisable) {
            using V=SIMDVector<T,simd_abi_type>;
            constexpr FASTOR_INDEX stride = V::Size;
            V _vec = (T)num;
            V _vec_out;
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                _vec_out.load(&_data[ind],false);
                _vec_out -= _vec;
                _vec_out.store(&_data[ind],false);

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    if (jt == _dims.size()-1) as[jt]+=stride;
                    else as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        else {
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                _data[ind] -= (T)num;

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator*=(U num) {

        T *_data = _expr.data();
        std::array<int,DIMS> as = {};
        int total = size();
        int jt, counter = 0;

        if (_is_vectorisable) {
            using V=SIMDVector<T,simd_abi_type>;
            constexpr FASTOR_INDEX stride = V::Size;
            V _vec = (T)num;
            V _vec_out;
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                _vec_out.load(&_data[ind],false);
                _vec_out *= _vec;
                _vec_out.store(&_data[ind],false);

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    if (jt == _dims.size()-1) as[jt]+=stride;
                    else as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        else {
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                _data[ind] *= (T)num;

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_HINT_INLINE void operator/=(U num) {

        T *_data = _expr.data();
        std::array<int,DIMS> as = {};
        int total = size();
        int jt, counter = 0;

        if (_is_vectorisable) {
            using V=SIMDVector<T,simd_abi_type>;
            constexpr FASTOR_INDEX stride = V::Size;
            V _vec = (T)num;
            V _vec_out;
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                _vec_out.load(&_data[ind],false);
                _vec_out /= _vec;
                _vec_out.store(&_data[ind],false);

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    if (jt == _dims.size()-1) as[jt]+=stride;
                    else as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        else {
            while(counter < total)
            {
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                }
                _data[ind] /= (T)num;

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
    }
    //----------------------------------------------------------------------------------//

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX idx) const {

        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,DIMS> as = {};
        std::array<int,SIMDVector<U,simd_abi_type>::Size> inds;
        for (auto j=0; j<SIMDVector<U,simd_abi_type>::Size; ++j) {
            int remaining = size();
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( int(idx+j) / remaining ) % _dims[n];
            }
            inds[j] = 0;
            for(int it = 0; it< DIMS; it++) {
                inds[j] += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }
        }
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }


    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx) const {

        std::array<int,DIMS> as = {};
        int remaining = size();
        for (int n = 0; n < DIMS; ++n) {
            remaining /= _dims[n];
            as[n] = ( (int)idx / remaining ) % _dims[n];
        }
        int ind = 0;
        for(int it = 0; it< DIMS; it++) {
            ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
        }

        return _expr.data()[ind];
    }


    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX idx, FASTOR_INDEX j) const {

        idx += j;
        SIMDVector<U,simd_abi_type> _vec;
        std::array<int,DIMS> as = {};

        std::array<int,SIMDVector<U,simd_abi_type>::Size> inds;
        for (auto j=0; j<SIMDVector<U,simd_abi_type>::Size; ++j) {
            int remaining = size();
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( int(idx+j) / remaining ) % _dims[n];
            }
            inds[j] = 0;
            for(int it = 0; it< DIMS; it++) {
                inds[j] += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }
        }
        vector_setter(_vec,_expr.data(),inds);
        return _vec;
    }


    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx, FASTOR_INDEX j) const {

        idx += j;
        std::array<int,DIMS> as = {};
        int remaining = size();
        for (int n = 0; n < DIMS; ++n) {
            remaining /= _dims[n];
            as[n] = ( (int)idx / remaining ) % _dims[n];
        }
        int ind = 0;
        for(int it = 0; it< DIMS; it++) {
            ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
        }

        return _expr.data()[ind];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int,DIMS>& as) const {
        int ind = 0;
        for(int it = 0; it< DIMS; it++) {
            ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
        }
        if (_is_vectorisable) return SIMDVector<T,simd_abi_type>(&_expr.data()[ind],false);
        else if (_is_strided_vectorisable) {
            SIMDVector<U,simd_abi_type> _vec;
            vector_setter(_vec,_expr.data(),ind,_seqs[DIMS-1]._step);
            return _vec;
        }
        else {
            // return eval(ind);

            SIMDVector<U,simd_abi_type> _vec;
            std::array<int,SIMDVector<U,simd_abi_type>::Size> inds;
            std::array<int,DIMS> as_ = as;
            for (auto j=0; j<SIMDVector<U,simd_abi_type>::Size; ++j) {
                int _sum = 0;
                for(int it = 0; it< DIMS; it++) {
                    _sum += products_[it]*as_[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
                }
                inds[j] = _sum;

                for(int jt = (int)DIMS-1; jt>=0; jt--)
                {
                  as_[jt] +=1;
                  if(as_[jt]<_dims[jt])
                      break;
                  else
                      as_[jt]=0;
                }
            }

            vector_setter(_vec,_expr.data(),inds);
            return _vec;
        }
    }

    template<typename U=T>
    FASTOR_INLINE U teval_s(const std::array<int,DIMS>& as) const {
        int ind = 0;
        for(int it = 0; it< DIMS; it++) {
            ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
        }
        return _expr.data()[ind];
    }
};


template<template<typename,size_t...> class TensorType, typename T, size_t DIMS, size_t ... Rest>
constexpr std::array<size_t,DIMS> TensorViewExpr<TensorType<T,Rest...>,DIMS>::products_;


} // end of namespace Fastor


#endif // TENSOR_VIEWS_ND_H
