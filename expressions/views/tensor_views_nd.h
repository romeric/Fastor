#ifndef TENSOR_VIEWS_ND_H
#define TENSOR_VIEWS_ND_H


#include "tensor/Tensor.h"
#include "tensor/ranges.h"

namespace Fastor {



// Generic const tensor views based on sequences/slices
//----------------------------------------------------------------------------------------------//
template<typename T, size_t DIMS, size_t ... Rest>
struct TensorConstViewExpr<Tensor<T,Rest...>,DIMS>: public AbstractTensor<TensorConstViewExpr<Tensor<T,Rest...>,DIMS>,DIMS> {
private:
    const Tensor<T,Rest...> &expr;
    std::array<seq,sizeof...(Rest)> _seqs;
    std::array<int,DIMS> _dims;
public:
    using scalar_type = T;
    static constexpr FASTOR_INDEX Dimension = DIMS;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    static constexpr std::array<size_t,DIMS> products_ = nprods_views<Index<Rest...>,
        typename std_ext::make_index_sequence<DIMS>::type>::values;
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

        for (FASTOR_INDEX i=0; i<DIMS; ++i) _dims[i] = dimension(i);
    }


    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX idx) const {

        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,DIMS> as = {};
        std::array<int,SIMDVector<U,DEFAULT_ABI>::Size> inds;
        for (auto j=0; j<SIMDVector<U,DEFAULT_ABI>::Size; ++j) {
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
        vector_setter(_vec,expr.data(),inds);
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

        return expr.data()[ind];
    }


    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX idx, FASTOR_INDEX j) const {

        idx += j;
        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,DIMS> as = {};

        std::array<int,SIMDVector<U,DEFAULT_ABI>::Size> inds;
        for (auto j=0; j<SIMDVector<U,DEFAULT_ABI>::Size; ++j) {
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
        vector_setter(_vec,expr.data(),inds);
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

        return expr.data()[ind];
    }
};
//----------------------------------------------------------------------------------------------//



// Generic non-const tensor views based on sequences/slices
//----------------------------------------------------------------------------------------------//
template<typename T, size_t DIMS, size_t ... Rest>
struct TensorViewExpr<Tensor<T,Rest...>,DIMS>: public AbstractTensor<TensorViewExpr<Tensor<T,Rest...>,DIMS>,DIMS> {
private:
    Tensor<T,Rest...> &expr;
    std::array<seq,sizeof...(Rest)> _seqs;
    bool does_alias = false;
    std::array<int,DIMS> _dims;

    constexpr FASTOR_INLINE Tensor<T,Rest...> get_tensor() const {return expr;};
    constexpr FASTOR_INLINE std::array<seq,sizeof...(Rest)> get_sequences() const {return _seqs;}

public:
    using scalar_type = T;
    static constexpr FASTOR_INDEX Dimension = DIMS;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    static constexpr std::array<size_t,DIMS> products_ = nprods_views<Index<Rest...>,
        typename std_ext::make_index_sequence<DIMS>::type>::values;
    FASTOR_INLINE FASTOR_INDEX size() const {
        int sizer = 1;
        for (auto &_seq: _seqs) sizer *= _seq.size();
        return sizer;
    }
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return _seqs[i].size();}

    FASTOR_INLINE TensorViewExpr<Tensor<T,Rest...>,DIMS>& noalias() {
        does_alias = true;
        return *this;
    }

    TensorViewExpr(Tensor<T,Rest...> &_ex, std::array<seq,sizeof...(Rest)> _s) : expr(_ex), _seqs(std::move(_s)) {
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

        for (FASTOR_INDEX i=0; i<DIMS; ++i) _dims[i] = dimension(i);
    }

    // View evalution operators
    // Copy assignment operators [Needed in addition to generic AbstractTensor overload]
    //----------------------------------------------------------------------------------//
    void operator=(const TensorViewExpr<Tensor<T,Rest...>,DIMS> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
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
#ifndef NDEBUG
        FASTOR_ASSERT(other.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();

        std::array<int,DIMS> as = {};
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= _dims[n];
                    as[n] = ( (i+j) / remaining ) % _dims[n];
                }
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
                }
                _data[ind] = _vec_other[j];
            }
        }
        // Remaining scalar ops
        for (; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] = other.template eval_s<T>(i);
        }
#else
        for (int i = 0; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] = other.template eval_s<T>(i);
        }
#endif
    }

    void operator+=(const TensorViewExpr<Tensor<T,Rest...>,DIMS> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
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
#ifndef NDEBUG
        FASTOR_ASSERT(other.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();
        std::array<int,DIMS> as = {};
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= _dims[n];
                    as[n] = ( (i+j) / remaining ) % _dims[n];
                }
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
                }
                _data[ind] += _vec_other[j];
            }
        }
        // Remaining scalar ops
        for (; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] += other.template eval_s<T>(i);
        }
#else
        for (int i = 0; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] += other.template eval_s<T>(i);
        }
#endif
    }

    void operator-=(const TensorViewExpr<Tensor<T,Rest...>,DIMS> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
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
#ifndef NDEBUG
        FASTOR_ASSERT(other.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();

        std::array<int,DIMS> as = {};
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= _dims[n];
                    as[n] = ( (i+j) / remaining ) % _dims[n];
                }
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
                }
                _data[ind] -= _vec_other[j];
            }
        }
        // Remaining scalar ops
        for (; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] -= other.template eval_s<T>(i);
        }
#else
        for (int i = 0; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] -= other.template eval_s<T>(i);
        }
#endif
    }

    void operator*=(const TensorViewExpr<Tensor<T,Rest...>,DIMS> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
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
#ifndef NDEBUG
        FASTOR_ASSERT(other.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();
        std::array<int,DIMS> as = {};
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= _dims[n];
                    as[n] = ( (i+j) / remaining ) % _dims[n];
                }
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
                }
                _data[ind] *= _vec_other[j];
            }
        }
        // Remaining scalar ops
        for (; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] *= other.template eval_s<T>(i);
        }
#else
        for (int i = 0; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] *= other.template eval_s<T>(i);
        }
#endif
    }

    void operator/=(const TensorViewExpr<Tensor<T,Rest...>,DIMS> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
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
#ifndef NDEBUG
        FASTOR_ASSERT(other.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();
        std::array<int,DIMS> as = {};
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= _dims[n];
                    as[n] = ( (i+j) / remaining ) % _dims[n];
                }
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
                }
                _data[ind] /= _vec_other[j];
            }
        }
        // Remaining scalar ops
        for (; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] /= other.template eval_s<T>(i);
        }
#else
        for (int i = 0; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] /= other.template eval_s<T>(i);
        }
#endif
    }
    //----------------------------------------------------------------------------------//

    // AbstractTensor binders
    //----------------------------------------------------------------------------------//
    template<typename Derived, size_t OTHER_DIMS>
    void operator=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
        if (does_alias) {
            does_alias = false;
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
        T *_data = expr.data();
        std::array<int,DIMS> as;
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= _dims[n];
                    as[n] = ( (i+j) / remaining ) % _dims[n];
                }
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
                }
                _data[ind] = _vec_other[j];
            }
        }
        // Remaining scalar ops
        for (; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] = other_src.template eval_s<T>(i);
        }
#else
        for (int i = 0; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] = other_src.template eval_s<T>(i);
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
        T *_data = expr.data();
        std::array<int,DIMS> as = {};
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= _dims[n];
                    as[n] = ( (i+j) / remaining ) % _dims[n];
                }
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
                }
                _data[ind] += _vec_other[j];
            }
        }
        // Remaining scalar ops
        for (; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] += other_src.template eval_s<T>(i);
        }
#else
        for (int i = 0; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] += other_src.template eval_s<T>(i);
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
        T *_data = expr.data();
        std::array<int,DIMS> as = {};
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= _dims[n];
                    as[n] = ( (i+j) / remaining ) % _dims[n];
                }
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
                }
                _data[ind] -= _vec_other[j];
            }
        }
        // Remaining scalar ops
        for (; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] -= other_src.template eval_s<T>(i);
        }
#else
        for (int i = 0; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] -= other_src.template eval_s<T>(i);
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
        T *_data = expr.data();
        std::array<int,DIMS> as = {};
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= _dims[n];
                    as[n] = ( (i+j) / remaining ) % _dims[n];
                }
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
                }
                _data[ind] *= _vec_other[j];
            }
        }
        // Remaining scalar ops
        for (; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] *= other_src.template eval_s<T>(i);
        }
#else
        for (int i = 0; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] *= other_src.template eval_s<T>(i);
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
        T *_data = expr.data();
        std::array<int,DIMS> as = {};
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= _dims[n];
                    as[n] = ( (i+j) / remaining ) % _dims[n];
                }
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
                }
                _data[ind] /= _vec_other[j];
            }
        }
        // Remaining scalar ops
        for (; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] /= other_src.template eval_s<T>(i);
        }
#else
        for (int i = 0; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] /= other_src.template eval_s<T>(i);
        }
#endif
    }

    // scalar binders
    //----------------------------------------------------------------------------------//
    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator=(U num) {

        T *_data = expr.data();
        std::array<int,DIMS> as = {};
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i;
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= _dims[n];
                    as[n] = ( (i+j) / remaining ) % _dims[n];
                }
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
                }
                _data[ind] = _vec_other[j];
            }
        }
        for (; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }
            _data[ind] = num;
        }
#else
        for (int i = 0; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }
            _data[ind] = num;
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator+=(U num) {

        T *_data = expr.data();
        std::array<int,DIMS> as = {};
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i;
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= _dims[n];
                    as[n] = ( (i+j) / remaining ) % _dims[n];
                }
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
                }
                _data[ind] += _vec_other[j];
            }
        }
        for (; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }
            _data[ind] += num;
        }
#else
        for (int i = 0; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }
            _data[ind] += num;
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator-=(U num) {

        T *_data = expr.data();
        std::array<int,DIMS> as = {};
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i;
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= _dims[n];
                    as[n] = ( (i+j) / remaining ) % _dims[n];
                }
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
                }
                _data[ind] -= _vec_other[j];
            }
        }
        for (; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }
            _data[ind] -= num;
        }
#else
        for (int i = 0; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }
            _data[ind] -= num;
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator*=(U num) {

        T *_data = expr.data();
        std::array<int,DIMS> as = {};
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i;
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= _dims[n];
                    as[n] = ( (i+j) / remaining ) % _dims[n];
                }
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
                }
                _data[ind] *= _vec_other[j];
            }
        }
        for (; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }
            _data[ind] *= num;
        }
#else
        for (int i = 0; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }
            _data[ind] *= num;
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator/=(U num) {

        T *_data = expr.data();

        std::array<int,DIMS> as = {};
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i;
        T inum = T(1.)/num;
        SIMDVector<T,DEFAULT_ABI> _vec_other(inum);
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= _dims[n];
                    as[n] = ( (i+j) / remaining ) % _dims[n];
                }
                int ind = 0;
                for(int it = 0; it< DIMS; it++) {
                    ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
                }
                _data[ind] *= _vec_other[j];
            }
        }
        for (; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }
            _data[ind] *= inum;
        }
#else
        T inum = T(1.)/num;
        for (int i = 0; i <total; i++) {
            int remaining = total;
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( i / remaining ) % _dims[n];
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }
            _data[ind] *= inum;
        }
#endif
    }
    //----------------------------------------------------------------------------------//

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX idx) const {

        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,DIMS> as = {};
        std::array<int,SIMDVector<U,DEFAULT_ABI>::Size> inds;
        for (auto j=0; j<SIMDVector<U,DEFAULT_ABI>::Size; ++j) {
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

        vector_setter(_vec,expr.data(),inds);
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

        return expr.data()[ind];
    }


    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX idx, FASTOR_INDEX j) const {

        idx += j;
        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,DIMS> as = {};

        std::array<int,SIMDVector<U,DEFAULT_ABI>::Size> inds;
        for (auto j=0; j<SIMDVector<U,DEFAULT_ABI>::Size; ++j) {
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
        vector_setter(_vec,expr.data(),inds);
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

        return expr.data()[ind];
    }
};


template<typename T, size_t DIMS, size_t ... Rest>
constexpr std::array<size_t,DIMS> TensorViewExpr<Tensor<T,Rest...>,DIMS>::products_;


} // end of namespace Fastor


#endif // TENSOR_VIEWS_ND_H