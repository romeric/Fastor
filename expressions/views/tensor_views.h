#ifndef TENSOR_VIEWS_H
#define TENSOR_VIEWS_H


#include "tensor/Tensor.h"
#include "tensor/ranges.h"

namespace Fastor {



// Generic tensor views based on sequences/slices
//----------------------------------------------------------------------------------------------//
template<typename T, size_t DIMS, size_t ... Rest>
struct TensorViewExpr<Tensor<T,Rest...>,DIMS>: public AbstractTensor<TensorViewExpr<Tensor<T,Rest...>,DIMS>,DIMS> {
private:
    Tensor<T,Rest...> &expr;
    std::array<seq,sizeof...(Rest)> _seqs;
    bool does_alias = false;

    constexpr FASTOR_INLINE Tensor<T,Rest...> get_tensor() const {return expr;};
    constexpr FASTOR_INLINE std::array<seq,sizeof...(Rest)> get_sequences() const {return _seqs;}

public: 
    using scalar_type = T;
    static constexpr FASTOR_INDEX Dimension = DIMS;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
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

        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>, 
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= dimension(n);
                    as[n] = ( (i+j) / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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

        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>, 
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= dimension(n);
                    as[n] = ( (i+j) / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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

        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>, 
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= dimension(n);
                    as[n] = ( (i+j) / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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

        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>, 
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= dimension(n);
                    as[n] = ( (i+j) / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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

        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>, 
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= dimension(n);
                    as[n] = ( (i+j) / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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
    template<typename Derived>
    void operator=(const AbstractTensor<Derived,DIMS> &other) {
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
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other_src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();

        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>, 
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= dimension(n);
                    as[n] = ( (i+j) / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] = other_src.template eval_s<T>(i);
        }
#endif
    }

    template<typename Derived>
    void operator+=(const AbstractTensor<Derived,DIMS> &other) {
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
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other_src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();

        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>, 
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= dimension(n);
                    as[n] = ( (i+j) / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] += other_src.template eval_s<T>(i);
        }
#endif
    }

    template<typename Derived>
    void operator-=(const AbstractTensor<Derived,DIMS> &other) {
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
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other_src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();

        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>, 
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= dimension(n);
                    as[n] = ( (i+j) / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] -= other_src.template eval_s<T>(i);
        }
#endif
    }

    template<typename Derived>
    void operator*=(const AbstractTensor<Derived,DIMS> &other) {
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
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other_src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();

        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>, 
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= dimension(n);
                    as[n] = ( (i+j) / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
            }
            int ind = 0;
            for(int it = 0; it< DIMS; it++) {
                ind += products_[it]*as[it]*_seqs[it]._step + _seqs[it]._first*products_[it];
            }

            _data[ind] *= other_src.template eval_s<T>(i);
        }
#endif
    }

    template<typename Derived>
    void operator/=(const AbstractTensor<Derived,DIMS> &other) {
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
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other_src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();

        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>, 
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i=0;
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            auto _vec_other = other_src.template eval<T>(i);
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= dimension(n);
                    as[n] = ( (i+j) / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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

        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>, 
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i;
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= dimension(n);
                    as[n] = ( (i+j) / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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

        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>, 
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i;
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= dimension(n);
                    as[n] = ( (i+j) / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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

        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>, 
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i;
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= dimension(n);
                    as[n] = ( (i+j) / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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

        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>, 
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i;
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= dimension(n);
                    as[n] = ( (i+j) / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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

        std::array<size_t,DIMS> products_ = nprods<Index<Rest...>, 
            typename std_ext::make_index_sequence<DIMS>::type>::values;
        products_[DIMS-1]=1;

        std::array<int,DIMS> as;
        std::fill(as.begin(),as.end(),0);
        int total = size();

#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        int i;
        T inum = T(1.)/num;
        SIMDVector<T,DEFAULT_ABI> _vec_other(inum);
        for (i = 0; i <ROUND_DOWN(total,Stride); i+=Stride) {
            for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
                int remaining = total;
                for (int n = 0; n < DIMS; ++n) {
                    remaining /= dimension(n);
                    as[n] = ( (i+j) / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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
                remaining /= dimension(n);
                as[n] = ( i / remaining ) % dimension(n);
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

















//----------------------------------------------------------------------------------------------//















// 1D Views
template<typename T, size_t N>
struct TensorViewExpr<Tensor<T,N>,1>: public AbstractTensor<TensorViewExpr<Tensor<T,N>,1>,1> {
private:
    Tensor<T,N> &expr;
    seq _seq;
    bool does_alias = false;

    constexpr FASTOR_INLINE Tensor<T,N> get_tensor() const {return expr;}
    constexpr FASTOR_INLINE seq get_sequence() const {return _seq;}
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
#ifdef FASTOR_DISALLOW_ALIASING
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
#ifdef FASTOR_DISALLOW_ALIASING
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
#ifdef FASTOR_DISALLOW_ALIASING
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
#ifdef FASTOR_DISALLOW_ALIASING
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
#ifdef FASTOR_DISALLOW_ALIASING
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
    template<typename Derived>
    void operator=(const AbstractTensor<Derived,1> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
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

    template<typename Derived>
    void operator+=(const AbstractTensor<Derived,1> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
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

    template<typename Derived>
    void operator-=(const AbstractTensor<Derived,1> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
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

    template<typename Derived>
    void operator*=(const AbstractTensor<Derived,1> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
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

    template<typename Derived>
    void operator/=(const AbstractTensor<Derived,1> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
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
};










//-----------------------------------------------------------------------------------------------------------//













// 2D Views
template<typename T, size_t M, size_t N>
struct TensorViewExpr<Tensor<T,M,N>,2>: public AbstractTensor<TensorViewExpr<Tensor<T,M,N>,2>,2> {
private:
    Tensor<T,M,N>& expr;
    seq _seq0;
    seq _seq1;
    bool does_alias = false;

    constexpr FASTOR_INLINE Tensor<T,M,N> get_tensor() const {return expr;};
    // constexpr FASTOR_INLINE std::array<seq,sizeof...(Rest)> get_sequences() {return _seqs;}

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
    }

    // View evalution operators
    // Copy assignment operators
    //----------------------------------------------------------------------------------//
    void operator=(const TensorViewExpr<Tensor<T,M,N>,2> &other_src) {
#ifdef FASTOR_DISALLOW_ALIASING
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
#ifdef FASTOR_DISALLOW_ALIASING
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
#ifdef FASTOR_DISALLOW_ALIASING
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
#ifdef FASTOR_DISALLOW_ALIASING
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
#ifdef FASTOR_DISALLOW_ALIASING
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


    // AbstractTensor binders
    //----------------------------------------------------------------------------------//
    template<typename Derived>
    void operator=(const AbstractTensor<Derived,2> &other) {
#ifdef FASTOR_DISALLOW_ALIASING
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
        T *__restrict__ _data = expr.data();
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
#ifdef FASTOR_DISALLOW_ALIASING
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
#ifdef FASTOR_DISALLOW_ALIASING
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
#ifdef FASTOR_DISALLOW_ALIASING
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
#ifdef FASTOR_DISALLOW_ALIASING
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
                data_setter(_data,_vec_other,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
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
                data_setter(_data,_vec_other,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
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
                data_setter(_data,_vec_other,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
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
                data_setter(_data,_vec_other,(_seq0._step*i+_seq0._first)*N+_seq1._step*j+_seq1._first,_seq1._step);
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
            // auto it = (idx+j) / _seq0.size(), jt = (idx+j) % _seq0.size();
            auto it = (idx+j) / _seq1.size(), jt = (idx+j) % _seq1.size();
            inds[j] = _seq0._step*it*N+_seq1._step*jt + _seq0._first*N + _seq1._first;
        }
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        SIMDVector<U,DEFAULT_ABI> _vec; 
        if (_seq1._step==1) _vec.load(expr.data()+_seq0._step*i*N+_seq1._step*j + _seq0._first*N + _seq1._first,false);       
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
    
};

}


#endif // TENSOR_VIEWS_H