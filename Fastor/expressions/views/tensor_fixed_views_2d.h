#ifndef TENSOR_FIXED_VIEWS_2D_H
#define TENSOR_FIXED_VIEWS_2D_H


#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/Ranges.h"

namespace Fastor {


// 2D const fixed views
//----------------------------------------------------------------------------------------------//
template<typename T, size_t M, size_t N, int F0, int L0, int S0, int F1, int L1, int S1>
struct TensorConstFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2> :
    public AbstractTensor<TensorConstFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2>,2> {
private:
    const Tensor<T,M,N> &expr;
public:
    using scalar_type = T;
    static constexpr FASTOR_INDEX Dimension = 2;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {
        return range_detector<F0,L0,S0>::value*range_detector<F1,L1,S1>::value;
    }
    static constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) {
        return i==0 ? range_detector<F0,L0,S0>::value : range_detector<F1,L1,S1>::value;
    }
    static constexpr FASTOR_INDEX Padding = F0*N+F1;

    constexpr FASTOR_INLINE TensorConstFixedViewExpr2D(const Tensor<T,M,N> &_ex) : expr(_ex) {}

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX idx) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,SIMDVector<U,DEFAULT_ABI>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<U,DEFAULT_ABI>::Size; ++j) {
            auto it = (idx+j) / range_detector<F1,L1,S1>::value, jt = (idx+j) % range_detector<F1,L1,S1>::value;
            inds[j] = S0*it*N+S1*jt + Padding;
        }
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx) const {
        auto it = idx / range_detector<F1,L1,S1>::value, jt = idx % range_detector<F1,L1,S1>::value;
        auto ind = S0*it*N+S1*jt + Padding;
        return expr.data()[ind];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        vector_setter(_vec,expr.data(),S0*i*N+S1*j + Padding,S1);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return expr(S0*i+F0,S1*j+F1);
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> teval(const std::array<int,2>& as) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        vector_setter(_vec,expr.data(),S0*as[0]*N+S1*as[1] + Padding,S1);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,2>& as) const {
        return expr(S0*as[0]+F0,S1*as[1]+F1);
    }
};





// 2D non-const fixed views
//----------------------------------------------------------------------------------------------//
template<typename T, size_t M, size_t N, int F0, int L0, int S0, int F1, int L1, int S1>
struct TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2> :
    public AbstractTensor<TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2>,2> {

private:
    Tensor<T,M,N> &expr;
public:
    using scalar_type = T;
    static constexpr FASTOR_INDEX Dimension = 2;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {
        return range_detector<F0,L0,S0>::value*range_detector<F1,L1,S1>::value;
    }
    static constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) {
        return i==0 ? range_detector<F0,L0,S0>::value : range_detector<F1,L1,S1>::value;
    }
    static constexpr FASTOR_INDEX Padding = F0*N+F1;

    FASTOR_INLINE TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2>& nolias() {
        FASTOR_ASSERT(false,"FIXED 2D VIEWS DO NOT SUPPORT OVERLAPPING ASSIGNMENTS");
    }

    constexpr FASTOR_INLINE TensorFixedViewExpr2D(Tensor<T,M,N> &_ex) : expr(_ex) {}

    //----------------------------------------------------------------------------------//
    void operator=(const TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2> &other_src) {
#ifndef NDEBUG
        FASTOR_ASSERT(other_src.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other_src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *FASTOR_RESTRICT _data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        // FASTOR_INDEX i;
        // for (i = 0; i <ROUND_DOWN(size(),Stride); i+=Stride) {
        //     auto _vec_other = other_src.template eval<T>(i);
        //     for (auto j=0; j<SIMDVector<T,DEFAULT_ABI>::Size; ++j) {
        //         auto it = (i+j) / range_detector<F1,L1,S1>::value, jt = (i+j) % range_detector<F1,L1,S1>::value;
        //         auto idx = S0*it*N+S1*jt + Padding;
        //         _data[idx] = _vec_other[j];
        //     }
        // }
        // for (; i <size(); i++) {
        //     auto it = i / range_detector<F1,L1,S1>::value, jt = i % range_detector<F1,L1,S1>::value;
        //     auto idx = S0*it*N+S1*jt + Padding;
        //     _data[idx] = other_src.template eval_s<T>(i);
        // }
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                auto _vec = other_src.template eval<T>(i,j);
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
            }
            for (; j <dimension(1); ++j) {
                expr(S0*i+F0,S1*j+F1) = other_src.template eval_s<T>(i,j);
            }
        }
#else
        // for (FASTOR_INDEX i = 0; i <size(); i++) {
        //     auto it = i / range_detector<F1,L1,S1>::value, jt = i % range_detector<F1,L1,S1>::value;
        //     auto idx = S0*it*N+S1*jt + Padding;
        //     _data[idx] = other_src.template eval_s<T>(i);
        // }
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) = other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }

    void operator+=(const TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2> &other_src) {
#ifndef NDEBUG
        FASTOR_ASSERT(other_src.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other_src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        for (FASTOR_INDEX i = 0; i < dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j < ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(i,j) + other_src.template eval<T>(i,j);
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
            }
            for (; j <dimension(1); ++j) {
                expr(S0*i+F0,S1*j+F1) += other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) += other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }

    void operator-=(const TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2> &other_src) {
#ifndef NDEBUG
        FASTOR_ASSERT(other_src.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other_src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(i,j) - other_src.template eval<T>(i,j);
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
            }
            for (; j <dimension(1); ++j) {
                expr(S0*i+F0,S1*j+F1) -= other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) -= other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }

    void operator*=(const TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2> &other_src) {
#ifndef NDEBUG
        FASTOR_ASSERT(other_src.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other_src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(i,j) * other_src.template eval<T>(i,j);
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
            }
            for (; j <dimension(1); ++j) {
                expr(S0*i+F0,S1*j+F1) *= other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) *= other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }

    void operator/=(const TensorFixedViewExpr2D<Tensor<T,M,N>,fseq<F0,L0,S0>,fseq<F1,L1,S1>,2> &other_src) {
#ifndef NDEBUG
        FASTOR_ASSERT(other_src.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other_src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(i,j) / other_src.template eval<T>(i,j);
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
            }
            for (; j <dimension(1); ++j) {
                expr(S0*i+F0,S1*j+F1) /= other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) /= other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }

    // AbstractTensor binders - this is a special case for assigning another 2D expressions
    //----------------------------------------------------------------------------------//
    template<typename Derived>
    void operator=(const AbstractTensor<Derived,2> &other) {
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
        constexpr FASTOR_INDEX UNROLL_UPTO = ROUND_DOWN(dimension(1),Stride);
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <UNROLL_UPTO; j+=Stride) {
                auto _vec = other_src.template eval<T>(i,j);
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
            }
            for (; j <range_detector<F1,L1,S1>::value; ++j) {
                expr(S0*i+F0,S1*j+F1) = other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) = other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }

    template<typename Derived>
    void operator+=(const AbstractTensor<Derived,2> &other) {
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
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(i,j) + other_src.template eval<T>(i,j);
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
            }
            for (; j <dimension(1); ++j) {
                expr(S0*i+F0,S1*j+F1) += other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) += other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }

    template<typename Derived>
    void operator-=(const AbstractTensor<Derived,2> &other) {
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
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(i,j) - other_src.template eval<T>(i,j);
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
            }
            for (; j <dimension(1); ++j) {
                expr(S0*i+F0,S1*j+F1) -= other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) -= other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }

    template<typename Derived>
    void operator*=(const AbstractTensor<Derived,2> &other) {
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
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(i,j) * other_src.template eval<T>(i,j);
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
            }
            for (; j <dimension(1); ++j) {
                expr(S0*i+F0,S1*j+F1) *= other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) *= other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }

    template<typename Derived>
    void operator/=(const AbstractTensor<Derived,2> &other) {
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
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(i,j) / other_src.template eval<T>(i,j);
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
            }
            for (; j <dimension(1); ++j) {
                expr(S0*i+F0,S1*j+F1) /= other_src.template eval_s<T>(i,j);
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) /= other_src.template eval_s<T>(i,j);
            }
        }
#endif
    }
    //----------------------------------------------------------------------------------//


    // AbstractTensor binders for other nth rank tensors
    //----------------------------------------------------------------------------------//
    template<typename Derived, size_t DIMS>
    void operator=(const AbstractTensor<Derived,DIMS> &other) {
        const Derived& other_src = other.self();
#ifndef NDEBUG
        FASTOR_ASSERT(other_src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
        T *FASTOR_RESTRICT _data = expr.data();
        FASTOR_INDEX counter = 0;
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        constexpr FASTOR_INDEX UNROLL_UPTO = ROUND_DOWN(dimension(1),Stride);
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <UNROLL_UPTO; j+=Stride) {
                auto _vec = other_src.template eval<T>(counter);
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                counter += Stride;
            }
            for (; j <range_detector<F1,L1,S1>::value; ++j) {
                expr(S0*i+F0,S1*j+F1) = other_src.template eval_s<T>(counter);
                counter++;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) = other_src.template eval_s<T>(counter);
                counter++;
            }
        }
#endif
    }

    template<typename Derived, size_t DIMS>
    void operator+=(const AbstractTensor<Derived,DIMS> &other) {
        const Derived& other_src = other.self();
#ifndef NDEBUG
        FASTOR_ASSERT(other_src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
        T *FASTOR_RESTRICT _data = expr.data();
        FASTOR_INDEX counter = 0;
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(counter) + other_src.template eval<T>(counter);
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                counter += Stride;
            }
            for (; j <dimension(1); ++j) {
                expr(S0*i+F0,S1*j+F1) += other_src.template eval_s<T>(counter);
                counter++;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) += other_src.template eval_s<T>(counter);
                counter++;
            }
        }
#endif
    }

    template<typename Derived, size_t DIMS>
    void operator-=(const AbstractTensor<Derived,DIMS> &other) {
        const Derived& other_src = other.self();
#ifndef NDEBUG
        FASTOR_ASSERT(other_src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
        T *FASTOR_RESTRICT _data = expr.data();
        FASTOR_INDEX counter = 0;
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(counter) - other_src.template eval<T>(counter);
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                counter += Stride;
            }
            for (; j <dimension(1); ++j) {
                expr(S0*i+F0,S1*j+F1) -= other_src.template eval_s<T>(counter);
                counter++;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) -= other_src.template eval_s<T>(counter);
                counter++;
            }
        }
#endif
    }

    template<typename Derived, size_t DIMS>
    void operator*=(const AbstractTensor<Derived,DIMS> &other) {
        const Derived& other_src = other.self();
#ifndef NDEBUG
        FASTOR_ASSERT(other_src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
        T *FASTOR_RESTRICT _data = expr.data();
        FASTOR_INDEX counter = 0;
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(counter) * other_src.template eval<T>(counter);
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                counter += Stride;
            }
            for (; j <dimension(1); ++j) {
                expr(S0*i+F0,S1*j+F1) *= other_src.template eval_s<T>(counter);
                counter++;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) *= other_src.template eval_s<T>(counter);
                counter++;
            }
        }
#endif
    }

    template<typename Derived, size_t DIMS>
    void operator/=(const AbstractTensor<Derived,DIMS> &other) {
        const Derived& other_src = other.self();
#ifndef NDEBUG
        FASTOR_ASSERT(other_src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
        T *FASTOR_RESTRICT _data = expr.data();
        FASTOR_INDEX counter = 0;
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(counter) / other_src.template eval<T>(counter);
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
                counter += Stride;
            }
            for (; j <dimension(1); ++j) {
                expr(S0*i+F0,S1*j+F1) /= other_src.template eval_s<T>(counter);
                counter++;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) /= other_src.template eval_s<T>(counter);
                counter++;
            }
        }
#endif
    }
    //----------------------------------------------------------------------------------//


    // Scalar binders
    //----------------------------------------------------------------------------------//
    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator=(U num) {
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                data_setter(_data,_vec_other,S0*i*N+S1*j+Padding,S1);
            }
            for (; j <dimension(1); ++j) {
                expr(S0*i+F0,S1*j+F1) = num;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) = num;
            }
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator+=(U num) {
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(i,j) + _vec_other;
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
            }
            for (; j <dimension(1); ++j) {
                expr(S0*i+F0,S1*j+F1) += num;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) += num;
            }
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator-=(U num) {
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(i,j) - _vec_other;
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
            }
            for (; j <dimension(1); ++j) {
                expr(S0*i+F0,S1*j+F1) -= num;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) -= num;
            }
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator*=(U num) {
        T *_data = expr.data();
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(num));
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(i,j) * _vec_other;
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
            }
            for (; j <dimension(1); ++j) {
                expr(S0*i+F0,S1*j+F1) *= num;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) *= num;
            }
        }
#endif
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    void operator/=(U num) {
        T *_data = expr.data();
        T inum = T(1)/num;
#ifdef FASTOR_USE_VECTORISED_EXPR_ASSIGN
        SIMDVector<T,DEFAULT_ABI> _vec_other(static_cast<T>(inum));
        for (FASTOR_INDEX i = 0; i <dimension(0); i++) {
            FASTOR_INDEX j;
            for (j = 0; j <ROUND_DOWN(dimension(1),Stride); j+=Stride) {
                auto _vec = this->template eval<T>(i,j) * _vec_other;
                data_setter(_data,_vec,S0*i*N+S1*j+Padding,S1);
            }
            for (; j <dimension(1); ++j) {
                expr(S0*i+F0,S1*j+F1) *= inum;
            }
        }
#else
        for (FASTOR_INDEX i = 0; i <range_detector<F0,L0,S0>::value; i++) {
            for (FASTOR_INDEX j = 0; j <range_detector<F1,L1,S1>::value; j++) {
                expr(S0*i+F0,S1*j+F1) *= inum;
            }
        }
#endif
    }
    //----------------------------------------------------------------------------------//

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX idx) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,SIMDVector<U,DEFAULT_ABI>::Size> inds;
        for (FASTOR_INDEX j=0; j<SIMDVector<U,DEFAULT_ABI>::Size; ++j) {
            auto it = (idx+j) / range_detector<F1,L1,S1>::value, jt = (idx+j) % range_detector<F1,L1,S1>::value;
            inds[j] = S0*it*N+S1*jt + Padding;
        }
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }

    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx) const {
        auto it = idx / range_detector<F1,L1,S1>::value, jt = idx % range_detector<F1,L1,S1>::value;
        auto ind = S0*it*N+S1*jt + Padding;
        return expr.data()[ind];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        // if (S1==1) _vec.load(expr.data()+S0*i*N+S1*j + Padding, false);
        FASTOR_IF_CONSTEXPR (S1==1) _vec.load(expr.data()+S0*i*N+j + Padding, false);
        else vector_setter(_vec,expr.data(),S0*i*N+S1*j + Padding,S1);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        // return expr(S0*i+F0,S1*j+F1);
        return expr.data()[S0*i*N+S1*j + Padding];
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> teval(const std::array<int,2>& as) const {
        SIMDVector<U,DEFAULT_ABI> _vec;
        FASTOR_IF_CONSTEXPR (S1==1) _vec.load(expr.data()+S0*as[0]*N+as[1] + Padding, false);
        else vector_setter(_vec,expr.data(),S0*as[0]*N+S1*as[1] + Padding,S1);
        return _vec;
    }

    template<typename U=T>
    constexpr FASTOR_INLINE U teval_s(const std::array<int,2>& as) const {
        return expr.data()[S0*as[0]*N+S1*as[1] + Padding];
    }
};


}


#endif // TENSOR_FIXED_VIEWS_2D_H