#ifndef ABSTRACT_TENSOR_FUNCTIONS_H
#define ABSTRACT_TENSOR_FUNCTIONS_H

#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/TensorTraits.h"

namespace Fastor {

/* The implementation of the evaluate function that evaluates any expression in to a tensor
*/
//----------------------------------------------------------------------------------------------------------//
template<typename T, size_t ... Rest>
FASTOR_INLINE const Tensor<T,Rest...>& evaluate(const Tensor<T,Rest...> &src) {
    return src;
}
template<class Derived, size_t DIMS>
FASTOR_INLINE typename Derived::result_type evaluate(const AbstractTensor<Derived,DIMS> &src) {
    typename Derived::result_type out(src);
    return out;
}
//----------------------------------------------------------------------------------------------------------//


/* These are the set of functions that work on any expression that evaluate immediately
*/

/* Add all the elements of the tensor in a flattened sense
*/
template<class Derived, size_t DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type sum(const AbstractTensor<Derived,DIMS> &_src) {
    const Derived &src = _src.self();
    using result_type = typename Derived::result_type;
    const result_type out(src);
    return out.sum();
}
template<class Derived, size_t DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type sum(const AbstractTensor<Derived,DIMS> &_src) {

    const Derived &src = _src.self();
    using T = typename Derived::scalar_type;
    using V = typename Derived::simd_vector_type;
    FASTOR_INDEX i;
    T _scal=0; V _vec(_scal);
    for (i = 0; i < ROUND_DOWN(src.size(),V::Size); i+=V::Size) {
        _vec += src.template eval<T>(i);
    }
    for (; i < src.size(); ++i) {
        _scal += src.template eval_s<T>(i);
    }
    return _vec.sum() + _scal;
}

/* Multiply all the elements of the tensor in a flattened sense
*/
template<class Derived, size_t DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type product(const AbstractTensor<Derived,DIMS> &_src) {
    const Derived &src = _src.self();
    using result_type = typename Derived::result_type;
    const result_type out(src);
    return out.product();
}
template<class Derived, size_t DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type product(const AbstractTensor<Derived,DIMS> &_src) {

    const Derived &src = _src.self();
    using T = typename Derived::scalar_type;
    using V = typename Derived::simd_vector_type;
    FASTOR_INDEX i;
    T _scal=1; V _vec(_scal);
    for (i = 0; i < ROUND_DOWN(src.size(),V::Size); i+=V::Size) {
        _vec *= src.template eval<T>(i);
    }
    for (; i < src.size(); ++i) {
        _scal *= src.template eval_s<T>(i);
    }
    return _vec.product() * _scal;
}

/* Get the lower triangular matrix from a 2D expression
*/
template<class Derived, size_t DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type tril(const AbstractTensor<Derived,DIMS> &_src, int k = 0) {
    static_assert(DIMS==2,"TENSOR HAS TO BE 2D FOR TRIL");
    const Derived &src = _src.self();
    using result_type = typename Derived::result_type;
    const result_type out(src);
    return tril(out);
}
template<class Derived, size_t DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::result_type tril(const AbstractTensor<Derived,DIMS> &_src, int k = 0) {
    static_assert(DIMS==2,"TENSOR HAS TO BE 2D FOR TRIL");
    const Derived &src = _src.self();
    using T = typename Derived::scalar_type;
    typename Derived::result_type out(0);

    int M = int(src.dimension(0));
    int N = int(src.dimension(1));
    for (int i = 0; i < M; ++i) {
        int jcount =  k + i < N ? k + i : N - 1;
        for (int j = 0; j <= jcount; ++j) {
            out(i,j) = src.template eval_s<T>(i,j);
        }
    }
    return out;
}

/* Get the upper triangular matrix from a 2D expression
*/
template<class Derived, size_t DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type triu(const AbstractTensor<Derived,DIMS> &_src, int k = 0) {
    static_assert(DIMS==2,"TENSOR HAS TO BE 2D FOR TRIU");
    const Derived &src = _src.self();
    using result_type = typename Derived::result_type;
    const result_type out(src);
    return triu(out);
}
template<class Derived, size_t DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::result_type triu(const AbstractTensor<Derived,DIMS> &_src, int k = 0) {
    static_assert(DIMS==2,"TENSOR HAS TO BE 2D FOR TRIU");
    const Derived &src = _src.self();
    using T = typename Derived::scalar_type;
    typename Derived::result_type out(0);

    int M = int(src.dimension(0));
    int N = int(src.dimension(1));
    for (int i = 0; i < M; ++i) {
        int jcount =  k + i < 0 ? 0 : k + i;
        for (int j = jcount; j < N; ++j) {
            out(i,j) = src.template eval_s<T>(i,j);
        }
    }
    return out;
}
//----------------------------------------------------------------------------------------------------------//


// Boolean functions
//----------------------------------------------------------------------------------------------------------//
//----------------------------------------------------------------------------------------------------------//
template<class Derived, size_t DIMS, enable_if_t_<is_binary_cmp_op_v<Derived>,bool> = false>
FASTOR_INLINE bool all_of(const AbstractTensor<Derived,DIMS> &_src) {
    const Derived &src = _src.self();
    bool val = true;
    for (FASTOR_INDEX i = 0; i < src.size(); ++i) {
        if (src.template eval_s<bool>(i) == false) {
            val = false;
            break;
        }
    }
    return val;
}

template<class Derived, size_t DIMS, enable_if_t_<is_binary_cmp_op_v<Derived>,bool> = false>
FASTOR_INLINE bool any_of(const AbstractTensor<Derived,DIMS> &_src) {
    const Derived &src = _src.self();
    bool val = false;
    for (FASTOR_INDEX i = 0; i < src.size(); ++i) {
        if (src.template eval_s<bool>(i) == true) {
            val = true;
            break;
        }
    }
    return val;
}

template<class Derived, size_t DIMS, enable_if_t_<is_binary_cmp_op_v<Derived>,bool> = false>
FASTOR_INLINE bool none_of(const AbstractTensor<Derived,DIMS> &_src) {
    const Derived &src = _src.self();
    bool val = false;
    for (FASTOR_INDEX i = 0; i < src.size(); ++i) {
        if (src.template eval_s<bool>(i) == true) {
            val = true;
            break;
        }
    }
    return val;
}


/* Is a second order tensor expression a uniform
   A tensor expression is uniform if it spans equally in all dimensions,
   i.e. generalisation of square matrix to N-dimensions
*/
template<class Derived, size_t DIMS>
constexpr FASTOR_INLINE bool isuniform(const AbstractTensor<Derived,DIMS> &_src) {
    return is_tensor_uniform_v<typename Derived::result_type>;
}

/* Is a second order tensor expression a square matrix
*/
template<class Derived, size_t DIMS, enable_if_t_<DIMS==2,bool> = false>
constexpr FASTOR_INLINE bool issquare(const AbstractTensor<Derived,DIMS> &_src) {
    return is_tensor_uniform_v<typename Derived::result_type>;
}

/* Is a second order tensor expression orthogonal
*/
template<class Derived, size_t DIMS, enable_if_t_<DIMS==2,bool> = false>
FASTOR_INLINE bool isorthogonal(const AbstractTensor<Derived,DIMS> &_src, const double Tol=PRECI_TOL) {
    typename Derived::result_type tmp1(_src.self());
    typename Derived::result_type tmp2(matmul(transpose(tmp1),tmp1));
    typename Derived::result_type I; I.eye2();
    return isequal(tmp2,I,Tol);
}

/* Is a tensor expression symmetric - for higher order tensor two axes can defining a plane
    provided to determine if a tensor expression is symmertric in that plane
*/
template<size_t axis0 = 0, size_t axis1 = 1, class Derived, size_t DIMS,
    enable_if_t_<DIMS==2 && requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE bool issymmetric(const AbstractTensor<Derived,DIMS> &_src, const double Tol=PRECI_TOL) {
    if (!issquare(_src.self())) return false;
    return all_of( abs(evaluate(trans(_src.self()) - _src.self())) < Tol);
}
template<size_t axis0 = 0, size_t axis1 = 1, class Derived, size_t DIMS,
    enable_if_t_<DIMS==2 && !requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE bool issymmetric(const AbstractTensor<Derived,DIMS> &_src, const double Tol=PRECI_TOL) {
    if (!issquare(_src.self())) return false;
    // Avoid copies
    const Derived& src = _src.self();
    using T = typename Derived::scalar_type;
    using result_type = typename Derived::result_type;
    constexpr size_t M = get_tensor_dimension_v<0,result_type>;
    constexpr size_t N = get_tensor_dimension_v<1,result_type>;

    bool _issym = true;
    for (size_t i=0; i<M; ++i) {
        for (size_t j=0; j<N; ++j) {
            if (std::abs(  src.template eval_s<T>(i*N+j) - src.template eval_s<T>(j*N+i)  ) > Tol ) {
                _issym = false;
                break;
            }
        }
    }
    return _issym;
}

/* Is a second order tensor expression deviatoric - a 2D tensor expression is deviatoric if it is
    trace free
*/
template<class Derived, size_t DIMS, enable_if_t_<DIMS==2,bool> = false>
constexpr FASTOR_INLINE bool isdeviatoric(const AbstractTensor<Derived,DIMS> &_src, const double Tol=PRECI_TOL) {
    return std::abs( trace(_src.self()) ) < Tol ? true : false;
}

/* Is a second order tensor expression volumetric - a 2D tensor expression is volumetric if 1/3*[A:I]*I = A
*/
template<class Derived, size_t DIMS, enable_if_t_<DIMS==2,bool> = false>
constexpr FASTOR_INLINE bool isvolumetric(const AbstractTensor<Derived,DIMS> &_src, const double Tol=PRECI_TOL) {
    typename Derived::result_type I; I.eye2();
    typename Derived::result_type tmp = trace(_src.self()) * I;
    return all_of( abs(tmp - _src.self()) < Tol);
}

/* A second order tensor expression belongs to the special linear group if
it's determinant is +1
*/
template<class Derived, size_t DIMS, enable_if_t_<DIMS==2,bool> = false>
FASTOR_INLINE bool doesbelongtoSL3(const AbstractTensor<Derived,DIMS> &_src, const double Tol=PRECI_TOL) {
    // Expression must be square
    if ( !issquare(_src.self()) ) return false;
    return std::abs(determinant(_src.self()) - 1) < Tol ? true : false;
}

/* A second order tensor/expression belongs to the special orthogonal group if
it is orthogonal and it's determinant is +1
*/
template<class Derived, size_t DIMS, enable_if_t_<DIMS==2,bool> = false>
FASTOR_INLINE bool doesbelongtoSO3(const AbstractTensor<Derived,DIMS> &_src, const double Tol=PRECI_TOL) {
    // Expression must be square and orthogonal
    return issquare(_src.self()) && isorthogonal(_src.self()) ? true : false;
}

/* Are two tensor expressions approximately equal
*/
template<class Derived0, size_t DIMS0, class Derived1, size_t DIMS1,
    enable_if_t_<!requires_evaluation_v<Derived0> && !requires_evaluation_v<Derived1>,bool> = false>
FASTOR_INLINE bool isequal(
        const AbstractTensor<Derived0,DIMS0> &_src0,
        const AbstractTensor<Derived1,DIMS1> &_src1,
        const double Tol=PRECI_TOL) {
    if ( DIMS0 != DIMS1) return false;
    if ( _src0.self().size() != _src1.self().size()) return false;
    return all_of( abs(_src0.self() - _src1.self()) < Tol);
}
template<class Derived0, size_t DIMS0, class Derived1, size_t DIMS1,
    enable_if_t_<requires_evaluation_v<Derived0> || requires_evaluation_v<Derived1>,bool> = false>
FASTOR_INLINE bool isequal(
        const AbstractTensor<Derived0,DIMS0> &_src0,
        const AbstractTensor<Derived1,DIMS1> &_src1,
        const double Tol=PRECI_TOL) {
    if ( DIMS0 != DIMS1) return false;
    if ( _src0.self().size() != _src1.self().size()) return false;
    return all_of( abs(evaluate(_src0.self() - _src1.self())) < Tol);
}
//----------------------------------------------------------------------------------------------------------//
//----------------------------------------------------------------------------------------------------------//

} // end of namespace Fastor


#endif // #ifndef TENSOR_FUNCTIONS_H
