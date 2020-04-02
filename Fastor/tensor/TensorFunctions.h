#ifndef TENSOR_FUNCTIONS_H
#define TENSOR_FUNCTIONS_H

#include "Fastor/tensor/Tensor.h"
#include "Fastor/meta/tensor_post_meta.h"

namespace Fastor {

// BLAS/LAPACK/Tensor cross routines
template<typename T, size_t I, size_t J>
FASTOR_INLINE Tensor<T,J,I> transpose(const Tensor<T,I,J> &a) {
    Tensor<T,J,I> out;
    _transpose<T,I,J>(a.data(),out.data());
    return out;
}

template<typename T, size_t I>
FASTOR_INLINE T trace(const Tensor<T,I,I> &a) {
    return _trace<T,I,I>(static_cast<const T *>(a.data()));
}

template<typename T, size_t I>
FASTOR_INLINE T determinant(const Tensor<T,I,I> &a) {
    return _det<T,I,I>(static_cast<const T *>(a.data()));
}

template<typename T, size_t I>
FASTOR_INLINE Tensor<T,I,I> cofactor(const Tensor<T,I,I> &a) {
    Tensor<T,I,I> out;
    _cofactor<T,I,I>(a.data(),out.data());
    return out;
}

template<typename T, size_t I>
FASTOR_INLINE Tensor<T,I,I> adjoint(const Tensor<T,I,I> &a) {
    Tensor<T,I,I> out;
    _adjoint<T,I,I>(a.data(),out.data());
    return out;
}

template<typename T, size_t I>
FASTOR_INLINE Tensor<T,I,I> inverse(const Tensor<T,I,I> &a) {
    Tensor<T,I,I> out;
    _inverse<T,I>(a.data(),out.data());
    return out;
}

template<typename T, size_t ... Rest,
         typename std::enable_if<std::is_floating_point<T>::value,bool>::type=0>
FASTOR_INLINE T norm(const Tensor<T,Rest...> &a) {
    if (sizeof...(Rest) == 0)
        return *a.data();
    return _norm<T,prod<Rest...>::value>(a.data());
}

template<typename T, size_t ... Rest,
         typename std::enable_if<!std::is_floating_point<T>::value,bool>::type=0>
FASTOR_INLINE double norm(const Tensor<T,Rest...> &a) {
    if (sizeof...(Rest) == 0)
        return *a.data();
    return _norm_nonfloating<T,prod<Rest...>::value>(a.data());
}

// matmul - matvec overloads
template<typename T, size_t I, size_t J, size_t K>
FASTOR_INLINE Tensor<T,I,K> matmul(const Tensor<T,I,J> &a, const Tensor<T,J,K> &b) {
    Tensor<T,I,K> out;
    _matmul<T,I,J,K>(a.data(),b.data(),out.data());
    return out;
}

template<typename T, size_t I, size_t J>
FASTOR_INLINE Tensor<T,I> matmul(const Tensor<T,I,J> &a, const Tensor<T,J> &b) {
// Hack clang to get around alignment
#if defined(__llvm__) || defined(__clang__)
    unused(a);
#endif
    Tensor<T,I> out;
    _matmul<T,I,J,1>(a.data(),b.data(),out.data());
    return out;
}

template<typename T, size_t J, size_t K>
FASTOR_INLINE Tensor<T,K> matmul(const Tensor<T,J> &a, const Tensor<T,J,K> &b) {
    Tensor<T,K> out;
    _matmul<T,1,J,K>(a.data(),b.data(),out.data());
    return out;
}

template<typename T, size_t K>
FASTOR_INLINE Tensor<T,1,1> matmul(const Tensor<T,1,K> &a, const Tensor<T,K,1> &b) {
    Tensor<T,1,1> out(_doublecontract<T,K,1>(a.data(),b.data()));
    return out;
}

template<typename T, size_t I, size_t K>
FASTOR_INLINE Tensor<T,I,K> matmul(const Tensor<T,I,1> &a, const Tensor<T,1,K> &b) {
    Tensor<T,I,K> out;
    _dyadic<T,I,K>(a.data(),b.data(),out.data());
    return out;
}

template<typename T, size_t ... Rest>
FASTOR_INLINE T dot(const Tensor<T,Rest...> &b, const Tensor<T,Rest...> &a) {
    return _doublecontract<T,sizeof...(Rest),1>(a.data(),b.data());
}

template<typename T, size_t I, size_t J>
FASTOR_INLINE Tensor<T,J> solve(const Tensor<T,I,J> &A, const Tensor<T,J> &b) {
    return matmul(inverse(A),b);
}

template<typename T, size_t I, size_t J>
FASTOR_INLINE Tensor<T,J,1> solve(const Tensor<T,I,J> &A, const Tensor<T,J,1> &b) {
    return matmul(inverse(A),b);
}

// Tensor cross product of two 2nd order tensors
template<typename T, size_t I, size_t J, typename std::enable_if<I==3 && J==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J> cross(const Tensor<T,I,J> &b, const Tensor<T,I,J> &a) {
    Tensor<T,I,J> out;
    _crossproduct<T,I,I,J>(a.data(),b.data(),out.data());
    return out;
}

template<typename T, size_t I, size_t J, typename std::enable_if<I==2 && J==2,bool>::type=0>
FASTOR_INLINE Tensor<T,I+1,J+1> cross(const Tensor<T,I,J> &b, const Tensor<T,I,J> &a) {
    Tensor<T,I+1,J+1> out;
    _crossproduct<T,I,I,J>(a.data(),b.data(),out.data());
    return out;
}

template<int Plane, typename T, size_t I, size_t J, typename std::enable_if<I==2 && J==2 && Plane==FASTOR_PlaneStrain,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J> cross(const Tensor<T,I,J> &b, const Tensor<T,I,J> &a) {
    // Plane strain case
    Tensor<T,I,J> out;
    Tensor<T,I+1,J+1> a3, b3;
    a3(0,0) = a(0,0);
    a3(0,1) = a(0,1);
    a3(1,0) = a(1,0);
    a3(1,1) = a(1,1);
    a3(2,2) = 1;
    b3(0,0) = b(0,0);
    b3(0,1) = b(0,1);
    b3(1,0) = b(1,0);
    b3(1,1) = b(1,1);
    b3(2,2) = 1;
    _crossproduct<T,FASTOR_PlaneStrain>(a.data(),b.data(),out.data());
    return out;
}

// Tensor cross product of a vector with 2nd order tensor
template<typename T, size_t I, size_t J, typename std::enable_if<I==3 && J==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J> cross(const Tensor<T,I> &b, const Tensor<T,I,J> &a) {
    Tensor<T,I,J> out;
    _crossproduct<T,I,1,J>(a.data(),b.data(),out.data());
    return out;
}

template<typename T, size_t I, size_t J, typename std::enable_if<I==2 && J==2,bool>::type=0>
FASTOR_INLINE Tensor<T,I+1,J+1> cross(const Tensor<T,I> &b, const Tensor<T,I,J> &a) {
    Tensor<T,I+1,J+1> out;
    _crossproduct<T,I,1,J>(a.data(),b.data(),out.data());
    return out;
}

// Tensor cross product of a 2nd order tensor with a vector
template<typename T, size_t I, size_t J, typename std::enable_if<I==3 && J==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J> cross(const Tensor<T,I,J> &b, const Tensor<T,J> &a) {
    Tensor<T,I,J> out;
    _crossproduct<T,I,J,1>(a.data(),b.data(),out.data());
    return out;
}

template<typename T, size_t I, size_t J, typename std::enable_if<I==2 && J==2,bool>::type=0>
FASTOR_INLINE Tensor<T,I+1,J+1> cross(const Tensor<T,I,J> &b, const Tensor<T,J> &a) {
    Tensor<T,I+1,J+1> out;
    _crossproduct<T,I,J,1>(a.data(),b.data(),out.data());
    return out;
}


// Tensor cross product of a 3rd order tensors
template<typename T, size_t I, size_t J, size_t K, size_t L, size_t M, size_t N,
         typename std::enable_if<I==3 && J==3 && K==3 && L==3 && M==3 && N==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,3,3,N> cross(const Tensor<T,I,J,K> &A, const Tensor<T,L,M,N> &B) {
    Tensor<T,I,3,3,N> C;
    _crossproduct<T,I,J,K,L,M,N>(A.data(),B.data(),C.data());
    return C;
}


// Tensor cross product of a 4th order tensors
template<typename T, size_t I, size_t J, size_t K, size_t L, size_t M, size_t N, size_t O, size_t P,
         typename std::enable_if<I==3 && J==3 && K==3 && L==3 && M==3 && N==3 && O==3 && P==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J,3,3,O,P> cross(const Tensor<T,I,J,K,L> &A, const Tensor<T,M,N,O,P> &B) {
    Tensor<T,I,J,3,3,O,P> C;
    _crossproduct<T,I,J,K,L,M,N,O,P>(A.data(),B.data(),C.data());
    return C;
}


// Tensor cross product of a 4th order tensor with 2nd order tensor
template<typename T, size_t I, size_t J, size_t K, size_t L, size_t M, size_t N,
         typename std::enable_if<I==3 && J==3 && K==3 && L==3 && M==3 && N==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J,K,L> cross(const Tensor<T,I,J,K,L> &A, const Tensor<T,M,N> &B) {
    Tensor<T,I,J,K,L> C;
    _crossproduct42<T,I,J,K,L,M,N>(A.data(),B.data(),C.data());
    return C;
}


// Tensor cross product of a 2nd order tensor with 4th order tensor
template<typename T, size_t I, size_t J, size_t K, size_t L, size_t M, size_t N,
         typename std::enable_if<I==3 && J==3 && K==3 && L==3 && M==3 && N==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J,K,L> cross(const Tensor<T,M,N> &A, const Tensor<T,I,J,K,L> &B) {
    Tensor<T,I,J,K,L> C;
    _crossproduct24<T,I,J,K,L,M,N>(B.data(),A.data(),C.data());
    return C;
}


// Tensor cross product of a 3rd order tensor with 2nd order tensor
template<typename T, size_t I, size_t J, size_t K, size_t L, size_t M,
         typename std::enable_if<I==3 && J==3 && K==3 && L==3 && M==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J,K> cross(const Tensor<T,I,J,K> &A, const Tensor<T,L,M> &B) {
    Tensor<T,I,J,K> C;
    _crossproduct32<T,I,J,K,L,M>(A.data(),B.data(),C.data());
    return C;
}


// Tensor cross product of a 2nd order tensor with 3rd order tensor
template<typename T, size_t I, size_t J, size_t K, size_t L, size_t M,
         typename std::enable_if<I==3 && J==3 && K==3 && L==3 && M==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J,K> cross(const Tensor<T,L,M> &A, const Tensor<T,I,J,K> &B) {
    Tensor<T,I,J,K> C;
    _crossproduct23<T,I,J,K,L,M>(B.data(),A.data(),C.data());
    return C;
}




// Overloads for high order tensors
// Only inverse, cofactor, adjoint, determinant and trace are overloaded
// at the moment. In the future a generice overloads allowing for
// optional axis/axes (like numpy) should be implemented for all high order
// tensors. At the moment the last two dimensions of the tensors
// (last matrix) are used as matrices to perform these operations.
// Other functions like norm don't make sense to be overloaded
// unless allowing for axis since norm of high order tensors is already
// taken care of with _norm function

template<typename T, size_t ... Rest, typename std::enable_if<sizeof...(Rest)>=3,bool>::type=0>
FASTOR_INLINE
typename LastMatrixExtracter<Tensor<T,Rest...>, typename std_ext::make_index_sequence<sizeof...(Rest)-2>::type>::type
determinant(const Tensor<T,Rest...> &a) {

    using OutTensor = typename LastMatrixExtracter<Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Rest)-2>::type>::type;
    constexpr size_t remaining_product = LastMatrixExtracter<Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Rest)-2>::type>::remaining_product;

    constexpr size_t I = get_value<sizeof...(Rest)-1,Rest...>::value;
    constexpr size_t J = get_value<sizeof...(Rest),Rest...>::value;
    static_assert(I==J,"THE LAST TWO DIMENSIONS OF TENSOR MUST BE THE SAME");

    OutTensor out;
    T *a_data = a.data();
    T *out_data = out.data();

    for (size_t i=0; i<remaining_product; ++i) {
        out_data[i] = _det<T,J,J>(static_cast<const T *>(a_data+i*J*J));
    }

    return out;
}


template<typename T, size_t ... Rest, typename std::enable_if<sizeof...(Rest)>=3,bool>::type=0>
FASTOR_INLINE
typename LastMatrixExtracter<Tensor<T,Rest...>, typename std_ext::make_index_sequence<sizeof...(Rest)-2>::type>::type
trace(const Tensor<T,Rest...> &a) {

    using OutTensor = typename LastMatrixExtracter<Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Rest)-2>::type>::type;
    constexpr size_t remaining_product = LastMatrixExtracter<Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Rest)-2>::type>::remaining_product;

    constexpr size_t I = get_value<sizeof...(Rest)-1,Rest...>::value;
    constexpr size_t J = get_value<sizeof...(Rest),Rest...>::value;
    static_assert(I==J,"THE LAST TWO DIMENSIONS OF TENSOR MUST BE THE SAME");

    OutTensor out;
    T *a_data = a.data();
    T *out_data = out.data();

    for (size_t i=0; i<remaining_product; ++i) {
        out_data[i] = _trace<T,J,J>(static_cast<const T *>(a_data+i*J*J));
    }

    return out;
}


template<typename T, size_t ... Rest, typename std::enable_if<sizeof...(Rest)>=3,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest...>
transpose(const Tensor<T,Rest...> &a) {

    constexpr size_t remaining_product = LastMatrixExtracter<Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Rest)-2>::type>::remaining_product;

    constexpr size_t I = get_value<sizeof...(Rest)-1,Rest...>::value;
    constexpr size_t J = get_value<sizeof...(Rest),Rest...>::value;
    static_assert(I==J,"THE LAST TWO DIMENSIONS OF TENSOR MUST BE THE SAME");

    Tensor<T,Rest...> out;
    T *a_data = a.data();
    T *out_data = out.data();

    for (size_t i=0; i<remaining_product; ++i) {
        _transpose<T,J,J>(a_data+i*J*J,out_data+i*J*J);
    }

    return out;
}


template<typename T, size_t ... Rest, typename std::enable_if<sizeof...(Rest)>=3,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest...>
adjoint(const Tensor<T,Rest...> &a) {

    constexpr size_t remaining_product = LastMatrixExtracter<Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Rest)-2>::type>::remaining_product;

    constexpr size_t I = get_value<sizeof...(Rest)-1,Rest...>::value;
    constexpr size_t J = get_value<sizeof...(Rest),Rest...>::value;
    static_assert(I==J,"THE LAST TWO DIMENSIONS OF TENSOR MUST BE THE SAME");

    Tensor<T,Rest...> out;
    T *a_data = a.data();
    T *out_data = out.data();

    for (size_t i=0; i<remaining_product; ++i) {
        _adjoint<T,J,J>(a_data+i*J*J,out_data+i*J*J);
    }

    return out;
}


template<typename T, size_t ... Rest, typename std::enable_if<sizeof...(Rest)>=3,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest...>
cofactor(const Tensor<T,Rest...> &a) {

    constexpr size_t remaining_product = LastMatrixExtracter<Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Rest)-2>::type>::remaining_product;

    constexpr size_t I = get_value<sizeof...(Rest)-1,Rest...>::value;
    constexpr size_t J = get_value<sizeof...(Rest),Rest...>::value;
    static_assert(I==J,"THE LAST TWO DIMENSIONS OF TENSOR MUST BE THE SAME");

    Tensor<T,Rest...> out;
    T *a_data = a.data();
    T *out_data = out.data();

    for (size_t i=0; i<remaining_product; ++i) {
        _cofactor<T,J,J>(a_data+i*J*J,out_data+i*J*J);
    }

    return out;
}

template<typename T, size_t ... Rest, typename std::enable_if<sizeof...(Rest)>=3,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest...>
inverse(const Tensor<T,Rest...> &a) {

    constexpr size_t remaining_product = LastMatrixExtracter<Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Rest)-2>::type>::remaining_product;

    constexpr size_t I = get_value<sizeof...(Rest)-1,Rest...>::value;
    constexpr size_t J = get_value<sizeof...(Rest),Rest...>::value;
    static_assert(I==J,"THE LAST TWO DIMENSIONS OF TENSOR MUST BE THE SAME");

    Tensor<T,Rest...> out;
    T *a_data = a.data();
    T *out_data = out.data();

    for (size_t i=0; i<remaining_product; ++i) {
        T det = _det<T,J,J>(static_cast<const T *>(a_data+i*J*J));
        _adjoint<T,J,J>(a_data+i*J*J,out_data+i*J*J);

        for (size_t j=i*J*J; j<(i+1)*J*J; ++j) {
            out_data[j] /= det;
        }
    }

    return out;
}
//



//
template<template<typename,size_t...> class TensorType, typename T, size_t ... Rest>
FASTOR_INLINE Tensor<T,Rest...> tocolumnmajor(const TensorType<T,Rest...> &a) {
    constexpr int Dimension = sizeof...(Rest);
    if (Dimension < 2) {
        return a;
    }
    else {
        Tensor<T,Rest...> out;
        T *arr_out = out.data();
        const T *a_data = a.data();

        if (Dimension == 2) {
            constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
            constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
            for (FASTOR_INDEX i=0; i<M; ++i) {
                for (FASTOR_INDEX j=0; j<N; ++j) {
                    arr_out[i*N+j] = a_data[j*M+i];
                }
            }
        }
        else {
            constexpr int Size = prod<Rest...>::value;
            std::array<size_t,Dimension> products_ = nprods_views<Index<Rest...>,
                typename std_ext::make_index_sequence<Dimension>::type>::values;
            FASTOR_INDEX DimensionHolder[Dimension] = {Rest...};
            std::reverse(DimensionHolder,DimensionHolder+Dimension);
            std::reverse(products_.begin(),products_.end());
            std::array<int,Dimension> as = {};

            int jt;
            FASTOR_INDEX counter=0;
            while(counter < Size)
            {
                FASTOR_INDEX index = 0;
                for (int ii=0; ii<Dimension; ++ii) {
                    index += products_[ii]*as[ii];
                }

                arr_out[index] = a_data[counter];

                counter++;
                for(jt = Dimension-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<DimensionHolder[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        return out;
    }
}

template<template<typename,size_t...> class TensorType, typename T, size_t ... Rest>
FASTOR_INLINE Tensor<T,Rest...> torowmajor(const TensorType<T,Rest...> &a) {
    constexpr int Dimension = sizeof...(Rest);
    if (Dimension < 2) {
        return a;
    }
    else {
        Tensor<T,Rest...> out;
        T *arr_out = out.data();
        const T *a_data = a.data();

        if (Dimension == 2) {
            constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
            constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
            for (FASTOR_INDEX i=0; i<M; ++i) {
                for (FASTOR_INDEX j=0; j<N; ++j) {
                    arr_out[j*M+i] = a_data[i*N+j];
                }
            }
        }
        else {
            constexpr int Size = prod<Rest...>::value;
            std::array<size_t,Dimension> products_ = nprods_views<Index<Rest...>,
                typename std_ext::make_index_sequence<Dimension>::type>::values;
            FASTOR_INDEX DimensionHolder[Dimension] = {Rest...};
            std::reverse(DimensionHolder,DimensionHolder+Dimension);
            std::reverse(products_.begin(),products_.end());
            std::array<int,Dimension> as = {};

            int jt;
            FASTOR_INDEX counter=0;
            while(counter < Size)
            {
                FASTOR_INDEX index = 0;
                for (int ii=0; ii<Dimension; ++ii) {
                    index += products_[ii]*as[ii];
                }

                arr_out[counter] = a_data[index];

                counter++;
                for(jt = Dimension-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<DimensionHolder[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        return out;
    }
}
//




// Constant tensors
static FASTOR_INLINE
Tensor<float,3,3,3> levi_civita_ps() {
    Tensor<float,3,3,3> LeCi_ps;
    LeCi_ps(0,1,2) = 1.f;
    LeCi_ps(1,2,0) = 1.f;
    LeCi_ps(2,0,1) = 1.f;
    LeCi_ps(1,0,2) = -1.f;
    LeCi_ps(2,1,0) = -1.f;
    LeCi_ps(0,2,1) = -1.f;

    return LeCi_ps;
}

static FASTOR_INLINE
Tensor<double,3,3,3> levi_civita_pd() {
    Tensor<double,3,3,3> LeCi_pd;
    LeCi_pd(0,1,2) = 1.;
    LeCi_pd(1,2,0) = 1.;
    LeCi_pd(2,0,1) = 1.;
    LeCi_pd(1,0,2) = -1.;
    LeCi_pd(2,1,0) = -1.;
    LeCi_pd(0,2,1) = -1.;

    return LeCi_pd;
}

//template<typename T>
//Tensor<T,3,3,3> levi_civita() {
//    Tensor<T,3,3,3> out; out.zeros();
//}

template<typename T, size_t ... Rest>
static FASTOR_INLINE
Tensor<T,Rest...> kronecker_delta() {
    Tensor<T,Rest...> out; out.eye();
    return out;
}

}

#endif // TENSOR_FUNCTIONS_H

