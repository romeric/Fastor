#ifndef EINSUM_H
#define EINSUM_H

#include "backend/backend.h"
#include "tensor/Tensor.h"
#include "meta/einsum_meta.h"
#include "indicial.h"

#include "reshape.h"
#include "permutation.h"

namespace Fastor {



//template<typename Index_I, typename Index_J,
//         template<typename,size_t,size_t,size_t...Rest> class Tensor,
//         typename T, size_t M, size_t N, size_t ... Rest,
//         typename std::enable_if<prod<Rest...>::value==1 && Index_I::_IndexHolder[0]==Index_J::_IndexHolder[0]
//                                 && Index_I::_IndexHolder[1]==Index_J::_IndexHolder[1]>::type=0 >
//FASTOR_INLINE Tensor<T,M,N,Rest...> einsum(const Tensor<T,M,N,Rest...> &a, const Tensor<T,M,N,Rest...> &b) {
//    return doublecontract<T,M,N>(a.data(),b.data());
//}

// trace
template<typename Index_I,
         template<typename,size_t ... Rest> class Tensor, typename T, size_t ... Rest,
         typename std::enable_if< sizeof...(Rest)==0 && Index_I::_IndexHolder[0]==Index_I::_IndexHolder[1] &&
                                 get_value<1,Rest...>::value==get_value<2,Rest...>::value, bool>::type=0 >
FASTOR_INLINE T einsum(const Tensor<T,Rest...> &a) {
    return _trace<T,Rest...>(a.data());
}

// matmul
template<typename Index_I, typename Index_J,
         template<typename,size_t...Rest> class Tensor,
         typename T, size_t ... Rest,
         typename std::enable_if<sizeof...(Rest)==3 &&
                                 Index_I::_IndexHolder[1]==Index_J::_IndexHolder[0] &&
                                 Index_I::_IndexHolder[0]!=Index_J::_IndexHolder[1] &&
                                 Index_I::_IndexHolder[0]!=Index_I::_IndexHolder[1] &&
                                 Index_J::_IndexHolder[0]!=Index_J::_IndexHolder[1]
                                 ,bool>::type=0 >
FASTOR_INLINE Tensor<T,Rest...> einsum(const Tensor<T,Rest...> &a, const Tensor<T,Rest...> &b) {
    Tensor<T,Rest...> c;
    constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
    constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
    matmul<T,M,N,N>(a.data(),b.data(),c.data());
    return c;
}

//// dyadic
//template<typename Index_I, typename Index_J,
//         template<typename,size_t...Rest> class Tensor, typename T, size_t ... Rest,
//         typename std::enable_if<sizeof...(Rest)==2 && get_value<1,Rest...>::value==3 && get_value<2,Rest...>::value==3
//                                 && Index_I::_IndexHolder[0]!=Index_J::_IndexHolder[0]
//                                 && Index_I::_IndexHolder[0]!=Index_J::_IndexHolder[1]
//                                 && Index_I::_IndexHolder[1]!=Index_J::_IndexHolder[0]
//                                 && Index_I::_IndexHolder[1]!=Index_J::_IndexHolder[1]
//                                 ,bool>::type=0 >
//FASTOR_INLINE Tensor<T,Rest...,Rest...> einsum(const Tensor<T,Rest...> &a, const Tensor<T,Rest...> &b) {

//    constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
//    constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;

//    Tensor<T,Rest...,Rest...> c;
//    c.zeros();

//    // The branches are resolved at compile time
//    if (Index_I::_IndexHolder[0]<Index_I::_IndexHolder[1] &&
//            Index_J::_IndexHolder[0]<Index_J::_IndexHolder[1] &&
//            Index_I::_IndexHolder[0]<Index_J::_IndexHolder[0] &&
//            Index_I::_IndexHolder[1]<Index_J::_IndexHolder[1] &&
//            Index_I::_IndexHolder[1]<Index_J::_IndexHolder[0] &&
//            Index_I::_IndexHolder[0]<Index_J::_IndexHolder[1]) {
//        for (FASTOR_INDEX i=0; i<M; ++i)
//            for (FASTOR_INDEX j=0; j<N; ++j)
//                for (FASTOR_INDEX k=0; k<M; ++k)
//                    for (FASTOR_INDEX l=0; l<N; ++l)
//                        c(i,j,k,l) += a(i,j)*b(k,l);
//    }
//    else if (Index_I::_IndexHolder[0]>Index_I::_IndexHolder[1] &&
//             Index_J::_IndexHolder[0]<Index_J::_IndexHolder[1] &&
//             Index_I::_IndexHolder[0]<Index_J::_IndexHolder[0] &&
//             Index_I::_IndexHolder[1]<Index_J::_IndexHolder[1] &&
//             Index_I::_IndexHolder[1]<Index_J::_IndexHolder[0] &&
//             Index_I::_IndexHolder[0]<Index_J::_IndexHolder[1]) {
//         for (FASTOR_INDEX i=0; i<M; ++i)
//             for (FASTOR_INDEX j=0; j<N; ++j)
//                 for (FASTOR_INDEX k=0; k<M; ++k)
//                     for (FASTOR_INDEX l=0; l<N; ++l)
//                         c(i,j,k,l) += a(j,i)*b(k,l);
//         }
//    else if (Index_I::_IndexHolder[0]<Index_I::_IndexHolder[1] &&
//             Index_J::_IndexHolder[0]>Index_J::_IndexHolder[1] &&
//             Index_I::_IndexHolder[0]<Index_J::_IndexHolder[0] &&
//             Index_I::_IndexHolder[1]<Index_J::_IndexHolder[1] &&
//             Index_I::_IndexHolder[1]<Index_J::_IndexHolder[0] &&
//             Index_I::_IndexHolder[0]<Index_J::_IndexHolder[1]) {
//         for (FASTOR_INDEX i=0; i<M; ++i)
//             for (FASTOR_INDEX j=0; j<N; ++j)
//                 for (FASTOR_INDEX k=0; k<M; ++k)
//                     for (FASTOR_INDEX l=0; l<N; ++l)
//                         c(i,j,k,l) += a(i,j)*b(l,k);
//         }
//    else if (Index_I::_IndexHolder[0]>Index_I::_IndexHolder[1] &&
//             Index_J::_IndexHolder[0]>Index_J::_IndexHolder[1] &&
//             Index_I::_IndexHolder[0]<Index_J::_IndexHolder[0] &&
//             Index_I::_IndexHolder[1]<Index_J::_IndexHolder[1] &&
//             Index_I::_IndexHolder[1]<Index_J::_IndexHolder[0] &&
//             Index_I::_IndexHolder[0]<Index_J::_IndexHolder[1]) {
//         for (FASTOR_INDEX i=0; i<M; ++i)
//             for (FASTOR_INDEX j=0; j<N; ++j)
//                 for (FASTOR_INDEX k=0; k<M; ++k)
//                     for (FASTOR_INDEX l=0; l<N; ++l)
//                         c(i,j,k,l) += a(j,i)*b(l,k);
//         }
//    else if(Index_I::_IndexHolder[0]<Index_I::_IndexHolder[1] &&
//            Index_J::_IndexHolder[0]<Index_J::_IndexHolder[1] &&
//            Index_I::_IndexHolder[0]<Index_J::_IndexHolder[0] &&
//            Index_I::_IndexHolder[1]<Index_J::_IndexHolder[1] &&
//            Index_I::_IndexHolder[1]>Index_J::_IndexHolder[0] &&
//            Index_I::_IndexHolder[0]<Index_J::_IndexHolder[1]) {
//        for (FASTOR_INDEX i=0; i<M; ++i)
//            for (FASTOR_INDEX j=0; j<N; ++j)
//                for (FASTOR_INDEX k=0; k<M; ++k)
//                    for (FASTOR_INDEX l=0; l<N; ++l)
//                        c(i,j,k,l) += a(i,k)*b(j,l);
//    }
//    else if(Index_I::_IndexHolder[0]<Index_I::_IndexHolder[1] &&
//            Index_J::_IndexHolder[0]>Index_J::_IndexHolder[1] &&
//            Index_I::_IndexHolder[0]<Index_J::_IndexHolder[0] &&
//            Index_I::_IndexHolder[1]>Index_J::_IndexHolder[1] &&
//            Index_I::_IndexHolder[1]<Index_J::_IndexHolder[0] &&
//            Index_I::_IndexHolder[0]<Index_J::_IndexHolder[1]) {
//        for (FASTOR_INDEX i=0; i<M; ++i)
//            for (FASTOR_INDEX j=0; j<N; ++j)
//                for (FASTOR_INDEX k=0; k<M; ++k)
//                    for (FASTOR_INDEX l=0; l<N; ++l)
//                        c(i,j,k,l) += a(i,k)*b(l,j);
//    }
//    else if(Index_I::_IndexHolder[0]<Index_I::_IndexHolder[1] &&
//            Index_J::_IndexHolder[0]<Index_J::_IndexHolder[1] &&
//            Index_I::_IndexHolder[0]<Index_J::_IndexHolder[0] &&
//            Index_I::_IndexHolder[1]>Index_J::_IndexHolder[1] &&
//            Index_I::_IndexHolder[1]>Index_J::_IndexHolder[0] &&
//            Index_I::_IndexHolder[0]<Index_J::_IndexHolder[1]) {
//        for (FASTOR_INDEX i=0; i<M; ++i)
//            for (FASTOR_INDEX j=0; j<N; ++j)
//                for (FASTOR_INDEX k=0; k<M; ++k)
//                    for (FASTOR_INDEX l=0; l<N; ++l)
//                        c(i,j,k,l) += a(i,l)*b(j,k);
//    }
//    else if(Index_I::_IndexHolder[0]<Index_I::_IndexHolder[1] &&
//            Index_J::_IndexHolder[0]>Index_J::_IndexHolder[1] &&
//            Index_I::_IndexHolder[0]<Index_J::_IndexHolder[0] &&
//            Index_I::_IndexHolder[1]>Index_J::_IndexHolder[1] &&
//            Index_I::_IndexHolder[1]>Index_J::_IndexHolder[0] &&
//            Index_I::_IndexHolder[0]<Index_J::_IndexHolder[1]) {
//        for (FASTOR_INDEX i=0; i<M; ++i)
//            for (FASTOR_INDEX j=0; j<N; ++j)
//                for (FASTOR_INDEX k=0; k<M; ++k)
//                    for (FASTOR_INDEX l=0; l<N; ++l)
//                        c(i,j,k,l) += a(i,l)*b(k,j);
//    }
//    return c;
//}

// dyadic voigt
template<typename Index_I, typename Index_J, int VoigtNotation,
         template<typename,size_t...Rest> class Tensor, typename T, size_t ... Rest,
         typename std::enable_if<get_value<1,Rest...>::value==3 && get_value<2,Rest...>::value==3
                                 && Index_I::_IndexHolder[0]!=Index_J::_IndexHolder[0]
                                 && Index_I::_IndexHolder[0]!=Index_J::_IndexHolder[1]
                                 && Index_I::_IndexHolder[1]!=Index_J::_IndexHolder[0]
                                 && Index_I::_IndexHolder[1]!=Index_J::_IndexHolder[1]
                                 && VoigtNotation==Voigt
                                 ,bool>::type=0 >
FASTOR_INLINE Tensor<T,6,6> einsum(const Tensor<T,Rest...> &a, const Tensor<T,Rest...> &b) {
    Tensor<T,6,6> c;
    outer<T,Rest...,Rest...>(a.data(),b.data(),c.data());
    return c;
}

// double contract
template<typename Index_I, typename Index_J,
         template<typename,size_t...Rest> class Tensor,
         typename T, size_t ... Rest,
         typename std::enable_if<
             sizeof...(Rest)==2
             && Index_I::_IndexHolder[0]==Index_J::_IndexHolder[0] && Index_I::_IndexHolder[1]==Index_J::_IndexHolder[1]
             && sizeof...(Rest)==Index_I::NoIndices && sizeof...(Rest)==Index_J::NoIndices,bool>::type=0 >
FASTOR_INLINE T einsum(const Tensor<T,Rest...> &a, const Tensor<T,Rest...> &b) {
    return _doublecontract<T,Rest...>(a.data(),b.data());
}

// triple contract
template<typename Index_I, typename Index_J,
         template<typename,size_t...Rest> class Tensor,
         typename T, size_t ... Rest,
         typename std::enable_if<
             sizeof...(Rest)==3
             && Index_I::_IndexHolder[0]==Index_J::_IndexHolder[0] && Index_I::_IndexHolder[1]==Index_J::_IndexHolder[1]
             && Index_I::_IndexHolder[2]==Index_J::_IndexHolder[2]
             && sizeof...(Rest)==Index_I::NoIndices && sizeof...(Rest)==Index_J::NoIndices,bool>::type=0 >
FASTOR_INLINE T einsum(const Tensor<T,Rest...> &a, const Tensor<T,Rest...> &b) {
    T tc = static_cast<T>(0);
    for (FASTOR_INDEX i=0; i<get_value<1,Rest...>::value; ++i)
        for (FASTOR_INDEX j=0; j<get_value<2,Rest...>::value; ++j)
            for (FASTOR_INDEX k=0; k<get_value<3,Rest...>::value; ++k)
                tc += a(i,j,k)*b(i,j,k);
    return tc;
}

// four index contract
template<typename Index_I, typename Index_J,
         template<typename,size_t,size_t,size_t...Rest> class Tensor,
         typename T, size_t M, size_t N, size_t ... Rest,
         typename std::enable_if<
             prod<Rest...>::value != 1 && sizeof...(Rest)==2
             && Index_I::_IndexHolder[0]==Index_J::_IndexHolder[0] && Index_I::_IndexHolder[1]==Index_J::_IndexHolder[1]
             && Index_I::_IndexHolder[2]==Index_J::_IndexHolder[2] && Index_I::_IndexHolder[3]==Index_J::_IndexHolder[3]
             && sizeof...(Rest)+2==Index_I::NoIndices && sizeof...(Rest)+2==Index_J::NoIndices,bool>::type=0 >
FASTOR_INLINE T einsum(const Tensor<T,M,N,Rest...> &a, const Tensor<T,M,N,Rest...> &b) {
    T tc = static_cast<T>(0);
    for (FASTOR_INDEX i=0; i<M; ++i)
        for (FASTOR_INDEX j=0; j<N; ++j)
            for (FASTOR_INDEX k=0; k<get_value<1,Rest...>::value; ++k)
                for (FASTOR_INDEX l=0; l<get_value<1,Rest...>::value; ++l)
                    tc += a(i,j,k,l)*b(i,j,k,l);
    return tc;
}

// triple reduction
template<typename Index_I,
         template<typename,size_t,size_t,size_t...Rest> class Tensor,
         typename T, size_t M, size_t N, size_t ... Rest,
         typename std::enable_if<
             prod<Rest...>::value != 1 && sizeof...(Rest)==1 && M==N && M==get_value<1,Rest...>::value
             && Index_I::_IndexHolder[0]==Index_I::_IndexHolder[1] && Index_I::_IndexHolder[0]==Index_I::_IndexHolder[2]
             && sizeof...(Rest)+2==Index_I::NoIndices,bool>::type=0 >
FASTOR_INLINE T einsum(const Tensor<T,M,N,Rest...> &a) {
    T tc = static_cast<T>(0);
    for (FASTOR_INDEX i=0; i<M; ++i)
        tc += a(i,i,i);
    return tc;
}

// four index reduction
//template<typename Index_I,
//         template<typename,size_t,size_t,size_t...Rest> class Tensor,
//         typename T, size_t M, size_t N, size_t ... Rest,
//         typename std::enable_if<
//             prod<Rest...>::value != 1 && sizeof...(Rest)==2 && M==N && M==get_value<1,Rest...>::value //&& M==get_value<2,Rest...>::value
//             && Index_I::_IndexHolder[0]==Index_I::_IndexHolder[1] && Index_I::_IndexHolder[0]==Index_I::_IndexHolder[2]
//             && Index_I::_IndexHolder[0]==Index_I::_IndexHolder[3]
//             && sizeof...(Rest)+2==Index_I::NoIndices,bool>::type=0 >
//FASTOR_INLINE T einsum(const Tensor<T,M,N,Rest...> &a) {
//    T tc = static_cast<T>(0);
//    for (FASTOR_INDEX i=0; i<M; ++i)
//        tc += a(i,i,i,i);
//    return tc;
//}



//template<template<size_t...Idx0> class Index_I,
//         template<size_t...Idx1> class Index_J,
//         size_t ... Idx0, size_t ... Idx1,
//         template<typename,size_t...Rest0> class Tensor0,
//         template<typename,size_t...Rest1> class Tensor1,
//         typename T, size_t ... Rest0, size_t ... Rest1,
//         typename std::enable_if<sizeof...(Rest0)>=2 && sizeof...(Rest0)>=2 &&
//                                 sizeof...(Rest0)==sizeof...(Idx0) &&
//                                 sizeof...(Rest1)==sizeof...(Idx0),bool>::type=0 >
//FASTOR_INLINE Tensor<T,Rest0...,Rest1...> einsum(const Tensor0<T,Rest0...> &a, const Tensor1<T,Rest1...> &b) {

//    static_assert(sizeof...(Rest0)==sizeof...(Idx0), "CONTRACTION INDICES DO NOT MATCH WITH TENSOR RANK");
//    static_assert(sizeof...(Rest1)==sizeof...(Idx1), "CONTRACTION INDICES DO NOT MATCH WITH TENSOR RANK");



// Most generic overload  - place holder at the moment
template<typename Index_I, typename Index_J,
         template<typename,size_t...Rest0> class Tensor0,
         template<typename,size_t...Rest1> class Tensor1,
         typename T, size_t ... Rest0, size_t ... Rest1,
         typename std::enable_if<sizeof...(Rest0)>=2 && sizeof...(Rest0)>=2 &&
                                 sizeof...(Rest0)==Index_I::NoIndices &&
                                 sizeof...(Rest1)==Index_J::NoIndices,bool>::type=0 >
FASTOR_INLINE Tensor<T,Rest0...,Rest1...> einsum(const Tensor0<T,Rest0...> &a, const Tensor1<T,Rest1...> &b) {
//FASTOR_INLINE Tensor<T,Rest0...> einsum(const Tensor0<T,Rest0...> &a, const Tensor1<T,Rest1...> &b) {
//FASTOR_INLINE AbstractTensor<Tensor<T,3,3,3>,3> einsum(const Tensor0<T,M0,N0,Rest0...> &a, const Tensor1<T,M1,N1,Rest1...> &b) {
//FASTOR_INLINE Tensor<T,M0,N0,Rest0...,M1,N1,Rest1...> einsum(const Tensor0<T,M0,N0,Rest0...> &a, const Tensor1<T,M1,N1,Rest1...> &b) {
    static_assert(sizeof...(Rest0)==Index_I::NoIndices, "CONTRACTION INDICES DO NOT MATCH WITH TENSOR RANK");
    static_assert(sizeof...(Rest1)==Index_J::NoIndices, "CONTRACTION INDICES DO NOT MATCH WITH TENSOR RANK");

    Tensor<T,Rest0...,Rest1...> out;
    out.zeros();
    T *a_data = a.data();
    T *b_data = b.data();
    T *out_data = out.data();

//    constexpr FASTOR_INDEX a_dim = sizeof...(Rest0);
//    constexpr FASTOR_INDEX b_dim = sizeof...(Rest1);
//    constexpr FASTOR_INDEX out_dim = a_dim+b_dim;
//    constexpr std::array<FASTOR_INDEX,a_dim> maxes_a = {Rest0...};
//    constexpr std::array<FASTOR_INDEX,b_dim> maxes_b = {Rest1...};
//    constexpr std::array<FASTOR_INDEX,out_dim> maxes_out = {Rest0...,Rest1...};


    constexpr int a_dim = sizeof...(Rest0);
    constexpr int b_dim = sizeof...(Rest1);
    constexpr int out_dim = a_dim+b_dim;
    constexpr std::array<int,a_dim> maxes_a = {Rest0...};
    constexpr std::array<int,b_dim> maxes_b = {Rest1...};
    constexpr std::array<int,out_dim> maxes_out = {Rest0...,Rest1...};

//    std::vector<int> maxes_a = {Rest0...};
//    std::vector<int> maxes_b = {Rest1...};
//    std::vector<int> maxes_out = {Rest0...,Rest1...};

    std::array<int,a_dim> products_a;
//    std::vector<int> products_a(a_dim);
    for (int j=a_dim-1; j>0; --j) {
        int num = maxes_a[a_dim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_a[a_dim-1-k-1];
        }
        products_a[j] = num;
    }
    std::array<int,b_dim> products_b;
//    std::vector<int> products_b(b_dim);
    for (int j=b_dim-1; j>0; --j) {
        int num = maxes_b[b_dim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_b[b_dim-1-k-1];
        }
        products_b[j] = num;
    }
    std::array<int,out_dim> products_out;
//    std::vector<int> products_out(out_dim);
    for (int j=out_dim-1; j>0; --j) {
        int num = maxes_out[out_dim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_out[out_dim-1-k-1];
        }
        products_out[j] = num;
    }

    std::reverse(products_a.begin(),products_a.end());
    std::reverse(products_b.begin(),products_b.end());
    std::reverse(products_out.begin(),products_out.end());
//    print(products_a,products_b,products_out);

//    std::array<FASTOR_INDEX,a_dim> largs_a;
//    std::array<FASTOR_INDEX,a_dim> largs_b;
//    std::array<FASTOR_INDEX,a_dim+b_dim> largs_out;
//    std::fill(largs_a.begin(),largs_a.end(),0);
//    std::fill(largs_b.begin(),largs_b.end(),0);
//    std::fill(largs_out.begin(),largs_out.end(),0);


//    std::array<int,out_dim> as;
//    std::fill(as.begin(),as.end(),0);
//    std::vector<int> as(out_dim); as.resize(out_dim);
//    std::fill(as.begin(),as.end(),0);
    int as[out_dim];
    std::fill(as,as+out_dim,0);
    int it,jt;

//    out_data[2] = a_data[5]+b_data[3]+maxes_a[1]+maxes_b[1]+maxes_out[1]+products_a[1]+products_b[2];
//    auto counter =0;
    while(true)
    {
        int index_a = as[a_dim-1];
        for(it = 0; it< a_dim; it++) {
            index_a += products_a[it]*as[it];
        }
        int index_b = as[out_dim-1];
        for(it = a_dim; it< out_dim; it++) {
            index_b += products_b[it-a_dim]*as[it];
        }
        int index_out = as[out_dim-1];
        for(it = 0; it< out_dim; it++) {
            index_out += products_out[it]*as[it];
        }
        out_data[index_out] += a_data[index_a]*b_data[index_b];
//        out_data[index_out] = a_data[index_a]*b_data[index_b];
//        if (as)
//        auto dd = _mm_mul_sd(_mm_load_sd(a_data+index_a),_mm_load_sd(b_data+index_b));
//        _mm_store_sd(out_data+index_out,dd);

//        std::cout << a_data[index_a] << " " << b_data[index_b] << "  " << out_data[index_out] <<  "\n";
//        out_data[728] += a_data[index_a]*b_data[index_b];
//        out_data[it] += a_data[index_a]*b_data[index_b];
//        std::cout << index_a << " " << index_b << "  " << index_out <<  "\n";
//        print(index_a);

//        for(it = 0; it< out_dim; it++) {
//            std::cout << as[it] << " ";
//        }
//        print();
//        counter++;
//        if (counter % 4 ==0) {
//            for(it = 0; it< out_dim; it++) {
//                std::cout << as[it] << " ";
//            }
//            print();
//        }

        for(jt = out_dim-1 ; jt>=0 ; jt--)
//        for(jt = out_dim+2-1 ; jt>=0 ; jt--)
        {
            if(++as[jt]<maxes_out[jt])
                break;
            else
                as[jt]=0;
        }
        if(jt<0)
            break;
    }

//    print(out);
//    print<T,729>(out.data());
//    print("\n");
//    return a;
    return out;
}






//----------------------------------------------------------------------------------------------
// reduction
template<typename T, size_t ... Rest>
T reduction(const Tensor<T,Rest...> &a) {
    //! Reduces a multi-dimensional tensor to a scalar
    //!
    //! If a is scalar/Tensor<T> returns the value itself
    //! If a is a vector Tensor<T,N> returns the sum of values
    //! If a is a second order tensor Tensor<T,N,N> returns the trace
    //! If a is a third order tensor Tensor<T,N,N,N> returns a_iii
    //! ...
    //!
    //! The size of the tensor in all dimensions should be equal

    static_assert(no_of_unique<Rest...>::value<=1, "REDUCTION IS ONLY POSSIBLE FOR TENSORS WITH EQUAL SIZES IN ALL DIMENSIONS");
    constexpr int ndim = sizeof...(Rest);

    T *a_data = a.data();
    if (ndim==0) {
        return a_data[0];
    }
    else if (ndim==1) {
        return a.sum();
    }
    else {
        constexpr std::array<int,ndim> maxes_a = {Rest...};
        std::array<int,ndim> products;
        std::fill(products.begin(),products.end(),0);

        for (int j=ndim-1; j>0; --j) {
            int num = maxes_a[ndim-1];
            for (int k=0; k<j-1; ++k) {
                num *= maxes_a[ndim-1-k-1];
            }
            products[j] = num;
        }
        std::reverse(products.begin(),products.end());

        T reductor = static_cast<T>(0);
        for (int i=0; i<a.dimension(0); ++i) {
            int index_a = i;
            for(int it = 0; it< ndim; it++) {
                index_a += products[it]*i;
            }
            reductor += a_data[index_a];
        }
        return reductor;
    }
}



//----------------------------------------------------------------------------------------------
// summation
template<typename T, size_t ... Rest>
FASTOR_INLINE T summation(const Tensor<T,Rest...> &a) {
    return a.sum();
}



//----------------------------------------------------------------------------------------------
//// permutation
//template<typename Index_I, typename T, size_t ... Rest>
//FASTOR_INLINE Tensor<T,Rest...> permutation(const Tensor<T,Rest...> &a) {
//    // TODO
//    return a;
//}


} // end of namespace

#endif // EINSUM_H

