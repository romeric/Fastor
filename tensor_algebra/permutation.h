#ifndef PERMUTATION_H
#define PERMUTATION_H

#include "tensor/Tensor.h"
#include "indicial.h"

namespace Fastor {



template<size_t N>
constexpr size_t count_less(const size_t (&seq)[N], size_t i, size_t cur = 0) {
    return cur == N ? 0
                    : (count_less(seq, i, cur + 1) + (seq[cur] < i ? 1 : 0));
}

template<typename T, class List, class Tensor, class Seq>
struct permute_impl;

template<typename T, size_t ... ls, size_t ... fs, size_t... ss>
struct permute_impl<T,Index<ls...>, Tensor<T, fs...>, std_ext::index_sequence<ss...>>{
    constexpr static size_t lst[sizeof...(ls)] = { ls... };
    constexpr static size_t fvals[sizeof...(ls)] = {fs...};
    using type = Tensor<T,fvals[count_less(lst, lst[ss])]...>;
};





template<class T>
struct extractor_perm {};

template<size_t ... Idx>
struct extractor_perm<Index<Idx...> > {
  template<typename T, size_t ... Rest>
    static
    typename permute_impl<T,Index<Idx...>, Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Idx)>::type>::type
    permutation_impl(const Tensor<T,Rest...> &a) {

        typename permute_impl<T,Index<Idx...>, Tensor<T,Rest...>,
            typename std_ext::make_index_sequence<sizeof...(Idx)>::type>::type out;
        out.zeros();

        T *a_data = a.data();
        T *out_data = out.data();

        constexpr int a_dim = sizeof...(Rest);
        constexpr int out_dim = a_dim;
        constexpr std::array<int,a_dim> maxes_a = {Rest...};
        constexpr std::array<int,a_dim> idx = {Idx...};
        std::array<int,out_dim> maxes_idx = argsort(idx);
        std::array<int,out_dim> maxes_out;
//        std::fill(maxes_out.begin(),maxes_out.end(),0);
        for (int i=0; i<out_dim; ++i) {
            maxes_out[i] = maxes_a[maxes_idx[i]];
        }

        std::array<int,a_dim> products_a;
        std::fill(products_a.begin(),products_a.end(),0);
        for (int j=a_dim-1; j>0; --j) {
            int num = maxes_a[a_dim-1];
            for (int k=0; k<j-1; ++k) {
                num *= maxes_a[a_dim-1-k-1];
            }
            products_a[j] = num;
        }
        std::array<int,out_dim> products_out;
        std::fill(products_out.begin(),products_out.end(),0);
        for (int j=out_dim-1; j>0; --j) {
            int num = maxes_out[out_dim-1];
            for (int k=0; k<j-1; ++k) {
                num *= maxes_out[out_dim-1-k-1];
            }
            products_out[j] = num;
        }

        std::reverse(products_a.begin(),products_a.end());
        std::reverse(products_out.begin(),products_out.end());

        int as[out_dim];
        std::fill(as,as+out_dim,0);
        int it,jt;

        while(true)
        {
            int index_a = as[a_dim-1];
            for(it = 0; it< a_dim; it++) {
                index_a += products_a[it]*as[it];
            }
            int index_out = as[maxes_idx[out_dim-1]];
            for(it = 0; it< out_dim-1; it++) {
                index_out += products_out[it]*as[maxes_idx[it]];
            }

            out_data[index_out] = a_data[index_a];

            for(jt = out_dim-1 ; jt>=0 ; jt--)
            {
                if(++as[jt]<maxes_a[jt])
                    break;
                else
                    as[jt]=0;
            }
            if(jt<0)
                break;
        }

        return out;
    }
};



template<class Index_I, typename T, size_t ... Rest>
typename permute_impl<T,Index_I, Tensor<T,Rest...>,
    typename std_ext::make_index_sequence<sizeof...(Rest)>::type>::type permutation(const Tensor<T, Rest...> &a) {
    return extractor_perm<Index_I>::permutation_impl(a);
}




// Specialised dispatcher as the above generic version can be expensive

// IKJL
template<class Ind,
         typename T, size_t I, size_t J, size_t K, size_t L,
         typename std::enable_if<Ind::NoIndices==4 &&
                                 is_less<Ind::_IndexHolder[0], Ind::_IndexHolder[1]>::value && 
                                 is_less<Ind::_IndexHolder[0], Ind::_IndexHolder[2]>::value && 
                                 is_less<Ind::_IndexHolder[0], Ind::_IndexHolder[3]>::value &&

                                 is_greater<Ind::_IndexHolder[1], Ind::_IndexHolder[2]>::value && 
                                 is_less<Ind::_IndexHolder[1], Ind::_IndexHolder[3]>::value &&

                                 is_less<Ind::_IndexHolder[2], Ind::_IndexHolder[3]>::value,bool>::type = 0>
FASTOR_INLINE Tensor<T,I,K,J,L>
permutation(const Tensor<T,I,J,K,L> &a) {

    Tensor<T,I,K,J,L> out;
    for (size_t i=0; i<I; ++i) {
        for (size_t j=0; j<J; ++j) {
            for (size_t k=0; k<K; ++k) {
                for (size_t l=0; l<L; ++l) {
                    out(i,j,k,l) = a(i,k,j,l);
                }
            }
        }
    }
    return out;
}

// ILJK
template<class Ind,
         typename T, size_t I, size_t J, size_t K, size_t L,
         typename std::enable_if<Ind::NoIndices==4 &&
                                 is_less<Ind::_IndexHolder[0], Ind::_IndexHolder[1]>::value && 
                                 is_less<Ind::_IndexHolder[0], Ind::_IndexHolder[2]>::value && 
                                 is_less<Ind::_IndexHolder[0], Ind::_IndexHolder[3]>::value &&

                                 is_greater<Ind::_IndexHolder[1], Ind::_IndexHolder[2]>::value && 
                                 is_greater<Ind::_IndexHolder[1], Ind::_IndexHolder[3]>::value &&

                                 is_less<Ind::_IndexHolder[2], Ind::_IndexHolder[3]>::value,bool>::type = 0>
FASTOR_INLINE Tensor<T,I,L,J,K>
permutation(const Tensor<T,I,J,K,L> &a) {

    Tensor<T,I,L,J,K> out;
    for (size_t i=0; i<I; ++i) {
        for (size_t j=0; j<J; ++j) {
            for (size_t k=0; k<K; ++k) {
                for (size_t l=0; l<L; ++l) {
                    out(i,j,k,l) = a(i,l,j,k);
                }
            }
        }
    }
    return out;
}


// IKJ
template<class Ind,
         typename T, size_t I, size_t J, size_t K,
         typename std::enable_if<Ind::NoIndices==3 &&
                                 is_less<Ind::_IndexHolder[0], Ind::_IndexHolder[1]>::value && 
                                 is_less<Ind::_IndexHolder[0], Ind::_IndexHolder[2]>::value && 

                                 is_greater<Ind::_IndexHolder[1], Ind::_IndexHolder[2]>::value,bool>::type = 0>
FASTOR_INLINE Tensor<T,I,K,J>
permutation(const Tensor<T,I,J,K> &a) {

    Tensor<T,I,K,J> out;
    for (size_t i=0; i<I; ++i) {
        for (size_t j=0; j<J; ++j) {
            for (size_t k=0; k<K; ++k) {
                out(i,j,k) = a(i,k,j);
            }
        }
    }
    return out;
}


// JKI
template<class Ind,
         typename T, size_t I, size_t J, size_t K,
         typename std::enable_if<Ind::NoIndices==3 &&
                                 is_less<Ind::_IndexHolder[0], Ind::_IndexHolder[1]>::value && 
                                 is_greater<Ind::_IndexHolder[0], Ind::_IndexHolder[2]>::value && 

                                 is_greater<Ind::_IndexHolder[1], Ind::_IndexHolder[2]>::value,bool>::type = 0>
FASTOR_INLINE Tensor<T,J,K,I>
permutation(const Tensor<T,I,J,K> &a) {

    Tensor<T,J,K,I> out;
    for (size_t i=0; i<I; ++i) {
        for (size_t j=0; j<J; ++j) {
            for (size_t k=0; k<K; ++k) {
                out(i,j,k) = a(j,k,i);
            }
        }
    }
    return out;
}


// KJI
template<class Ind,
         typename T, size_t I, size_t J, size_t K,
         typename std::enable_if<Ind::NoIndices==3 &&
                                 is_greater<Ind::_IndexHolder[0], Ind::_IndexHolder[1]>::value && 
                                 is_greater<Ind::_IndexHolder[0], Ind::_IndexHolder[2]>::value && 

                                 is_greater<Ind::_IndexHolder[1], Ind::_IndexHolder[2]>::value,bool>::type = 0>
FASTOR_INLINE Tensor<T,K,J,I>
permutation(const Tensor<T,K,J,I> &a) {

    Tensor<T,K,J,I> out;
    for (size_t i=0; i<I; ++i) {
        for (size_t j=0; j<J; ++j) {
            for (size_t k=0; k<K; ++k) {
                out(i,j,k) = a(k,j,i);
            }
        }
    }
    return out;
}


}
#endif // PERMUTATION_H

