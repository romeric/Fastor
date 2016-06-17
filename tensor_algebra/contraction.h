#ifndef CONTRACTION_H
#define CONTRACTION_H

#include "tensor/Tensor.h"
#include "indicial.h"


namespace Fastor {

/////////

// is ind[i] unique in ind?
template<size_t N>
constexpr bool is_uniq(const int (&ind)[N], size_t i, size_t cur = 0){
    return cur == N ? true :
           (cur == i || ind[cur] != ind[i]) ? is_uniq(ind, i, cur + 1) : false;
}

// For every i where ind[i] == index, is dim[i] == dimension?
template<size_t N>
constexpr bool check_all_eq(int index, int dimension,
                            const int (&ind)[N], const int (&dim)[N], size_t cur = 0) {
    return cur == N ? true :
           (ind[cur] != index || dim[cur] == dimension) ?
                check_all_eq(index, dimension, ind, dim, cur + 1) : false;
}

// if position i should be contracted away, return -1, otherwise return dim[i].
// triggers a compile-time error when used in a constant expression on mismatch.
template<size_t N>
constexpr int calc(size_t i, const int (&ind)[N], const int (&dim)[N]){
    return is_uniq(ind, i) ? dim[i] :
           check_all_eq(ind[i], dim[i], ind, dim) ? -1 : throw "dimension mismatch";
}
//Now we need a way to get rid of the -1s:
template<class Ind, class... Inds>
struct concat_ { using type = Ind; };
template<size_t... I1, size_t... I2, class... Inds>
struct concat_<Index<I1...>, Index<I2...>, Inds...>
    :  concat_<Index<I1..., I2...>, Inds...> {};

// filter out all instances of I from Is...,
// return the rest as an Indices
template<size_t I, size_t... Is>
struct filter_
    :  concat_<typename std::conditional<Is == I, Index<>, Index<Is>>::type...> {};
//Use them:
template<class Ind, class Arr, class Seq>
struct contraction_impl;

template<class T, size_t... Ind, size_t... Dim, size_t... Seq>
struct contraction_impl<Index<Ind...>, Tensor<T, Dim...>, std_ext::index_sequence<Seq...>>{
    static constexpr int ind[] = { Ind... };
    static constexpr int dim[] = { Dim... };
    static constexpr int result[] = {calc(Seq, ind, dim)...};

    template<size_t... Dims>
    static auto unpack_helper(Index<Dims...>) -> Tensor<T, Dims...>;

    using type = decltype(unpack_helper(typename filter_<-1,  result[Seq]...>::type{}));
};
////////////////////



template<class T>
struct prod_index;
//template<class T, class U>
//struct prod_index;

template<size_t ... rest>
struct prod_index<Index<rest...>> {
    const static size_t value = prod<rest...>::value;
};

template<class T, class U>
struct prod_index2;

template<size_t ... rest0, size_t ... rest1, typename T>
struct prod_index2<Index<rest0...>,Tensor<T,rest1...>> {
    const static size_t value = prod<rest0...,rest1...>::value;
};

template<class T, class U>
struct find_loop_type;

template<size_t ... rest0, size_t ... rest1, typename T>
struct find_loop_type<Index<rest0...>,Tensor<T,rest1...>> {
    using type = apply_typelist_t<quote_c<size_t, Index>,
                        uniq_t<typelist_c<size_t, rest0...,rest1...>>>;
};



////////////////

template<int a_dim, int out_dim>
std::vector<int> which_where(const size_t * __restrict__ a, const size_t * __restrict__ out) {

    std::vector<int> vec;
    for (auto i=0; i<a_dim; ++i) {
        for (auto j=0; j<out_dim; ++j) {
            if (a[i]==out[j]) {
                vec.push_back(j);
            }
        }
    }
    return vec;
}


template<class T, class U>
struct extractor_contract_2 {};

template<size_t ... Idx0, size_t ... Idx1>
struct extractor_contract_2<Index<Idx0...>, Index<Idx1...>> {
  template<typename T, size_t ... Rest0, size_t ... Rest1>
    static
    typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
                              typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type
    contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) {

//        Tensor<T,Rest0...,Rest1...> out;
//        out.zeros();
        T *a_data = a.data();
        T *b_data = b.data();
//        T *out_data = out.data();

        constexpr int a_dim = sizeof...(Rest0);
        constexpr int b_dim = sizeof...(Rest1);
        constexpr int out_dim = no_of_unique<Rest0...,Rest1...>::value;
//        constexpr std::array<int,a_dim> maxes_a = {Rest0...};
//        constexpr std::array<int,b_dim> maxes_b = {Rest1...};
        constexpr size_t maxes_a[a_dim] = {Rest0...};
        constexpr size_t maxes_b[b_dim] = {Rest1...};
        using unique_indices = apply_typelist_t<quote_c<size_t, Index>,
            uniq_t<typelist_c<size_t, Rest0...,Rest1...>>>;
        const size_t *maxes_out = unique_indices::_IndexHolder;
//        const int *maxes_out = static_cast<const int[5]>(unique_indices::_IndexHolder);
//        const size_t maxes_out[out_dim] = unique_indices::_IndexHolder;
//        print(type_name<loop_type>());
//        size_t *maxes_out = loop_type::_IndexHolder;
//        print(maxes_out);
//        print(type_name<decltype(loop_type::_IndexHolder)>());
//        print(maxes_out);
//        constexpr std::array<int,out_dim> maxes_out = {Rest0...,Rest1...};

        std::array<int,a_dim> products_a; products_a[0]=0;
        for (int j=a_dim-1; j>0; --j) {
            int num = maxes_a[a_dim-1];
            for (int k=0; k<j-1; ++k) {
                num *= maxes_a[a_dim-1-k-1];
            }
            products_a[j] = num;
        }
        std::array<int,b_dim> products_b; products_b[0]=0;
        for (int j=b_dim-1; j>0; --j) {
            int num = maxes_b[b_dim-1];
            for (int k=0; k<j-1; ++k) {
                num *= maxes_b[b_dim-1-k-1];
            }
            products_b[j] = num;
        }
        std::array<int,out_dim> products_out; products_out[0]=0;
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
//        print(products_a,products_b,products_out);

        int as[out_dim];
        std::fill(as,as+out_dim,0);

        std::vector<int> vec_a = which_where<a_dim,out_dim>(maxes_a,maxes_out);
        std::vector<int> vec_b = which_where<b_dim,out_dim>(maxes_b,maxes_out);
////        int it,jt;

//        while(true)
//        {
//            int index_a = as[a_dim-1];
//            for(it = 0; it< a_dim; it++) {
//                index_a += products_a[it]*as[it];
//            }
//            int index_b = as[out_dim-1];
//            for(it = a_dim; it< out_dim; it++) {
//                index_b += products_b[it-a_dim]*as[it];
//            }
//            int index_out = as[out_dim-1];
//            for(it = 0; it< out_dim; it++) {
//                index_out += products_out[it]*as[it];
//            }
//    //        out_data[index_out] += a_data[index_a]*b_data[index_b];
//            out_data[index_out] = a_data[index_a]*b_data[index_b];
//    //        if (as)
//    //        auto dd = _mm_mul_sd(_mm_load_sd(a_data+index_a),_mm_load_sd(b_data+index_b));
//    //        _mm_store_sd(out_data+index_out,dd);

//    //        std::cout << a_data[index_a] << " " << b_data[index_b] << "  " << out_data[index_out] <<  "\n";
//    //        std::cout << index_a << " " << index_b << "  " << index_out <<  "\n";

//    //        for(it = 0; it< out_dim; it++) {
//    //            std::cout << as[it] << " ";
//    //        }
//    //        print();
//    //        counter++;
//    //        if (counter % 4 ==0) {
//    //            for(it = 0; it< out_dim; it++) {
//    //                std::cout << as[it] << " ";
//    //            }
//    //            print();
//    //        }

//            for(jt = out_dim-1 ; jt>=0 ; jt--)
//            {
//                if(++as[jt]<maxes_out[jt])
//                    break;
//                else
//                    as[jt]=0;
//            }
//            if(jt<0)
//                break;
//        }

    }

};

template<class Index_I, class Index_J,
         typename T, size_t ... Rest0, size_t ... Rest1>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b)
-> decltype(extractor_contract_2<Index_I,Index_J>::contract_impl(a,b)) {
    return extractor_contract_2<Index_I,Index_J>::contract_impl(a,b);
}



/////////////////////////





//template<size_t ... rest>
//struct extract_tensor {

//};


//template<class T, T ... Rest>
//struct extractor_contract {};
template<class T, class U, class V>
struct extractor_contract {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2>
struct extractor_contract<Index<Idx0...>, Index<Idx1...>, Index<Idx2...> > {
  template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2>
    static
    typename contraction_impl<Index<Idx0...,Idx1...,Idx2...>, Tensor<T,Rest0...,Rest1...,Rest2...>,
                              typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+sizeof...(Rest2)>::type>::type
    contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c) {

        // Perform depth-first search
        // first two tensors contracted first
        using loop_type_0 = apply_typelist_t<quote_c<size_t, Index>,
                    uniq_t<typelist_c<size_t, Rest0...,Rest1...>>>;
        using contract_0 = typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
                                  typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type;
        using loop_type_1 = typename find_loop_type<Index<Rest2...>,contract_0>::type;
        constexpr size_t flop_count_0 = prod_index<loop_type_0>::value + prod_index<loop_type_1>::value;
        // second and third tensors contracted first
        using loop_type_2 = apply_typelist_t<quote_c<size_t, Index>,
                    uniq_t<typelist_c<size_t, Rest1...,Rest2...>>>;
        using contract_1 = typename contraction_impl<Index<Idx1...,Idx2...>, Tensor<T,Rest1...,Rest2...>,
                                  typename std_ext::make_index_sequence<sizeof...(Rest1)+sizeof...(Rest2)>::type>::type;
        using loop_type_3 = typename find_loop_type<Index<Rest0...>,contract_1>::type;
        constexpr size_t flop_count_1 = prod_index<loop_type_2>::value + prod_index<loop_type_3>::value;
        // first and third tensors contracted first
        using loop_type_4 = apply_typelist_t<quote_c<size_t, Index>,
                    uniq_t<typelist_c<size_t, Rest0...,Rest2...>>>;
        using contract_2 = typename contraction_impl<Index<Idx0...,Idx2...>, Tensor<T,Rest0...,Rest2...>,
                                  typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest2)>::type>::type;
        using loop_type_5 = typename find_loop_type<Index<Rest1...>,contract_2>::type;
        constexpr size_t flop_count_2 = prod_index<loop_type_4>::value + prod_index<loop_type_5>::value;
        // all tensors contracted at once
        using loop_type_6 = apply_typelist_t<quote_c<size_t, Index>,
                    uniq_t<typelist_c<size_t, Rest1...,Rest1...,Rest2...>>>;
        constexpr size_t flop_count_3 = prod_index<loop_type_6>::value;

//        print(flop_count_0,flop_count_1,flop_count_2,flop_count_3);
//        print(meta_min<flop_count_0,flop_count_1,flop_count_2,flop_count_3>::value);
//        print(meta_argmin<flop_count_0,flop_count_1,flop_count_2,flop_count_3>::value);

//        print(meta_argmin<40,30,20,6>::value);

        constexpr int which_variant = meta_argmin<flop_count_0,flop_count_1,flop_count_2,flop_count_3>::value;
//        print(which_variant);

        if (which_variant == 0) {
//            return contraction<Index<>>(contraction(a,b),c);
//            return
            contraction<Index<Idx0...>,Index<Idx1...>>(a,b);
        }
        else if (which_variant == 1) {
//            return contraction(a,contraction(b,c));
        }
        else if (which_variant == 2) {
//            return contraction(contraction(a,c),b);
        }
        else {
            // actual implementation goes here
        }



    }

};

template<class Index_I, class Index_J, class Index_K,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c)
-> decltype(extractor_contract<Index_I,Index_J,Index_K>::contract_impl(a,b,c)) {
    return extractor_contract<Index_I,Index_J,Index_K>::contract_impl(a,b,c);
}


}

#endif // CONTRACTION_H

