#ifndef SINGLEVALUE_TENSOR_H
#define SINGLEVALUE_TENSOR_H


#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/TensorIO.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/meta/tensor_meta.h"
#include <limits>

namespace Fastor {

template<typename T, size_t ...Rest>
class SingleValueTensor : public AbstractTensor<SingleValueTensor<T,Rest...>,sizeof...(Rest)> {
public:
    using scalar_type      = T;
    using simd_vector_type = choose_best_simd_vector_t<T>;
    using simd_abi_type    = typename simd_vector_type::abi_type;
    using result_type      = SingleValueTensor<T,Rest...>;
    using dimension_t      = std::integral_constant<FASTOR_INDEX, sizeof...(Rest)>;
    static constexpr FASTOR_INLINE FASTOR_INDEX rank() {return sizeof...(Rest);}
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return pack_prod<Rest...>::value;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX dim) const {
#if FASTOR_SHAPE_CHECK
        FASTOR_ASSERT(dim>=0 && dim < sizeof...(Rest), "TENSOR SHAPE MISMATCH");
#endif
        const FASTOR_INDEX DimensionHolder[sizeof...(Rest)] = {Rest...};
        return DimensionHolder[dim];
    }

    template<typename U=int>
    SingleValueTensor(U num) : _data{(T)num} {}
    SingleValueTensor(const SingleValueTensor<T,Rest...> &a) : _data{(T)a.data()[0]} {}
    FASTOR_INLINE SingleValueTensor(const AbstractTensor<SingleValueTensor<T,Rest...>,sizeof...(Rest)>& src_) : _data{T(0)} {
    }

    constexpr FASTOR_INLINE T* data() const { return const_cast<T*>(this->_data.data());}

    // Index retriever
    //----------------------------------------------------------------------------------------------------------//
    template<typename U>
    FASTOR_INLINE int get_mem_index(U index) const {
#if FASTOR_BOUNDS_CHECK
        FASTOR_ASSERT((index>=0 && index<size()), "INDEX OUT OF BOUNDS");
#endif
        return index;
    }

    template<typename... Args, typename std::enable_if<sizeof...(Args)==dimension_t::value &&
                                is_arithmetic_pack<Args...>::value,bool>::type =0>
    FASTOR_INLINE int get_flat_index(Args ... args) const {
#if FASTOR_BOUNDS_CHECK
        int largs[sizeof...(Args)] = {args...};
        constexpr int DimensionHolder[dimension_t::value] = {Rest...};
        for (int i=0; i<dimension_t::value; ++i) {
            if (largs[i]==-1) largs[i] += DimensionHolder[i];
            assert( (largs[i]>=0 && largs[i]<DimensionHolder[i]) && "INDEX OUT OF BOUNDS");
        }
#endif
        return 0;
    }

    FASTOR_INLINE int get_flat_index(const std::array<int, dimension_t::value> &as) const {
#if FASTOR_BOUNDS_CHECK
        constexpr std::array<size_t,dimension_t::value> products_ = nprods_views<Index<Rest...>,
            typename std_ext::make_index_sequence<dimension_t::value>::type>::values;
        int index = 0;
        for (int i=0; i<dimension_t::value; ++i) {
            index += products_[i]*as[i];
        }
        FASTOR_ASSERT((index>=0 && index<size()), "INDEX OUT OF BOUNDS");
#endif
        return 0;
    }
    //----------------------------------------------------------------------------------------------------------//


    // Scalar indexing const
    //----------------------------------------------------------------------------------------------------------//
#undef SCALAR_INDEXING_CONST_H
    #include <Fastor/tensor/ScalarIndexing.h>
#define SCALAR_INDEXING_CONST_H
    //----------------------------------------------------------------------------------------------------------//

    // Expression templates evaluators
    //----------------------------------------------------------------------------------------------------------//
#undef TENSOR_EVALUATOR_H
    #include "Fastor/tensor/TensorEvaluator.h"
#define TENSOR_EVALUATOR_H
    //----------------------------------------------------------------------------------------------------------//

    // Tensor methods
    //----------------------------------------------------------------------------------------------------------//
#undef TENSOR_METHODS_CONST_H
    #include "Fastor/tensor/TensorMethods.h"
#define TENSOR_METHODS_CONST_H
    //----------------------------------------------------------------------------------------------------------//

    // Converters
    //----------------------------------------------------------------------------------------------------------//
#undef PODCONVERTERS_H
    #include "Fastor/tensor/PODConverters.h"
#define PODCONVERTERS_H
    //----------------------------------------------------------------------------------------------------------//

    // Cast method
    //----------------------------------------------------------------------------------------------------------//
    template<typename U>
    FASTOR_INLINE SingleValueTensor<U,Rest...> cast() const {
        SingleValueTensor<U,Rest...> out(static_cast<U>(_data[0]));
        return out;
    }
    //----------------------------------------------------------------------------------------------------------//

    //----------------------------------------------------------------------------------------------------------//
private:
    const FASTOR_ALIGN std::array<T,1> _data;
    //----------------------------------------------------------------------------------------------------------//
};

// template<typename T, size_t ...Rest>
// constexpr const T SingleValueTensor<T,Rest...>::_data[pack_prod<Rest...>::value];

template<typename T, size_t ... Rest>
struct tensor_type_finder<SingleValueTensor<T,Rest...>> {
    using type = SingleValueTensor<T,Rest...>;
};

template<typename T, size_t ... Rest>
struct scalar_type_finder<SingleValueTensor<T,Rest...>> {
    using type = T;
};


FASTOR_MAKE_OS_STREAM_TENSOR0(SingleValueTensor)
FASTOR_MAKE_OS_STREAM_TENSOR1(SingleValueTensor)
FASTOR_MAKE_OS_STREAM_TENSOR2(SingleValueTensor)
FASTOR_MAKE_OS_STREAM_TENSORn(SingleValueTensor)


template<typename T, size_t M, size_t N>
FASTOR_INLINE SingleValueTensor<T,N,M> transpose(const SingleValueTensor<T,M,N> &a) {
    return SingleValueTensor<T,N,M>(a(0,0));
}

template<typename T, size_t M>
T trace(const SingleValueTensor<T,M,M> &a) {
    return M*a(0,0);
}

template<typename T, size_t M>
FASTOR_INLINE T determinant(const SingleValueTensor<T,M,M> &a) {
    // determinant of a single value tensor is 0
    return 0.;
}

template<typename T, size_t M, size_t N>
FASTOR_INLINE double norm(const SingleValueTensor<T,M,N> &a) {
    return a(0,0)*std::sqrt(double(M*N));
}

template<typename T, size_t I>
FASTOR_INLINE Tensor<T,I,I> inverse(const SingleValueTensor<T,I,I> &a) {
    // A single value tensor is not invertible
    Tensor<T,I,I> out(std::numeric_limits<T>::quiet_NaN());
    return out;
}

template<typename T, size_t M, size_t K, size_t N>
FASTOR_INLINE Tensor<T,M,N> matmul(const Tensor<T,M,K> &a, const SingleValueTensor<T,K,N> &b) {

    using V = SIMDVector<T,DEFAULT_ABI>;

    Tensor<T,M,N> out;
    T *out_data = out.data();
    const T *a_data = a.data();
    const T b_value = b(0,0);

    for (size_t i=0; i<M; ++i) {
        V vec_out;
        size_t j=0;
        for (; j<ROUND_DOWN(K,V::Size); j+=V::Size) {
            vec_out += V(&a_data[i*K+j])*b_value;
        }
        T out_value = 0.;
        for (; j<K; j++) {
            out_value += a_data[i*K+j]*b_value;
        }
        out_value += vec_out.sum();
        V out_vec_value(out_value);

        j=0;
        for (; j<ROUND_DOWN(N,V::Size); j+=V::Size) {
            out_vec_value.store(&out_data[i*N+j],false);
        }
        for (; j<N; ++j) {
            out_data[i*N+j] = out_value;
        }
    }

    return out;
}

template<typename T, size_t M, size_t K, size_t N>
FASTOR_INLINE Tensor<T,M,N> matmul(const SingleValueTensor<T,M,K> &a, const Tensor<T,K,N> &b) {
    return transpose(matmul(transpose(b),transpose(a)));
}

template<typename T, size_t M, size_t K, size_t N>
FASTOR_INLINE SingleValueTensor<T,M,N> matmul(const SingleValueTensor<T,M,K> &a, const SingleValueTensor<T,K,N> &b) {

    const T a_value = a(0,0);
    const T b_value = b(0,0);
    // matmul is just this
    SingleValueTensor<T,M,N> out(a_value*b_value*K);

    // Not necessary
    // using V = SIMDVector<T,DEFAULT_ABI>;
    // V vec_out;
    // size_t j=0;
    // for (; j<ROUND_DOWN(K,V::Size); j+=V::Size) {
    //     vec_out = vec_out + V(a_value)*b_value;
    // }
    // T out_value = 0.;
    // for (; j<K; j++) {
    //     out_value += a_value*b_value;
    // }
    // out_value += vec_out.sum();
    // SingleValueTensor<T,M,N> out(out_value);

    return out;
}




// This one is almost like a compile time einsum
template<class Index_I, class Index_J, typename T, size_t ... Rest0, size_t ... Rest1>
FASTOR_INLINE
typename contraction_impl<typename concat_<Index_I,Index_J>::type,SingleValueTensor<T,Rest0...,Rest1...>,
                typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type
einsum(const SingleValueTensor<T,Rest0...> &a, const SingleValueTensor<T,Rest1...> &b) {

    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");

    std::array<size_t,Index_I::NoIndices> idx0; std::copy_n(Index_I::_IndexHolder,Index_I::NoIndices,idx0.begin());
    std::array<size_t,Index_J::NoIndices> idx1; std::copy_n(Index_J::_IndexHolder,Index_J::NoIndices,idx1.begin());
    std::array<size_t,Index_I::NoIndices> dims0 = {Rest0...};

    // n^2 but it is okay as this is a small loop with compile time spans
    size_t total = 1;
    for (int i=0; i<idx0.size(); ++i) {
        for (int j=0; j<idx1.size(); ++j) {
            if (idx0[i]==idx1[j]) {
                total *= dims0[i];
            }
        }
    }
    const T a_value = a.eval_s(0);
    const T b_value = b.eval_s(0);
    const T out_value = total*a_value*b_value;

    using OutTensor = typename contraction_impl<typename concat_<Index_I,Index_J>::type,SingleValueTensor<T,Rest0...,Rest1...>,
                typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type;
    OutTensor out(out_value);
    return out;
}


} // end of namespace Fastor


#endif // SINGLEVALUE_TENSOR_H
