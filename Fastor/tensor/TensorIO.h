#ifndef TENSOR_PRINT_H
#define TENSOR_PRINT_H

#include "Fastor/tensor/Tensor.h"

namespace Fastor {


namespace internal {

template<typename T>
using std_matrix = typename std::vector<std::vector<T>>::type;

// Generate combinations
template<size_t M, size_t N, size_t ... Rest>
FASTOR_INLINE std::vector<std::vector<int>> index_generator() {
    // Do NOT change int to size_t, comparison overflows
    std::vector<std::vector<int>> idx; idx.resize(pack_prod<M,N,Rest...>::value);
    std::array<int,sizeof...(Rest)+2> maxes = {M,N,Rest...};
    std::array<int,sizeof...(Rest)+2> a;
    int i,j;
    std::fill(a.begin(),a.end(),0);

    auto counter=0;
    while(1)
    {
        std::vector<int> current_idx; //current_idx.reserve(sizeof...(Rest)+2);
        for(i = 0; i< sizeof...(Rest)+2; i++) {
            current_idx.push_back(a[i]);
        }
        idx[counter] = current_idx;
        counter++;
        for(j = sizeof...(Rest)+2-1 ; j>=0 ; j--)
        {
            if(++a[j]<maxes[j])
                break;
            else
                a[j]=0;
        }
        if(j<0)
            break;
    }
    return idx;
}


template<typename T>
int get_row_width(const std::ostream &os, const T *a_data, size_t size) {
    // compute the largest width
    int width = 0;
    for(size_t j = 0; j < size; ++j)
    {
        std::stringstream sstr;
        sstr.copyfmt(os);
        sstr << a_data[j];
        width = std::max<int>(width, int(sstr.str().length()));
    }
    return width;
}


template<template<typename,size_t...> class t_type, typename T, size_t ...Rest>
int get_row_width(const std::ostream &os, const t_type<T,Rest...>& a) {
    int width = 0;
    for(size_t j = 0; j < pack_prod<Rest...>::value; ++j)
    {
        std::stringstream sstr;
        sstr.copyfmt(os);
        sstr << a.eval_s(j);
        width = std::max<int>(width, int(sstr.str().length()));
    }
    return width;
}

} // end of namespace internal


#define FASTOR_MAKE_OS_STREAM_TENSOR0(t_type) \
template<typename T>\
FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, const t_type<T> &a) {\
    IOFormat fmt = FASTOR_DEFINE_IO_FORMAT;\
    os.precision(fmt._precision);\
    os << *a.data();\
    return os;\
}\

#define FASTOR_MAKE_OS_STREAM_TENSOR1(t_type) \
template<typename T, size_t M> \
FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, const t_type<T,M> &a) {\
    IOFormat fmt = FASTOR_DEFINE_IO_FORMAT;\
    os.precision(fmt._precision);\
    int width = internal::get_row_width(os, a);\
    for(size_t i = 0; i < M; ++i)\
    {\
        os << fmt._rowprefix;\
        if(width) os.width(width);\
        os << a(i);\
        os << fmt._rowsuffix;\
        os << fmt._rowsep;\
    }\
    return os;\
}\

#define FASTOR_MAKE_OS_STREAM_TENSOR2(t_type) \
template<typename T, size_t M, size_t N> \
FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, const t_type<T,M,N> &a) { \
    IOFormat fmt = FASTOR_DEFINE_IO_FORMAT; \
    os.precision(fmt._precision); \
    int width = internal::get_row_width(os, a); \
    for(size_t i = 0; i < M; ++i) \
    {\
        os << fmt._rowprefix;\
        if(width) os.width(width);\
        os << a(i, 0);\
        for(size_t j = 1; j < N; ++j)\
        {\
            os << fmt._colsep;\
            if(width) os.width(width);\
            os << a(i, j);\
        }\
        os << fmt._rowsuffix;\
        if( i < M - 1)\
            os << fmt._rowsep;\
    }\
    return os;\
}\


#define FASTOR_MAKE_OS_STREAM_TENSORn(t_type) \
template<typename T, size_t ... Rest, typename std::enable_if<sizeof...(Rest)>=3,bool>::type=0> \
FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, const t_type<T,Rest...> &a) {\
    IOFormat fmt = FASTOR_DEFINE_IO_FORMAT;\
    constexpr std::array<int,sizeof...(Rest)> DimensionHolder = {Rest...};\
    constexpr int M = get_value<sizeof...(Rest)-1,Rest...>::value;\
    constexpr int N = get_value<sizeof...(Rest),Rest...>::value;\
    constexpr int lastrowcol = M*N;\
    constexpr size_t prods = pack_prod<Rest...>::value;\
    std::vector<std::vector<int>> combs = internal::index_generator<Rest...>();\
    os.precision(fmt._precision);\
    int width = internal::get_row_width(os, a);\
    for (size_t dims=0; dims<prods/M/N; ++dims) {\
        if (fmt._print_dimensions)\
        {\
            os << "[";\
            for (size_t kk=0; kk<sizeof...(Rest)-2; ++kk) {\
                os << combs[lastrowcol*dims][kk] << ",";\
            }\
            os << ":,:]\n";\
        }\
        else {\
            if (dims)\
                os << "\n";\
        }\
        for(size_t i = 0; i < DimensionHolder[a.Dimension-2]; ++i)\
        {\
            os << fmt._rowprefix;\
            if(width) os.width(width);\
            os << a.eval_s(i*DimensionHolder[a.Dimension-1]+0+lastrowcol*dims);\
            for(int j = 1; j < DimensionHolder[a.Dimension-1]; ++j)\
            {\
                os << fmt._colsep;\
                if(width) os.width(width);\
                os << a.eval_s(i*DimensionHolder[a.Dimension-1]+j+lastrowcol*dims);\
            }\
            os << fmt._rowsuffix + fmt._rowsep;\
        }\
    }\
    return os;\
}\

FASTOR_MAKE_OS_STREAM_TENSOR0(Tensor)
FASTOR_MAKE_OS_STREAM_TENSOR1(Tensor)
FASTOR_MAKE_OS_STREAM_TENSOR2(Tensor)
FASTOR_MAKE_OS_STREAM_TENSORn(Tensor)


}

#endif // TENSOR_PRINT_H

