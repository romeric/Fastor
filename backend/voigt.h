#ifndef VOIGT_H
#define VOIGT_H

#include "tensor/Tensor.h"

namespace Fastor {

template<typename T, size_t ... Rest>
struct VoigtType;
template<typename T>
struct VoigtType<T,2,2,2,2> {
    using return_type = Tensor<T,3,3>;
};
template<typename T>
struct VoigtType<T,3,3,3,3> {
    using return_type = Tensor<T,6,6>;
};
template<typename T>
struct VoigtType<T,2,2,2> {
    using return_type = Tensor<T,3,2>;
};
template<typename T>
struct VoigtType<T,3,3,3> {
    using return_type = Tensor<T,6,3>;
};


template<typename T, size_t ... Rest,
         typename std::enable_if<sizeof...(Rest)==4 && prod<Rest...>::value == 16
                                 ,bool>::type=0>
FASTOR_INLINE void _voigt(const T * __restrict__ a_data, T * __restrict__ VoigtA) {
    VoigtA[0] = a_data[0];
    VoigtA[1] = a_data[3];
    VoigtA[2] = 0.5*(a_data[1]+a_data[2]);
    VoigtA[3] = VoigtA[1];
    VoigtA[4] = a_data[15];
    VoigtA[5] = 0.5*(a_data[13]+a_data[14]);
    VoigtA[6] = VoigtA[2];
    VoigtA[7] = VoigtA[5];
    VoigtA[8] = 0.5*(a_data[5]+a_data[6]);
}


template<typename T, size_t ... Rest,
         typename std::enable_if<sizeof...(Rest)==4 && prod<Rest...>::value == 81
                                 ,bool>::type=0>
FASTOR_INLINE void _voigt(const T * __restrict__ a_data, T * __restrict__ VoigtA) {

    VoigtA[0] = a_data[0];
    VoigtA[1] = a_data[4];
    VoigtA[2] = a_data[8];
    VoigtA[3] = 0.5*(a_data[1]+a_data[3]);
    VoigtA[4] = 0.5*(a_data[2]+a_data[6]);
    VoigtA[5] = 0.5*(a_data[5]+a_data[7]);
    VoigtA[6] = VoigtA[1];
    VoigtA[7] = a_data[40];
    VoigtA[8] = a_data[44];
    VoigtA[9] = 0.5*(a_data[37]+a_data[39]);
    VoigtA[10] = 0.5*(a_data[38]+a_data[42]);
    VoigtA[11] = 0.5*(a_data[41]+a_data[43]);
    VoigtA[12] = VoigtA[2];
    VoigtA[13] = VoigtA[8];
    VoigtA[14] = a_data[80];
    VoigtA[15] = 0.5*(a_data[73]+a_data[75]);
    VoigtA[16] = 0.5*(a_data[74]+a_data[78]);
    VoigtA[17] = 0.5*(a_data[77]+a_data[79]);
    VoigtA[18] = VoigtA[3];
    VoigtA[19] = VoigtA[9];
    VoigtA[20] = VoigtA[15];
    VoigtA[21] = 0.5*(a_data[10]+a_data[12]);
    VoigtA[22] = 0.5*(a_data[11]+a_data[15]);
    VoigtA[23] = 0.5*(a_data[14]+a_data[16]);
    VoigtA[24] = VoigtA[4];
    VoigtA[25] = VoigtA[10];
    VoigtA[26] = VoigtA[16];
    VoigtA[27] = VoigtA[22];
    VoigtA[28] = 0.5*(a_data[20]+a_data[24]);
    VoigtA[29] = 0.5*(a_data[23]+a_data[25]);
    VoigtA[30] = VoigtA[5];
    VoigtA[31] = VoigtA[11];
    VoigtA[32] = VoigtA[17];
    VoigtA[33] = VoigtA[23];
    VoigtA[34] = VoigtA[29];
    VoigtA[35] = 0.5*(a_data[50]+a_data[52]);
}


template<typename T, size_t ... Rest,
         typename std::enable_if<sizeof...(Rest)==3 && prod<Rest...>::value == 8
                                 ,bool>::type=0>
FASTOR_INLINE void _voigt(const T * __restrict__ a_data, T * __restrict__ VoigtA) {
    // TODO
    VoigtA[0] = a_data[0];
}


template<typename T, size_t ... Rest,
         typename std::enable_if<sizeof...(Rest)==3 && prod<Rest...>::value == 27
                                 ,bool>::type=0>
FASTOR_INLINE void _voigt(const T * __restrict__ a_data, T * __restrict__ VoigtA) {
    // TODO
    VoigtA[0] = a_data[0];
}


template<typename T, size_t ... Rest>
FASTOR_INLINE auto voigt(const Tensor<T,Rest...> &a)
-> typename VoigtType<T,Rest...>::return_type {
    T *a_data = a.data();
    using ret_type = typename VoigtType<T,Rest...>::return_type;
    ret_type voigt_a;
    T *VoigtA = voigt_a.data();
    _voigt<T,Rest...>(a_data,VoigtA);

    return voigt_a;
}

}

#endif // VOIGT_H

