#ifndef HELPER_H
#define HELPER_H

#include <Fastor.h>
#include <vector>
#include <fstream>
#include <sstream>

#ifndef _WIN32
#include <sys/resource.h>
#endif

using namespace Fastor;
using real = double;


template<typename T>
T random() {
    return (T)rand()/RAND_MAX;
}



template<typename T, size_t rows, size_t cols>
Tensor<T,rows,cols> loadtxt(const std::string &filename)
{
    // Read to a Tensor

    T temp;
    std::ifstream datafile;
    datafile.open(filename.c_str());

    if(!datafile)
    {
        warn("Unable to read file");
    }

    Tensor<T,rows,cols> out_arr;

    for (int row=0; row<rows; ++row) {
        for (int col=0; col<cols; ++col) {
            datafile >> temp;
            out_arr(row,col) = temp;        
        }
    }

    datafile.close();

    return out_arr;
}



template<typename T, size_t M, size_t N, size_t O, size_t P, typename std::enable_if<M==N && M==2,bool>::type=0>
static inline Tensor<T,M,N,O,P> eye() {
    Tensor<T,M,N,O,P> out; out.zeros();
    out(0,0,0,0) = 1.0;
    out(0,0,1,1) = 1.0;
    out(1,1,0,0) = 1.0;
    out(1,1,1,1) = 1.0;
    return out;
}


template<typename T, size_t M, size_t N, size_t O, size_t P, typename std::enable_if<M==N && M==3,bool>::type=0>
static inline Tensor<T,M,N,O,P> eye() {
    Tensor<T,M,N,O,P> out; out.zeros();
    out(0,0,0,0) = 1.0;
    out(0,0,1,1) = 1.0;
    out(0,0,2,2) = 1.0;
    out(1,1,0,0) = 1.0;
    out(1,1,1,1) = 1.0;
    out(1,1,2,2) = 1.0;
    out(2,2,0,0) = 1.0;
    out(2,2,1,1) = 1.0;
    out(2,2,2,2) = 1.0;
    return out;
}


template<typename T, size_t M, size_t N, typename std::enable_if<M==N && M==3,bool>::type=0>
static inline Tensor<T,M,N> eye2() {
    Tensor<T,M,N> out; out.zeros();
    out(0,0) = 1.0;
    out(1,1) = 1.0;
    out(2,2) = 1.0;
    return out;
}

template<typename T, size_t M, size_t N, typename std::enable_if<M==N && M==2,bool>::type=0>
static inline Tensor<T,M,N> eye2() {
    Tensor<T,M,N> out; out.zeros();
    out(0,0) = 1.0;
    out(1,1) = 1.0;
    return out;
}


#endif // HELPER_H