#ifndef TENSOR_PRINT_H
#define TENSOR_PRINT_H

#include "tensor/Tensor.h"

namespace Fastor {

template<typename T>
std::ostream& operator<<(std::ostream &os, const Tensor<T> &a) {
    os.precision(9);
    os << *a.data();
    return os;
}

template<typename T, size_t M>
std::ostream& operator<<(std::ostream &os, const Tensor<T,M> &a) {

    os.precision(9);
//    auto &&w = std::setw(7);
    auto &&w = std::fixed;
    os << "⎡" << w << a(0) << " ⎤\n";
    for (size_t i = 1; i + 1 < M; ++i) {
        os << "⎢" << w << a(i) << " ⎥\n";
    }
    if (M > 1)
        os << "⎣" << w << a(M - 1) << " ⎦\n";

    return os;
}

template<typename T, size_t M, size_t N>
std::ostream& operator<<(std::ostream &os, const Tensor<T,M,N> &a) {

    os.precision(9);
//    auto &&w = std::setw(7);
    auto &&w = std::fixed;
    if (M>1) {
        os << "⎡" << w << a(0,0);
        for (size_t j = 1; j < N; ++j) {
            os << ", " << w << a(0,j);
        }
        os << " ⎤\n";
        for (size_t i = 1; i + 1 < M; ++i) {
            os << "⎢" << w << a(i,0);
            for (size_t j = 1; j < N; ++j) {
                os << ", " << w << a(i,j);
            }
            os << " ⎥\n";
        }
        os << "⎣" << w << a(M - 1,0);
        for (size_t j = 1; j < N; ++j) {
            os << ", " << w << a(M - 1,j);
        }
    }
    else {
        os << "⎡" << w << a(0,0);
        for (size_t j = 1; j < N; ++j) {
            os << ", " << w << a(0,j);
        }
    }

    return os << " ⎦\n";
}


template<typename T, size_t P, size_t M, size_t ... Rest,
         typename std::enable_if<sizeof...(Rest)==1,bool>::type=0>
std::ostream& operator<<(std::ostream &os, const Tensor<T,P,M,Rest...> &a) {
    constexpr size_t N = get_value<3,P,M,Rest...>::value;
    os.precision(9);
    auto &&w = std::fixed;
    for (size_t k=0; k<P; ++k) {
        os << "["<< k << ",:,:]\n";
        if (M>1) {
            os << "⎡" << w << a(k,0,0);
            for (size_t j = 1; j < N; ++j) {
                os << ", " << w << a(k,0,j);
            }
            os << " ⎤\n";
            for (size_t i = 1; i + 1 < M; ++i) {
                os << "⎢" << w << a(k,i,0);
                for (size_t j = 1; j < N; ++j) {
                    os << ", " << w << a(k,i,j);
                }
                os << " ⎥\n";
            }
            os << "⎣" << w << a(k,M - 1,0);
            for (size_t j = 1; j < N; ++j) {
                os << ", " << w << a(k,M - 1,j);
            }
        }
        else {
            os << "⎡" << w << a(k,0,0);
            for (size_t j = 1; j < N; ++j) {
                os << ", " << w << a(k,0,j);
            }
        }
        os << " ⎦\n";
    }

    return os;
}

template<typename T, size_t P, size_t Q, size_t ... Rest,
         typename std::enable_if<sizeof...(Rest)==2,bool>::type=0>
std::ostream& operator<<(std::ostream &os, const Tensor<T,Q,P,Rest...> &a) {
    constexpr size_t M = get_value<3,P,Q,Rest...>::value;
    constexpr size_t N = get_value<4,P,Q,Rest...>::value;
    os.precision(9);
    auto &&w = std::fixed;
    for (size_t l=0; l<Q; ++l) {
        for (size_t k=0; k<P; ++k) {
            os << "["<< l << "," << k << ",:,:]\n";
            os << "⎡" << w << a(l,k,0,0);
            for (size_t j = 1; j < N; ++j) {
                os << ", " << w << a(l,k,0,j);
            }
            os << " ⎤\n";
            for (size_t i = 1; i + 1 < M; ++i) {
                os << "⎢" << w << a(l,k,i,0);
                for (size_t j = 1; j < N; ++j) {
                    os << ", " << w << a(l,k,i,j);
                }
                os << " ⎥\n";
            }
            os << "⎣" << w << a(l,k,M - 1,0);
            for (size_t j = 1; j < N; ++j) {
                os << ", " << w << a(l,k,M - 1,j);
            }
            os << " ⎦\n";
        }
    }
    return os;
}



template<typename T>
using std_matrix = typename std::vector<std::vector<T>>::type;

// Generate combinations
template<size_t M, size_t N, size_t ... Rest>
FASTOR_INLINE std::vector<std::vector<int>> index_generator() {
    // Do NOT change int to size_t, comparison overflows
    std::vector<std::vector<int>> idx; idx.resize(prod<M,N,Rest...>::value);
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


template<typename T, size_t M, size_t N, size_t ... Rest,
         typename std::enable_if<sizeof...(Rest)>=3,bool>::type=0>
std::ostream& operator<<(std::ostream &os, const Tensor<T,M,N,Rest...> &a) {

    T *a_data = a.data();

    constexpr int DimensionHolder[sizeof...(Rest)+2] = {M,N,Rest...};
    int prods = 1;
    for (int i=0; i<a.Dimension-2; ++i) {
        prods *= DimensionHolder[i];
    }
    int lastrowcol = 1;
    for (int i=a.Dimension-2; i<a.Dimension; ++i) {
        lastrowcol *= DimensionHolder[i];
    }

    std::vector<std::vector<int>> combs = index_generator<M,N,Rest...>();
    os.precision(9);
    auto &&w = std::fixed;
    size_t dims_2d = DimensionHolder[a.Dimension-2]*DimensionHolder[a.Dimension-1];
    for (int dims=0; dims<prods; ++dims) {
        os << "[";
        for (size_t kk=0; kk<sizeof...(Rest); ++kk) {
            os << combs[dims_2d*dims][kk] << ",";
        }
        os << ":,:]\n";
        if (DimensionHolder[a.Dimension-2] > 1) {
            os << "⎡" << w << a_data[lastrowcol*dims];
            for (size_t j = 1; j < DimensionHolder[a.Dimension-1]; ++j) {
                os << ", " << w << a_data[j+lastrowcol*dims];
            }
            os << " ⎤\n";
            for (size_t i = 1; i + 1 < DimensionHolder[a.Dimension-2]; ++i) {
                os << "⎢" << w << a_data[i*DimensionHolder[a.Dimension-1]+lastrowcol*dims];
                for (size_t j = 1; j < DimensionHolder[a.Dimension-1]; ++j) {
                    os << ", " << w << a_data[i*DimensionHolder[a.Dimension-1]+j+lastrowcol*dims];
                }
                os << " ⎥\n";
            }
            os << "⎣" << w << a_data[(DimensionHolder[a.Dimension-2]-1)*DimensionHolder[a.Dimension-1]+lastrowcol*dims];
            for (size_t j = 1; j < DimensionHolder[a.Dimension-1]; ++j) {
                os << ", " << w << a_data[(DimensionHolder[a.Dimension-2]-1)*DimensionHolder[a.Dimension-1]+j+lastrowcol*dims];
            }
        }
        else {
            os << "⎡" << w << a_data[lastrowcol*dims];
            for (size_t j = 1; j < DimensionHolder[a.Dimension-1]; ++j) {
                os << ", " << w << a_data[j+lastrowcol*dims];
            }
        }
        os << " ⎦\n";
    }

    return os;
}

}

#endif // TENSOR_PRINT_H

