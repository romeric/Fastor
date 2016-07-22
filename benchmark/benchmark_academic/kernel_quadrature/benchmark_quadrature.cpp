#include <Fastor.h>
#include <vector>
#include <fstream>
#include <sstream>

using namespace Fastor;
using real = double;

namespace sv {

template<typename T, size_t M, size_t N, 
    typename std::enable_if<M==3 && N==3,bool>::type=0>
inline Tensor<T,M,N> cofactor(const Tensor<T,M,N> &a, const Tensor<T,M,N> &b) {
    constexpr T levi_civita[27] = { 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,
                                0., -1.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.};
    Tensor<T,M,N> out;
    constexpr size_t size = N;
    for (size_t i=0; i<N; ++i)
        for (size_t j=0; j<N; ++j)
            for (size_t k=0; k<N; ++k)
                for (size_t I=0; I<N; ++I)
                    for (size_t J=0; J<N; ++J)
                        for (size_t K=0; K<N; ++K)
                            out(i,I) += 0.5*levi_civita[i*size*size+j*size+k]*levi_civita[I*size*size+J*size+K]*a(j,J)*b(k,K);
    return out;
}

template<typename T, size_t M, size_t N, 
    typename std::enable_if<M==2 && N==2,bool>::type=0>
inline Tensor<T,M+1,N+1> cofactor(const Tensor<T,M,N> &a, const Tensor<T,M,N> &b) {
    constexpr T levi_civita[27] = { 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,
                                0., -1.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.};

    Tensor<T,3,3> a3d; a.zeros();
    Tensor<T,3,3> b3d; b.zeros();

    for (size_t i=0; i<2; ++i)
        for (size_t j=0; j<2; ++j)
            a3d(i,j) = a(i,j);
            b3d(i,j) = b(i,j); 

    Tensor<T,M+1,N+1> out;
    constexpr size_t size = N+1;
    for (size_t i=0; i<N; ++i)
        for (size_t j=0; j<N; ++j)
            for (size_t k=0; k<N; ++k)
                for (size_t I=0; I<N; ++I)
                    for (size_t J=0; J<N; ++J)
                        for (size_t K=0; K<N; ++K)
                            out(i,I) += 0.5*levi_civita[i*size*size+j*size+k]*levi_civita[I*size*size+J*size+K]*a3d(j,J)*b3d(k,K);

    return out;
}


template<typename T, size_t M, size_t N, size_t K>
inline Tensor<T,M,N> matmul(const Tensor<T,M,K> &a, const Tensor<T,K,N> &b) {
    Tensor<T,M,N> out;
    constexpr size_t size = N;
    for (size_t i=0; i<M; ++i)
        for (size_t j=0; j<K; ++j)
            for (size_t k=0; k<N; ++k)
                out(i,k) += a(i,j)*b(j,k);
    return out;
}


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


void MooneyRivlin() {

}


int main() {

    std::string efile = "/home/roman/Dropbox/Fastor/benchmark/benchmark_academic/kernel_quadrature/meshes/course_mesh_2d_elements.dat";
    std::string pfile = "/home/roman/Dropbox/Fastor/benchmark/benchmark_academic/kernel_quadrature/meshes/course_mesh_2d_points.dat";
    std::string gfile0 = "/home/roman/Dropbox/Fastor/benchmark/benchmark_academic/kernel_quadrature/meshes/p2_2d_Jm.dat";
    std::string gfile1 = "/home/roman/Dropbox/Fastor/benchmark/benchmark_academic/kernel_quadrature/meshes/p2_2d_AllGauss.dat";


    constexpr int ndim          = 2;
    constexpr int nnode         = 173; 
    constexpr int nelem         = 75;
    constexpr int nodeperelem   = 6;
    constexpr int ngauss        = 4;


    auto elements = loadtxt<int,nelem,nodeperelem>(efile);
    auto points = loadtxt<real,nnode,ndim>(pfile);

    // Fill with uniformly distributed random values
    decltype(points) Eulerx; Eulerx.random();
    // print(elements);
    // print(points);

    // Tensor<real,ndim, nodeperelem, ngauss> Jm;
    Tensor<real,ndim*nodeperelem*ngauss,1> Jm_temp = loadtxt<real,ndim*nodeperelem*ngauss,1>(gfile0);
    Tensor<real,ndim,nodeperelem,ngauss> Jm;
    std::copy(Jm_temp.data(),Jm_temp.data()+Jm_temp.Size,Jm.data()); 

    Tensor<real,ngauss,1> AllGauss = loadtxt<real,ngauss,1>(gfile1);


    for (int elem=0; elem<nelem; ++elem) {
        Tensor<real,nodeperelem,ndim> LagrangeElemCoords;
        Tensor<real,nodeperelem,ndim> EulerElemCoords;

        for (int i=0; i<nodeperelem; ++i) {
            for (int j=0; j<ndim; ++j) {
                LagrangeElemCoords(i,j) = points(elements(elem,i),j);
                EulerElemCoords(i,j) = points(elements(elem,i),j);
            }
        }
        // print(LagrangeElemCoords);

        for (int g=0; g<ngauss; ++g) {
            // Get Gauss point Jm
            Tensor<real,ndim,nodeperelem> Jm_g;
            for (int i=0; i<ndim; ++i)
                for (int j=0; j<ndim; ++j)
                    Jm_g(i,j) = Jm(i,j,g);

            // Compute gradient of shape functions
            auto ParentGradientX = matmul(Jm_g,LagrangeElemCoords);
            // Compute material gradient
            auto MaterialGradient = matmul(inverse(ParentGradientX),Jm_g);

            // Compute the deformation gradient tensor
            auto F = matmul(MaterialGradient,EulerElemCoords);
            // Compute H
            auto H = cofactor(F);
            // Comput J
            auto J = determinant(F);

            // Compute work-conjugates

            // Compute first Piola-Kirchhoff stress tensor

            // Compute Hessian


            // dump

            // print(type_name<decltype(MaterialGradient)>());
            // print(type_name<decltype(F)>());

        }
    }

    return 0;
}