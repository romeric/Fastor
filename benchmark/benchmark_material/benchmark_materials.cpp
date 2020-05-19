#include <ElectromechanicalModel_0.h>


template<typename T>
void em_0() {

    const T mu = 1.0e5;
    const T lamb = 1.0e5;
    const T eps_1 = 1.28e-11;
    auto material = ElectromechanicalModel_0<T>(mu,lamb,eps_1);

    const int ngauss = 100;
    const int ndim = 3;
    T *Enp = (T*) malloc(ngauss*ndim*sizeof(T));
    T *Dnp = (T*) malloc(ngauss*ndim*sizeof(T));
    T *Fnp = (T*) malloc(ngauss*ndim*ndim*sizeof(T));
    T *Snp = (T*) malloc(ngauss*ndim*ndim*sizeof(T));
    T *Hnp = (T*) malloc(ngauss*ndim*ndim*ndim*ndim*sizeof(T));

    std::iota(Enp,Enp+ngauss*ndim,2.);
    std::iota(Fnp,Fnp+ngauss*ndim*ndim,2.);

    material.KineticMeasures(Dnp,Snp,Hnp,ndim,ngauss,Fnp,Enp);

    free(Enp);
    free(Dnp);
    free(Fnp);
    free(Snp);
    free(Hnp);
}


template<typename T>
void run() {
    timeit(static_cast<void (*)()>(&em_0<T>));
}


int main() {

    print(FBLU(BOLD("Numerical intergration time: single precision")));
    run<float>();
    print(FBLU(BOLD("Numerical intergration time: double precision")));
    run<double>();
    return 0;
}
