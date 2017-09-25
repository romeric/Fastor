#include "_MaterialBase_.h"
#include "_LegendreTransform_.h"

//! Numerical integration of the following material model
//          W(C,D0) = W_neo(C) + 1/2/eps_1 (FD0*FD0)

template<typename U>
class ElectromechanicalModel_0: public _MaterialBase_<U> {
public:
    U mu;
    U lamb;
    U eps_1;

    ElectromechanicalModel_0() = default;

    FASTOR_INLINE
    ElectromechanicalModel_0(U mu, U lamb, U eps_1) {
        this->mu = mu;
        this->lamb = lamb;
        this->eps_1 = eps_1;
    }

    FASTOR_INLINE
    void SetParameters(U mu, U lamb, U eps_1){
        this->mu = mu;
        this->lamb = lamb;
        this->eps_1 = eps_1;
    }


    template<typename T=U, size_t ndim>
    FASTOR_INLINE
    std::tuple<Tensor<T,ndim>,Tensor<T,ndim,ndim>, typename ElectroMechanicsHessianType<T,ndim>::return_type>
    _KineticMeasures_(const T *Fnp, const T *Enp) {


        // CREATE FASTOR TENSORS
        Tensor<T,ndim,ndim> F;
        Tensor<T,ndim> E;
        // COPY NUMPY ARRAY TO FASTOR TENSORS
        copy_numpy(F,Fnp);
        copy_numpy(E,Enp);

        // FIND THE KINEMATIC MEASURES
        // auto I = EYE2<T,ndim>();
        Tensor<T,ndim,ndim> I; I.eye();
        T J = determinant(F);
        // auto H = cofactor(F);
        auto b = matmul(F,transpose(F));

        // COMPUTE ELECTRIC DISPLACEMENT
        Tensor<T,ndim> D = (eps_1/J)*E;

        // COMPUTE CAUCHY STRESS TENSOR
        Tensor<T,ndim,ndim> sigma = mu/J*(b - I) + lamb*(J-1)*I + J/eps_1*outer(D,D);

        // FIND ELASTICITY TENSOR
        auto II_ijkl = outer(I,I);
        auto II_ikjl = permutation<Index<i,k,j,l>>(II_ijkl);
        auto II_iljk = permutation<Index<i,l,j,k>>(II_ijkl);

        Tensor<T,ndim,ndim,ndim,ndim> elasticity = lamb*(2.*J-1.)*II_ijkl + (mu/J - lamb*(J-1))*(II_ikjl+II_iljk);

        // FIND COUPLING TENSOR
        auto ID_ijk = outer(I,D);
        auto ID_ikj = permutation<Index<i,k,j>>(ID_ijk);
        auto ID_jki = permutation<Index<j,k,i>>(ID_ijk);

        Tensor<T,ndim,ndim,ndim> coupling =  J/eps_1*(ID_ikj + ID_jki);

        // FIND DIELELCTRIC TENSOR
        Tensor<T,ndim,ndim> dielectric = (J/eps_1)*I;

        // PERFORM LEGENDRE TRANSFORM
        auto legendre_transform = _LegendreTransform_<T>();
        auto hessian = legendre_transform.InternalEnergyToEnthalpy(elasticity,coupling,dielectric);

        auto kinetics = std::make_tuple(D,sigma,hessian);
        return kinetics;
    }


    template<typename T>
    void KineticMeasures(T* Dnp, T *Snp, T* Hnp, int ndim, int ngauss, const T *Fnp, const T *Enp);


};


template<> template<>
void ElectromechanicalModel_0<float>::KineticMeasures<float>(float *Dnp, float *Snp, float* Hnp,
    int ndim, int ngauss, const float *Fnp, const float *Enp) {

    if (ndim==3) {
        Tensor<float,3> D;
        Tensor<float,3,3> stress;
        Tensor<float,9,9> hessian;
        for (int g=0; g<ngauss; ++g) {
            std::tie(D,stress,hessian) =_KineticMeasures_<float,3>(Fnp+9*g, Enp+3*g);
            copy_fastor(Dnp,D,g*3);
            copy_fastor(Snp,stress,g*9);
            copy_fastor(Hnp,hessian,g*81);
        }
    }
    else if (ndim==2) {
        Tensor<float,2> D;
        Tensor<float,2,2> stress;
        Tensor<float,5,5> hessian;
        for (int g=0; g<ngauss; ++g) {
            std::tie(D,stress,hessian) =_KineticMeasures_<float,2>(Fnp+4*g, Enp+2*g);
            copy_fastor(Dnp,D,g*2);
            copy_fastor(Snp,stress,g*4);
            copy_fastor(Hnp,hessian,g*25);
        }
    }
}


template<> template<>
void ElectromechanicalModel_0<double>::KineticMeasures<double>(double *Dnp, double *Snp, double* Hnp,
    int ndim, int ngauss, const double *Fnp, const double *Enp) {

    if (ndim==3) {
        Tensor<double,3> D;
        Tensor<double,3,3> stress;
        Tensor<double,9,9> hessian;
        for (int g=0; g<ngauss; ++g) {
            std::tie(D,stress,hessian) =_KineticMeasures_<double,3>(Fnp+9*g, Enp+3*g);
            copy_fastor(Dnp,D,g*3);
            copy_fastor(Snp,stress,g*9);
            copy_fastor(Hnp,hessian,g*81);
        }
    }
    else if (ndim==2) {
        Tensor<double,2> D;
        Tensor<double,2,2> stress;
        Tensor<double,5,5> hessian;
        for (int g=0; g<ngauss; ++g) {
            std::tie(D,stress,hessian) =_KineticMeasures_<double,2>(Fnp+4*g, Enp+2*g);
            copy_fastor(Dnp,D,g*2);
            copy_fastor(Snp,stress,g*4);
            copy_fastor(Hnp,hessian,g*25);
        }
    }
}
