#ifndef _LEGENDRETRANSFORM__H
#define _LEGENDRETRANSFORM__H

#include "_MaterialBase_.h"

template<typename U>
class _LegendreTransform_ {
public:
   
    U newton_raphson_max_iter;
    U newton_raphson_tolerance;

    _LegendreTransform_() {
        newton_raphson_tolerance = 1e-7;
        newton_raphson_max_iter = 50;
    }

    _LegendreTransform_(U tolerance, int max_iter) {
        newton_raphson_tolerance = tolerance;
        newton_raphson_max_iter = max_iter;
    }

    void SetParameters(U tolerance, int max_iter) {
        newton_raphson_tolerance = tolerance;
        newton_raphson_max_iter = max_iter;
    }

    template<typename T=U, size_t N, template<typename> class Material>
    FASTOR_INLINE Tensor<T,N>
    QuadratureNewtonRaphson(const Material<T> &material, const Tensor<T,N,N> &F, const Tensor<T,N> &E) {

        // Assume that electric displacement is zero
        Tensor<T,N> D, deltaD; D.zeros();
        // Initial Residual
        Tensor<T,N> Residual = -E;
        // Norm of residual
        auto norm_forces = norm(Residual);
        if (std::fabs(norm_forces) < 1.0e-14) {
            norm_forces = 1.0e-14;
        }

        // Get the initial dielectric tensor 
        auto dielectric = material.DielectricTensor(F,D);
        int iter = 0;

        while (std::fabs(norm(Residual)/norm_forces) > newton_raphson_tolerance) {

            // SOLVE THE SYSTEM AND GET ITERATIVE D (deltaD)
            deltaD = solve(dielectric,static_cast<Tensor<T,N>>(-Residual));
            // UPDATE ELECTRIC DISPLACEMENT
            D += deltaD;
            // UPDATE DIELECTRIC TENSOR 
            dielectric = material.DielectricTensor(F,D);
            // RECOMPUTE RESIDUAL
            Residual = matmul(dielectric,D) - E;
            // STORE CONVERGENCE RESULT
            iter += 1;

            FASTOR_EXIT_ASSERT(iter < newton_raphson_max_iter,"Quadrature point based Newton-Raphson did not converge");

        }

        return D;
    }


    template<typename T=U, size_t N>
    FASTOR_INLINE
    typename ElectroMechanicsHessianType<T,N>::return_type
    InternalEnergyToEnthalpy(const Tensor<T,N,N,N,N> &W_elasticity, const Tensor<T,N,N,N> &W_coupling, const Tensor<T,N,N> &W_dielectric) {

        Tensor<T,N,N> H_dielectric = - inverse(W_dielectric);
        Tensor<T,N,N,N> H_coupling = - einsum<Index<k,l,i>,Index<i,j>>(W_coupling,H_dielectric);
        auto H_coupling_T = permutation<Index<k,j,i>>(H_coupling);
        Tensor<T,N,N,N,N> H_elasticity = W_elasticity - einsum<Index<i,j,k>,Index<k,l,m>>(W_coupling, H_coupling_T);

        // Make Hessian in Voigt form
        auto C_Voigt = voigt(H_elasticity);
        auto P_Voigt = voigt(H_coupling);

        return make_electromechanical_hessian(C_Voigt,P_Voigt,H_dielectric);
    }

};


#endif