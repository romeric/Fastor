#include <Fastor.h>
using namespace Fastor;

template<typename T>
void run() {

    timer<double> time_;
#ifdef ONE_D
    print(FBLU(BOLD("1D")));

    print(FBLU(BOLD("WITH CLASSICAL ARRAYS AND LOOPS")));
    {
        constexpr int NITER = 1000000;
        constexpr size_t N = 200, M=185;
        T a[N], b[M];
        for (auto i=0; i<N; ++i) a[i] = (T)rand()/RAND_MAX;
        for (auto i=0; i<M; ++i) b[i] = (T)rand()/RAND_MAX;
        int it[M]; std::iota(it,it+M,7);

        time_.tic();
        for (auto i=0; i<NITER; ++i) {
            for (auto j=0; j<M; ++j) {
                a[it[j]] = b[j];
                unused(b);
            } 
        } 
        time_.toc("Assigning to a view");
        
        time_.tic();
        for (auto i=0; i<NITER; ++i) {
            for (auto j=0; j<M; ++j) {
                a[it[j]] += 5*b[j];
                unused(b);
            } 
        }
        time_.toc("Assigning to view + 2*Add + Mul ");
    }

    print(FBLU(BOLD("USING VECTORISED NOTATION [MATLAB/NUMPY VECTORISATION IS IMPLIED]")));
    {
        constexpr int NITER = 1000000;
        constexpr size_t N = 200, M=185;
        Tensor<T,N> a; a.random();
        Tensor<int,M> it; it.iota(7);
        Tensor<T,M> b; b.random();

        time_.tic();
        for (auto i=0; i<NITER; ++i) {
            a(it) = b;
            unused(b);
        } 
        time_.toc("Assigning to a view");
        
        time_.tic();
        for (auto i=0; i<NITER; ++i) {
            a(it) += 5*b;
            unused(b);
        }
        time_.toc("Assigning to view + 2*Add + Mul ");
    }
#endif
#ifdef TWO_D
    print(FBLU(BOLD("2D")));

    print(FBLU(BOLD("WITH CLASSICAL ARRAYS AND LOOPS")));
    {
        constexpr int NITER = 100000;
        constexpr size_t M=100, N = 111;
        T a[M][N]; //T b[M][N];
        for (auto i=0; i<M; ++i)
            for (auto j=0; j<N; ++j) 
                a[i][j] = (T)rand()/RAND_MAX;
        // for (auto i=0; i<M*N; ++i) b[i] = (T)rand()/RAND_MAX;

        time_.tic();
         for (auto i=0; i<NITER; ++i) {
            for (auto j=0; j<M; ++j) {
                for (auto k=0; k<N; ++k) {
                    T b[M][N];
                    b[j][k] = a[j][k];
                    unused(a,b);
                    // unused(b);
                }
            // unused(a);
            } 
        } 
        // unused(a);
        time_.toc("Constructing from a view");

        time_.tic();
        for (auto i=0; i<NITER; ++i) {
            for (auto j=1; j<90; ++j) {
                for (auto k=4; k<104; k+=3) {
                    a[j][k] += 0.04;
                }
            } 
            unused(a);
        } 
        time_.toc("+= Assigning to a view");
    }

    print(FBLU(BOLD("USING VECTORISED NOTATION [MATLAB/NUMPY VECTORISATION IS IMPLIED]")));
    {
        constexpr int NITER = 100000;
        constexpr size_t M=100, N = 111;
        Tensor<T,M,N> a; a.random();

        time_.tic();
        // Tensor<T,M,N> b;
        for (auto i=0; i<NITER; ++i) {
            Tensor<T,M,N> b = a(all,all);
            // b = a(all,all);
            // Tensor<T,M,N> b = a(seq(0,M),seq(0,N));
            unused(b);
        } 
        time_.toc("Constructing from a view");
        
        time_.tic();
        for (auto i=0; i<NITER; ++i) 
            a(seq(1,90),seq(4,104,3)) += 0.04;
        time_.toc("+= Assigning to a view");
    }
#endif
#ifdef MULTI_D
    print(FBLU(BOLD("nD")));

    print(FBLU(BOLD("WITH CLASSICAL ARRAYS AND LOOPS")));
    {
        int i,j,k;
        constexpr int dim = 100;
        T A[dim][dim][3];
        for (i = 0; i < dim; i++)
            for (j = 0; j < dim; j++)
                for (k = 0; k < 3; k++)
                A[i][j][k] = (T)rand()/RAND_MAX;

        time_.tic();
        for (i = 0; i < dim; i++) {
            for (j = 0; j < dim; j++) {
                A[i][j][0] = A[i][j][1];
                A[i][j][2] = A[i][j][0];
                A[i][j][1] = A[i][j][2];
                unused(A,i,j);
            }
        }

        // for (i = 0; i < dim; i++) {
        //     for (j = 0; j < dim; j++) {
        //         A[i][j][0] += A[i][j][1];
        //         unused(A);
        //     }
        // }
        // for (i = 0; i < dim; i++) {
        //     for (j = 0; j < dim; j++) {
        //         A[i][j][2] += A[i][j][0];
        //         unused(A);
        //     }
        // }
        // for (i = 0; i < dim; i++) {
        //     for (j = 0; j < dim; j++) {
        //         A[i][j][1] += A[i][j][2];
        //         unused(A,i,j);
        //     }
        // }
        time_.toc("Tensor view copy assignment");
    }

    print(FBLU(BOLD("USING VECTORISED NOTATION [MATLAB/NUMPY VECTORISATION IS IMPLIED]")));
    {
        constexpr int dim = 100;
        Tensor<T,dim,dim,3> A;
        A.random();
        
        time_.tic(); 
 
        A(all,all,0) = A(all,all,1);
        A(all,all,2) = A(all,all,0);
        A(all,all,1) = A(all,all,2);

        // A(seq(0,dim),seq(0,dim),0) = A(seq(0,dim),seq(0,dim),1);
        // A(seq(0,dim),seq(0,dim),2) = A(seq(0,dim),seq(0,dim),0);
        // A(seq(0,dim),seq(0,dim),1) = A(seq(0,dim),seq(0,dim),2);
        
        time_.toc("Tensor view copy assignment");
    }
#endif
}

int main() {

    print(FBLU(BOLD("Benchmarking tensor views: single precision")));
    run<float>();
    print(FBLU(BOLD("Benchmarking tensor views: double precision")));
    run<double>();

    return 0;
}