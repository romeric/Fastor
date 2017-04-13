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

        time_.tic();
        for (auto i=0; i<NITER; ++i) {
            for (auto j=0; j<M; ++j) {
                a[7+j] = b[j];
                unused(b);
            } 
        } 
        time_.toc("Assigning to a view");
        
        time_.tic();
        for (auto i=0; i<NITER; ++i) {
            for (auto j=0; j<M; ++j) {
                a[7+j] += 5*b[j];
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
        Tensor<T,M> b; b.random();

        time_.tic();
        for (auto i=0; i<NITER; ++i) {
            a(fseq<7,7+M>{}) = b;
            // unused(b);
        } 
        time_.toc("Assigning to a view");
        
        time_.tic();
        for (auto i=0; i<NITER; ++i) 
            a(fseq<7,7+M>{}) += 5*b;
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
            Tensor<T,M,N> b = a(fall,fall);
            unused(b);
        } 
        time_.toc("Constructing from a view");
        
        time_.tic();
        for (auto i=0; i<NITER; ++i) 
            a(fseq<1,90>{},fseq<4,104,3>{}) += 0.04;
        time_.toc("+= Assigning to a view");
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