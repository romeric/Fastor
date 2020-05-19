#include <Fastor/Fastor.h>
using namespace Fastor;


// A non-sensical 4th order finite difference scheme
// for Laplace equation
// There are variance in iteration counts with -Ofast
// due to Fastor's internal FAST_MATH switch

// This benchmark is included, since it truly tests
// the inlining capabilities of compilers. There can
// be a huge slow-down/speed-up in performance
// if the whole expression is inlined 

template<typename T, size_t num>
T finite_difference_loop_impl_c(T u[num][num]) {

    T u_old[num][num];
    for (auto i=0; i<num; ++i) {
        for (auto j=0; j<num; ++j) {
            u_old[i][j] = u[i][j];
        }
    }

    T err = 0.;
    for (auto i=0; i<num-2; ++i) {
        for (auto j=0; j<num-2; ++j) {
            u[i+1][j+1] = (( u_old[i][j+1] + u_old[i+2][j+1] + 
                            u_old[i+1][j] + u_old[i+1][2+j] ) * 4.0 +
                            u_old[i][j]   + u_old[i][j+2]   +
                            u_old[2+i][j] + u_old[2+i][2+j]) / 20.0;
            auto diff = u[i+1][j+1] - u_old[i+1][j+1];
            err += diff*diff;  
        }
    }
    return sqrts(err);
}


template<typename T, size_t num>
void run_finite_difference() {

    T pi = 4*std::atan(1.0);
    T err = 2.;
    int iter = 0;
    
    T x[num];
    for (auto i=0; i<num; ++i) {
        x[i] = i*pi/(num-1);
    }

    T u[num][num];
    for (auto i=0; i<num; ++i) {
        for (auto j=0; j<num; ++j) {
            u[i][j] = 0.;
        }
    }
    for (auto i=0; i<num; ++i) {
        u[0][i] = sin(x[i]);
        u[num-1][i] = sin(x[i])*std::exp(-pi);
    }


    while (iter <100000 && err>1e-6) {
        err = finite_difference_loop_impl_c(u);
        iter++;
    }

    println(" Relative error is: ", err, '\n');
    println("Number of iterations: ", iter, '\n');

}

int main(int argc, char *argv[]) {

    const rlim_t stacksize = 1024*1024*1024;
    struct rlimit rl;
    int result;
    result = getrlimit(RLIMIT_STACK, &rl);
    if (result==0) {
        if (rl.rlim_cur < stacksize) {
            rl.rlim_cur = stacksize;
            result = setrlimit(RLIMIT_STACK,&rl);
            if (result !=0) {
                FASTOR_ASSERT(result !=0, "CHANGING STACK SIZE FAILED");
            }
        }
    }

    int N;
    if (argc == 2) {
       N = atoi(argv[1]);
    }
    else {
       print("Usage: \n");
       print("      ./exe N \n", argv[0]);
       exit(-1);
    }


    timer<double> t_j;
    t_j.tic();
    // Putting all this in stack might hurt the performance
    // as opposed to compiling for each individual benchmarks
    if (N==100) run_finite_difference<double,100>();
    if (N==150) run_finite_difference<double,150>();
    if (N==200) run_finite_difference<double,200>();
    t_j.toc();

    return 0;
}
