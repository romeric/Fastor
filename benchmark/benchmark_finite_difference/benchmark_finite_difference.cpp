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

/*
// include for historical reason. This scalar variant
// does not do the same thing as the vectorised variants
// and heavily suffers from aliasing and the compiler
// can't optimise that
template<typename T, size_t num>
T finite_difference_loop_impl(Tensor<T,num,num> &u) {

    T err = 0.;
    for (auto i=0; i<num-2; ++i) {
        for (auto j=0; j<num-2; ++j) {
            auto u_old = u(i+1,j+1);
            u(i+1,j+1) = (( u(i,j+1) + u(i+2,j+1) +
                            u(i+1,j) + u(i+1,2+j) ) * 4.0 +
                            u(i,j)   + u(i,j+2)   +
                            u(2+i,j) + u(2+i,2+j)) / 20.0;
            auto diff = u(i+1,j+1) - u_old;
            err += diff*diff;
        }
    }
    return sqrts(err);
}
*/


template<typename T, size_t num>
T finite_difference_loop_impl(Tensor<T,num,num> &u) {

    Tensor<T,num,num> u_old;
    for (auto i=0; i<num; ++i) {
        for (auto j=0; j<num; ++j) {
            u_old(i,j) = u(i,j);
        }
    }

    T err = 0.;
    for (auto i=0; i<num-2; ++i) {
        for (auto j=0; j<num-2; ++j) {
            u(i+1,j+1) = (( u_old(i,j+1) + u_old(i+2,j+1) +
                            u_old(i+1,j) + u_old(i+1,2+j) ) * 4.0 +
                            u_old(i,j)   + u_old(i,j+2)   +
                            u_old(2+i,j) + u_old(2+i,2+j)) / 20.0;
            auto diff = u(i+1,j+1) - u_old(i+1,j+1);
            err += diff*diff;
        }
    }
    return sqrts(err);
}


template<typename T, size_t num>
T finite_difference_seq_alias_impl(Tensor<T,num,num> &u) {

    Tensor<T,num,num> u_old = u;
    u(seq(1,last-1),seq(1,last-1)).noalias() =
        ((  u(seq(0,last-2),seq(1,last-1)) + u(seq(2,last),seq(1,last-1)) +
            u(seq(1,last-1),seq(0,last-2)) + u(seq(1,last-1),seq(2,last)) )*4.0 +
            u(seq(0,last-2),seq(0,last-2)) + u(seq(0,last-2),seq(2,last)) +
            u(seq(2,last),seq(0,last-2))   + u(seq(2,last),seq(2,last)) ) /20.0;

    return norm(u - u_old);
}


template<typename T, size_t num>
T finite_difference_seq_noalias_impl(Tensor<T,num,num> &u) {

    Tensor<T,num,num> u_old = u;

    u(seq(1,last-1),seq(1,last-1)) =
        ((  u_old(seq(0,last-2),seq(1,last-1)) + u_old(seq(2,last),seq(1,last-1)) +
            u_old(seq(1,last-1),seq(0,last-2)) + u_old(seq(1,last-1),seq(2,last)) )*4.0 +
            u_old(seq(0,last-2),seq(0,last-2)) + u_old(seq(0,last-2),seq(2,last)) +
            u_old(seq(2,last),seq(0,last-2))   + u_old(seq(2,last),seq(2,last)) ) /20.0;

    return norm(u-u_old);
}

template<typename T, size_t num>
T finite_difference_iseq_impl(Tensor<T,num,num> &u) {

    Tensor<T,num,num> u_old = u;

    u(seq(1,last-1),seq(1,last-1)) =
        ((  u_old(iseq<0,num-2>{},iseq<1,num-1>{}) + u_old(iseq<2,num>{},iseq<1,num-1>{}) +
            u_old(iseq<1,num-1>{},iseq<0,num-2>{}) + u_old(iseq<1,num-1>{},iseq<2,num>{}) ) * 4.0 +
            u_old(iseq<0,num-2>{},iseq<0,num-2>{}) + u_old(iseq<0,num-2>{},iseq<2,num>{}) +
            u_old(iseq<2,num>{},iseq<0,num-2>{})   + u_old(iseq<2,num>{},iseq<2,num>{})
        ) / 20.0;

    return norm(u-u_old);
}

template<typename T, size_t num>
T finite_difference_fseq_impl(Tensor<T,num,num> &u) {

    Tensor<T,num,num> u_old = u;
    u(fseq<1,num-1>{},fseq<1,num-1>{}) =
        ((  u_old(fseq<0,num-2>{},fseq<1,num-1>{}) + u_old(fseq<2,num>{},fseq<1,num-1>{}) +
            u_old(fseq<1,num-1>{},fseq<0,num-2>{}) + u_old(fseq<1,num-1>{},fseq<2,num>{}) ) * 4.0 +
            u_old(fseq<0,num-2>{},fseq<0,num-2>{}) + u_old(fseq<0,num-2>{},fseq<2,num>{}) +
            u_old(fseq<2,num>{},fseq<0,num-2>{})   + u_old(fseq<2,num>{},fseq<2,num>{})
        )  / 20.0;

    return norm(u-u_old);
}


template<typename T, size_t num>
void run_finite_difference() {

    T pi = 4*std::atan(1.0);
    T err = 2.;
    int iter = 0;

    Tensor<T,num> x;
    for (auto i=0; i<num; ++i) {
        x(i) = i*pi/(num-1);
    }

    Tensor<T,num,num> u; u.zeros();
#if defined(USE_SEQ_ALIAS) || defined(USE_SEQ_NOALIAS)
    u(0,all) = sin(x);
    u(num-1,all) = sin(x)*std::exp(-pi);
#else
    u(fseq<0,1>(),fall) = sin(x);
    u(fseq<num-1,num>(),fall) = sin(x)*std::exp(-pi);
#endif

    while (iter <100000 && err>1e-6) {
#if defined(USE_LOOPS)
        err = finite_difference_loop_impl(u);
#elif defined(USE_SEQ_ALIAS)
        err = finite_difference_seq_alias_impl(u);
#elif defined(USE_SEQ_NOALIAS)
        err = finite_difference_seq_noalias_impl(u);
#elif defined(USE_ISEQ)
        err = finite_difference_iseq_impl(u);
#elif defined(USE_FSEQ)
        err = finite_difference_fseq_impl(u);
#endif
        iter++;
    }

    println(" Relative error is: ", err, '\n');
    println("Number of iterations: ", iter, '\n');

}

int main(int argc, char *argv[]) {

    // const rlim_t stacksize = 1024*1024*1024;
    // struct rlimit rl;
    // int result;
    // result = getrlimit(RLIMIT_STACK, &rl);
    // if (result==0) {
    //     if (rl.rlim_cur < stacksize) {
    //         rl.rlim_cur = stacksize;
    //         result = setrlimit(RLIMIT_STACK,&rl);
    //         if (result !=0) {
    //             FASTOR_ASSERT(result !=0, "CHANGING STACK SIZE FAILED");
    //         }
    //     }
    // }

    using T = double;
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
    if (N==100) run_finite_difference<T,100>();
    if (N==150) run_finite_difference<T,150>();
    if (N==200) run_finite_difference<T,200>();
    t_j.toc();

    return 0;
}
