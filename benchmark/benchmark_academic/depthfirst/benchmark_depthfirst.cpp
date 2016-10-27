#include <Fastor.h>
#include <sys/resource.h>

using namespace Fastor;
using std::size_t;
using real = double;
enum {I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z};


constexpr int costs_3(int a,int b, int c, int d, int e, int f) {
    return (a*b*c*d*e*f)*3 - (2*(a*b*d*e*f)+2*(a*b*c*e*f));
}

template<typename T>
double temp_3(int a,int b, int e, int f) {
    return (double)(a*b*e*f*sizeof(T))/1024.;
}
//////////////////////////////



template<typename T, int a, int b, int c, int d, int e, int f>
void run_benchmark_3() {

    Tensor<T,a,b,c> ss1;
    Tensor<T,a,b,d> ss2;
    Tensor<T,e,f,d> ss3;

//    ss1.iota(0); ss2.iota(0); ss3.iota(0);
    ss1.random(); ss2.random(); ss3.random();

    double time_dp, time_nodp;
    std::tie(time_nodp,std::ignore) = rtimeit(static_cast<Tensor<real,c,e,f> (*)(const Tensor<real,a,b,c>&,
                                              const Tensor<real,a,b,d>&,
                                              const Tensor<real,e,f,d>&)>(&contraction_<Index<I,J,K>,
                                                                         Index<I,J,L>,Index<M,N,L>,NoDepthFirst>),
                                                                         ss1,ss2,ss3);
    std::tie(time_dp,std::ignore) = rtimeit(static_cast<Tensor<real,c,e,f> (*)(const Tensor<real,a,b,c>&,
                                              const Tensor<real,a,b,d>&,
                                              const Tensor<real,e,f,d>&)>(&contraction<Index<I,J,K>,
                                                                         Index<I,J,L>,Index<M,N,L>>),
                                                                         ss1,ss2,ss3);

   println(time_nodp,time_dp,"\n");

    // contraction<Index<I,J,K>,Index<I,J,L>,Index<M,N,L>>(ss1,ss2,ss3);
}

void runner_3() {
    // one temporary created

    // 0.5*L1
    // {
    //     constexpr int a=16,b=16,c=2,d=1,e=2,f=4;
    //     constexpr int cost_diff = costs_3(a,b,c,d,e,f);
    //     // println("reduction in FLOP count",cost_diff,"\n");
    //     println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
    //     println("size of temporary",temp_3<double>(a,b,e,f),"\n");
    //     run_benchmark_3<double,a,b,c,d,e,f>();
    // }

    {
        constexpr int a=16,b=16,c=2,d=2,e=2,f=4;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=16,c=8,d=2,e=2,f=4;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=16,c=16,d=2,e=2,f=4;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=16,c=128,d=2,e=2,f=4;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=16,c=128,d=16,e=2,f=4;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=16,c=128,d=144,e=2,f=4;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }


    // L1
    {
        constexpr int a=16,b=32,c=2,d=2,e=2,f=4;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=32,c=8,d=2,e=2,f=4;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=32,c=16,d=2,e=2,f=4;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=32,c=128,d=2,e=2,f=4;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=32,c=128,d=16,e=2,f=4;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=32,c=128,d=144,e=2,f=4;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    // 0.5*L2
    {
        constexpr int a=16,b=32,c=2,d=2,e=4,f=8;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=32,c=8,d=2,e=4,f=8;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=32,c=16,d=2,e=4,f=8;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=32,c=128,d=2,e=4,f=8;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=32,c=128,d=16,e=4,f=8;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=32,c=128,d=144,e=4,f=8;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }


    // L2
    {
        constexpr int a=16,b=32,c=2,d=2,e=8,f=8;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=32,c=8,d=2,e=8,f=8;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=32,c=16,d=2,e=8,f=8;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=32,c=128,d=2,e=8,f=8;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=32,c=128,d=16,e=8,f=8;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=16,b=32,c=128,d=144,e=8,f=8;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        // println("reduction in FLOP count",cost_diff,"\n");
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f),"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }


    // 0.5*L3
    {
        constexpr int a=32,b=20,c=4,d=1,e=32,f=64;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f)/1024.,"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=32,b=20,c=8,d=2,e=32,f=64;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f)/1024.,"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=32,b=20,c=64,d=2,e=32,f=64;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f)/1024.,"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    // L3
    {
        constexpr int a=32,b=40,c=4,d=1,e=32,f=64;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f)/1024.,"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=32,b=40,c=8,d=2,e=32,f=64;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f)/1024.,"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=32,b=40,c=64,d=2,e=32,f=64;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f)/1024.,"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }


    // 4*L3
    {
        constexpr int a=32,b=160,c=4,d=1,e=32,f=64;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f)/1024.,"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    {
        constexpr int a=32,b=160,c=8,d=2,e=32,f=64;
        constexpr int cost_diff = costs_3(a,b,c,d,e,f);
        println("FLOP count with no opt",(a*b*c*d*e*f)*3, "FLOP cost with opt", (2*(a*b*d*e*f)+2*(a*b*c*e*f)));
        println("size of temporary",temp_3<double>(a,b,e,f)/1024.,"\n");
        run_benchmark_3<double,a,b,c,d,e,f>();
    }

    return;
}



int main() {

    const rlim_t stacksize = 160*1024*1024;
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

   runner_3();
    return 0;
}