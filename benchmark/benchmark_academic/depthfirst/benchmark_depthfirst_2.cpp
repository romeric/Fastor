#include <Fastor/Fastor.h>
#include <sys/resource.h>

using namespace Fastor;
using real = double;
enum {I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z};


constexpr int costs_4(int a,int b, int c, int d, int e, int f, int g) {
    return (a*b*c*d*e*f*g)*4 - 2*(a*b*c*d+e*f*c*g+c*d*g);
}

template<typename T>
double temp_4(int c,int d, int g) {
    return (double)((c*d+c*g)*sizeof(T))/1024.;
}
//////////////////////////////



template<typename T, int a, int b, int c, int d, int e, int f, int g>
void run_benchmark_4() {

    Tensor<T,a,b,c> ss1;
    Tensor<T,a,b,d> ss2;
    Tensor<T,e,f,c> ss3;
    Tensor<T,e,f,g> ss4;

    ss1.random(); ss2.random(); ss3.random(); ss4.random();

    double time;
    std::tie(time,std::ignore) = rtimeit(static_cast<Tensor<T,d,g> (*)(const Tensor<T,a,b,c>&,
                                              const Tensor<T,a,b,d>&,
                                              const Tensor<T,e,f,c>&, const Tensor<T,e,f,g>&)>(&contraction<Index<I,J,K>,
                                                                         Index<I,J,L>,Index<M,N,K>,Index<M,N,O>>),
                                                                         ss1,ss2,ss3,ss4);

   println(time,"\n");
}

void runner_4() {
    // two temporaries created

    // 0.5*L1
    {
        constexpr int a=1,b=1,e=1,f=1,c=32,d=32,g=32;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        // println("reduction in FLOP count",cost_diff);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=4,b=1,e=1,f=1,c=32,d=32,g=32;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=32,b=1,e=1,f=1,c=32,d=32,g=32;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=32,b=8,e=1,f=1,c=32,d=32,g=32;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=64,b=16,e=1,f=1,c=32,d=32,g=32;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=128,b=64,e=1,f=1,c=32,d=32,g=32;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }    


    // L1
    {
        constexpr int a=1,b=1,e=1,f=1,c=32,d=64,g=64;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=4,b=1,e=1,f=1,c=32,d=64,g=64;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=32,b=1,e=1,f=1,c=32,d=64,g=64;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=32,b=8,e=1,f=1,c=32,d=64,g=64;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=64,b=32,e=1,f=1,c=32,d=64,g=64;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }


    // 0.5*L2
    {
        constexpr int a=1,b=1,e=1,f=1,c=16,d=512,g=512;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=4,b=1,e=1,f=1,c=16,d=512,g=512;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=32,b=1,e=1,f=1,c=16,d=512,g=512;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }


    // L2
    {
        constexpr int a=1,b=1,e=1,f=1,c=512,d=64,g=64;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=8,b=1,e=1,f=1,c=512,d=64,g=64;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=32,b=2,e=1,f=1,c=512,d=64,g=64;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }


    // 0.5*L3
    {
        constexpr int a=1,b=1,e=1,f=1,c=640*128,d=8,g=8;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=4,b=1,e=1,f=1,c=640*128,d=8,g=8;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=32,b=1,e=1,f=1,c=640*128,d=8,g=8;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    // L3
    {
        constexpr int a=1,b=1,e=1,f=1,c=640*256,d=8,g=8;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=4,b=1,e=1,f=1,c=640*256,d=8,g=8;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=32,b=1,e=1,f=1,c=640*256,d=8,g=8;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }


    // 4*L3
    {
        constexpr int a=1,b=1,e=1,f=1,c=640*512*4,d=4,g=4;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=4,b=1,e=1,f=1,c=640*512*4,d=4,g=4;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    {
        constexpr int a=16,b=1,e=1,f=1,c=640*512*4,d=4,g=4;
        constexpr int cost_diff = costs_4(a,b,c,d,e,f,g);
        println("FLOP count with no opt",(a*b*c*d*e*f*g)*4, "FLOP cost with opt", 2*(a*b*c*d+e*f*c*g+c*d*g));
        println("size of temporary",temp_4<real>(c,d,g),"\n");
        run_benchmark_4<real,a,b,c,d,e,f,g>();
    }

    return;
}



int main() {

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

   runner_4();
    return 0;
}
