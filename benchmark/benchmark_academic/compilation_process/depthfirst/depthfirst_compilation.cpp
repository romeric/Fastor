#include <Fastor/Fastor.h>

using namespace Fastor;

enum {I,J,K,L,M,N,O,P,Q,R,S,T};

template<typename T, size_t a,size_t b,size_t c,size_t d,size_t e,size_t f,size_t g,size_t h>
void run() {
    Tensor<T,a,b,c> A; Tensor<T,a,b,d> B; Tensor<T,e,f,d> C;
    A.random(); B.random(); C.random();

#ifdef THREE_TENSOR
    auto out = contraction<Index<I,J,K>,Index<I,J,L>,Index<M,N,L>>(A,B,C);
    unused(out);
#endif
#ifdef FOUR_TENSOR
    Tensor<T,e,f,g> D;
    D.random();
    auto out = contraction<Index<I,J,K>,Index<I,J,L>,Index<M,N,L>,Index<M,N,O>>(A,B,C,D);
    unused(out,D);
#endif
#ifdef FIVE_TENSOR
    Tensor<T,e,f,g> D; Tensor<T,g,h> E;
    D.random(); E.random();
    auto out = contraction<Index<I,J,K>,Index<I,J,L>,Index<M,N,L>,Index<M,N,O>,Index<O,P>>(A,B,C,D,E);
    unused(out,D,E);
#endif
    // Avoid dead code elimination
    unused(A,B,C);
    unused(a,b,c,d,e,f,g,h);
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


#ifdef SPAN_0
    {
        // 3: 2**4, 4: 2**5, 5: 2**6 
        constexpr size_t a=8,b=8,c=16,d=2,e=2,f=4,g=4,h=2;
        run<double,a,b,c,d,e,f,g,h>();
    }
#endif
#ifdef SPAN_1
    {
        // 3: 2**5, 4: 2**6, 5: 2**7 
        constexpr size_t a=8,b=8,c=32,d=2,e=2,f=4,g=4,h=2;
        run<double,a,b,c,d,e,f,g,h>();
    }
#endif
#ifdef SPAN_2
    {
        // 3: 2**6, 4: 2**7, 5: 2**8 
        constexpr size_t a=8,b=8,c=64,d=2,e=2,f=4,g=4,h=2;
        run<double,a,b,c,d,e,f,g,h>();
    }
#endif
#ifdef SPAN_3
    {
        // 3: 2**7, 4: 2**8, 5: 2**9 
        constexpr size_t a=8,b=8,c=128,d=2,e=2,f=4,g=4,h=2;
        run<double,a,b,c,d,e,f,g,h>();
    }
#endif
#ifdef SPAN_4
    {
        // 3: 2**8, 4: 2**9, 5: 2**10
        constexpr size_t a=8,b=8,c=256,d=2,e=2,f=4,g=4,h=2;
        run<double,a,b,c,d,e,f,g,h>();
    }
#endif
#ifdef SPAN_5
    {
        // 3: 2**9, 4: 2**10, 5: 2**11
        constexpr size_t a=8,b=8,c=512,d=2,e=2,f=4,g=4,h=2;
        run<double,a,b,c,d,e,f,g,h>();
    }
#endif
#ifdef SPAN_6
    {
        // 3: 2**10, 4: 2**11, 5: 2**12 
        constexpr size_t a=8,b=8,c=1024,d=2,e=2,f=4,g=4,h=2;
        run<double,a,b,c,d,e,f,g,h>();
    }
#endif
// #ifdef SPAN_7
//     {
//         // 3: 2**11, 4: 2**12, 5: 2**13 
//         constexpr size_t a=8,b=8,c=2048,d=2,e=2,f=4,g=4,h=2;
//         run<double,a,b,c,d,e,f,g,h>();
//     }
// #endif
// #ifdef SPAN_8
//     {
//         // 3: 2**12, 4: 2**13, 5: 2**14 
//         constexpr size_t a=8,b=8,c=4096,d=2,e=2,f=4,g=4,h=2;
//         run<double,a,b,c,d,e,f,g,h>();
//     }
// #endif

    return 0;
}
