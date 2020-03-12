#include <Fastor/Fastor.h>

using namespace Fastor;

enum {I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z};

template<typename dtype, size_t a, size_t b, size_t c, size_t d, size_t e, 
    size_t f, size_t g, size_t h>
void run() {

    Tensor<dtype,a,b,c,d,e,f,g> A; Tensor<dtype,a,b,c,d,e,f,g,h> B;
    A.random(); B.random();

#if SEVEN_INDEX

    auto C = contraction<Index<I,J,K,L,M,N,O>,Index<I,J,K,L,M,N,O,P>>(A,B);
    unused(C);

    #ifdef TEST_TIME
        double time;
        std::tie(time,std::ignore) = rtimeit(static_cast<decltype(C) (*)(const Tensor<dtype,a,b,c,d,e,f,g>&, 
            const Tensor<dtype,a,b,c,d,e,f,g,h>&)>(&contraction<Index<I,J,K,L,M,N,O>,Index<I,J,K,L,M,N,O,P>>),A,B);
        print(time);
    #endif

#elif SIX_INDEX

    auto C = contraction<Index<I,J,K,L,M,N,O>,Index<I,J,K,L,M,N,Q,P>>(A,B);
    unused(C);

    #ifdef TEST_TIME
        double time;
        std::tie(time,std::ignore) = rtimeit(static_cast<decltype(C) (*)(const Tensor<dtype,a,b,c,d,e,f,g>&, 
            const Tensor<dtype,a,b,c,d,e,f,g,h>&)>(&contraction<Index<I,J,K,L,M,N,O>,Index<I,J,K,L,M,N,Q,P>>),A,B);
        print(time);
    #endif

#elif FIVE_INDEX

    auto C = contraction<Index<I,J,K,L,M,N,O>,Index<I,J,K,L,M,R,Q,P>>(A,B);
    unused(C);

    #ifdef TEST_TIME
        double time;
        std::tie(time,std::ignore) = rtimeit(static_cast<decltype(C) (*)(const Tensor<dtype,a,b,c,d,e,f,g>&, 
            const Tensor<dtype,a,b,c,d,e,f,g,h>&)>(&contraction<Index<I,J,K,L,M,N,O>,Index<I,J,K,L,M,R,Q,P>>),A,B);
        print(time);
    #endif

#elif FOUR_INDEX

    auto C = contraction<Index<I,J,K,L,M,N,O>,Index<I,J,K,L,S,R,Q,P>>(A,B);
    unused(C);

    #ifdef TEST_TIME
        double time;
        std::tie(time,std::ignore) = rtimeit(static_cast<decltype(C) (*)(const Tensor<dtype,a,b,c,d,e,f,g>&, 
            const Tensor<dtype,a,b,c,d,e,f,g,h>&)>(&contraction<Index<I,J,K,L,M,N,O>,Index<I,J,K,L,S,R,Q,P>>),A,B);
        print(time);
    #endif


#elif THREE_INDEX

    auto C = contraction<Index<I,J,K,L,M,N,O>,Index<I,J,K,T,S,R,Q,P>>(A,B);
    unused(C);

    #ifdef TEST_TIME
        double time;
        std::tie(time,std::ignore) = rtimeit(static_cast<decltype(C) (*)(const Tensor<dtype,a,b,c,d,e,f,g>&, 
            const Tensor<dtype,a,b,c,d,e,f,g,h>&)>(&contraction<Index<I,J,K,L,M,N,O>,Index<I,J,K,T,S,R,Q,P>>),A,B);
        print(time);
    #endif


#elif TWO_INDEX

    auto C = contraction<Index<I,J,K,L,M,N,O>,Index<I,J,U,T,S,R,Q,P>>(A,B);
    unused(C);

    #ifdef TEST_TIME
        double time;
        std::tie(time,std::ignore) = rtimeit(static_cast<decltype(C) (*)(const Tensor<dtype,a,b,c,d,e,f,g>&, 
            const Tensor<dtype,a,b,c,d,e,f,g,h>&)>(&contraction<Index<I,J,K,L,M,N,O>,Index<I,J,U,T,S,R,Q,P>>),A,B);
        print(time);
    #endif


#elif ONE_INDEX

    auto C = contraction<Index<I,J,K,L,M,N,O>,Index<I,V,U,T,S,R,Q,P>>(A,B);
    unused(C);

    #ifdef TEST_TIME
        double time;
        std::tie(time,std::ignore) = rtimeit(static_cast<decltype(C) (*)(const Tensor<dtype,a,b,c,d,e,f,g>&, 
            const Tensor<dtype,a,b,c,d,e,f,g,h>&)>(&contraction<Index<I,J,K,L,M,N,O>,Index<I,V,U,T,S,R,Q,P>>),A,B);
        print(time);
    #endif


    unused(A,B);

#endif
}


int main() {

    constexpr size_t a=1,b=2,c=2,d=2,e=2,f=2,g=2,h=4;
    run<double,a,b,c,d,e,f,g,h>();
    return 0;

}