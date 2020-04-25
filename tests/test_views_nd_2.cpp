#include <Fastor/Fastor.h>
using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5
#define HugeTol 1e-2


template<typename T>
void run_vectorisable() {

    Tensor<T,2,2,2,16> a; a.iota(1);
    Tensor<T,2,2,2,16> b; b.iota(11);
    Tensor<T,3,2,2,16> c; c.iota(101);

    FASTOR_EXIT_ASSERT(abs(a.sum() - 8256) < Tol);
    FASTOR_EXIT_ASSERT(abs(b.sum() - 9536) < Tol);
    FASTOR_EXIT_ASSERT(abs(c.sum() - 37728) < Tol);

    // Assigning the same view to a view [check copy operators for views]
    a(all,all,all,all) = b(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 9536) < Tol);
    a.iota(1);
    a(all,all,all,all) += b(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 17792) < Tol);
    a(all,all,all,all) -= b(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8256) < Tol);
    a(all,all,all,all) *= b(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 789824) < Tol);
    a(all,all,all,all) /= b(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8256) < BigTol);

    // Assigning different view to a view [checks copy operators for abstract expressions]
    a(0,all,all,all) = c(fix<1>,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 18752) < Tol);
    a.iota(1);
    a(0,all,all,all) += c(fix<1>,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 20832) < Tol);
    a(0,all,all,all) -= c(1,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8256) < Tol);
    a(0,all,all,all) *= c(1,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 436736) < Tol);
    a(0,all,all,all) /= c(1,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8256) < BigTol);

    // Assigning different view to a view [checks copy operators for abstract expressions]
    a(0,all,all,all) = c(1,all,all,all)+1-1;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 18752) < Tol);
    a.iota(1);
    a(0,all,all,all) += c(1,all,all,all)+1-1;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 20832) < Tol);
    a(0,all,all,all) -= c(1,all,all,all)+1-1;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8256) < Tol);
    a(0,all,all,all) *= c(1,all,all,all)+1-1;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 436736) < Tol);
    a(0,all,all,all) /= c(1,all,all,all)+1-1;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8256) < BigTol);

    // Assigning numbers to a view [checks copy operators for numbers]
    a(0,all,1,all) = 2;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7024) < Tol);
    a.iota(1);
    a(0,all,1,all) += 2;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8320) < Tol);
    a(0,all,1,all) -= 2;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8256) < Tol);
    a(0,all,1,all) *= 2;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 9552) < Tol);
    a(0,all,1,all) /= 2;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8256) < BigTol);

    // Assigning different view to a view (non-equal order) [checks copy operators for abstract expressions]
    a.iota(1);
    Tensor<T,4,2,16> d; d.iota(1);
    a(0,0,all,all) = d(0,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8256) < BigTol);
    a(0,0,all,all) += d(0,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8784) < BigTol);
    a(0,0,all,all) -= d(0,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8256) < BigTol);
    a(0,0,all,all) *= d(0,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 19168) < BigTol);
    a(0,0,all,all) /= d(0,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8256) < BigTol);

    // Assigning different view to a view (non-equal order) [checks copy operators for abstract expressions]
    a.iota(1);
    a(0,0,all,all) = d(0,all,all) + 3 - 3;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8256) < BigTol);
    a(0,0,all,all) += d(0,all,all) + 3 - 3;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8784) < BigTol);
    a(0,0,all,all) -= d(0,all,all) + 3 - 3;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8256) < BigTol);
    a(0,0,all,all) *= d(0,all,all) + 3 - 3;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 19168) < BigTol);
    a(0,0,all,all) /= d(0,all,all) + 3 - 3;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8256) < BigTol);

    // Constructing tensor from a tensor view [checks specialised constructors]
    a.iota(1);
    a = a(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8256) < BigTol);
    decltype(a) e = a(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(e.sum() - 8256) < BigTol);
    e += b(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(e.sum() - 17792) < Tol);
    e -= b(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(e.sum() - 8256) < Tol);
    e *= b(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(e.sum() - 789824) < Tol);
    e /= b(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(e.sum() - 8256) < BigTol);

    // Constructing tensor from an expression that has tensor view [checks specialised constructors]
    a.iota(1);
    a = a(all,all,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8256) < BigTol);
    e = a(all,all,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(e.sum() - 8256) < BigTol);
    e += b(all,all,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(e.sum() - 17792) < Tol);
    e -= b(all,all,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(e.sum() - 8256) < Tol);
    e *= b(all,all,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(e.sum() - 789824) < Tol);
    e /= b(all,all,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(e.sum() - 8256) < BigTol);

    // Constructing tensor from a tensor view (non-equal order)
    a.iota(1);
    Tensor<T,8> f = a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(f.sum() - 64) < Tol);
    f += a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(f.sum() - 128) < Tol);
    f -= a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(f.sum() - 64) < Tol);
    f *= a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(f.sum() - 680) < Tol);
    f /= a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(f.sum() - 64) < BigTol);

    // Constructing tensor from a tensor view (equal order)
    a.iota(1);
    Tensor<T,1,1,1,8> g; g.zeros();
    g = a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(g.sum() - 64) < Tol);
    g += a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(g.sum() - 128) < Tol);
    g -= a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(g.sum() - 64) < Tol);
    g *= a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(g.sum() - 680) < Tol);
    g /= a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(g.sum() - 64) < BigTol);

    print(FGRN(BOLD("All tests passed successfully")));
}


template<typename T>
void run_non_vectorisable() {

    Tensor<T,2,2,2,15> a; a.iota(1);
    Tensor<T,2,2,2,15> b; b.iota(11);
    Tensor<T,3,2,2,15> c; c.iota(101);

    FASTOR_EXIT_ASSERT(abs(a.sum() - 7260) < Tol);
    FASTOR_EXIT_ASSERT(abs(b.sum() - 8460) < Tol);
    FASTOR_EXIT_ASSERT(abs(c.sum() - 34290) < Tol);

    // Assigning the same view to a view [check copy operators for views]
    a(all,all,all,all) = b(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8460) < Tol);
    a.iota(1);
    a(all,all,all,all) += b(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 15720) < Tol);
    a(all,all,all,all) -= b(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7260) < Tol);
    a(all,all,all,all) *= b(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 655820) < Tol);
    a(all,all,all,all) /= b(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7260) < BigTol);

    // Assigning different view to a view [checks copy operators for abstract expressions]
    a(0,all,all,all) = c(1,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 16860) < Tol);
    a.iota(1);
    a(0,all,all,all) += c(1,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 18690) < Tol);
    a(0,all,all,all) -= c(1,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7260) < Tol);
    a(0,all,all,all) *= c(1,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 372040) < Tol);
    a(0,all,all,all) /= c(1,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7260) < BigTol);

    // // Assigning different view to a view [checks copy operators for abstract expressions]
    a(0,all,all,all) = c(1,all,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 16860) < Tol);
    a.iota(1);
    a(0,all,all,all) += c(1,all,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 18690) < Tol);
    a(0,all,all,all) -= c(1,all,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7260) < Tol);
    a(0,all,all,all) *= c(1,all,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 372040) < Tol);
    a(0,all,all,all) /= c(1,all,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7260) < BigTol);

    // Assigning numbers to a view [checks copy operators for numbers]
    a(0,all,1,all) = 2;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 6180) < Tol);
    a.iota(1);
    a(0,all,1,all) += 2;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7320) < Tol);
    a(0,all,1,all) -= 2;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7260) < Tol);
    a(0,all,1,all) *= 2;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 8400) < Tol);
    a(0,all,1,all) /= 2;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7260) < BigTol);

    // Assigning different view to a view (non-equal order) [checks copy operators for abstract expressions]
    a.iota(1);
    Tensor<T,4,2,15> d; d.iota(1);
    a(0,0,all,all) = d(0,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7260) < BigTol);
    a(0,0,all,all) += d(0,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7725) < BigTol);
    a(0,0,all,all) -= d(0,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7260) < BigTol);
    a(0,0,all,all) *= d(0,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 16250) < BigTol);
    a(0,0,all,all) /= d(0,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7260) < BigTol);

    // Assigning different view to a view (non-equal order) [checks copy operators for abstract expressions]
    a.iota(1);
    a(0,0,all,all) = d(0,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7260) < BigTol);
    a(0,0,all,all) += d(0,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7725) < BigTol);
    a(0,0,all,all) -= d(0,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7260) < BigTol);
    a(0,0,all,all) *= d(0,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 16250) < BigTol);
    a(0,0,all,all) /= d(0,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7260) < BigTol);

    // Constructing tensor from a tensor view [checks specialised constructors]
    a.iota(1);
    a = a(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7260) < BigTol);
    decltype(a) e = a(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(e.sum() - 7260) < BigTol);
    e += b(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(e.sum() - 15720) < Tol);
    e -= b(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(e.sum() - 7260) < Tol);
    e *= b(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(e.sum() - 655820) < Tol);
    e /= b(all,all,all,all);
    FASTOR_EXIT_ASSERT(abs(e.sum() - 7260) < BigTol);

    // Constructing tensor from an expression that has tensor view [checks specialised constructors]
    a.iota(1);
    a = a(all,all,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(a.sum() - 7260) < BigTol);
    e = a(all,all,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(e.sum() - 7260) < BigTol);
    e += b(all,all,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(e.sum() - 15720) < Tol);
    e -= b(all,all,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(e.sum() - 7260) < Tol);
    e *= b(all,all,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(e.sum() - 655820) < Tol);
    e /= b(all,all,all,all) + 0;
    FASTOR_EXIT_ASSERT(abs(e.sum() - 7260) < BigTol);

    // Constructing tensor from a tensor view (non-equal order)
    a.iota(1);
    Tensor<T,8> f = a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(f.sum() - 64) < Tol);
    f += a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(f.sum() - 128) < Tol);
    f -= a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(f.sum() - 64) < Tol);
    f *= a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(f.sum() - 680) < Tol);
    f /= a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(f.sum() - 64) < BigTol);

    // Constructing tensor from a tensor view (equal order)
    a.iota(1);
    Tensor<T,1,1,1,8> g; g.zeros();
    g = a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(g.sum() - 64) < Tol);
    g += a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(g.sum() - 128) < Tol);
    g -= a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(g.sum() - 64) < Tol);
    g *= a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(g.sum() - 680) < Tol);
    g /= a(0,0,0,seq(0,last,2));
    FASTOR_EXIT_ASSERT(abs(g.sum() - 64) < BigTol);

    print(FGRN(BOLD("All tests passed successfully")));
}



int main() {

    print(FBLU(BOLD("Testing multi-dimensional tensor views: single precision")));
    run_vectorisable<float>();
    run_non_vectorisable<float>();
    print(FBLU(BOLD("Testing multi-dimensional tensor views: double precision")));
    run_vectorisable<double>();
    run_non_vectorisable<double>();
    print(FBLU(BOLD("Testing multi-dimensional tensor views: int 32")));
    run_vectorisable<int>();
    run_non_vectorisable<int>();
    print(FBLU(BOLD("Testing multi-dimensional tensor views: int 64")));
    run_vectorisable<Int64>();
    run_non_vectorisable<Int64>();

    return 0;
}