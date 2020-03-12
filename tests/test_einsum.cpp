#include <Fastor/Fastor.h>
using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5
#define HugeTol 1e-2


template<typename T>
void run() {

    using std::abs;
    enum {i,j,k,l,m,n};
    {
        Tensor<T,2,2> II; II.eye();
        auto II_ijkl = einsum<Index<i,j>,Index<k,l>>(II,II);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(0,0,0,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(0,0,1,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(1,1,0,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(1,1,1,1) - 1) < Tol);
        auto II_ijkl2 = outer(II,II);
        FASTOR_EXIT_ASSERT(II_ijkl==II_ijkl2);
        auto II_ikjl = permutation<Index<i,k,j,l>>(II_ijkl);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(0,0,0,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(0,1,0,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(1,0,1,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(1,1,1,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(norm(II_ijkl) - norm(II_ikjl)) < Tol);
        auto II_iljk = permutation<Index<i,l,j,k>>(II_ijkl);
        FASTOR_EXIT_ASSERT(abs(II_iljk(0,0,0,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(0,1,1,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(1,0,0,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(1,1,1,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(norm(II_ijkl) - norm(II_iljk)) < Tol);


        Tensor<T,2,2> A, B; A.iota(101); B.iota(77);
        Tensor<T,2> D = {45.5, 73.82};
        auto bb_ijkl = einsum<Index<i,j>,Index<k,l>>(A,B);
        FASTOR_EXIT_ASSERT(abs(norm(bb_ijkl) - 32190.178937) < HugeTol);
        auto bb_ikjl = permutation<Index<i,k,j,l>>(bb_ijkl);
        FASTOR_EXIT_ASSERT(abs(norm(bb_ikjl) - 32190.178937) < HugeTol);
        auto bb_iljk = permutation<Index<i,l,j,k>>(bb_ijkl);
        FASTOR_EXIT_ASSERT(abs(norm(bb_iljk) - 32190.178937) < HugeTol);
        auto bb_ijkl2 = outer(A,B);
        FASTOR_EXIT_ASSERT(abs(norm(bb_ijkl2) - 32190.178937) < HugeTol);


        FASTOR_EXIT_ASSERT(abs(bb_ijkl(0,1,0,1)-bb_ikjl(0,0,1,1)) < BigTol);
        FASTOR_EXIT_ASSERT(abs(bb_ijkl(0,1,0,1)-bb_iljk(0,0,1,1)) < BigTol);

        for (auto ii=0; ii< 2; ii++) {
            for (auto jj=0; jj< 2; jj++) {
                for (auto kk=0; kk< 2; kk++) {
                    for (auto ll=0; ll< 2; ll++) {
                        FASTOR_EXIT_ASSERT(abs( bb_ijkl(ii,jj,kk,ll) - bb_ikjl(ii,kk,jj,ll) ) < BigTol);
                        // FASTOR_EXIT_ASSERT(abs( bb_ijkl(ii,jj,kk,ll) - bb_iljk(ii,ll,jj,kk) ) < BigTol); // By definition this cannot be
                        FASTOR_EXIT_ASSERT(abs( bb_ijkl(ii,ll,jj,kk) - bb_iljk(ii,jj,kk,ll) ) < BigTol);
                    }
                }
            }
        }

        auto bD_ijk = einsum<Index<i,j>,Index<k>>(A,D);
        FASTOR_EXIT_ASSERT(abs(norm(bD_ijk) - 17777.8111) < HugeTol);
        auto bD_ikj = permutation<Index<i,k,j>>(bD_ijk);
        FASTOR_EXIT_ASSERT(abs(norm(bD_ikj) - 17777.8111) < HugeTol);
        auto bD_jki = permutation<Index<j,k,i>>(bD_ijk);
        FASTOR_EXIT_ASSERT(abs(norm(bD_jki) - 17777.8111) < HugeTol);
        auto bD_kji = permutation<Index<k,j,i>>(bD_ijk);
        FASTOR_EXIT_ASSERT(abs(norm(bD_kji) - 17777.8111) < HugeTol);


        auto invA = inverse(A);
        FASTOR_EXIT_ASSERT(abs(norm(invA) - 102.506) < HugeTol);
        FASTOR_EXIT_ASSERT(abs(norm(transpose(invA)) - 102.506) < HugeTol);
        FASTOR_EXIT_ASSERT(abs(norm(permutation<Index<j,i>>(invA)) - 102.506) < HugeTol);
        FASTOR_EXIT_ASSERT(abs(trace(invA) + 102.4999) < HugeTol);
        FASTOR_EXIT_ASSERT(abs(trace(matmul(invA,II)) + 102.4999) < HugeTol);
        FASTOR_EXIT_ASSERT(abs(norm(determinant(A)*transpose(invA)) - norm(cofactor(A))) < HugeTol);
        FASTOR_EXIT_ASSERT(abs(inner(invA,A) - 2.5) < BigTol);
        FASTOR_EXIT_ASSERT(abs(inner(D,D) - 7519.642) < HugeTol);
        FASTOR_EXIT_ASSERT(abs((outer(D,D)).sum() - 14237.2624) < HugeTol);
    }


    {
        Tensor<T,3,3> II; II.eye();
        auto II_ijkl = einsum<Index<i,j>,Index<k,l>>(II,II);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(0,0,0,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(0,0,1,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(0,0,2,2) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(1,1,0,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(1,1,1,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(1,1,2,2) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(2,2,0,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(2,2,1,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(2,2,2,2) - 1) < Tol);
        auto II_ijkl2 = outer(II,II);
        FASTOR_EXIT_ASSERT(II_ijkl==II_ijkl2);
        auto II_ikjl = permutation<Index<i,k,j,l>>(II_ijkl);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(0,0,0,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(0,1,0,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(0,2,0,2) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(1,0,1,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(1,1,1,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(1,2,1,2) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(2,0,2,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(2,1,2,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(2,2,2,2) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(norm(II_ijkl) - norm(II_ikjl)) < Tol);
        auto II_iljk = permutation<Index<i,l,j,k>>(II_ijkl);
        FASTOR_EXIT_ASSERT(abs(II_iljk(0,0,0,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(0,1,1,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(0,2,2,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(1,0,0,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(1,1,1,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(1,2,2,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(2,0,0,2) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(2,1,1,2) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(2,2,2,2) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(norm(II_ijkl) - norm(II_iljk)) < Tol);


        Tensor<T,3,3> A, B; A.iota(65); B.iota(13.2);
        Tensor<T,3> D = {124.36, -37.29, 5.61};
        auto bb_ijkl = einsum<Index<i,j>,Index<k,l>>(A,B);
        FASTOR_EXIT_ASSERT(abs(norm(bb_ijkl) - 10808.437) < HugeTol);
        auto bb_ikjl = permutation<Index<i,k,j,l>>(bb_ijkl);
        FASTOR_EXIT_ASSERT(abs(norm(bb_ikjl) - 10808.437) < HugeTol);
        auto bb_iljk = permutation<Index<i,l,j,k>>(bb_ijkl);
        FASTOR_EXIT_ASSERT(abs(norm(bb_iljk) - 10808.437) < HugeTol);
        auto bb_ijkl2 = outer(A,B);
        FASTOR_EXIT_ASSERT(abs(norm(bb_ijkl2) - 10808.437) < HugeTol);

        FASTOR_EXIT_ASSERT(abs(bb_ijkl(0,1,0,1)-bb_ikjl(0,0,1,1)) < BigTol);
        FASTOR_EXIT_ASSERT(abs(bb_ijkl(0,1,0,1)-bb_iljk(0,0,1,1)) < BigTol);

        for (auto ii=0; ii< 3; ii++) {
            for (auto jj=0; jj< 3; jj++) {
                for (auto kk=0; kk< 3; kk++) {
                    for (auto ll=0; ll< 3; ll++) {
                        FASTOR_EXIT_ASSERT(abs( bb_ijkl(ii,jj,kk,ll) - bb_ikjl(ii,kk,jj,ll) ) < BigTol);
                        // FASTOR_EXIT_ASSERT(abs( bb_ijkl(ii,jj,kk,ll) - bb_iljk(ii,ll,jj,kk) ) < BigTol); // By definition this cannot be
                        FASTOR_EXIT_ASSERT(abs( bb_ijkl(ii,ll,jj,kk) - bb_iljk(ii,jj,kk,ll) ) < BigTol);
                    }
                }
            }
        }

        auto bD_ijk = einsum<Index<i,j>,Index<k>>(A,D);
        FASTOR_EXIT_ASSERT(abs(norm(bD_ijk) - 26918.8141) < HugeTol);
        auto bD_ikj = permutation<Index<i,k,j>>(bD_ijk);
        FASTOR_EXIT_ASSERT(abs(norm(bD_ikj) - 26918.8141) < HugeTol);
        auto bD_jki = permutation<Index<j,k,i>>(bD_ijk);
        FASTOR_EXIT_ASSERT(abs(norm(bD_jki) - 26918.8141) < HugeTol);

        A.iota(5); A(0,0) = 100;
        A = matmul(A,inverse(A));
        FASTOR_EXIT_ASSERT(abs(norm(A) - sqrts(T(3))) < HugeTol);
        A.iota(5); A(0,0) = 100;
        auto invA = inverse(A);
        FASTOR_EXIT_ASSERT(abs(norm(invA) - 7.359) < HugeTol);
        FASTOR_EXIT_ASSERT(abs(norm(transpose(invA)) - 7.359) < HugeTol);
        FASTOR_EXIT_ASSERT(abs(norm(permutation<Index<j,i>>(invA)) - 7.359) < HugeTol);
        FASTOR_EXIT_ASSERT(abs(trace(invA) + 7.2701) < HugeTol);
        FASTOR_EXIT_ASSERT(abs(trace(matmul(invA,II)) + 7.2701) < HugeTol);
        FASTOR_EXIT_ASSERT(abs(norm(determinant(A)*transpose(invA)) - norm(cofactor(A))) < HugeTol);

        FASTOR_EXIT_ASSERT(abs(inner(invA,A) - 4.3333333) < BigTol);
        FASTOR_EXIT_ASSERT(abs(inner(D,D) - 16887.4258) < HugeTol);
        FASTOR_EXIT_ASSERT(abs((outer(D,D)).sum() - 8589.5824) < HugeTol);
    }

    {
        Tensor<T,3,5> a1; a1.iota(1);
        FASTOR_EXIT_ASSERT(norm(transpose(a1) - permutation<Index<j,i>>(a1)) < Tol);
        Tensor<T,7,13> a2; a2.iota(14.5);
        FASTOR_EXIT_ASSERT(norm(transpose(a2) - permutation<Index<j,i>>(a2)) < Tol);

        Tensor<T,3,4,5> a3; a3.iota(1.2);

        Tensor<T,3,5,4> a4 = permutation<Index<i,k,j>>(a3);
        FASTOR_EXIT_ASSERT(abs(norm(a3(0,all,all))-norm(a4(0,all,all))) < HugeTol);
        FASTOR_EXIT_ASSERT(abs(norm(a3(1,all,all))-norm(a4(1,all,all))) < HugeTol);
        FASTOR_EXIT_ASSERT(abs(norm(a3(2,all,all))-norm(a4(2,all,all))) < HugeTol);

        Tensor<T,4,5,3> a5 = permutation<Index<j,k,i>>(a3);
        FASTOR_EXIT_ASSERT(abs(norm(a3) - norm(a5)) < HugeTol);
        FASTOR_EXIT_ASSERT(abs(a5(0,0,1) - 6.2)  < BigTol);
        FASTOR_EXIT_ASSERT(abs(a5(0,0,2) - 11.2) < BigTol);
        FASTOR_EXIT_ASSERT(abs(a5(0,1,0) - 16.2) < BigTol);
        FASTOR_EXIT_ASSERT(abs(a5(0,2,0) - 31.2) < BigTol);
        FASTOR_EXIT_ASSERT(abs(a5(0,3,0) - 46.2) < BigTol);
        FASTOR_EXIT_ASSERT(abs(a5(0,4,0) - 2.2)  < BigTol);
        FASTOR_EXIT_ASSERT(abs(a5(3,3,1) - 40.2) < BigTol);
        FASTOR_EXIT_ASSERT(abs(a5(3,4,2) - 60.2) < BigTol);

        Tensor<T,4,3,5> a6 = permutation<Index<j,i,k>>(a3);
        FASTOR_EXIT_ASSERT(abs(norm(a3) - norm(a6)) < HugeTol);
        FASTOR_EXIT_ASSERT(abs(a6(0,0,1) - 2.2)   < BigTol);
        FASTOR_EXIT_ASSERT(abs(a6(0,1,0) - 21.2)  < BigTol);
        FASTOR_EXIT_ASSERT(abs(a6(3,2,3) - 59.2)  < BigTol);

        Tensor<T,3,4,5,6> a7; a7.iota(2);
        Tensor<T,3,4,6,5> a8 = permutation<Index<i,j,l,k>>(a7);
        for (int ii=0; ii<3; ++ii)
            for (int jj=0; jj<3; ++jj)
                FASTOR_EXIT_ASSERT(abs(norm(a7(ii,jj,all,all))-norm(a8(ii,jj,all,all))) < HugeTol);
    }



    {
        Tensor<T,5,5,5> A; Tensor<T,5> B;
        A.iota(1); B.iota(2);

        auto C = einsum<Index<i,j,k>,Index<j>>(A,B);
        FASTOR_EXIT_ASSERT(abs(C.sum() - 32750) < Tol);
        auto D = einsum<Index<j>,Index<i,j,k>>(B,A);
        FASTOR_EXIT_ASSERT(abs(D.sum() - 32750) < Tol);
        auto E = einsum<Index<i,j,k>,Index<i,j,l>>(A,A);
        FASTOR_EXIT_ASSERT(abs(E.sum() - 3293125) < Tol);
        auto F = einsum<Index<i>,Index<k>>(B,B);
        FASTOR_EXIT_ASSERT(abs(F.sum() - 400) < Tol);
    }

    {
        Tensor<double,5,5> A; Tensor<double,5,5,5,5> B; Tensor<double,5> C;
        A.iota(); B.iota(); C.iota();
        auto D = einsum<Index<k,j>,Index<k,i,l,j>,Index<l>>(A,B,C);
        FASTOR_EXIT_ASSERT(abs(D.sum() - 6.32e+06) < BigTol);
    }

    // Test generic tensors
    {
        Tensor<T,5,5,5> A; Tensor<T,5> B;
        A.iota(1); B.iota(2);

        auto C = einsum<Index<i,j,k>,Index<j>>(A-0,1+B-1);
        FASTOR_EXIT_ASSERT(abs(C.sum() - 32750) < Tol);
        auto D = einsum<Index<j>,Index<i,j,k>>(B*1,-A+A+A);
        FASTOR_EXIT_ASSERT(abs(D.sum() - 32750) < Tol);
        auto E = einsum<Index<i,j,k>,Index<i,j,l>>(1*A,A*1);
        FASTOR_EXIT_ASSERT(abs(E.sum() - 3293125) < Tol);
        auto F = einsum<Index<i>,Index<k>>(2*B-B+B-B,sqrt(B)+B-sqrt(B));
        FASTOR_EXIT_ASSERT(abs(F.sum() - 400) < BigTol);
    }

    {
        Tensor<double,5,5> A; Tensor<double,5,5,5,5> B; Tensor<double,5> C;
        A.iota(); B.iota(); C.iota();
        auto D = einsum<Index<k,j>,Index<k,i,l,j>,Index<l>>(2*sin(A)+A-sin(A)-sin(A),B,C+0);
        FASTOR_EXIT_ASSERT(abs(D.sum() - 6.32e+06) < BigTol);
    }

    {
        Tensor<T,2,3,4,5> As; As.iota();
        auto Bs1 = permutation<Index<k,i,j,l>>(As);
        auto Bs2 = permutation<Index<k,i,j,l>>(As-0);
        auto Bs3 = permutation<Index<k,i,j,l>>(1+As-1);
        auto Bs4 = permutation<Index<k,i,j,l>>(2*As-As-sqrt(As)+sqrt(As));
        FASTOR_EXIT_ASSERT(abs(As.sum() - Bs1.sum()) < BigTol);
        FASTOR_EXIT_ASSERT(abs(As.sum() - Bs2.sum()) < BigTol);
        FASTOR_EXIT_ASSERT(abs(As.sum() - Bs3.sum()) < BigTol);
        FASTOR_EXIT_ASSERT(abs(As.sum() - Bs4.sum()) < BigTol);
    }

    {
        Tensor<T,3,2> As; As.iota(1);
        Tensor<T,3> bs; bs.fill(1);
        Tensor<T,2> cs; cs.fill(2);

        FASTOR_EXIT_ASSERT((einsum<Index<i,j>,Index<j>>(As,cs)).sum() - 42. < Tol);
        FASTOR_EXIT_ASSERT((einsum<Index<i,j>,Index<i>>(As,bs)).sum() - 21. < Tol);
        FASTOR_EXIT_ASSERT((einsum<Index<i>,Index<i,j>>(bs,As)).sum() - 21. < Tol);
        FASTOR_EXIT_ASSERT((einsum<Index<j>,Index<i,j>>(cs,As)).sum() - 42. < Tol);
    }

    {
        // Test strided_contraction when second tensor disappears
        Tensor<T,4,4,4> a; a.iota(1);
        Tensor<T,4,4> b; b.iota(1);

        Tensor<T,4> c1 = einsum<Index<i,j,k>,Index<j,k> >(a,b);
        Tensor<T,4> c2 = einsum<Index<i,j,k>,Index<i,k> >(a,b);
        Tensor<T,4> c3 = einsum<Index<i,j,k>,Index<i,j> >(a,b);

        FASTOR_EXIT_ASSERT(abs(c1(0) - 1496.) < Tol);
        FASTOR_EXIT_ASSERT(abs(c1(1) - 3672.) < Tol);
        FASTOR_EXIT_ASSERT(abs(c1(2) - 5848.) < Tol);
        FASTOR_EXIT_ASSERT(abs(c1(3) - 8024.) < Tol);

        FASTOR_EXIT_ASSERT(abs(c2(0) - 4904.) < Tol);
        FASTOR_EXIT_ASSERT(abs(c2(1) - 5448.) < Tol);
        FASTOR_EXIT_ASSERT(abs(c2(2) - 5992.) < Tol);
        FASTOR_EXIT_ASSERT(abs(c2(3) - 6536.) < Tol);

        FASTOR_EXIT_ASSERT(abs(c3(0) - 5576.) < Tol);
        FASTOR_EXIT_ASSERT(abs(c3(1) - 5712.) < Tol);
        FASTOR_EXIT_ASSERT(abs(c3(2) - 5848.) < Tol);
        FASTOR_EXIT_ASSERT(abs(c3(3) - 5984.) < Tol);
    }

    print(FGRN(BOLD("All tests passed successfully")));
}

int main() {

    print(FBLU(BOLD("Testing all einsum features (contractions, permutations, reductions): single precision")));
    run<float>();
    print(FBLU(BOLD("Testing all einsum features (contractions, permutations, reductions): double precision")));
    run<double>();

    return 0;
}