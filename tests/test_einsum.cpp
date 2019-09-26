#include <Fastor.h>
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
        assert(abs(II_ijkl(0,0,0,0) - 1) < Tol); 
        assert(abs(II_ijkl(0,0,1,1) - 1) < Tol); 
        assert(abs(II_ijkl(1,1,0,0) - 1) < Tol); 
        assert(abs(II_ijkl(1,1,1,1) - 1) < Tol); 
        auto II_ijkl2 = outer(II,II);
        assert(II_ijkl==II_ijkl2);
        auto II_ikjl = permutation<Index<i,k,j,l>>(II_ijkl);
        assert(abs(II_ikjl(0,0,0,0) - 1) < Tol); 
        assert(abs(II_ikjl(0,1,0,1) - 1) < Tol); 
        assert(abs(II_ikjl(1,0,1,0) - 1) < Tol); 
        assert(abs(II_ikjl(1,1,1,1) - 1) < Tol); 
        assert(abs(norm(II_ijkl) - norm(II_ikjl)) < Tol);
        auto II_iljk = permutation<Index<i,l,j,k>>(II_ijkl);
        assert(abs(II_iljk(0,0,0,0) - 1) < Tol); 
        assert(abs(II_iljk(0,1,1,0) - 1) < Tol); 
        assert(abs(II_iljk(1,0,0,1) - 1) < Tol); 
        assert(abs(II_iljk(1,1,1,1) - 1) < Tol); 
        assert(abs(norm(II_ijkl) - norm(II_iljk)) < Tol); 


        Tensor<T,2,2> A, B; A.iota(101); B.iota(77);
        Tensor<T,2> D = {45.5, 73.82};
        auto bb_ijkl = einsum<Index<i,j>,Index<k,l>>(A,B);
        assert(abs(norm(bb_ijkl) - 32190.178937) < HugeTol);
        auto bb_ikjl = permutation<Index<i,k,j,l>>(bb_ijkl);
        assert(abs(norm(bb_ikjl) - 32190.178937) < HugeTol);
        auto bb_iljk = permutation<Index<i,l,j,k>>(bb_ijkl);
        assert(abs(norm(bb_iljk) - 32190.178937) < HugeTol);
        auto bb_ijkl2 = outer(A,B);
        assert(abs(norm(bb_ijkl2) - 32190.178937) < HugeTol);


        assert(abs(bb_ijkl(0,1,0,1)-bb_ikjl(0,0,1,1)) < BigTol);
        assert(abs(bb_ijkl(0,1,0,1)-bb_iljk(0,0,1,1)) < BigTol);

        for (auto ii=0; ii< 2; ii++) {
            for (auto jj=0; jj< 2; jj++) {
                for (auto kk=0; kk< 2; kk++) {
                    for (auto ll=0; ll< 2; ll++) {
                        assert(abs( bb_ijkl(ii,jj,kk,ll) - bb_ikjl(ii,kk,jj,ll) ) < BigTol);
                        // assert(abs( bb_ijkl(ii,jj,kk,ll) - bb_iljk(ii,ll,jj,kk) ) < BigTol); // By definition this cannot be
                        assert(abs( bb_ijkl(ii,ll,jj,kk) - bb_iljk(ii,jj,kk,ll) ) < BigTol); 
                    }
                }
            }
        }

        auto bD_ijk = einsum<Index<i,j>,Index<k>>(A,D);
        assert(abs(norm(bD_ijk) - 17777.8111) < HugeTol);
        auto bD_ikj = permutation<Index<i,k,j>>(bD_ijk);
        assert(abs(norm(bD_ikj) - 17777.8111) < HugeTol);
        auto bD_jki = permutation<Index<j,k,i>>(bD_ijk);
        assert(abs(norm(bD_jki) - 17777.8111) < HugeTol);
        auto bD_kji = permutation<Index<k,j,i>>(bD_ijk);
        assert(abs(norm(bD_kji) - 17777.8111) < HugeTol);


        auto invA = inverse(A);
        assert(abs(norm(invA) - 102.506) < HugeTol);
        assert(abs(norm(transpose(invA)) - 102.506) < HugeTol);
        assert(abs(norm(permutation<Index<j,i>>(invA)) - 102.506) < HugeTol);
        assert(abs(trace(invA) + 102.4999) < HugeTol);
        assert(abs(trace(matmul(invA,II)) + 102.4999) < HugeTol);
        assert(abs(norm(determinant(A)*transpose(invA)) - norm(cofactor(A))) < HugeTol);
        assert(abs(inner(invA,A) - 2.5) < BigTol);
        assert(abs(inner(D,D) - 7519.642) < HugeTol);
        assert(abs((outer(D,D)).sum() - 14237.2624) < HugeTol);
    }


    {
        Tensor<T,3,3> II; II.eye();
        auto II_ijkl = einsum<Index<i,j>,Index<k,l>>(II,II);
        assert(abs(II_ijkl(0,0,0,0) - 1) < Tol); 
        assert(abs(II_ijkl(0,0,1,1) - 1) < Tol);
        assert(abs(II_ijkl(0,0,2,2) - 1) < Tol); 
        assert(abs(II_ijkl(1,1,0,0) - 1) < Tol); 
        assert(abs(II_ijkl(1,1,1,1) - 1) < Tol);
        assert(abs(II_ijkl(1,1,2,2) - 1) < Tol); 
        assert(abs(II_ijkl(2,2,0,0) - 1) < Tol); 
        assert(abs(II_ijkl(2,2,1,1) - 1) < Tol);
        assert(abs(II_ijkl(2,2,2,2) - 1) < Tol); 
        auto II_ijkl2 = outer(II,II);
        assert(II_ijkl==II_ijkl2);
        auto II_ikjl = permutation<Index<i,k,j,l>>(II_ijkl);
        assert(abs(II_ikjl(0,0,0,0) - 1) < Tol); 
        assert(abs(II_ikjl(0,1,0,1) - 1) < Tol); 
        assert(abs(II_ikjl(0,2,0,2) - 1) < Tol); 
        assert(abs(II_ikjl(1,0,1,0) - 1) < Tol); 
        assert(abs(II_ikjl(1,1,1,1) - 1) < Tol); 
        assert(abs(II_ikjl(1,2,1,2) - 1) < Tol);
        assert(abs(II_ikjl(2,0,2,0) - 1) < Tol);  
        assert(abs(II_ikjl(2,1,2,1) - 1) < Tol); 
        assert(abs(II_ikjl(2,2,2,2) - 1) < Tol); 
        assert(abs(norm(II_ijkl) - norm(II_ikjl)) < Tol);
        auto II_iljk = permutation<Index<i,l,j,k>>(II_ijkl);
        assert(abs(II_iljk(0,0,0,0) - 1) < Tol); 
        assert(abs(II_iljk(0,1,1,0) - 1) < Tol);
        assert(abs(II_iljk(0,2,2,0) - 1) < Tol); 
        assert(abs(II_iljk(1,0,0,1) - 1) < Tol); 
        assert(abs(II_iljk(1,1,1,1) - 1) < Tol); 
        assert(abs(II_iljk(1,2,2,1) - 1) < Tol); 
        assert(abs(II_iljk(2,0,0,2) - 1) < Tol); 
        assert(abs(II_iljk(2,1,1,2) - 1) < Tol); 
        assert(abs(II_iljk(2,2,2,2) - 1) < Tol); 
        assert(abs(norm(II_ijkl) - norm(II_iljk)) < Tol); 


        Tensor<T,3,3> A, B; A.iota(65); B.iota(13.2);
        Tensor<T,3> D = {124.36, -37.29, 5.61};
        auto bb_ijkl = einsum<Index<i,j>,Index<k,l>>(A,B);
        assert(abs(norm(bb_ijkl) - 10808.437) < HugeTol);
        auto bb_ikjl = permutation<Index<i,k,j,l>>(bb_ijkl);
        assert(abs(norm(bb_ikjl) - 10808.437) < HugeTol);
        auto bb_iljk = permutation<Index<i,l,j,k>>(bb_ijkl);
        assert(abs(norm(bb_iljk) - 10808.437) < HugeTol);
        auto bb_ijkl2 = outer(A,B);
        assert(abs(norm(bb_ijkl2) - 10808.437) < HugeTol);

        assert(abs(bb_ijkl(0,1,0,1)-bb_ikjl(0,0,1,1)) < BigTol);
        assert(abs(bb_ijkl(0,1,0,1)-bb_iljk(0,0,1,1)) < BigTol);

        for (auto ii=0; ii< 3; ii++) {
            for (auto jj=0; jj< 3; jj++) {
                for (auto kk=0; kk< 3; kk++) {
                    for (auto ll=0; ll< 3; ll++) {
                        assert(abs( bb_ijkl(ii,jj,kk,ll) - bb_ikjl(ii,kk,jj,ll) ) < BigTol);
                        // assert(abs( bb_ijkl(ii,jj,kk,ll) - bb_iljk(ii,ll,jj,kk) ) < BigTol); // By definition this cannot be
                        assert(abs( bb_ijkl(ii,ll,jj,kk) - bb_iljk(ii,jj,kk,ll) ) < BigTol); 
                    }
                }
            }
        }

        auto bD_ijk = einsum<Index<i,j>,Index<k>>(A,D);
        assert(abs(norm(bD_ijk) - 26918.8141) < HugeTol);
        auto bD_ikj = permutation<Index<i,k,j>>(bD_ijk);
        assert(abs(norm(bD_ikj) - 26918.8141) < HugeTol);
        auto bD_jki = permutation<Index<j,k,i>>(bD_ijk);
        assert(abs(norm(bD_jki) - 26918.8141) < HugeTol);

        A.iota(5); A(0,0) = 100;
        A = matmul(A,inverse(A));
        assert(abs(norm(A) - sqrts(T(3))) < HugeTol);
        A.iota(5); A(0,0) = 100;
        auto invA = inverse(A);
        assert(abs(norm(invA) - 7.359) < HugeTol);
        assert(abs(norm(transpose(invA)) - 7.359) < HugeTol);
        assert(abs(norm(permutation<Index<j,i>>(invA)) - 7.359) < HugeTol);
        assert(abs(trace(invA) + 7.2701) < HugeTol);
        assert(abs(trace(matmul(invA,II)) + 7.2701) < HugeTol);
        assert(abs(norm(determinant(A)*transpose(invA)) - norm(cofactor(A))) < HugeTol);

        assert(abs(inner(invA,A) - 4.3333333) < BigTol);
        assert(abs(inner(D,D) - 16887.4258) < HugeTol);
        assert(abs((outer(D,D)).sum() - 8589.5824) < HugeTol);
    }



    {
        Tensor<T,5,5,5> A; Tensor<T,5> B;
        A.iota(1); B.iota(2);

        auto C = einsum<Index<i,j,k>,Index<j>>(A,B);
        assert(abs(C.sum() - 32750) < Tol); 
        auto D = einsum<Index<j>,Index<i,j,k>>(B,A);
        assert(abs(D.sum() - 32750) < Tol); 
        auto E = einsum<Index<i,j,k>,Index<i,j,l>>(A,A);
        assert(abs(E.sum() - 3293125) < Tol); 
        auto F = einsum<Index<i>,Index<k>>(B,B);
        assert(abs(F.sum() - 400) < Tol); 
    }

    {
        Tensor<double,5,5> A; Tensor<double,5,5,5,5> B; Tensor<double,5> C;
        A.iota(); B.iota(); C.iota();
        auto D = einsum<Index<k,j>,Index<k,i,l,j>,Index<l>>(A,B,C);
        assert(abs(D.sum() - 6.32e+06) < BigTol); 
    }

    // Test generic tensors
    {
        Tensor<T,5,5,5> A; Tensor<T,5> B;
        A.iota(1); B.iota(2);

        auto C = einsum<Index<i,j,k>,Index<j>>(A-0,1+B-1);
        assert(abs(C.sum() - 32750) < Tol); 
        auto D = einsum<Index<j>,Index<i,j,k>>(B*1,-A+A+A);
        assert(abs(D.sum() - 32750) < Tol); 
        auto E = einsum<Index<i,j,k>,Index<i,j,l>>(1*A,A*1);
        assert(abs(E.sum() - 3293125) < Tol); 
        auto F = einsum<Index<i>,Index<k>>(2*B-B+B-B,sqrt(B)+B-sqrt(B));
        assert(abs(F.sum() - 400) < BigTol); 
    }

    {
        Tensor<double,5,5> A; Tensor<double,5,5,5,5> B; Tensor<double,5> C;
        A.iota(); B.iota(); C.iota();
        auto D = einsum<Index<k,j>,Index<k,i,l,j>,Index<l>>(2*sin(A)+A-sin(A)-sin(A),B,C+0);
        assert(abs(D.sum() - 6.32e+06) < BigTol); 
    }

    {
        Tensor<T,2,3,4,5> As; As.iota();
        auto Bs1 = permutation<Index<k,i,j,l>>(As);
        auto Bs2 = permutation<Index<k,i,j,l>>(As-0);
        auto Bs3 = permutation<Index<k,i,j,l>>(1+As-1);
        auto Bs4 = permutation<Index<k,i,j,l>>(2*As-As-sqrt(As)+sqrt(As));
        assert(abs(As.sum() - Bs1.sum()) < BigTol); 
        assert(abs(As.sum() - Bs2.sum()) < BigTol); 
        assert(abs(As.sum() - Bs3.sum()) < BigTol); 
        assert(abs(As.sum() - Bs4.sum()) < BigTol); 
    }

    {
        Tensor<T,3,2> As; As.iota(1);
        Tensor<T,3> bs; bs.fill(1);
        Tensor<T,2> cs; cs.fill(2);

        assert((einsum<Index<i,j>,Index<j>>(As,cs)).sum() - 42. < Tol);
        assert((einsum<Index<i,j>,Index<i>>(As,bs)).sum() - 21. < Tol);
        assert((einsum<Index<i>,Index<i,j>>(bs,As)).sum() - 21. < Tol);
        assert((einsum<Index<j>,Index<i,j>>(cs,As)).sum() - 42. < Tol);
    }

    {
        // Test strided_contraction when second tensor disappears
        Tensor<T,4,4,4> a; a.iota(1);
        Tensor<T,4,4> b; b.iota(1);

        Tensor<T,4> c1 = einsum<Index<i,j,k>,Index<j,k> >(a,b);
        Tensor<T,4> c2 = einsum<Index<i,j,k>,Index<i,k> >(a,b);
        Tensor<T,4> c3 = einsum<Index<i,j,k>,Index<i,j> >(a,b);

        assert (abs(c1(0) - 1496.) < Tol);
        assert (abs(c1(1) - 3672.) < Tol);
        assert (abs(c1(2) - 5848.) < Tol);
        assert (abs(c1(3) - 8024.) < Tol);

        assert (abs(c2(0) - 4904.) < Tol);
        assert (abs(c2(1) - 5448.) < Tol);
        assert (abs(c2(2) - 5992.) < Tol);
        assert (abs(c2(3) - 6536.) < Tol);

        assert (abs(c3(0) - 5576.) < Tol);
        assert (abs(c3(1) - 5712.) < Tol);
        assert (abs(c3(2) - 5848.) < Tol);
        assert (abs(c3(3) - 5984.) < Tol);
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