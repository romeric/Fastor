# Fastor
**Fastor** is a stack-based high performance tensor (multi-dimensional array) library written in modern C++ [C++11/14/17] with powerful in-built tensor algebraic functionalities (tensor contraction, permutation, reductions, special tensor groups etc) and focus on specialised highly efficient tensor product kernels. There are multiple paradigms that Fastor exploits:

- **Operation minimisation or FLOP reducing algorithms:** Fastor relies on a domain-aware Expression Template (ET) engine that can not only perform lazy and delayed evaluation but also sophisticated mathematical transformations at *compile time* such as graph optimisation, nearly symbolic tensor algebraic manipulation to reduce the complexity of evaluation of BLAS and/or non-BLAS type expressions by orders of magnitude. Some of these functionalities are non-existent in other available C++ ET linear algebra libraries. As an example of what Fastor can do with expressions at compile time see the section on [smart expression templates](###Smart-expression-templates).
- **Data parallelism/Stream computing** Fastor utilises explicit SIMD instructions (from SSE all the way to AVX512 and FMA) through it's built-in `SIMDVector` layer. This backend is configurable and one can switch to a different implementation of SIMD types for instance to [Vc](https://github.com/VcDevel/Vc) or even to C++20 SIMD data types [std::experimental::simd](https://en.cppreference.com/w/cpp/experimental/simd/simd) which will cover ARM NEON, AltiVec and other potential streaming architectures like GPUs.
- **High performance zero overhead tensor kernels** Combining sophisticated metaprogramming capabilities with statically dispatched bespoke kernels, makes Fastor a highly efficient framework for tensor operations that can rival specialised vendor libraries like [MKL-JIT](https://software.intel.com/en-us/articles/intel-math-kernel-library-improved-small-matrix-performance-using-just-in-time-jit-code) and [LIBXSMM](https://github.com/hfp/libxsmm).

### Documentation
Documenation can be found under the [Wiki](https://github.com/romeric/Fastor/wiki) pages.

### High-level API
Fastor provides a high level interface for tensor algebra. As a first example consider the following
~~~c++
Tensor<double> scalar; // A scalar
Tensor<double,6> vector6; //  A vector
Tensor<double,4,5> matrix; // A second order tensor
Tensor<double,3,3,3> tensor_3; // A third order tensor with dimension 3x3x3
tensor_3.arange(0); // fill tensor with sequentially ascending numbers
print(tensor_3); // print out the tensor
tensor_3(0,2,1); // index a tensor
tensor_3(all,last,seq(0,2)); // slice a tensor
tensor_3.rank(); // get rank of tensor, 3 in this case
Tensor<float,2,2,2,2,1,2,2,4,3,2,3,3,6> tensor_13; // A 13th order tensor
~~~
will output the following
~~~bash
[0,:,:]
⎡      0,       1,       2 ⎤
⎢      3,       4,       5 ⎥
⎣      6,       7,       8 ⎦
[1,:,:]
⎡      9,      10,      11 ⎤
⎢     12,      13,      14 ⎥
⎣     15,      16,      17 ⎦
[2,:,:]
⎡     18,      19,      20 ⎤
⎢     21,      22,      23 ⎥
⎣     24,      25,      26 ⎦
~~~
Einstein summation as well as summing over multiple (i.e. more than two) indices are supported. As a complete example, for instance, consider
~~~c++
#include <Fastor/Fastor.h>
using namespace Fastor;
enum {I,J,K,L,M,N};

int main() {
    // An example of Einstein summation
    Tensor<double,2,3,5> A; Tensor<double,3,5,2,4> B;
    // fill A and B
    A.random(); B.random();
    auto C = einsum<Index<I,J,K>,Index<J,L,M,N>>(A,B);

    // An example of summing over three indices
    Tensor<double,5,5,5> D; D.random();
    auto E = reduction(D);

    // An example of tensor permutation
    Tensor<float,3,4,5,2> F; F.random();
    auto G = permutation<Index<J,K,I,L>>(F);

    // Output the results
    print("Our big tensors:",C,E,G);

    return 0;
}
~~~
You can compile and run this by providing the following (or equivalent) flags to your compiler `-std=c++14 -O3 -march=native -DNDEBUG`.

### Tensor views: A powerful indexing, slicing and broadcasting mechanism
Fastor introduces powerful tensor views which make tensor indexing, slicing and broadcating look and feel native to scientific programmers. Consider the following examples
~~~c++
Tensor<double,4,3,10> A, B;
A.random(); B.random();
Tensor<double,2,2,5> C; Tensor<double,4,3,1> D;

// Dynamic views -> seq(first,last,step)
C = A(seq(0,2),seq(0,2),seq(0,last,2));                              // C = A[0:2,0:2,0::2]
D = B(all,all,0) + A(all,all,last);                                  // D = B[:,:,0] + A[:,:,-1]
A(2,all,3) = 5.0;                                                    // A[2,:,3] = 5.0

// Static views -> fseq<first,last,step>
C = A(fseq<0,2>(),fseq<0,2>(),fseq<0,last,2>());                     // C = A[0:2,0:2,0::2]
D = B(fall,fall,fseq<0,1>()) + A(fall,fall,fseq<9,10>());            // D = B[:,:,0] + A[:,:,-1]
A(2,fall,3) = 5.0;                                                   // A[2,:,3] = 5.0

// Overlapping is also allowed without having undefined behaviour
A(seq(2,last),all,all).noalias() += A(seq(0,last-2),all,all);        // A[2::,:,:] += A[::-2,:,:]
// Note that in case of perfect overlapping noalias is not required
A(seq(0,last-2),all,all) += A(seq(0,last-2),all,all);                // A[::2,:,:] += A[::2,:,:]

// If instead of a tensor view, one needs an actual tensor the iseq could be used
// iseq<first,last,step>
C = A(iseq<0,2>(),iseq<0,2>(),iseq<0,last,2>());                     // C = A[0:2,0:2,0::2]
// Note that iseq returns an immediate tensor rather than a tensor view and hence cannot appear
// on the left hand side, for instance
A(iseq<0,2>(),iseq<0,2>(),iseq<0,last,2>()) = 2; // Will not compile, as left operand is an rvalue

// One can also index a tensor with another tensor(s)
Tensor<float,10,10> E; E.fill(2);
Tensor<int,5> it = {0,1,3,6,8};
Tensor<size_t,10,10> t_it; t_it.arange();
E(it,0) = 2;
E(it,seq(0,last,3)) /= -1000.;
E(all,it) += E(all,it) * 15.;
E(t_it) -= 42 + E;
~~~
Aside from `iseq` (which pretty much immediately returns another tensor), all other possible combination of slicing and broadcasting types are possible. For instance, one complex slicing and broadcasting example is given below
~~~c++
A(all,all) -= log(B(all,all,0)) + abs(B(all,fall,1)) + sin(C(all,0,all,0)) - 102. - cos(B(all,all,0));
~~~

<!-- It should be mentioned that since tensor views work on a view of (reference to) a tensor and do not copy any data in the background, the use of the keyword `auto` can be dangerous at times
~~~c++
auto B = A(all,all,seq(0,5),seq(0,3)); // the scope of view expressions ends with ; as view is a refrerence to an rvalue
auto C = B + 2; // Hence this will sigfault as B refers to a non-existing piece of memory
~~~
To solve this issue, use immediate construction from a view
~~~c++
Tensor<double,2,2,5,3> B = A(all,all,seq(0,5),seq(0,3)); // B is now permanent
auto C = B + 2; // This will behave as expected
~~~ -->
From a performance point of view, Fastor tries very hard to vectorise (read SIMD vectorisation) tensor views, but this heavily depends on the compilers ability to inline multiple recursive functions [as is the case for all expression templates]. If a view appears on the right hand side of an assignment, but not on the left, Fastor automatically vectorises the expression. However if a view appears on the left hand side of an assignment, Fastor does not by default vectorise the expression. To enable vectorisation across all tensor views use the compiler flag `-DFASTOR_USE_VECTORISE_EXPR_ASSIGN`. Also for performance reasons it is beneficial to avoid overlapping assignments, otherwise a copy will be made. If your code does not use any overlapping assignments, then this feature can be turned off completely by issusing `-DFASTOR_NO_ALIAS`. At this stage it is also beneficial to consider that while compiling complex and big expressions the inlining limit of the compiler should be increased and tested i.e. `-finline-limit=<big number>` for GCC, `-mllvm -inline-threshold=<big number>` for Clang and `-inline-forceinline` for ICC.

To see how efficient tensor views can be vectorised, as an example consider the following 4th order finite difference example for Laplace equation
~~~c++
Tensor<double,100,100> u, v;
// fill u and v
// A complex assignment expression involving multiple tensor views
u(seq(1,last-1),seq(1,last-1)) =
    ((  v(seq(0,last-2),seq(1,last-1)) + v(seq(2,last),seq(1,last-1)) +
        v(seq(1,last-1),seq(0,last-2)) + v(seq(1,last-1),seq(2,last)) )*4.0 +
        v(seq(0,last-2),seq(0,last-2)) + v(seq(0,last-2),seq(2,last)) +
        v(seq(2,last),seq(0,last-2))   + v(seq(2,last),seq(2,last)) ) / 20.0;
~~~
using `-O3 -mavx2 -mfma -DNDEBUG -DFASTOR_NO_ALIAS -DFASTOR_USE_VECTORISE_EXPR_ASSIGN` the above expression compiles to
~~~assembly
L129:
  leaq  -768(%rcx), %rdx
  movq  %rsi, %rax
  .align 4,0x90
L128:
  vmovupd 8(%rax), %ymm0
  vmovupd (%rax), %ymm1
  addq  $32, %rdx
  addq  $32, %rax
  vaddpd  1576(%rax), %ymm0, %ymm0
  vaddpd  768(%rax), %ymm0, %ymm0
  vaddpd  784(%rax), %ymm0, %ymm0
  vfmadd132pd %ymm3, %ymm1, %ymm0
  vaddpd  -16(%rax), %ymm0, %ymm0
  vaddpd  1568(%rax), %ymm0, %ymm0
  vaddpd  1584(%rax), %ymm0, %ymm0
  vdivpd  %ymm2, %ymm0, %ymm0
  vmovupd %ymm0, -32(%rdx)
  cmpq  %rdx, %rcx
  jne L128
  vmovupd 2376(%rsi), %xmm0
  vaddpd  776(%rsi), %xmm0, %xmm0
  addq  $800, %rcx
  addq  $800, %rsi
  vaddpd  768(%rsi), %xmm0, %xmm0
  vaddpd  784(%rsi), %xmm0, %xmm0
  vfmadd213pd -32(%rsi), %xmm5, %xmm0
  vaddpd  -16(%rsi), %xmm0, %xmm0
  vaddpd  1568(%rsi), %xmm0, %xmm0
  vaddpd  1584(%rsi), %xmm0, %xmm0
  vdivpd  %xmm4, %xmm0, %xmm0
  vmovups %xmm0, -800(%rcx)
  cmpq  %r13, %rcx
  jne L129
~~~
Aside from unaligned load and store instructions (which are in fact equally fast as aligned load and store) which are also unavoidable in this specific case the rest of the generated code is as efficient as it gets for an `AVX2` architecture beating the perforamnce of Fortran. With the help of an optimising compiler, Fastor's functionalities come closest to the ideal metal performance for numerical tensor algebra code.

### Specialised tensors
A set of specialised tensors are available that provide optimised tensor algebraic computations, for instance `SingleValueTensor`. Some of the computations performed on these tensors have almost zero cost no matter how big the tensor is. These tensors work in the exact same way as the `Tensor` class and can be assigned to one another. Consider for example the einsum between two `SingleValueTensor`s. A `SingleValueTensor` is a tensor of any dimension and size whose elements are all the same (a matrix of ones for instance).

~~~c++
SingleValueTensor<double,20,20,30> a(3.51);
SingleValueTensor<double,20,30> b(2.76);
auto c = einsum<Index<0,1,2>,Index<0,2>>(a,b);
~~~

This will incur almost no runtime cost. As where if the tensors were of type `Tensor` then a heavy computation would ensue.


### Basic expression templates
Expression templates are archetypal of array/tensor libraries in C++ as they provide a means for lazy evaluation of arbitrary chained operations. Consider the following expression

~~~c++
Tensor<float,16,16,16,16> tn1 ,tn2, tn3;
tn1.random(); tn2.random(); tn3.random();
auto tn4 = 2*tn1+sqrt(tn2-tn3);
~~~

Here `tn4` is not another tensor but rather an expression that is not yet evaluated. The expression is evaluated if you explicitly assign it to another tensor or call the free function `evaluate` on the expression

~~~c++
Tensor<float,16,16,16,16> tn5 = tn4;
// or
auto tn6 = evaluate(tn4);
~~~

this mechanism helps chain the operations to avoid the need for intermediate memory allocations. Various re-structuring of the expression before evaluation is possible depending on the chosen policy.

### Smart expression templates

Aside from basic expression templates, by employing further template metaprogrommaing techniques Fastor can mathematically transform expressions and/or apply compile time graph optimisation to find optimal contraction indices of complex tensor networks, for instance. This gives Fastor the ability to re-structure or completely re-write an expression and simplify it rather symbolically. As an example, consider the expression `trace(matmul(transpose(A),B))` which is `O(n^3)` in computational complexity. Fastor can determine this to be inefficient and will statically dispatch the call to an equivalent but much more efficient routine, in this case `A_ij*B_ij` or `inner(A,B)` which is `O(n^2)`. Further examples of such mathemtical transformation include (not inclusive)
~~~c++
determinant(inverse(A)); // transformed to 1/determinant(A), O(n^3) reduction in computation
transpose(cofactor(A));  // transformed to adjoint(A), O(n^2) reduction in memory access
transpose(adjoint(A));   // transformed to cofactor(A), O(n^2) reduction in memory access
matmul(matmul(A,B),b);   // transformed to matmul(A,matmul(B,b)), O(n) reduction in computation
// and many more
~~~
These expressions are not treated as special cases but rather the **Einstein indicial notation** of the whole expression is constructed under the hood and by simply simplifying/collapsing the indices one obtains the most efficient form that an expression can be evaluated. The expression is then sent to an optimised kernel for evaluation. Note that there are situations that the user may write a complex chain of operations in the most verbose/obvious way perhaps for readibility purposes, but Fastor delays the evaluation of the expression and checks if an equivalent but efficient expression can be computed.

For tensor networks comprising of many higher rank tensors, a full generalisation of the above mathematical transformation can be performed through a constructive graph search optimisation. This typically involves finding the most optimal pattern of tensor contraction by studying the indices of contraction wherein tensor pairs are multiplied, summed over and factorised out in all possible combinations in order to come up with a cost model. Once again, knowing the dimensions of the tensor and the contraction pattern, Fastor performs this operation minimisation step at *compile time* and further checks the SIMD vectorisability of the tensor contraction loop nest (i.e. full/partial/broadcast vectorisation). In nutshell, it not only minimises the the number of floating point operations but also generates the most optimum vectorisable loop nest for computing those FLOPs. The following figures show the run time benefit of operation minimisation (FLOP optimal) over a single expression evaluation (Memory-saving) approach in contracting a three-tensor-network fitting in `L1`, `L2` and `L3` caches, respectively
<p align="left">
  <img src="docs/imgs/05l1.png" width="280">
  <img src="docs/imgs/05l2.png" width="280">
  <img src="docs/imgs/05l3.png" width="280">
</p>
The X-axis shows the number FLOPS saved/reduced over single expression evaluation scheme. Certainly, the bigger the size of tensors the more reduction in FLOPs is necessary to compensate for the temporaries created during by-pair evalution.


### Domain-aware numerical analysis
In nonlinear mechanics, it is customary to transform high order tensors to low rank tensors using Voigt transformation. Fastor has domain-specific features for such tensorial operations. For example, consider the dyadic product `A_ik*B_jl`, that can be computed in Fastor like
~~~c++
Tensor<double,3,3> A,B;
A.random(); B.random();
Tensor<double,6,6> C = einsum<Index<0,2>,Index<1,3>,FASTOR_Voigt>(A,B);
// or alternatively
enum {I,J,K,L};
Tensor<double,6,6> D = einsum<Index<I,K>,Index<J,L>,FASTOR_Voigt>(A,B);
~~~

As you notice, all indices are resolved and the Voigt transformation is performed at compile time, keeping only the cost of computation at runtime. Equivalent implementation of this in C/Fortran requires either low-level for loop style programming that has an O(n^4) computational complexity and non-contiguous memory access, or if a function like einsum is desired the indices will need to be passed requiring potentially extra register allocation. Here is performance benchmark between Ctran (C/Fortran) for loop code and the equivalent Fastor implementation for the above example, run over a million times (both compiled using `-O3 -mavx`, on `Intel(R) Xeon(R) CPU E5-2650 v2 @2.60GHz` running `Ubuntu 14.04`):


<p align="center">
  <img src="docs/imgs/cyclic_bench.png" width="600" align="middle">
</p>


Notice that by compiling with the same flags, it is meant that the compiler is permitted to auto-vectorise the C/tran code as well. The real performance of Fastor comes from the fact, that when a Voigt transformation is requested, Fastor does not compute the elements which are not needed.
### The tensor cross product and its associated algebra
Building upon its domain specific features, Fastor implements the tensor cross product family of algebra recently introduced by [Bonet et. al.](http://dx.doi.org/10.1016/j.ijsolstr.2015.12.030) in the context of numerical analysis of nonlinear classical mechanics which can significantly reduce the amount algebra involved in tensor derivatives of functionals which are forbiddingly complex to derive using a standard approach. The tensor cross product of two second order tensors is defined as `C_iI = e_ijk*e_IJK*A_jJ*b_kK` where `e` is the third order permutation tensor. As can be seen this product is O(n^6) in computational complexity (furthermore a cross product is essentially defined in 3-dimensional space i.e. perfectly suitable for stack allocation). Using Fastor the equivalent code is only 81 SSE intrinsics
~~~c++
// A and B are second order tensors
using Fastor::LeviCivita_pd;
Tensor<double,3,3> E = einsum<Index<i,j,k>,Index<I,J,K>,Index<j,J>,Index<k,K>>
                       (LeviCivita_pd,LeviCivita_pd,A,B);
// or simply
Tensor<double,3,3> F = cross(A,B);
~~~
Here is performance benchmark between Ctran (C/Fortran) code and the equivalent Fastor implementation for the above example, run over a million times (both compiled using `-O3 -mavx`, on `Intel(R) Xeon(R) CPU E5-2650 v2 @2.60GHz` running `Ubuntu 14.04`):


<p align="center">
  <img src="docs/imgs/tensor_cross_bench.png" width="600" align="middle">
</p>


Notice the almost two orders of magnitude performance gain using Fastor. Again the real performance gain comes from the fact that Fastor eliminates zeros from the computation.


### Boolean tensor algebra
A set of boolean tensor routines are available in Fastor. Note that, whenever possible most of these operations are performed at compile time
~~~c++
is_uniform();   // does the tensor span equally in all spatial dimensions, generalisation of square matrices
is_orthogonal();
does_belong_to_sl3(); // does the tensor belong to special linear 3D group
does_belong_to_so3(); // does the tensor belong to special orthogonal 3D group
is_symmetric(int axis_1, int axis_2); // is the tensor symmetric in the axis_1 x axis_2 plane
is_equal(B); // equality check with another tensor
is_identity();
~~~

### Basic SIMD optimised linear algebra routines for small tensors
All basic numerical linear algebra subroutines for small tensors (where the overhead of calling vendor/optimised `BLAS` is typically not worth it) are fully SIMD optimised and efficiently implemented
~~~c++
Tensor<double,3,3> A,B;
// fill A and B
auto ab = matmul(A,B);          // matrix matrix multiplication of A*B
auto a_norm = norm(A);          // Frobenious norm of A
auto b_det = determinant(B);    // determinant of B
auto a_inv = inverse(A);        // inverse of A
auto b_cof = cofactor(B);       // cofactor of B
~~~

### Template meta-programming for powerful tensor contraction/permutation
Fastor utilises a bunch of meta-functions to perform most operations at compile time, consider the following examples
~~~c++
Tensor<double,3,4,5> A;
Tensor<double,5,3,4> B;
Tensor<double,3,3,3> C;
auto D = permutation<Index<2,0,1>>(A); // type of D is deduced at compile time as Tensor<double,5,3,4>
auto E = einsum<Index<I,J,K>,Index<L,M,N>>(D,B); // type of E is deduced at compile time as Tensor<double,5,3,4,5,3,4>
auto F = einsum<Index<I,I,J>>(C); // type of F is deduced at compile time as Tensor<double,3>
auto F2 = reduction(C); // type of F2 is deduced at compile time as scalar i.e. Tensor<double>
auto E2 = reduction(D,B); // type of E2 is deduced at compile time as Tensor<double>
Tensor<float,2,2> G,H;
trace(H); // trace of H, in other words H_II
reduction(G,H); // double contraction of G and H i.e. G_IJ*H_IJ
~~~
As you can observe with combination of `permutation`, `contraction`, `reduction` and `einsum` (which itself is a glorified wrapper over the first three) any type of tensor contraction, and permutation is possible, and using meta-programming the right amount of stack memory to be allocated is deduced at compile time.

### A minimal framework
Fastor is extremely light weight, it is a *header-only* library, requires no build or compilation process and has no external dependencies. It is written in pure C++11 from the foundation.

### Tested Compilers
Fastor has been tested against the following compilers (on Ubuntu 14.04/16.04/18.04 and macOS). While compiling on macOS with Clang, `-std=c++14` is necessary
- GCC 4.8, GCC 4.9, GCC 5.1, GCC 5.2, GCC 5.3, GCC 5.4, GCC 6.2, GCC 7.3, GCC 8, GCC 9.1
- Clang 3.6, Clang 3.7, Clang 3.8, Clang 3.9, Clang 5, Clang 7, Clang 8
- Intel 16.0.1, Intel 16.0.2, Intel 16.0.3, Intel 17.0.1, Intel 18.2, Intel 19.2

### Reference/Citation
Fastor can be cited as
````latex
@Article{Poya2017,
    author="Poya, Roman and Gil, Antonio J. and Ortigosa, Rogelio",
    title = "A high performance data parallel tensor contraction framework: Application to coupled electro-mechanics",
    journal = "Computer Physics Communications",
    year="2017",
    doi = "http://dx.doi.org/10.1016/j.cpc.2017.02.016",
    url = "http://www.sciencedirect.com/science/article/pii/S0010465517300681"
}
````