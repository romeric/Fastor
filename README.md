# Fastor
**Fastor** is a **FA**st **S**IMD op**T**imised tens**OR** algebra framework, with emphasis on tensor contraction algorithms typically arising in the mechanics of nonlinear solids, fluids and coupled continua. There are two paradigms that Fastor exploits:

- **Complexity reducing/low flop algorithms** via statically dispatched (zero-overhead) bespoke kernels for a set of tensor products using either a priori knowledge of tensors or smart expression templates or both
- **Data parallelism/stream computing** by utilising explicit SIMD (SSE/AVX) intrinsics

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

### No heap allocation
Fastor is essentially designed for small mutlidimensional tensors, that can appear in computing stresses, work conjugates, Hessian etc, during numerical integration in a finite element framework. As can be seen from the above examples, Fastor is based on fixed size static arrays (entirely stack allocation). The dimensions of the tensors must be known at compile time, which is typically the case for the use-cases it is designed for. 
### Static disptaching for absolute branchless code
This is a strong statement to make, but Fastor strives to generate optimised SIMD code by utilising the static nature of tensors and the `SFINAE` (Substitution Failure Is Not an Error) feature of `C++11` to statically dispatch calls to bespoke kernels, which completely avoids the need for runtime branching. For example the double contraction of two second order double precision tensors  `A` and `B`, `A_ij*B_ij` with dimensions `2x2`, is statically dispatched to 
~~~c++
return _mm256_sum_pd(_mm256_mul_pd(_mm256_load_pd(A._data),_mm256_load_pd(B._data)));
~~~
(Notice that the double contraction of two second order tensors of `2x2` requires `4 multiplication + 3 addition` which using SIMD lanes can be reduced to `1 multiplication + 1 addition`. Also `_mm256_sum_pd` is Fastor's in-built extension to SIMD intrinsics.) while for `3x3` double precision second order tensors the call is dispatched to 
~~~c++
__m256d r1 = _mm256_mul_pd(_mm256_load_pd(A._data),_mm256_load_pd(B._data));
__m256d r2 = _mm256_mul_pd(_mm256_load_pd(A._data+4),_mm256_load_pd(B._data+4));
__m128d r3 = _mm_mul_sd(_mm_load_sd(A._data+8),_mm_load_sd(B._data+8));
__m128d summ = _mm_add_pd(_add_pd(r3),_add_pd(_mm256_add_pd(r1,r2)));
return _mm_cvtsd_f64(summ);
~~~
without the need for a branch or a potential `jmp` instruction in assembly (again note that `9 multiplication + 8 addition` is reduced to `3 multiplication + 3 addition`). The main motivation behind customising/optimising these operations for such small tensors is that they are typically needed in the critical hotspots of finite element implementations (i.e. they almost always happen to appear at every quadrature point).  

### Tensor contraction example and performance benchmark
Consider the dyadic product `A_ik*B_jl`, that can be computed in Fastor like 
~~~c++
Tensor<double,3,3> A,B;
A.random(); B.random();
using Fastor::Voigt;
Tensor<double,6,6> C = einsum<Index<0,2>,Index<1,3>,Voigt>(A,B);
// or alternatively
enum {I,J,K,L};
Tensor<double,6,6> D = einsum<Index<I,K>,Index<J,L>,Voigt>(A,B);
~~~

As you notice, all indices are resolved and the Voigt transformation is performed at compile time, keeping only the cost of computation at runtime. Equivalent implementation of this in C/Fortran requires either low-level for loop style programming that has an O(n^4) computational complexity and non-contiguous memory access, or if a function like einsum is desired the indices will need to be passed requiring potentially extra register allocation. Here is performance benchmark between Ctran (C/Fortran) for loop code and the equivalent Fastor implementation all normalised by performance of `Fastor GCC 5.3.0`, for the above example, run over a million times (both compiled using `-O3 -mavx`, on `Intel(R) Xeon(R) CPU E5-2650 v2 @2.60GHz` running `Ubuntu 14.04`):       

|  Ctran GCC 5.3.0        | Fastor GCC 5.3.0    |  Ctran Clang 3.8.0       | Fastor Clang 3.8.0 | 
| ----------------------- |:-------------------:| ------------------------:| ------------------ |
| 6.163                   | 1                   | 4.461                    | 1.080              |

Notice that by compiling with the same flags, it is meant that the compiler is permitted to auto-vectorise the C/tran code as well, hence the 2-3 times speed-up should be considered a reasonally good speed-up over what the compiler can offer.
### The tensor cross product and its associated algebra
If not the main, one of the main motivations behind developing Fastor has been the recently introduced tensor cross product by [Bonet et. al.](http://dx.doi.org/10.1016/j.ijsolstr.2015.12.030) in the context of nonlinear solid mechanics which can significantly reduce the amount algebra involved in consistent linearisation of functionals which are forbiddingly complex to derive using the classical approach. The tensor cross product of two second order tensors is defined as `C_iI = e_ijk*e_IJK*A_jJ*b_kK` where `e` is the third order permutation tensor. As can be seen this product is O(n^6) in computational complexity (furthermore a cross product is essentially defined in 3-dimensional space i.e. perfectly suitable for stack allocation). Using Fastor the equivalent code is only 81 SSE intrinsics
~~~c++
// A and B are second order tensors
using Fastor::LeviCivita_pd;
Tensor<double,3,3> E = einsum<Index<i,j,k>,Index<I,J,K>,Index<j,J>,Index<k,K>>
                       (LeviCivita_pd,LeviCivita_pd,A,B);
// or simply
Tensor<double,3,3> F = cross(A,B);
~~~
Here is performance benchmark between Ctran (C/Fortran) for loop code and the equivalent Fastor implementation all normalised by performance of `Fastor GCC 5.3.0`, for the above example, run over a million times (both compiled using `-O3 -mavx`, on `Intel(R) Xeon(R) CPU E5-2650 v2 @2.60GHz` running `Ubuntu 14.04`):       

|  Ctran Gcc 5.3.0        | Fastor GCC 5.3.0    |  Ctran Clang 3.8.0       | Fastor Clang 3.8.0 | 
| ----------------------- |:-------------------:| ------------------------:| ------------------ |
| 119.327                 | 1                   | 246.859                  | 1.002              |

Notice over two orders of magnitude performance gain using Fastor!

### Smart expression templates
A must have feature of every numerical linear algebra and even more so tensor contraction frameworks is lazy evaluation of arbitrary chained operations. Consider the following expression

~~~c++
Tensor<float,16,16,16,16> tn1 ,tn2, tn3, tn4;
tn1.random(); tn2.random(); tn3.random();
tn4 = 2*tn1+sqrt(tn2-tn3);
~~~
The above code is transparently converted to a single `AVX` loop
~~~c++
for (size_t i=0; i<tn4.Size; i+=tn4.Stride)
    _mm256_store_ps(tn4._data+i,_mm256_set1_ps(static_cast<float>(2))*_mm256_load_ps(tn1._data+i)+
    _mm256_sqrt_ps(_mm256_sub_ps(_mm256_load_ps(tn2._data+i),_mm256_load_ps(tn3._data+i)));
~~~
avoiding any need for temporary memory allocation. Importantly, Fastor goes deeper into the realm of *smart* expression templates to find optimal networks of tensor contraction, for instance an expression like `trace(transpose(A),B)` which is `O(n^3)` in computational complexity is determined to be inefficient and Fastor statically dispatches the call to an equivalent but much more efficient routine, in this case `A_ijB_ij` or `doublecontract(A,B)` which is `O(n^2)`. Furthermore, Fastor is all about small stack-allocated on cache tensors, and the overhead of calling vendor/optimised `BLAS` is not worth it. Further examples include
~~~c++
// the l in-front of the names stands for 'lazy'
ldeterminant(linverse(A)); // transformed to 1/ldeterminant(A), O(n^3) reduction in computation
ltranspose(lcofactor(A));  // transformed to ladjoint(A), O(n^2) reduction in memory access
ltranspose(ladjoint(A));   // transformed to lcofactor(A), O(n^2) reduction in memory access
~~~
Note that there are situations that the user writes a complex chain of operations, perhaps for readibility purpose, but Fastor delays the evaluation of the expression and checks if an equivalent but efficient expression can be computed. The computed expression always binds back to the base tensor overhead free, without a runtime virtual table/pointer penalty.  

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
All basic numerical linear algebra subroutines for small tensors are completely SIMD optimised 
~~~c++
Tensor<double,3,3> A,B; 
// fill A and B
A+B;                            // element-wise addition A and B
A-B;                            // element-wise subtraction A and B
A*B;                            // element-wise multiplication A and B
A/B;                            // element-wise division A and B                   
auto ab = matmul(A,B);          // matrix matrix multiplication of A*B
auto a_norm = norm(A);          // Frobenious norm of A
auto a_det = determinant(B);    // determinant of B
auto a_inv = inverse(A);        // inverse of A
auto a_cof = cofactor(B);       // cofactor of A
~~~

### Template meta-programming for powerful tensor contraction/permutation
Fastor utilises a bunch of meta-functions to perform most operations at compile time, consider the following examples
~~~c++
Tensor<double,3,4,5> A;
Tensor<double,5,3,4> B;
Tensor<double,3,3,3> C;
auto D = permutation<Index<2,0,1>>(A); // type of D is deduced at compile time as Tensor<double,5,3,4>
auto E = einsum<Index<I,J,K>,Index<L,M,N>>(D,B); // type of E is deduced at compile time as Tensor<double,5,3,4,5,3,4>
auto F = einsum<Index<I,I,I>>(C); // type of F is deduced at compile time as scalar i.e. Tensor<double>
auto F2 = reduction(C); // same as above, returned value is a scalar Tensor<double>
auto E2 = einsum<Index<I,J,K>,Index<I,J,K>>(D,B); // type of E is deduced at compile time as Tensor<double>
Tensor<float,2,2> G,H;
einsum<Index<I,I>>(H); // trace of H
einsum<Index<I,J>,Index<I,J>>(G,H); // double contraction of G and H
~~~
As you can observe with combination of `permutation`, `contraction`, `reduction` and `einsum` (which itself is a glorified wrapper over the first three) any type of tensor contraction, and permutation that you can percieve of, is possible, and using meta-programming the right amount of stack memory to be allocated is deduced at compile time.

### Similar Projects:
Similar projects exist in particular
- [FTensor](http://www.wlandry.net/Projects/FTensor): Up to rank 4 dense tensor framework
- [LTensor](https://code.google.com/archive/p/ltensor/): Up to rank 4 dense tensor framework
- [libtensor](https://github.com/juanjosegarciaripoll/tensor): Up to rank 6 dense tensor framework
- [Eigen's unsupported tensor algebra package](http://eigen.tuxfamily.org/index.php?title=Tensor_support): Arbitrary rank dense tensor module
- [Blitz++'s tensor module](http://blitz.sourceforge.net/): Up to rank 11, dense tensor algebra 
- [TiledArray](https://github.com/ValeevGroup/tiledarray): Massively parallel arbitrary rank block sparse tensor framework based on Eigen
- [Cyclops Tensor Framework](https://github.com/solomonik/ctf): Distributed memory arbitrary rank sparse tensor algebra

It should be noted, that compared to the above projects Fastor is *minimal* and does not try to be a full-fledged tensor algebra framework, specifically like Eigen. It is designed with specific needs in mind. Some noteworthy differences are

- Fastor does not fall back to scalar code on non-SIMD architectures. That has just not been the goal of Fastor.
- Fastor is for small tensors and stack allocated and the limit to the dimensions of the tensor is dictated by the compilers template instantiation depth which is by default 500 in `gcc` at which point you would certainly exceed stack-allocation limit anyway. Some of the above libraries are limited to a few dimensional tensors. 
- Most of the points mentioned above, like resolving indices at compile time, Voigt transformation, the einsum feature is specific to and niceties of Fastor. Some of these design principles certainly make Fastor less flexible compared to the above mentioned projects.
- While stable, Fastor is in its infancy, whereas most of the aforementioned projects have reached a certain level maturity. Unless you find some features of Fastor appealing and work in the areas that we do, there is no reason why you shouldn't be using one of the above projects. In particular, Eigen is a really powerful alternative. 