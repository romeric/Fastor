# DO NOT ALLOW THE COMPILER TO BREAK IEEE COMPLIANCE
# ENSURE THE SCALAR CODE IS COMPILED AS REQUIRED BY THE BENCHMARK BY PASSING 
# APPROPRIATE FLAGS
# All the compilers are unable to vectorise the scalar code, passing these flags
# is just a pre-caution

# GCC = /opt/gcc-5.2/bin/g++ -fabi-version=6 -finline-functions -fno-fast-math
GCC = /opt/gcc-5.3.0/bin/g++ -fabi-version=6 -finline-functions -fno-fast-math -D_GLIBCXX_USE_CXX11_ABI=0
# CLANG = clang++ -fno-fast-math
# CLANG = /media/MATLAB/clang+llvm-3.8.0-x86_64-linux-gnu-ubuntu-14.04/bin/clang++ -fno-fast-math
CLANG = /home/roman/Downloads/clang+llvm-3.8.0-x86_64-linux-gnu-ubuntu-14.04/bin/clang++
# ICC = /media/MATLAB/intel/bin/icpc -fp-model=strict
ICC = /home/roman/intel_e2/bin/icpc -fp-model=strict

GCC_FLAGS_Fastor = -std=c++11 -O3 -mavx 
GCC_FLAGS_Scalar = -std=c++11 -O3 -mavx -fno-tree-vectorize

CLANG_FLAGS_Fastor = -std=c++11 -O3 -mavx 
CLANG_FLAGS_Scalar = -std=c++11 -O3 -mavx -fno-tree-vectorize -fno-vectorize

ICC_FLAGS_Fastor = -std=c++11 -O3 -mavx 
ICC_FLAGS_Scalar = -std=c++11 -O3 -mavx #-no-vec
# Note that -no-vec option in ICC seemingly disables some other optimisation
# that hurts the performance  

INCLUDES = -I../../../

all: gcc_runner clang_runner icc_runner

gcc_runner:    gcc_name_change bench_outerproducts_gcc bench_outerproducts_scalar_gcc
clang_runner:  clang_name_change bench_outerproducts_clang bench_outerproducts_scalar_clang
icc_runner:    icc_name_change bench_outerproducts_icc bench_outerproducts_scalar_icc name_change


bench_outerproducts_gcc:
	$(GCC) benchmark_outerproducts.cpp -o benchmark_outerproducts_fastor_gcc.exe $(GCC_FLAGS_Fastor) $(INCLUDES)

bench_outerproducts_scalar_gcc:
	$(GCC) benchmark_outerproducts_scalar.cpp -o benchmark_outerproducts_scalar_gcc.exe $(GCC_FLAGS_Scalar) $(INCLUDES)

bench_outerproducts_clang:
	$(CLANG) benchmark_outerproducts.cpp -o benchmark_outerproducts_fastor_clang.exe $(CLANG_FLAGS_Fastor) $(INCLUDES)

bench_outerproducts_scalar_clang:
	$(CLANG) benchmark_outerproducts_scalar.cpp -o benchmark_outerproducts_scalar_clang.exe $(CLANG_FLAGS_Scalar) $(INCLUDES)

bench_outerproducts_icc:
	$(ICC) benchmark_outerproducts.cpp -o benchmark_outerproducts_fastor_icc.exe $(ICC_FLAGS_Fastor) $(INCLUDES)

bench_outerproducts_scalar_icc:
	$(ICC) benchmark_outerproducts_scalar.cpp -o benchmark_outerproducts_scalar_icc.exe $(ICC_FLAGS_Scalar) $(INCLUDES)

gcc_name_change:
	sed -i -e 's/SIMD_products_results/SIMD_products_results_gcc/g' benchmark_outerproducts.cpp
	sed -i -e 's/Scalar_products_results/Scalar_products_results_gcc/g' benchmark_outerproducts_scalar.cpp
clang_name_change:
	sed -i -e 's/SIMD_products_results_gcc/SIMD_products_results_clang/g' benchmark_outerproducts.cpp
	sed -i -e 's/Scalar_products_results_gcc/Scalar_products_results_clang/g' benchmark_outerproducts_scalar.cpp
icc_name_change:
	sed -i -e 's/SIMD_products_results_clang/SIMD_products_results_icc/g' benchmark_outerproducts.cpp
	sed -i -e 's/Scalar_products_results_clang/Scalar_products_results_icc/g' benchmark_outerproducts_scalar.cpp
name_change:
	sed -i -e 's/SIMD_products_results_icc/SIMD_products_results/g' benchmark_outerproducts.cpp
	sed -i -e 's/Scalar_products_results_icc/Scalar_products_results/g' benchmark_outerproducts_scalar.cpp

# run benchmarks through make
run:
	./benchmark_outerproducts_scalar_gcc.exe
	./benchmark_outerproducts_fastor_gcc.exe
	./benchmark_outerproducts_scalar_clang.exe
	./benchmark_outerproducts_fastor_clang.exe
	./benchmark_outerproducts_scalar_icc.exe
	./benchmark_outerproducts_fastor_icc.exe


clean:
	rm -rf benchmark_outerproducts_fastor_gcc.exe benchmark_outerproducts_fastor_clang.exe benchmark_outerproducts_fastor_icc.exe \
	benchmark_outerproducts_scalar_gcc.exe benchmark_outerproducts_scalar_clang.exe benchmark_outerproducts_scalar_icc.exe \
	SIMD_products_results_gcc SIMD_products_results_clang SIMD_products_results_icc \
	Scalar_products_results_gcc Scalar_products_results_clang Scalar_products_results_icc









# # DO NOT ALLOW THE COMPILER TO BREAK IEEE COMPLIANCE
# # CXX = g++ -fabi-version=6 -finline-functions -fno-fast-math
# # CXX = /opt/gcc-5.3.0/bin/g++ -fabi-version=6 -finline-functions -fno-fast-math
# CXX = /opt/gcc-5.2/bin/g++ -fabi-version=6 -finline-functions -fno-fast-math
# # CXX = clang++ -fno-fast-math
# # CXX = /media/MATLAB/intel/bin/icpc -fp-model=strict
# # -fp-model=precise
# # CXX = /home/roman/Downloads/clang+llvm-3.8.0-x86_64-linux-gnu-ubuntu-14.04/bin/clang++
# # CXX_FLAGS = -std=c++11 -O3 -mavx -funroll-loops 
# # -fvectorize -mllvm -force-vector-width=8
# # CXX_FLAGS_Scalar = -std=c++11 -O3 -mavx -fno-unroll-loops -fno-tree-vectorize 
# CXX_FLAGS = -std=c++11 -O3 -mavx 
# # CXX_FLAGS_Scalar = -std=c++11 -O3 -mavx -fno-tree-vectorize -fno-vectorize
# CXX_FLAGS_Scalar = -std=c++11 -O3 -mavx -fno-tree-vectorize
# # CXX_FLAGS_Scalar = -std=c++11 -O3 -mavx
# # CXX_FLAGS_Scalar = -std=c++11 -O3 -mavx #-no-vec #-opt-repor
# # CXX_FLAGS = -std=c++11 
# # GCC_COMPAT = -fabi-version=6 -finline-functions
# INCLUDES = -I../../

# # all: bench_crossproduct
# all: bench_outerproducts bench_outerproducts_scalar

# bench_crossproduct:
# 	$(CXX) benchmark_crossproduct.cpp -o benchmark_crossproduct.exe $(CXX_FLAGS) $(INCLUDES)

# bench_outerproducts:
# 	$(CXX) benchmark_outerproducts.cpp -o benchmark_outerproducts.exe $(CXX_FLAGS) $(INCLUDES)

# bench_outerproducts_scalar:
# 	$(CXX) benchmark_outerproducts_scalar.cpp -o benchmark_outerproducts_scalar.exe $(CXX_FLAGS_Scalar) $(INCLUDES)
