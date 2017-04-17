# ENSURE THE SCALAR CODE IS COMPILED AS REQUIRED BY THE BENCHMARK BY PASSING 
# APPROPRIATE FLAGS. THAT IS THE AUTO-VECTORISER SHOULD BE TURNED OFF IN ORDER TO ASSESS
# FASTOR'S TRUE VECTORISATION CAPABILITY
# DO NOT ALLOW THE COMPILER TO BREAK IEEE COMPLIANCE

# Sandy-Bridge
# GCC = g++-6 -fabi-version=6 -fno-fast-math -fno-associative-math -D_GLIBCXX_USE_CXX11_ABI=0 
# CLANG = /media/MATLAB/clang+llvm-3.9.0-x86_64-linux-gnu-ubuntu-14.04/bin/clang++ -fno-fast-math -fno-associative-math
# ICC = /media/MATLAB/intel_2017/bin/icpc -fp-model=strict

# Xeon
GCC = g++-6 -fabi-version=6 -fno-fast-math -fno-associative-math -D_GLIBCXX_USE_CXX11_ABI=0 -DNDEBUG
CLANG = clang++-3.9 -fno-fast-math -fno-associative-math -DNDEBUG
ICC = /home/roman/intel_2017/bin/icpc -fp-model=strict -DNDEBUG

GCC_FLAGS_Fastor = -std=c++11 -O3 -mavx -DFASTOR_USE_OLD_OUTER
GCC_FLAGS_Scalar = -std=c++11 -O3 -mavx -fno-tree-vectorize -DFASTOR_USE_OLD_OUTER

CLANG_FLAGS_Fastor = -std=c++11 -O3 -mavx -DFASTOR_USE_OLD_OUTER
CLANG_FLAGS_Scalar = -std=c++11 -O3 -mavx -fno-tree-vectorize -fno-vectorize -DFASTOR_USE_OLD_OUTER

ICC_FLAGS_Fastor = -std=c++11 -O3 -mavx -DFASTOR_USE_OLD_OUTER
ICC_FLAGS_Scalar = -std=c++11 -O3 -mavx -DFASTOR_USE_OLD_OUTER #-no-vec

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