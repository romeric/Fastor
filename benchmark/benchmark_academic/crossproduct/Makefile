# DO NOT ALLOW THE COMPILER TO BREAK IEEE COMPLIANCE
# ENSURE THE SCALAR CODE IS COMPILED AS REQUIRED BY THE BENCHMARK BY PASSING 
# APPROPRIATE FLAGS

GCC = /opt/gcc-5.3.0/bin/g++ -fabi-version=6 -finline-functions -fno-fast-math -D_GLIBCXX_USE_CXX11_ABI=0
CLANG = /home/roman/Downloads/clang+llvm-3.8.0-x86_64-linux-gnu-ubuntu-14.04/bin/clang++
ICC = /home/roman/intel_e2/bin/icpc -fp-model=strict

GCC_FLAGS_Fastor = -std=c++11 -O3 -mavx 
CLANG_FLAGS_Fastor = -std=c++11 -O3 -mavx 
ICC_FLAGS_Fastor = -std=c++11 -O3 -mavx 

INCLUDES = -I../../../

all: bench_gcc bench_clang bench_icc

bench_gcc:
	$(GCC) benchmark_crossproduct.cpp -o benchmark_crossproduct_gcc.exe $(GCC_FLAGS_Fastor) $(INCLUDES)
	$(GCC) benchmark_cofactor.cpp -o benchmark_cofactor_gcc.exe $(GCC_FLAGS_Fastor) $(INCLUDES)

bench_clang:
	$(CLANG) benchmark_crossproduct.cpp -o benchmark_crossproduct_clang.exe $(CLANG_FLAGS_Fastor) $(INCLUDES)
	$(CLANG) benchmark_cofactor.cpp -o benchmark_cofactor_clang.exe $(CLANG_FLAGS_Fastor) $(INCLUDES)

bench_icc:
	$(ICC) benchmark_crossproduct.cpp -o benchmark_crossproduct_icc.exe $(ICC_FLAGS_Fastor) $(INCLUDES)
	$(ICC) benchmark_cofactor.cpp -o benchmark_cofactor_icc.exe $(ICC_FLAGS_Fastor) $(INCLUDES)


# run benchmarks through make
run:
	./benchmark_crossproduct_gcc.exe > gcc_res_cross
	./benchmark_crossproduct_clang.exe > clang_res_cross
	./benchmark_crossproduct_icc.exe > icc_res_cross
	./benchmark_cofactor_gcc.exe > gcc_res_cof
	./benchmark_cofactor_clang.exe.exe > clang_res_cof
	./benchmark_cofactor_icc.exe.exe > icc_res_cof


clean:
	rm -rf benchmark_crossproduct_gcc.exe benchmark_crossproduct_clang.exe benchmark_crossproduct_icc.exe \
	benchmark_cofactor_gcc.exe benchmark_cofactor_clang.exe benchmark_cofactor_icc.exe \
	gcc_res_cross gcc_res_cof clang_res_cross clang_res_cof icc_res_cross icc_res_cof