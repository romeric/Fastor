# DO NOT ALLOW THE COMPILER TO BREAK IEEE COMPLIANCE
# ENSURE THE SCALAR CODE IS COMPILED AS REQUIRED BY THE BENCHMARK BY PASSING 
# APPROPRIATE FLAGS
# All the compilers are unable to vectorise the scalar code, passing these flags
# is just a pre-caution

# GCC = /opt/gcc-5.2/bin/g++ -fabi-version=6 -finline-functions -fno-fast-math
GCC = /opt/gcc-5.3.0/bin/g++ -fabi-version=6 -finline-functions -fno-fast-math -D_GLIBCXX_USE_CXX11_ABI=0
CLANG = /home/roman/Downloads/clang+llvm-3.8.0-x86_64-linux-gnu-ubuntu-14.04/bin/clang++
ICC = /home/roman/intel_e2/bin/icpc -fp-model=strict

GCC_FLAGS_Fastor = -std=c++11 -O3 -mavx 
CLANG_FLAGS_Fastor = -std=c++11 -O3 -mavx 
ICC_FLAGS_Fastor = -std=c++11 -O3 -mavx 

INCLUDES = -I../../../

# all: bench_gcc bench_clang bench_icc
all: bench_gcc 

bench_gcc:
	$(GCC) benchmark_depthfirst.cpp -o benchmark_depthfirst_gcc.exe $(GCC_FLAGS_Fastor) $(INCLUDES)
	$(GCC) benchmark_depthfirst_2.cpp -o benchmark_depthfirst_2_gcc.exe $(GCC_FLAGS_Fastor) $(INCLUDES)

bench_clang:
	$(CLANG) benchmark_depthfirst.cpp -o benchmark_depthfirst_clang.exe $(CLANG_FLAGS_Fastor) $(INCLUDES)

bench_icc:
	$(ICC) benchmark_depthfirst.cpp -o benchmark_depthfirst_icc.exe $(ICC_FLAGS_Fastor) $(INCLUDES)

# run benchmarks through make
# run:
# 	./benchmark_depthfirst_gcc.exe > gcc_res
# 	./benchmark_depthfirst_clang.exe > clang_res
# 	./benchmark_depthfirst_icc.exe > icc_res
run:
	./benchmark_depthfirst_gcc.exe > gcc_res
	./benchmark_depthfirst_2_gcc.exe > gcc_res_2

# clean:
# 	rm -rf benchmark_depthfirst_gcc.exe benchmark_depthfirst_clang.exe benchmark_depthfirst_icc.exe \
# 		gcc_res clang_res icc_res 

clean:
	rm -rf benchmark_depthfirst_gcc.exe benchmark_depthfirst_2_gcc.exe \
		gcc_res gcc_res_2 

