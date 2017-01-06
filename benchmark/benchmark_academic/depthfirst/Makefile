# DO NOT ALLOW THE COMPILER TO BREAK IEEE COMPLIANCE

GCC = g++-6 -fabi-version=6 -fno-fast-math -D_GLIBCXX_USE_CXX11_ABI=0
CLANG = clang++-3.9 -fno-fast-math

# ICC DOESNOT NOT COMPILE THE TIMEIT FUNCTION WITH MORE THAN 2 ENTRIES
# ICC = /home/roman/intel_2017/bin/icpc -fp-model=strict
# ICC = /media/MATLAB/intel_2017/bin/icpc -fp-model=strict 


GCC_FLAGS_Fastor = -std=c++11 -O3 -mavx 
CLANG_FLAGS_Fastor = -std=c++11 -O3 -mavx 
ICC_FLAGS_Fastor = -std=c++11 -O3 -mavx 

INCLUDES = -I../../../

all: bench_gcc 

bench_gcc:
	$(GCC) benchmark_depthfirst.cpp -o benchmark_depthfirst_dp_gcc.exe $(GCC_FLAGS_Fastor) $(INCLUDES) -DFASTOR_KEEP_DP_FIXED
	$(GCC) benchmark_depthfirst_2.cpp -o benchmark_depthfirst_2_dp_gcc.exe $(GCC_FLAGS_Fastor) $(INCLUDES) -DFASTOR_KEEP_DP_FIXED
	$(GCC) benchmark_depthfirst.cpp -o benchmark_depthfirst_nodp_gcc.exe $(GCC_FLAGS_Fastor) $(INCLUDES) -DFASTOR_DONT_PERFORM_OP_MIN
	$(GCC) benchmark_depthfirst_2.cpp -o benchmark_depthfirst_2_nodp_gcc.exe $(GCC_FLAGS_Fastor) $(INCLUDES) -DFASTOR_DONT_PERFORM_OP_MIN

bench_clang:
	$(GCC) benchmark_depthfirst.cpp -o benchmark_depthfirst_dp_clang.exe $(GCC_FLAGS_Fastor) $(INCLUDES)
	$(GCC) benchmark_depthfirst_2.cpp -o benchmark_depthfirst_2_dp_clang.exe $(GCC_FLAGS_Fastor) $(INCLUDES)
	$(GCC) benchmark_depthfirst.cpp -o benchmark_depthfirst_nodp_clang.exe $(GCC_FLAGS_Fastor) $(INCLUDES) -DFASTOR_DONT_PERFORM_OP_MIN
	$(GCC) benchmark_depthfirst_2.cpp -o benchmark_depthfirst_2_nodp_clang.exe $(GCC_FLAGS_Fastor) $(INCLUDES) -DFASTOR_DONT_PERFORM_OP_MIN
bench_icc:
	$(ICC) benchmark_depthfirst.cpp -o benchmark_depthfirst_icc.exe $(ICC_FLAGS_Fastor) $(INCLUDES)
	$(ICC) benchmark_depthfirst_2.cpp -o benchmark_depthfirst_2_icc.exe $(GCC_FLAGS_Fastor) $(INCLUDES)

# run benchmarks through make
run:
	./benchmark_depthfirst_dp_gcc.exe > gcc_res_dp
	./benchmark_depthfirst_nodp_gcc.exe > gcc_res_nodp
	./benchmark_depthfirst_2_dp_gcc.exe > gcc_res_2_dp
	./benchmark_depthfirst_2_nodp_gcc.exe > gcc_res_2_nodp

clean:
	rm -rf benchmark_depthfirst_dp_gcc.exe benchmark_depthfirst_nodp_gcc.exe \
		benchmark_depthfirst_2_dp_gcc.exe benchmark_depthfirst_2_nodp_gcc.exe \
		gcc_res_dp gcc_res_nodp gcc_res_2_dp gcc_res_2_nodp 