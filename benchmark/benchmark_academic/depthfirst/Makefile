# DO NOT ALLOW THE COMPILER TO BREAK IEEE COMPLIANCE
# ICC DOESNOT NOT COMPILE THE TIMEIT FUNCTION WITH MORE THAN 2 ENTRIES
CXX_FLAGS = -std=c++11 -O3 -mavx -DNDEBUG

# GCC = g++-6 -fabi-version=6 -fno-fast-math -D_GLIBCXX_USE_CXX11_ABI=0
ifeq "$(CXX)" "g++"
    CXX_FLAGS += -D_GLIBCXX_USE_CXX11_ABI=0 -fabi-version=6
else ifeq "$(CXX)" "g++-6"
    CXX_FLAGS += -D_GLIBCXX_USE_CXX11_ABI=0 -fabi-version=6
endif


INCLUDES = -I../../../

all: 
	$(CXX) benchmark_depthfirst.cpp -o benchmark_depthfirst_dp_gcc.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_KEEP_DP_FIXED
	$(CXX) benchmark_depthfirst_2.cpp -o benchmark_depthfirst_2_dp_gcc.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_KEEP_DP_FIXED
	$(CXX) benchmark_depthfirst.cpp -o benchmark_depthfirst_nodp_gcc.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_DONT_PERFORM_OP_MIN
	$(CXX) benchmark_depthfirst_2.cpp -o benchmark_depthfirst_2_nodp_gcc.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_DONT_PERFORM_OP_MIN

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