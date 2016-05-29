# DO NOT ALLOW THE COMPILER TO BREAK IEEE COMPLIANCE
# CXX = g++ -fabi-version=6 -finline-functions -fno-fast-math
# CXX = /opt/gcc-5.3.0/bin/g++ -fabi-version=6 -finline-functions -fno-fast-math
# CXX = /opt/gcc-5.2/bin/g++ -fabi-version=6 -finline-functions -fno-fast-math
# CXX = clang++ -fno-fast-math
CXX = /media/MATLAB/intel/bin/icpc -fp-model=strict 
# -fp-model=precise
# CXX = /home/roman/Downloads/clang+llvm-3.8.0-x86_64-linux-gnu-ubuntu-14.04/bin/clang++
CXX_FLAGS = -std=c++11 -O3 -mavx -funroll-loops 
# CXX_FLAGS = -std=c++11 
# GCC_COMPAT = -fabi-version=6 -finline-functions
INCLUDES = -I../../

# all: bench_transpose
# all: bench_trace
# all: bench_doublecontract
# all: bench_norm
all: bench_crossproduct
# all: bench_outer
# all: bench_cyclic

# ifeq ($(CXX),g++)
# 	CXX = g++ -fabi-version=6 -finline-functions
# else
# 	CXX = CXX
# endif 



bench_transpose:
	$(CXX) benchmark_transpose.cpp -o benchmark_transpose.exe $(CXX_FLAGS) $(INCLUDES)

bench_trace:
	$(CXX) benchmark_trace.cpp -o benchmark_trace.exe $(CXX_FLAGS) $(INCLUDES)

bench_norm:
	$(CXX) benchmark_norm.cpp -o benchmark_norm.exe $(CXX_FLAGS) $(INCLUDES)

bench_doublecontract:
	$(CXX) benchmark_doublecontract.cpp -o benchmark_doublecontract.exe $(CXX_FLAGS) $(INCLUDES)

bench_crossproduct:
	$(CXX) benchmark_crossproduct.cpp -o benchmark_crossproduct.exe $(CXX_FLAGS) $(INCLUDES)

bench_outer:
	$(CXX) benchmark_outer.cpp -o benchmark_outer.exe $(CXX_FLAGS) $(INCLUDES)

bench_cyclic:
	$(CXX) benchmark_cyclic.cpp -o benchmark_cyclic.exe $(CXX_FLAGS) $(INCLUDES)
