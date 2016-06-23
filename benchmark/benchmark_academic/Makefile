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

# all: bench_crossproduct
all: bench_outerproducts

bench_crossproduct:
	$(CXX) benchmark_crossproduct.cpp -o benchmark_crossproduct.exe $(CXX_FLAGS) $(INCLUDES)

bench_outer:
	$(CXX) benchmark_outerproducts.cpp -o benchmark_outerproducts.exe $(CXX_FLAGS) $(INCLUDES)
