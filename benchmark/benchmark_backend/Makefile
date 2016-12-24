# DO NOT ALLOW THE COMPILER TO BREAK IEEE COMPLIANCE
CXX = g++ -fabi-version=6 -finline-functions -fno-fast-math
# CXX = clang++ -fno-fast-math
# CXX = icpc -fp-model=strict # -fp-model=precise
CXX_FLAGS = -std=c++11 -O3 -mavx 
# GCC_COMPAT = -fabi-version=6 -finline-functions
INCLUDES = -I../../

all: bench_transpose bench_trace bench_norm bench_doublecontract bench_crossproduct \
	bench_outer bench_cyclic bench_matmul



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

bench_matmul:
	$(CXX) benchmark_matmul.cpp -o benchmark_matmul.exe $(CXX_FLAGS) $(INCLUDES)

run:
	./benchmark_doublecontract.exe
	./benchmark_norm.exe 
	./benchmark_outer.exe 
	./benchmark_cyclic.exe 
	./benchmark_crossproduct.exe 
	./benchmark_transpose.exe 
	./benchmark_trace.exe 
	./benchmark_matmul.exe

clean:
	rm -rf benchmark_transpose.exe benchmark_trace.exe benchmark_norm.exe benchmark_doublecontract.exe \
	benchmark_crossproduct.exe benchmark_outer.exe benchmark_cyclic.exe benchmark_matmul.exe
