CXX_FLAGS = -std=c++14 -O3 -march=native -DNDEBUG
INCLUDES = -I../../


# DO NOT ALLOW THE COMPILER TO BREAK IEEE COMPLIANCE
ifeq "$(CXX)" "g++"
    CXX_FLAGS += -fno-fast-math -fabi-version=6
endif
ifeq "$(CXX)" "clang++"
    CXX_FLAGS += -fno-fast-math
endif
ifeq "$(CXX)" "icpc"
    CXX_FLAGS += -fp-model=strict
endif

# On some architectures -march=native does not define -mfma
HAS_FMA := $(shell $(CXX) -march=native -dM -E - < /dev/null | egrep "AVX2" | sort)
ifeq ($(HAS_FMA),)
else
CXX_FLAGS += -mfma
endif


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
	rm -rf *.exe
