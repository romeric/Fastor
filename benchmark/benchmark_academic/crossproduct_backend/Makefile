# DO NOT ALLOW THE COMPILER TO BREAK IEEE COMPLIANCE
# ENSURE THE SCALAR CODE IS COMPILED AS REQUIRED BY THE BENCHMARK BY PASSING
# APPROPRIATE FLAGS


CXX_FLAGS = -std=c++14 -O3 -mavx -fno-fast-math

INCLUDES = -I../../../

all: bench

bench:
	$(CXX) benchmark_crossproduct_backend.cpp -o benchmark_crossproduct.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) benchmark_crossproduct_vector_backend.cpp -o benchmark_crossproduct_vector.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) benchmark_cofactor_backend.cpp -o benchmark_cofactor.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) benchmark_crossproduct_3rd_backend.cpp -o benchmark_crossproduct_3rd.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) benchmark_crossproduct_4th_backend.cpp -o benchmark_crossproduct_4th.exe $(CXX_FLAGS) $(INCLUDES)

# run benchmarks through make
run:
	./benchmark_crossproduct.exe
	./benchmark_crossproduct_vector.exe
	./benchmark_cofactor.exe
	./benchmark_crossproduct_3rd.exe
	./benchmark_crossproduct_4th.exe


clean:
	rm -rf *.exe
