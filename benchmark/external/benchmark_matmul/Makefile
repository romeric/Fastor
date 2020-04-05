OPT = -O3 -march=native -mfma -DNDEBUG
CXX_FLAGS = -std=c++17 $(OPT)
CXX_FLAGS += -I../

EIGENROOT = /Users/roman/Downloads/eigen-master/
BLAZEROOT = /Users/roman/Downloads/blaze-3.7/
FASTORROOT = ../../../
LIBXSMMROOT = /Users/roman/Downloads/libxsmm

LIBXSMMINC = -I$(FASTORROOT) -DFASTOR_USE_LIBXSMM -DFASTOR_BLAS_SWITCH_MATRIX_SIZE=1 -I$(LIBXSMMROOT)/include
LIBXSMMLIB =  -L$(LIBXSMMROOT)/lib -lxsmm -lblas

RUN_SINGLE =

CXX_FLAGS += $(RUN_SINGLE)

all:
	$(CXX) benchmark_eigen.cpp -o benchmark_eigen.exe $(CXX_FLAGS) -I$(EIGENROOT)
	$(CXX) benchmark_blaze.cpp -o benchmark_blaze.exe $(CXX_FLAGS) -I$(BLAZEROOT)
	$(CXX) benchmark_fastor.cpp -o benchmark_fastor.exe $(CXX_FLAGS) -I$(FASTORROOT)
	$(CXX) benchmark_fastor.cpp -o benchmark_libxsmm.exe $(CXX_FLAGS) $(LIBXSMMINC) $(LIBXSMMLIB)


run:
	./benchmark_eigen.exe
	./benchmark_blaze.exe
	./benchmark_fastor.exe
	./benchmark_libxsmm.exe

clean:
	rm -rf *.exe