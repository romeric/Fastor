CXX_GCC = g++
CXX_CLANG = clang++
CXX_FLAGS = -std=c++11 -O3 -mavx -funroll-loops 
GCC_COMPAT = -fabi-version=6 -finline-functions
INCLUDES = -I../

all: gcc clang

gcc:
	$(CXX_GCC) test_basics.cpp -o test_basics_gcc.exe $(CXX_FLAGS) $(GCC_COMPAT) $(INCLUDES)	
clang:
	$(CXX_CLANG) test_basics.cpp -o test_basics_clang.exe $(CXX_FLAGS) $(INCLUDES)