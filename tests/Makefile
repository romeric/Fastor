CXX_FLAGS = -std=c++11 -O3 -mavx
INCLUDES = -I../

ifeq ($(CXX),g++)
CXX_FLAGS += -fabi-version=6 
endif

all: tests
tests: test_simd_vectors test_basics test_linalg

test_simd_vectors:
	$(CXX) test_simd_vectors.cpp -o test_simd_vectors.exe $(CXX_FLAGS) $(INCLUDES)

test_basics:
	$(CXX) test_basics.cpp -o test_basics.exe $(CXX_FLAGS) $(INCLUDES)	

test_linalg:
	$(CXX) test_linalg.cpp -o test_linalg.exe $(CXX_FLAGS) $(INCLUDES)	

run:
	./test_simd_vectors.exe
	./test_basics.exe 
	./test_linalg.exe

clean:
	rm -rf test_simd_vectors.exe test_basics.exe test_linalg.exe 
