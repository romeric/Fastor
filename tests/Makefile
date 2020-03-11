INCLUDES = -I../
CXX_FLAGS = -std=c++11 -O3 -march=native

# On some architectures -march=native does not define -mfma
HAS_FMA := $(shell $(CXX) -march=native -dM -E - < /dev/null | egrep "AVX2" | sort)
ifeq ($(HAS_FMA),)
else
CXX_FLAGS += -mfma
endif

ifeq ($(CXX),g++)
CXX_FLAGS += -fabi-version=6
endif


all: tests
tests: test_auxiliary test_simd_vectors test_basics test_linalg test_einsum test_views test_mixed_views test_complex_expressions

test_auxiliary:
	$(CXX) test_auxiliary.cpp -o test_auxiliary.exe $(CXX_FLAGS) $(INCLUDES)

test_simd_vectors:
	$(CXX) test_simd_vectors.cpp -o test_simd_vectors.exe $(CXX_FLAGS) $(INCLUDES)

test_basics:
	$(CXX) test_basics.cpp -o test_basics.exe $(CXX_FLAGS) $(INCLUDES)

test_einsum:
	$(CXX) test_einsum.cpp -o test_einsum_0.exe $(CXX_FLAGS) $(INCLUDES) -std=c++14
	$(CXX) test_einsum.cpp -o test_einsum_1.exe $(CXX_FLAGS) $(INCLUDES) -std=c++14 -DCONTRACT_OPT=1
	$(CXX) test_einsum.cpp -o test_einsum_2.exe $(CXX_FLAGS) $(INCLUDES) -std=c++14 -DCONTRACT_OPT=-1

test_linalg:
	$(CXX) test_linalg.cpp -o test_linalg.exe $(CXX_FLAGS) $(INCLUDES) -std=c++14

test_views:
	$(CXX) test_views_1d.cpp -o test_views_1d.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_views_1d.cpp -o test_views_1d_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

	$(CXX) test_views_2d.cpp -o test_views_2d.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_views_2d.cpp -o test_views_2d_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

	$(CXX) test_views_nd.cpp -o test_views_nd.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_views_nd.cpp -o test_views_nd_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN
	$(CXX) test_views_nd_2.cpp -o test_views_nd_2.exe $(CXX_FLAGS) $(INCLUDES)

	$(CXX) test_fixed_views_1d.cpp -o test_fixed_views_1d.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_fixed_views_1d.cpp -o test_fixed_views_1d_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

	$(CXX) test_fixed_views_2d.cpp -o test_fixed_views_2d.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_fixed_views_2d.cpp -o test_fixed_views_2d_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

	$(CXX) test_random_views_1d.cpp -o test_random_views_1d.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_random_views_1d.cpp -o test_random_views_1d_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

	$(CXX) test_random_views_nd.cpp -o test_random_views_nd.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_random_views_nd.cpp -o test_random_views_nd_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

test_mixed_views:
	$(CXX) test_mixed_views.cpp -o test_mixed_views.exe $(CXX_FLAGS) $(INCLUDES)

test_complex_expressions:
	$(CXX) test_complex_expressions.cpp -o test_complex_expressions.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_complex_expressions.cpp -o test_complex_expressions_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN



run: run_auxiliary run_simd_vectors run_basics run_linalg run_einsum run_views run_complex_expressions

run_auxiliary:
	./test_auxiliary.exe

run_simd_vectors:
	./test_simd_vectors.exe

run_basics:
	./test_basics.exe

run_linalg:
	./test_linalg.exe

run_einsum:
	./test_einsum_0.exe
	./test_einsum_1.exe
	./test_einsum_2.exe

run_views:
	./test_views_1d.exe
	./test_views_1d_vec.exe
	./test_views_2d.exe
	./test_views_2d_vec.exe
	./test_views_nd.exe
	./test_views_nd_vec.exe
	./test_views_nd_2.exe
	./test_fixed_views_1d.exe
	./test_fixed_views_1d_vec.exe
	./test_fixed_views_2d.exe
	./test_fixed_views_2d_vec.exe
	./test_random_views_1d.exe
	./test_random_views_1d_vec.exe
	./test_random_views_nd.exe
	./test_random_views_nd_vec.exe

run_mixed_views:
	./test_mixed_views.exe

run_complex_expressions:
	./test_complex_expressions.exe
	./test_complex_expressions_vec.exe



clean:
	rm -rf *.exe
