INCLUDES = -I../
CXX_FLAGS = -std=c++14 -O3 -march=native

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
tests: test_auxiliary \
test_simd_vectors \
test_basics \
test_matmul \
test_tmatmul \
test_linalg \
test_lu \
test_inverse \
test_solve \
test_einsum \
test_views \
test_mixed_views \
test_tensormap \
test_complex_expressions \
test_binary_cmp_ops

test_auxiliary:
	$(CXX) test_auxiliary.cpp -o test_auxiliary.exe $(CXX_FLAGS) $(INCLUDES)

test_simd_vectors:
	$(CXX) test_simd_vectors.cpp -o test_simd_vectors.exe $(CXX_FLAGS) $(INCLUDES)

test_basics:
	$(CXX) test_basics.cpp -o test_basics.exe $(CXX_FLAGS) $(INCLUDES)

test_matmul:
	$(CXX) test_matmul.cpp -o test_matmul_1.exe $(INCLUDES) -std=c++14
	$(CXX) test_matmul.cpp -o test_matmul_2.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_matmul_small.cpp -o test_matmul_small.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_lazy_matmul.cpp -o test_lazy_matmul.exe $(INCLUDES) -std=c++14

test_tmatmul:
	$(CXX) test_tmatmul_unmasked.cpp -o test_tmatmul_unmasked.exe $(INCLUDES) -std=c++14
	$(CXX) test_tmatmul_masked.cpp -o test_tmatmul_masked.exe $(INCLUDES) -std=c++14

test_linalg:
	$(CXX) test_linalg.cpp -o test_linalg.exe $(CXX_FLAGS) $(INCLUDES)

test_lu:
	$(CXX) test_lu.cpp -o test_lu.exe $(CXX_FLAGS) $(INCLUDES)

test_inverse:
	$(CXX) test_inverse.cpp -o test_inverse.exe $(CXX_FLAGS) $(INCLUDES)

test_solve:
	$(CXX) test_solve.cpp -o test_solve.exe $(CXX_FLAGS) $(INCLUDES)

test_einsum:
	$(CXX) test_contraction.cpp -o test_contraction_0.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_contraction.cpp -o test_contraction_1.exe $(CXX_FLAGS) $(INCLUDES) -DCONTRACT_OPT=1
	$(CXX) test_contraction.cpp -o test_contraction_2.exe $(CXX_FLAGS) $(INCLUDES) -DCONTRACT_OPT=-1
	$(CXX) test_einsum.cpp -o test_einsum_0.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_einsum.cpp -o test_einsum_1.exe $(CXX_FLAGS) $(INCLUDES) -DCONTRACT_OPT=1
	$(CXX) test_einsum.cpp -o test_einsum_2.exe $(CXX_FLAGS) $(INCLUDES) -DCONTRACT_OPT=-1


test_views:
	$(CXX) test_views_1d.cpp -o test_views_1d.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_views_1d.cpp -o test_views_1d_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

	$(CXX) test_views_2d.cpp -o test_views_2d.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_views_2d.cpp -o test_views_2d_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

	$(CXX) test_views_nd.cpp -o test_views_nd.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_views_nd_2.cpp -o test_views_nd_2.exe $(CXX_FLAGS) $(INCLUDES)

	$(CXX) test_fixed_views_1d.cpp -o test_fixed_views_1d.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_fixed_views_1d.cpp -o test_fixed_views_1d_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

	$(CXX) test_fixed_views_2d.cpp -o test_fixed_views_2d.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_fixed_views_2d.cpp -o test_fixed_views_2d_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

	$(CXX) test_fixed_views_nd.cpp -o test_fixed_views_nd.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_fixed_views_nd_2.cpp -o test_fixed_views_nd_2.exe $(CXX_FLAGS) $(INCLUDES)

	$(CXX) test_random_views_1d.cpp -o test_random_views_1d.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_random_views_1d.cpp -o test_random_views_1d_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

	$(CXX) test_random_views_nd.cpp -o test_random_views_nd.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_random_views_nd.cpp -o test_random_views_nd_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

test_mixed_views:
	$(CXX) test_mixed_views.cpp -o test_mixed_views.exe $(CXX_FLAGS) $(INCLUDES)

test_tensormap:
	$(CXX) test_tensormap.cpp -o test_tensormap.exe $(CXX_FLAGS) $(INCLUDES)

test_complex_expressions:
	$(CXX) test_complex_expressions.cpp -o test_complex_expressions.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_complex_expressions.cpp -o test_complex_expressions_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

test_binary_cmp_ops:
	$(CXX) test_binary_cmp_ops.cpp -o test_binary_cmp_ops.exe $(CXX_FLAGS) $(INCLUDES)



run: run_auxiliary \
run_simd_vectors \
run_basics \
run_matmul \
run_tmatmul \
run_linalg \
run_lu \
run_inverse \
run_solve \
run_einsum \
run_views \
run_mixed_views \
run_tensormap \
run_complex_expressions \
run_binary_cmp_ops

run_auxiliary:
	./test_auxiliary.exe

run_simd_vectors:
	./test_simd_vectors.exe

run_basics:
	./test_basics.exe

run_matmul:
	./test_matmul_1.exe
	./test_matmul_2.exe
	./test_matmul_small.exe
	./test_lazy_matmul.exe

run_tmatmul:
	./test_tmatmul_unmasked.exe
	./test_tmatmul_masked.exe

run_linalg:
	./test_linalg.exe

run_lu:
	./test_lu.exe

run_inverse:
	./test_inverse.exe

run_solve:
	./test_solve.exe

run_einsum:
	./test_contraction_0.exe
	./test_contraction_1.exe
	./test_contraction_2.exe
	./test_einsum_0.exe
	./test_einsum_1.exe
	./test_einsum_2.exe

run_views:
	./test_views_1d.exe
	./test_views_1d_vec.exe
	./test_views_2d.exe
	./test_views_2d_vec.exe
	./test_views_nd.exe
	./test_views_nd_2.exe
	./test_fixed_views_1d.exe
	./test_fixed_views_1d_vec.exe
	./test_fixed_views_2d.exe
	./test_fixed_views_2d_vec.exe
	./test_fixed_views_nd.exe
	./test_fixed_views_nd_2.exe
	./test_random_views_1d.exe
	./test_random_views_1d_vec.exe
	./test_random_views_nd.exe
	./test_random_views_nd_vec.exe

run_mixed_views:
	./test_mixed_views.exe

run_tensormap:
	./test_tensormap.exe

run_complex_expressions:
	./test_complex_expressions.exe
	./test_complex_expressions_vec.exe

run_binary_cmp_ops:
	./test_binary_cmp_ops.exe

clean:
	rm -rf *.exe
