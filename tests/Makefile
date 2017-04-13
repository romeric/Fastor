CXX_FLAGS = -std=c++11 -O3 -march=native
INCLUDES = -I../

ifeq ($(CXX),g++)
CXX_FLAGS += -fabi-version=6 
endif


all: tests
tests: test_simd_vectors test_basics test_linalg test_views
# tests: test_views

test_simd_vectors:
	$(CXX) test_simd_vectors.cpp -o test_simd_vectors.exe $(CXX_FLAGS) $(INCLUDES)

test_basics:
	$(CXX) test_basics.cpp -o test_basics.exe $(CXX_FLAGS) $(INCLUDES)	

test_linalg:
	$(CXX) test_linalg.cpp -o test_linalg.exe $(CXX_FLAGS) $(INCLUDES)	

test_views:
	$(CXX) test_views_1d.cpp -o test_views_1d.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_views_1d.cpp -o test_views_1d_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

	$(CXX) test_views_2d.cpp -o test_views_2d.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_views_2d.cpp -o test_views_2d_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

	$(CXX) test_views_nd.cpp -o test_views_nd.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_views_nd.cpp -o test_views_nd_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

	$(CXX) test_fixed_views_1d.cpp -o test_fixed_views_1d.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_fixed_views_1d.cpp -o test_fixed_views_1d_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

	$(CXX) test_fixed_views_2d.cpp -o test_fixed_views_2d.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_fixed_views_2d.cpp -o test_fixed_views_2d_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

	$(CXX) test_random_views_1d.cpp -o test_random_views_1d.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_random_views_1d.cpp -o test_random_views_1d_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

	$(CXX) test_random_views_nd.cpp -o test_random_views_nd.exe $(CXX_FLAGS) $(INCLUDES)
	$(CXX) test_random_views_nd.cpp -o test_random_views_nd_vec.exe $(CXX_FLAGS) $(INCLUDES) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN

run:
	./test_simd_vectors.exe
	./test_basics.exe 
	./test_linalg.exe
	./test_views_1d.exe
	./test_views_1d_vec.exe
	./test_views_2d.exe
	./test_views_2d_vec.exe
	./test_views_nd.exe
	./test_views_nd_vec.exe
	./test_fixed_views_1d.exe
	./test_fixed_views_1d_vec.exe
	./test_fixed_views_2d.exe
	./test_fixed_views_2d_vec.exe
	./test_random_views_1d.exe
	./test_random_views_1d_vec.exe
	./test_random_views_nd.exe
	./test_random_views_nd_vec.exe

clean:
	rm -rf *.exe
