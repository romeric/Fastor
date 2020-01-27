OPT=-O3 -march=native -DNDEBUG -DFASTOR_NO_ALIAS

ifeq "$(CXX)" "g++"
    OPT += -finline-limit=1000000 -ffp-contract=fast
endif
ifeq "$(CXX)" "g++-6"
    OPT += -finline-limit=1000000 -ffp-contract=fast
endif
ifeq "$(CXX)" "/usr/local/bin/g++-6"
    OPT += -finline-limit=1000000 -ffp-contract=fast
endif

ifeq "$(CXX)" "clang++"
    OPT += -mllvm -inline-threshold=1000000  -ffp-contract=fast
endif
ifeq "$(CXX)" "clang++-3.9"
    OPT += -mllvm -inline-threshold=1000000  -ffp-contract=fast
endif
ifeq "$(CXX)" "c++"
    OPT += -mllvm -inline-threshold=1000000  -ffp-contract=fast
endif

ifeq "$(CXX)" "icpc"
    OPT += -inline-forceinline -fp-model fast=2
endif
ifeq "$(CXX)" "/home/roman/intel_2017b/bin/icpc"
    OPT += -inline-forceinline -fp-model fast=2
endif

# On some architectures -march=native does not define -mfma
HAS_FMA := $(shell $(CXX) -march=native -dM -E - < /dev/null | egrep "AVX2" | sort)
ifeq ($(HAS_FMA),)
else
OPT += -mfma
endif

all:
	$(CXX) -std=c++14 -I../../ benchmark_views.cpp -o benchmark_views_1d.exe $(OPT) -DONE_D
	$(CXX) -std=c++14 -I../../ benchmark_views.cpp -o benchmark_views_vec_1d.exe $(OPT) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN -DONE_D

	$(CXX) -std=c++14 -I../../ benchmark_views.cpp -o benchmark_views_2d.exe $(OPT) -DTWO_D
	$(CXX) -std=c++14 -I../../ benchmark_views.cpp -o benchmark_views_vec_2d.exe $(OPT) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN -DTWO_D

	$(CXX) -std=c++14 -I../../ benchmark_views.cpp -o benchmark_views_nd.exe $(OPT) -DMULTI_D
	$(CXX) -std=c++14 -I../../ benchmark_views.cpp -o benchmark_views_vec_nd.exe $(OPT) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN -DMULTI_D


	$(CXX) -std=c++14 -I../../ benchmark_fix_views.cpp -o benchmark_fix_views_1d.exe $(OPT) -DONE_D
	$(CXX) -std=c++14 -I../../ benchmark_fix_views.cpp -o benchmark_fix_views_vec_1d.exe $(OPT) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN -DONE_D

	$(CXX) -std=c++14 -I../../ benchmark_fix_views.cpp -o benchmark_fix_views_2d.exe $(OPT) -DTWO_D
	$(CXX) -std=c++14 -I../../ benchmark_fix_views.cpp -o benchmark_fix_views_vec_2d.exe $(OPT) -DFASTOR_USE_VECTORISED_EXPR_ASSIGN -DTWO_D


run:
	./benchmark_views_1d.exe
	./benchmark_views_vec_1d.exe
	./benchmark_views_2d.exe
	./benchmark_views_vec_2d.exe
	./benchmark_views_nd.exe
	./benchmark_views_vec_nd.exe

	./benchmark_fix_views_1d.exe
	./benchmark_fix_views_vec_1d.exe
	./benchmark_fix_views_2d.exe
	./benchmark_fix_views_vec_2d.exe

clean:
	rm -rf *.exe