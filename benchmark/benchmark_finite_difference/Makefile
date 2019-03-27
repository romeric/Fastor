OPT = -O3
CXXFLAGS = -std=c++14 $(OPT) -march=native -DNDEBUG


ifeq "$(CXX)" "g++"
    CXXFLAGS += -finline-limit=1000000 -ffp-contract=fast 
endif
ifeq "$(CXX)" "g++-6"
    CXXFLAGS += -finline-limit=1000000 -ffp-contract=fast 
endif
ifeq "$(CXX)" "/usr/local/bin/g++-6"
    CXXFLAGS += -finline-limit=1000000 -ffp-contract=fast 
endif

ifeq "$(CXX)" "clang++"
    CXXFLAGS += -ftemplate-depth=10000000 -fconstexpr-depth=1000000000 -mllvm -inline-threshold=1000000 -foperator-arrow-depth=100000000  -ffp-contract=fast
endif
ifeq "$(CXX)" "clang++-3.9"
    CXXFLAGS += -ftemplate-depth=10000000 -fconstexpr-depth=1000000000 -mllvm -inline-threshold=1000000 -foperator-arrow-depth=100000000  -ffp-contract=fast
endif
ifeq "$(CXX)" "c++"
    CXXFLAGS += -ftemplate-depth=10000000 -fconstexpr-depth=1000000000 -mllvm -inline-threshold=1000000 -foperator-arrow-depth=100000000  -ffp-contract=fast
endif

ifeq "$(CXX)" "/home/roman/intel_2017b/bin/icpc"
    CXXFLAGS += -inline-forceinline -fp-model fast=2
endif

# On some architectures -march=native does not define -mfma
HAS_FMA := $(shell $(CXX) -march=native -dM -E - < /dev/null | egrep "AVX2" | sort)
ifeq ($(HAS_FMA),)
else
CXXFLAGS += -mfma
endif

all:
	$(CXX) -I../../ benchmark_finite_difference.cpp -o out_loops.exe $(CXXFLAGS) -DUSE_LOOPS

	$(CXX) -I../../ benchmark_finite_difference.cpp -o out_seq_alias.exe $(CXXFLAGS) -DUSE_SEQ_ALIAS
	$(CXX) -I../../ benchmark_finite_difference.cpp -o out_seq_noalias.exe $(CXXFLAGS) -DUSE_SEQ_NOALIAS -DFASTOR_NO_ALIAS
	$(CXX) -I../../ benchmark_finite_difference.cpp -o out_iseq.exe $(CXXFLAGS) -DUSE_ISEQ
	$(CXX) -I../../ benchmark_finite_difference.cpp -o out_fseq.exe $(CXXFLAGS) -DUSE_FSEQ

	$(CXX) -I../../ benchmark_finite_difference.cpp -o out_seq_alias_vec.exe $(CXXFLAGS) -DUSE_SEQ_ALIAS -DFASTOR_USE_VECTORISED_EXPR_ASSIGN
	$(CXX) -I../../ benchmark_finite_difference.cpp -o out_seq_noalias_vec.exe $(CXXFLAGS) -DUSE_SEQ_NOALIAS -DFASTOR_NO_ALIAS -DFASTOR_USE_VECTORISED_EXPR_ASSIGN
	$(CXX) -I../../ benchmark_finite_difference.cpp -o out_iseq_vec.exe $(CXXFLAGS) -DUSE_ISEQ -DFASTOR_USE_VECTORISED_EXPR_ASSIGN
	$(CXX) -I../../ benchmark_finite_difference.cpp -o out_fseq_vec.exe $(CXXFLAGS) -DUSE_FSEQ -DFASTOR_USE_VECTORISED_EXPR_ASSIGN 

N=100

run:
	./out_loops.exe $(N)

	./out_seq_alias.exe $(N)
	./out_seq_noalias.exe $(N)
	./out_iseq.exe $(N)
	./out_fseq.exe $(N)

	./out_seq_alias_vec.exe $(N)
	./out_seq_noalias_vec.exe $(N)
	./out_iseq_vec.exe $(N)
	./out_fseq_vec.exe $(N)


clean:
	rm -rf *.exe