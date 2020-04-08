CXX_FLAGS = -O3 -march=native -ffast-math -ffp-contract=fast -DNDEBUG

# On some architectures -march=native does not define -mfma
HAS_FMA := $(shell $(CXX) -march=native -dM -E - < /dev/null | egrep "AVX2" | sort)
ifeq ($(HAS_FMA),)
else
CXX_FLAGS += -mfma
endif

all:
	$(CXX) -std=c++14 benchmark_materials.cpp -o benchmark_materials.exe -I../../ -I. $(CXX_FLAGS)

run:
	./benchmark_materials.exe

clean:
	rm -rf *.exe