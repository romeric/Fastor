all:
	$(CXX) -std=c++14 benchmark_materials.cpp -o benchmark_materials.exe -I../../ -I. -O3 -march=native \
	-ffast-math -ffp-contract=fast -DNDEBUG -DCONTRACT_OPT=2

run:
	./benchmark_materials.exe

clean:
	rm -rf *.exe