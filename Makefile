all:
	$(CXX) -std=c++14 -I. Fastor.cpp -o Fastor -O3 -march=native -DNDEBUG $(VEC)