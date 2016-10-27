# CXX = /opt/gcc-5.2/bin/g++ -fabi-version=6 #-fno-aggressive-loop-optimizations
# CXX = /opt/gcc-5.3.0/bin/g++ -fabi-version=6 -D_GLIBCXX_USE_CXX11_ABI=0 #-funroll-loops
CXX = /home/roman/intel_e2/bin/icpc 
OPT = -std=c++11 -O3 -mavx
INC = -I../../../../ -I../

all:
	$(CXX) benchmark_quadrature_p1.cpp -o out_p1.exe $(INC) $(OPT)
	$(CXX) benchmark_quadrature_scalar_p1.cpp -o out_scalar_p1.exe $(INC) $(OPT) 

	$(CXX) benchmark_quadrature_p2.cpp -o out_p2.exe $(INC) $(OPT) 
	$(CXX) benchmark_quadrature_scalar_p2.cpp -o out_scalar_p2.exe $(INC) $(OPT)

	$(CXX) benchmark_quadrature_p3.cpp -o out_p3.exe $(INC) $(OPT)
	$(CXX) benchmark_quadrature_scalar_p3.cpp -o out_scalar_p3.exe $(INC) $(OPT)

	$(CXX) benchmark_quadrature_p4.cpp -o out_p4.exe $(INC) $(OPT)
	$(CXX) benchmark_quadrature_scalar_p4.cpp -o out_scalar_p4.exe $(INC) $(OPT) 

	$(CXX) benchmark_quadrature_p5.cpp -o out_p5.exe $(INC) $(OPT)
	$(CXX) benchmark_quadrature_scalar_p5.cpp -o out_scalar_p5.exe $(INC) $(OPT) 

	$(CXX) benchmark_quadrature_p6.cpp -o out_p6.exe $(INC) $(OPT)
	$(CXX) benchmark_quadrature_scalar_p6.cpp -o out_scalar_p6.exe $(INC) $(OPT)

run:
	./out_scalar_p1.exe
	./out_p1.exe
	./out_scalar_p2.exe
	./out_p2.exe
	./out_scalar_p3.exe
	./out_p3.exe
	./out_scalar_p4.exe
	./out_p4.exe
	./out_scalar_p5.exe
	./out_p5.exe
	./out_scalar_p6.exe
	./out_p6.exe

clean:
	rm -rf out_scalar_p1.exe out_scalar_p2.exe out_scalar_p3.exe out_scalar_p4.exe \
	out_scalar_p5.exe out_scalar_p6.exe \
	out_p1.exe out_p2.exe out_p3.exe out_p4.exe out_p5.exe out_p6.exe 