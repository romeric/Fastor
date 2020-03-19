#include <Fastor/Fastor.h>

using namespace Fastor;


template<typename T, FASTOR_INDEX mm, FASTOR_INDEX nn>
void run_fixed_size() {

    {
        Tensor<T,mm,nn> a; a.iota(5);
        Tensor<bool,mm,nn> ba1 = a == a;

        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba1(i,j) == true, "TEST FAILED");
            }
        }
        Tensor<bool,mm,nn> ba2 = a != a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba2(i,j) == false, "TEST FAILED");
            }
        }
        Tensor<bool,mm,nn> ba3 = a < a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba3(i,j) == false, "TEST FAILED");
            }
        }
        Tensor<bool,mm,nn> ba4 = a >= a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba4(i,j) == true, "TEST FAILED");
            }
        }
        Tensor<bool,mm,nn> ba5 = a <= a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }

        // expr
        ba5 = a + 1 == a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == false, "TEST FAILED");
            }
        }
        ba5 = a + 1 != a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }
        ba5 = a + 1 > a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }
        ba5 = a - 1 < a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }
        ba5 = a * 1 <= a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }
        ba5 = a * 2 >= a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }


        // fixed views
        ba5(fall,fall) = a == a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }
        ba5(fall,fall) = a < a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == false, "TEST FAILED");
            }
        }
        ba5(fall,fall) = a > a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == false, "TEST FAILED");
            }
        }
        ba5(fall,fall) = a <= a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }
        ba5(fall,fall) = a >= a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }

        // views
        ba5(all,all) = a == a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }
        ba5(all,all) = a < a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == false, "TEST FAILED");
            }
        }
        ba5(all,all) = a > a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == false, "TEST FAILED");
            }
        }
        ba5(all,all) = a <= a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }
        ba5(all,all) = a >= a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }

        // expr fixed view
        ba5(fall,fall) = a + 1 == a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == false, "TEST FAILED");
            }
        }
        ba5(fall,fall) = a + 1 != a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }
        ba5(fall,fall) = a + 1 > a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }
        ba5(fall,fall) = a - 1 < a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }
        ba5(fall,fall) = a * 1 <= a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }
        ba5(fall,fall) = a * 2 >= a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }

        // expr view
        ba5(all,all) = a + 1 == a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == false, "TEST FAILED");
            }
        }
        ba5(all,all) = a + 1 != a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }
        ba5(all,all) = a + 1 > a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }
        ba5(all,all) = a - 1 < a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }
        ba5(all,all) = a * 1 <= a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }
        ba5(all,all) = a * 2 >= a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(ba5(i,j) == true, "TEST FAILED");
            }
        }
    }

    {
        Tensor<T,mm,nn> a; a.iota(3);
        Tensor<T,mm,nn> b; b.iota(3);
        b(1,1) = 99;
        Tensor<bool,mm,nn> bab1 = b == a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                if (i==1 && j==1) {
                    FASTOR_EXIT_ASSERT(bab1(i,j) == false, "TEST FAILED");
                }
                else {
                    FASTOR_EXIT_ASSERT(bab1(i,j) == true, "TEST FAILED");
                }
            }
        }
        bab1 = b != a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                if (i==1 && j==1) {
                    FASTOR_EXIT_ASSERT(bab1(i,j) == true, "TEST FAILED");
                }
                else {
                    FASTOR_EXIT_ASSERT(bab1(i,j) == false, "TEST FAILED");
                }
            }
        }
        bab1 = a < b;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                if (i==1 && j==1) {
                    FASTOR_EXIT_ASSERT(bab1(i,j) == true, "TEST FAILED");
                }
                else {
                    FASTOR_EXIT_ASSERT(bab1(i,j) == false, "TEST FAILED");
                }
            }
        }
        bab1 = b > a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                if (i==1 && j==1) {
                    FASTOR_EXIT_ASSERT(bab1(i,j) == true, "TEST FAILED");
                }
                else {
                    FASTOR_EXIT_ASSERT(bab1(i,j) == false, "TEST FAILED");
                }
            }
        }
        bab1 = b >= a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(bab1(i,j) == true, "TEST FAILED");
            }
        }
        bab1 = a <= a;
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_EXIT_ASSERT(bab1(i,j) == true, "TEST FAILED");
            }
        }
    }

    {
        Tensor<T> a(2); 
        Tensor<bool> ba = a==2;
        FASTOR_EXIT_ASSERT(ba.toscalar() == true, "TEST FAILED");
        ba = a>2;
        FASTOR_EXIT_ASSERT(ba.toscalar() == false, "TEST FAILED");
        ba = a<2;
        FASTOR_EXIT_ASSERT(ba.toscalar() == false, "TEST FAILED");
        ba = a>=2;
        FASTOR_EXIT_ASSERT(ba.toscalar() == true, "TEST FAILED");
        ba = a<=2;
        FASTOR_EXIT_ASSERT(ba.toscalar() == true, "TEST FAILED");
    }
}


template<typename T>
void run() {
    run_fixed_size<T,2,2>();
    print(FGRN(BOLD("All tests passed successfully")));
    run_fixed_size<T,4,4>();
    print(FGRN(BOLD("All tests passed successfully")));
    run_fixed_size<T,8,8>();
    print(FGRN(BOLD("All tests passed successfully")));
    run_fixed_size<T,7,13>();
    print(FGRN(BOLD("All tests passed successfully")));
}


int main() {
    print(FBLU(BOLD("Testing binary comparison operators: single precision")));
    run<float>();
    print(FBLU(BOLD("Testing binary comparison operators: double precision")));
    run<double>();
    print(FBLU(BOLD("Testing binary comparison operators: int 32")));
    run<int>();
    print(FBLU(BOLD("Testing binary comparison operators: int 64")));
    run<Int64>();
}

