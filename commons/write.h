#ifndef WRITE_H
#define WRITE_H

#include <fstream>

template<typename T>
void write(const std::string &filename, const T &a) {
//    if ( filename.empty() )
//            filename = "output";
    std::ofstream outfile;
    outfile.open(filename,std::ios_base::app);
    outfile << a << "\n";
    outfile.close();
}

template<typename T, typename ... Rest>
void write(const std::string &filename, const T &first, const Rest& ... rest) {
    write(filename,first);
    write(filename,rest...);
}

void write(const std::string &filename) {
    std::ofstream outfile;
    outfile.open(filename,std::ios_base::app);
    outfile << "\n";
    outfile.close();
}

//void write() {
//    const std::string &filename = "output";
//    std::ofstream outfile;
//    outfile.open(filename,std::ios_base::app);
//    outfile << "\n";
//    outfile.close();
//}


#endif // WRITE_H

