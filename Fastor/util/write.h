#ifndef WRITE_H
#define WRITE_H

#include <fstream>

namespace Fastor {

template<typename T>
inline void write(const std::string &filename, const T &a) {
//    if ( filename.empty() )
//            filename = "output";
    std::ofstream outfile;
    outfile.open(filename,std::ios_base::app);
    outfile << a << "\n";
    outfile.close();
}

template<typename T, typename ... Rest>
inline void write(const std::string &filename, const T &first, const Rest& ... rest) {
    write(filename,first);
    write(filename,rest...);
}

inline void write(const std::string &filename) {
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


} // end of namespace

#endif // WRITE_H

