/*  This file is part of the FASTOR library. {{{
Copyright (c) 2020 Roman Poya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

}}}*/

#ifndef CPUID_H
#define CPUID_H

#ifdef _WIN32
#include <limits.h>
#include <intrin.h>
typedef unsigned __int32  uint32_t;
#else
#include <stdint.h>
#endif


namespace Fastor {

class CPUID {
  uint32_t regs[4];

public:
  explicit CPUID(unsigned i) {
#ifdef _WIN32
    __cpuid((int *)regs, (int)i);

#else
    asm volatile
      ("cpuid" : "=a" (regs[0]), "=b" (regs[1]), "=c" (regs[2]), "=d" (regs[3])
       : "a" (i), "c" (0));
    // ECX is set to zero for CPUID function 4
#endif
  }

  const uint32_t& EAX() const {return regs[0];} // cpu base frequency
  const uint32_t& EBX() const {return regs[1];} // cpu max frequency
  const uint32_t& ECX() const {return regs[2];} // bus (reference) frequency
  const uint32_t& EDX() const {return regs[3];}
};

// Usage:
// CPUID cpuID(0);
// std::string vendor;
// vendor += std::string((const char *)&cpuID.EBX(), 4);
// vendor += std::string((const char *)&cpuID.EDX(), 4);
// vendor += std::string((const char *)&cpuID.ECX(), 4);
// std::cout << "CPU vendor = " << vendor << std::endl;


} // Fastor

#endif // CPUID_H
