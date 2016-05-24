#include <string>
#include <typeinfo>
#include <type_traits>
#include <memory>
#if defined(__GNUC__)
#include <cxxabi.h>
#endif

template <class T>
std::string type_name()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own (nullptr,std::free);
#if defined(__GNUC__)
    int status = 0;
    char* demangled = abi::__cxa_demangle(typeid(TR).name(),0,0,&status);
    std::string r = own != nullptr ? own.get() : std::string(demangled);
#else
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
#endif
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}