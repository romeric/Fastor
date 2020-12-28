#ifndef TYPE_NAMES_H
#define TYPE_NAMES_H

#if __cplusplus >= 201703L

#include <string_view>
namespace Fastor {
namespace useless {
class probe_type;
inline void extract_type(std::string_view& name, std::string_view probe_type_name);
} // useless

template <typename T>
constexpr std::string_view type_name() {
  std::string_view probe_type_name("class useless::probe_type");
  const std::string_view class_specifier("class ");

  std::string_view name;
#ifdef __clang__
  name = __PRETTY_FUNCTION__;
  probe_type_name.remove_prefix (class_specifier.length ());
#elif defined(__GNUC__)
  name = __PRETTY_FUNCTION__;
  probe_type_name.remove_prefix (class_specifier.length ());
#elif defined(_MSC_VER)
  name = __FUNCSIG__;
#endif
  useless::extract_type(name, probe_type_name);
  return name;
}

namespace useless {
inline void extract_type(std::string_view& name, std::string_view probe_type_name) {
  if (name.find(probe_type_name) == std::string_view::npos) {
    //For known type probe_type get raw name and then prefix and suffix sizes
    const std::string_view probe_type_raw_name = type_name<probe_type> ();

    const size_t prefix_size = probe_type_raw_name.find(probe_type_name);
    const size_t suffix_size = probe_type_raw_name.length () - prefix_size - probe_type_name.length();

    name.remove_prefix (prefix_size);
    name.remove_suffix (suffix_size);
  }
}
} // useless
} // end of namespace Fastor

#else

#include <string>
#include <typeinfo>
#include <type_traits>
#include <memory>
#if defined(__GNUC__)
#include <cxxabi.h>
#endif

namespace Fastor {
template <class T>
std::string type_name()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own(nullptr, std::free);
#if defined(__GNUC__)
    int status = 0;
    char* demangled = abi::__cxa_demangle(typeid(TR).name(), nullptr, nullptr, &status);
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
} // end of namespace Fastor

#endif
#endif // TYPE_NAMES_H
