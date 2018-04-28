// a header file for simple template metaprogramming utilities
#pragma once
#include <type_traits>

// bool_constant is added in c++17
template <bool B>
using bool_constant = std::integral_constant<bool, B>;

// static_if. taken from:
// https://stackoverflow.com/questions/37617677/implementing-a-compile-time-static-if-logic-for-different-string-types-in-a-co

template <typename T, typename F>
auto static_if(std::true_type, T t, F) {return t;}

template <typename T, typename F>
auto static_if(std::false_type, T, F f) {return f;}

template <bool B, typename T, typename F>
auto static_if(T t, F f) {
  return static_if(bool_constant<B>{}, t, f);
}

template <bool B, typename T>
auto static_if(T t) {
  return static_if(bool_constant<B>{}, t, [](auto&&...){});
}
