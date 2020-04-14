#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

using namespace std;

template<typename T>
T convertTo(const int position, const T init, int argc, char *argv[]) {
  if (argc <= position) {
    std::cout
        << "Conversion of argument " << position
        << " failed, not enough parameters, using default parameter: "
        << init << std::endl;
    return init;
  }
  T arg;
  std::istringstream tmp(argv[position]);
  tmp >> arg;
  // tmp >> arg ?  (std::cout << "Conversion of argument " << position << "  successfull: " << arg)
  //               : (std::cout << "Conversion of argument " << position
  //                            << "  failed");

  return arg;
}