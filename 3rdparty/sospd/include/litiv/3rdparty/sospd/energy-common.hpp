#ifndef _ENERGY_COMMON_HPP_
#define _ENERGY_COMMON_HPP_

#include <cstdint>
#include <limits>
#include <vector>
#include <memory>
#include <cassert>
#include <stdexcept>
#include <functional>
#include <string>
#include <array>
#include <iostream>
#include <algorithm>
#include <chrono>

#ifndef DNO_ASSERT
#define ASSERT(cond) do { if (!(cond)) { throw std::logic_error((std::string("Assertion failure at " __FILE__ ":")+std::to_string(__LINE__)+std::string(" -- " #cond)).c_str() ); }} while(false)
#else
#define ASSERT(cond) (void)0
#endif

#endif
