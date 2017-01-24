
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2015 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
/////////////////////////////////////////////////////////////////////////////

#pragma once

#include "gtest/gtest.h"
#include "benchmark/benchmark.h"
#include <random>
#include <memory>
#include <fstream>

namespace lv {

    namespace test {

        /// generates a data array with 'n' random values between 'min' and 'max'
        template<typename T>
        std::unique_ptr<T[]> genarray(size_t n, T min, T max) {
            std::random_device rd;
            std::mt19937 gen(rd());
            typedef std::conditional_t<
                std::is_integral<T>::value,
                std::uniform_int_distribution<
                    std::conditional_t<
                        std::is_same<uint64_t,T>::value,
                        uint64_t,
                        int64_t
                    >
                >,
                std::uniform_real_distribution<double>
            > unif_distr;
            unif_distr uniform_dist(min,max);
            std::unique_ptr<T[]> v(new T[n]);
            for(size_t i=0; i<n; ++i)
                v[i] = (T)uniform_dist(gen);
            return std::move(v);
        }

        /// prints framework version/build info to stdout; always returns true
        bool info();

    } // namespace test

} // namespace lv
