
#include "gtest/gtest.h"
#include "benchmark/benchmark.h"
#include <random>
#include <fstream>

namespace {

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

}