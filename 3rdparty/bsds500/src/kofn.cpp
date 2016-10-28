
#include "litiv/3rdparty/bsds500/kofn.hpp"
#include <random>
#include <chrono>
#include <cassert>
#include <functional>

using namespace BSDS500;

static std::mt19937 s_oMT(std::chrono::system_clock::now().time_since_epoch().count());
static std::uniform_real_distribution<double> s_oURDistrib_0_1(0,std::nextafter(1,std::numeric_limits<double>::max()));
static auto s_oRand_0_1_Funct = std::bind(s_oURDistrib_0_1,s_oMT);

// O(n) implementation.
static void
_kOfN_largeK (int k, int n, int* values)
{
    assert (k > 0);
    assert (k <= n);
    int j = 0;
    for (int i = 0; i < n; i++) {
        double prob = (double) (k - j) / (n - i);
        assert (prob <= 1);
        double x = s_oRand_0_1_Funct();
        if (x < prob) {
            values[j++] = i;
        }
    }
    assert (j == k);
}

// O(k*lg(k)) implementation; constant factor is about 2x the constant
// factor for the O(n) implementation.
static void
_kOfN_smallK (int k, int n, int* values)
{
    assert (k > 0);
    assert (k <= n);
    if (k == 1) {
        std::uniform_int_distribution<int32_t> oUIDistrib(0,n-1);
        values[0] = oUIDistrib(s_oMT);
        return;
    }
    int leftN = n / 2;
    int rightN = n - leftN;
    int leftK = 0;
    int rightK = 0;
    for (int i = 0; i < k; i++) {
        std::uniform_int_distribution<int32_t> oUIDistrib(0,n-i-1);
        int x = oUIDistrib(s_oMT);
        if (x < leftN - leftK) {
            leftK++;
        } else {
            rightK++;
        }
    }
    if (leftK > 0) { _kOfN_smallK (leftK, leftN, values); }
    if (rightK > 0) { _kOfN_smallK (rightK, rightN, values + leftK); }
    for (int i = leftK; i < k; i++) {
        values[i] += leftN;
    }
}

// Return k randomly selected integers from the interval [0,n), in
// increasing sorted order.
void
BSDS500::kOfN (int k, int n, int* values)
{
    assert (k >= 0);
    assert (n >= 0);
    if (k == 0) { return; }
    static double log2 = log (2);
    double klogk = k * log (k) / log2;
    if (klogk < n / 2) {
        _kOfN_smallK (k, n, values);
    } else {
        _kOfN_largeK (k, n, values);
    }
}
