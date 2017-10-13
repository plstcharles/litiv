#pragma once

#include "litiv/3rdparty/sospd/energy-common.hpp"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>

namespace sospd {

    typedef uint32_t Assgn; // cannot change; uses built-in functions in some utils
    typedef std::chrono::system_clock::time_point TimePt;
    typedef std::chrono::duration<double> Duration;
    typedef std::chrono::system_clock Clock;

    template<typename REAL, typename IDX>
    using UpperBoundFunction =  void(*)(IDX,const std::vector<REAL>&,std::vector<REAL>&);
    template<typename REAL, typename IDX>
    REAL SubmodularLowerBound(IDX n, std::vector<REAL>& energyTable, bool early_finish=false);
    template<typename REAL, typename IDX>
    void UpperBoundCVPR14(IDX n, const std::vector<REAL>& origEnergy, std::vector<REAL>& energyTable);
    template<typename REAL, typename IDX>
    void ChenUpperBound(IDX n, const std::vector<REAL>& origEnergy, std::vector<REAL>& energyTable);

    // Takes in a set s (given by bitstring) and returns new energy such that
    // f(t | s) = f(t) for all t. Does not change f(t) for t disjoint from s
    // I.e., creates a set s whose members have zero marginal gain for all t
    template<typename REAL, typename IDX>
    void ZeroMarginalSet(IDX n, std::vector<REAL>& energyTable, Assgn s);

    // Updates f to f'(S) = f(S) + psi(S)
    template<typename REAL, typename IDX>
    void AddLinear(IDX n, std::vector<REAL>& energyTable, const std::vector<REAL>& psi);

    // Updates f to f'(S) = f(S) - psi1(S) - psi2(V\S)
    template<typename REAL, typename IDX, typename REALARRAY>
    void SubtractLinear(IDX n, std::vector<REAL>& energyTable, const REALARRAY& psi1, const REALARRAY& psi2);

    // Modifies an energy function to be >= 0, with f(0) = f(V) = 0
    // energyTable is modified in place, must be submodular
    // psi must be length n, gets filled so that
    //  f'(S) = f(S) + psi(S)
    // where f' is the new energyTable, and f is the old one
    template<typename REAL, typename IDX>
    void Normalize(IDX n, std::vector<REAL>& energyTable, std::vector<REAL>& psi);

    template<typename REAL, typename IDX>
    bool CheckSubmodular(IDX n, const std::vector<REAL>& energyTable);
    template<typename REAL, typename IDX>
    bool CheckUpperBoundInvariants(IDX n, const std::vector<REAL>& energyTable, const std::vector<REAL>& upperBound);

    template<typename REAL>
    double DiffL1(const std::vector<REAL>& e1, const std::vector<REAL>& e2);
    template<typename REAL>
    double DiffL2(const std::vector<REAL>& e1, const std::vector<REAL>& e2);
    template<typename REAL>
    double DiffLInfty(const std::vector<REAL>& e1, const std::vector<REAL>& e2);

    inline Assgn NextPerm(Assgn v) {
        Assgn t = v | (v - 1); // t gets v's least significant 0 bits set to 1
        // Next set to 1 the most significant bit to change,
        // set to 0 the least significant ones, and add the necessary 1 bits.
        return (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(v) + 1));
    }

} // namespace sospd

template<typename REAL, typename IDX>
inline REAL sospd::SubmodularLowerBound(IDX n, std::vector<REAL>& energyTable, bool early_finish) {
    static_assert(std::is_arithmetic<REAL>::value,"value type must be arithmetic");
    ASSERT(n < IDX(32));
    Assgn max_assgn = 1u << n;
    ASSERT(energyTable.size() == max_assgn);
    REAL max_diff = 0;

    // Need to iterate over all k bit subsets in increasing k
    for (IDX k = 1; k <= n; ++k) {
        bool changed = false;
        Assgn bound;
        if (k == 0) bound = 0;
        else bound = max_assgn - 1;
        Assgn s = (1u << k) - 1;
        do {
            REAL subtract_from_s = 0;
            for (IDX i = 0; i < n; ++i) {
                Assgn s_i = s ^ (1u << i); // Set s - i
                if (s_i >= s) continue;
                for (IDX j = i+1; j < n; ++j) {
                    Assgn s_j = s ^ (1u << j); // Set s - j
                    if (s_j >= s) continue;
                    Assgn s_ij = s_i & s_j;
                    REAL submodularity = energyTable[s] + energyTable[s_ij]
                                         - energyTable[s_i] - energyTable[s_j];
                    if (submodularity > subtract_from_s) {
                        subtract_from_s = submodularity;
                    }
                }
            }
            energyTable[s] -= subtract_from_s;
            changed |= (subtract_from_s > REAL(0));
            max_diff = std::max(max_diff, subtract_from_s);
            s = NextPerm(s);
        } while (s < bound);
        if (early_finish && !changed)
            break;
    }
    return max_diff;
}

template<typename REAL, typename IDX>
inline void sospd::UpperBoundCVPR14(IDX n, const std::vector<REAL>& origEnergy, std::vector<REAL>& energyTable) {
    static_assert(std::is_arithmetic<REAL>::value,"value type must be arithmetic");
    ASSERT(n < IDX(32));
    Assgn max_assgn = 1u << n;
    for (Assgn i = 0; i < max_assgn; ++i)
        energyTable[i] = origEnergy[i];
    std::vector<REAL> psi(max_assgn, 0);
    while (!CheckSubmodular(n, energyTable)) {
        // Reset psi
        for (auto& p : psi)
            p = 0;
        // Need to iterate over all k bit subsets in decreasing k
        for (IDX k = n-2; k >= 0; --k) {
            // Pattern to iterate over k bit subsets is: start with (1 << k) - 1
            // which has k bits set, and then increment each time with NextPerm
            // which gives the next k bit subset, until we get a subset with the
            // n-th bit set (which will be >= max_assgn)
            // We also special case 0, which needs a different bound
            Assgn bound;
            if (k == 0) bound = 0;
            else bound = max_assgn - 1;
            Assgn s = (1u << k) - 1;
            do {
                for (IDX i = 0; i < n; ++i) {
                    Assgn s_i = s | (1u << i); // Set s + i
                    if (s_i == s) continue;
                    for (IDX j = i+1; j < n; ++j) {
                        Assgn s_j = s | (1u << j); // Set s + j
                        if (s_j == s) continue;
                        Assgn s_ij = s_i | s_j;
                        REAL delta_Sij = energyTable[s] + energyTable[s_ij]
                            - energyTable[s_i] - energyTable[s_j];
                        if (delta_Sij > 0) {
                            REAL shift = (delta_Sij + 1) / 2;
                            //REAL rem = delta_Sij % 2;
                            psi[s_i] = std::max(psi[s_i], shift);
                            psi[s_j] = std::max(psi[s_j], shift);
                        }
                    }
                }
                s = NextPerm(s);
            } while (s < bound);
            // Then, add psi[s] to energyTable[s] for every k+1 bit subset s
            bound = max_assgn - 1;
            s = (1u << (k+1)) - 1;
            do {
                energyTable[s] += psi[s];
                s = NextPerm(s);
            } while (s < bound);

        }
    }
}

template<typename REAL, typename IDX>
inline void sospd::ChenUpperBound(IDX n, const std::vector<REAL>& origEnergy, std::vector<REAL>& energyTable) {
    static_assert(std::is_arithmetic<REAL>::value,"value type must be arithmetic");
    ASSERT(n < IDX(32));
    IDX max_assgn = IDX(1u << n);
    for (IDX i = 0; i < max_assgn; ++i)
        energyTable[i] = origEnergy[i];
    std::vector<REAL> oldEnergy(energyTable);
    std::vector<REAL> diffEnergy(max_assgn, 0);
    IDX loopIterations = 0;
    std::vector<REAL> sumEnergy;
    while (!CheckSubmodular(n, energyTable)) {
        loopIterations++;
        REAL iterSumEnergy = 0;
        for (IDX i = 0; i < max_assgn; ++i)
            iterSumEnergy += energyTable[i];
        sumEnergy.push_back(iterSumEnergy);

        if (loopIterations > IDX(1000)) {
            std::cout << "Infinite upper bound loop\n";
            std::cout << "nVars = " << n << "\n";
            std::cout << "Energy: [";
            for (IDX i = 0; i < max_assgn; ++i)
                std::cout << origEnergy[i] << ", ";
            std::cout << "]\n";

            std::cout << "New energy: [";
            for (IDX i = 0; i < max_assgn; ++i)
                std::cout << energyTable[i] << ", ";
            std::cout << "]\n";

            std::cout << "Zero Values: ";
            for (IDX i = 0; i < max_assgn; ++i) {
                if (origEnergy[i] == 0)
                    std::cout << i << ", ";
            }
            std::cout << "\n";

            std::cout << "Energy values: [";
            for (auto v : sumEnergy)
                std::cout << v << ",";
            std::cout << "]\n";
            exit(-1);
        }

        SubmodularLowerBound(n, energyTable);

        for (IDX i = 0; i < max_assgn; ++i)
            diffEnergy[i] = oldEnergy[i] - energyTable[i];
        for (IDX i = max_assgn - 2; i > 0; --i) {
            for (IDX k = 0; k < n; ++k) {
                if (i & IDX(1u << k)) continue;
                IDX ik = i | IDX(1u << k);
                bool t = false;
                for (IDX j = 0; j < n; ++j) {
                    if (i & IDX(1u << j)) {
                        REAL tmp = diffEnergy[ik ^ IDX(1u << j)];
                        if (tmp < diffEnergy[ik] / 2) {
                            t = true;
                            break;
                        }
                        if (diffEnergy[ik] - tmp > diffEnergy[i])
                            diffEnergy[i] = diffEnergy[ik] - tmp;
                    }
                }
                if (t) {
                    REAL tmp = (diffEnergy[i | IDX(1u << k)] + 1) / 2;
                    if (tmp > diffEnergy[i])
                        diffEnergy[i] = tmp;
                }
            }
        }
        for (IDX i = 0; i < max_assgn; ++i) {
            energyTable[i] += diffEnergy[i];
        }
        std::copy(energyTable.begin(), energyTable.end(), oldEnergy.begin());
    }
}

template<typename REAL, typename IDX>
inline void sospd::ZeroMarginalSet(IDX n, std::vector<REAL>& energyTable, Assgn s) {
    static_assert(std::is_arithmetic<REAL>::value,"value type must be arithmetic");
    Assgn base_set = (1u << n) - 1;
    Assgn not_s = base_set & (~s);
    for (Assgn t = 0; t <= base_set; ++t)
        energyTable[t] = energyTable[t & not_s];
}

template<typename REAL, typename IDX>
inline void sospd::AddLinear(IDX n, std::vector<REAL>& energyTable, const std::vector<REAL>& psi) {
    static_assert(std::is_arithmetic<REAL>::value,"value type must be arithmetic");
    Assgn max_assgn = 1u << n;
    ASSERT(max_assgn == energyTable.size());
    ASSERT(n == IDX(psi.size()));
    REAL sum = 0;
    Assgn last_gray = 0;
    for (Assgn a = 1; a < max_assgn; ++a) {
        Assgn gray = a ^ (a >> 1);
        Assgn diff = gray ^ last_gray;
        IDX changed_bit = __builtin_ctz(diff);
        if (gray & diff)
            sum += psi[changed_bit];
        else
            sum -= psi[changed_bit];
        energyTable[gray] += sum;
        last_gray = gray;
    }
}

template<typename REAL, typename IDX, typename REALARRAY>
inline void sospd::SubtractLinear(IDX n, std::vector<REAL>& energyTable, const REALARRAY& psi1, const REALARRAY& psi2) {
    static_assert(std::is_arithmetic<REAL>::value,"value type must be arithmetic");
    Assgn max_assgn = 1u << n;
    ASSERT(max_assgn == energyTable.size());
    ASSERT(n <= IDX(psi1.size()));
    ASSERT(n <= IDX(psi2.size()));
    REAL sum = 0;
    for (IDX i = 0; i < n; ++i)
        sum += psi2[i];
    energyTable[0] -= sum;
    Assgn last_gray = 0;
    for (Assgn a = 1; a < max_assgn; ++a) {
        Assgn gray = a ^ (a >> 1);
        Assgn diff = gray ^ last_gray;
        IDX changed_idx = __builtin_ctz(diff);
        if (gray & diff)
            sum += psi1[changed_idx] - psi2[changed_idx];
        else
            sum += psi2[changed_idx] - psi1[changed_idx];
        energyTable[gray] -= sum;
        last_gray = gray;
    }
}

template<typename REAL, typename IDX>
inline void sospd::Normalize(IDX n, std::vector<REAL>& energyTable, std::vector<REAL>& psi) {
    static_assert(std::is_arithmetic<REAL>::value,"value type must be arithmetic");
    Assgn max_assgn = 1u << n;
    ASSERT(max_assgn == energyTable.size());
    ASSERT(n == IDX(psi.size()));
    auto constTerm = energyTable[0];
    for (auto& e : energyTable)
        e -= constTerm;
    Assgn last_assgn = 0;
    Assgn this_assgn = 0;
    for (IDX i = 0; i < n; ++i) {
        this_assgn |= (1u << i);
        psi[i] = energyTable[last_assgn] - energyTable[this_assgn];
        last_assgn = this_assgn;
    }
    AddLinear(n, energyTable, psi);

    for (REAL e : energyTable)
        ASSERT(e >= 0);
    ASSERT(energyTable[0] == 0);
    ASSERT(energyTable[max_assgn-1] == 0);
}

template<typename REAL, typename IDX>
inline bool sospd::CheckSubmodular(IDX n, const std::vector<REAL>& energyTable) {
    static_assert(std::is_arithmetic<REAL>::value,"value type must be arithmetic");
    ASSERT(n < IDX(32));
    Assgn max_assgn = 1u << n;
    ASSERT(energyTable.size() == max_assgn);

    for (Assgn s = 0; s < max_assgn; ++s) {
        for (IDX i = 0; i < n; ++i) {
            Assgn s_i = s | (1u << i);
            if (s_i == s) continue;
            for (IDX j = i+1; j < n; ++j) {
                Assgn s_j = s | (1u << j);
                if (s_j == s) continue;
                Assgn s_ij = s_i | s_j;
                REAL submodularity = energyTable[s] + energyTable[s_ij] - energyTable[s_i] - energyTable[s_j];
                if (submodularity > 0) {
                    //std::cout << "Nonsubmodular: (" << s << ", " << i << ", " << j << "): ";
                    //std::cout << energyTable[s] << " " << energyTable[s_i] << " "
                    //    << energyTable[s_j] << " " << energyTable[s_ij] << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}

template<typename REAL, typename IDX>
inline bool sospd::CheckUpperBoundInvariants(IDX n, const std::vector<REAL>& energyTable, const std::vector<REAL>& upperBound) {
    static_assert(std::is_arithmetic<REAL>::value,"value type must be arithmetic");
    IDX energy_len = IDX(energyTable.size());
    ASSERT(energy_len == IDX(upperBound.size()));
    REAL max_energy = std::numeric_limits<REAL>::min();
    for (IDX i = 0; i < energy_len; ++i) {
        if (energyTable[i] > upperBound[i])
            return false;
        max_energy = std::max(energyTable[i], max_energy);
    }
    for (IDX i = 0; i < n; ++i) {
        if (upperBound[1u << i] > max_energy)
            return false;
    }
    return CheckSubmodular(n, upperBound);
}

template<typename REAL>
inline double sospd::DiffL1(const std::vector<REAL>& e1, const std::vector<REAL>& e2) {
    double norm = 0;
    for(size_t i = 0; i < e1.size(); ++i)
        norm += std::abs(static_cast<double>(e1[i] - e2[i]));
    return norm;
}

template<typename REAL>
inline double sospd::DiffL2(const std::vector<REAL>& e1, const std::vector<REAL>& e2) {
    double norm = 0;
    for(size_t i = 0; i < e1.size(); ++i) {
        double diff = std::abs(static_cast<double>(e1[i] - e2[i]));
        norm += diff*diff;
    }
    return norm;
}

template<typename REAL>
inline double sospd::DiffLInfty(const std::vector<REAL>& e1, const std::vector<REAL>& e2) {
    double norm = 0;
    for(size_t i = 0; i < e1.size(); ++i)
        norm = std::max(norm, std::abs(static_cast<double>(e1[i] - e2[i])));
    return norm;
}
