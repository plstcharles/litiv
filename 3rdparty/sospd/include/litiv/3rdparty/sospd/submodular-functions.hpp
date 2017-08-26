#ifndef _SUBMODULAR_FUNCTIONS_HPP_
#define _SUBMODULAR_FUNCTIONS_HPP_

#include "litiv/3rdparty/sospd/energy-common.hpp"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>

typedef uint32_t Assgn;

typedef void (*UpperBoundFunction)(int, const std::vector<REAL>&, std::vector<REAL>&);
void SubmodularUpperBound(int n, const std::vector<REAL>& oldEnergy, std::vector<REAL>& normalizedEnergy);
REAL SubmodularLowerBound(int n, std::vector<REAL>& energyTable, bool early_finish = false);
void UpperBoundCVPR14(int n, const std::vector<REAL>& origEnergy, std::vector<REAL>& energyTable);

// Takes in a set s (given by bitstring) and returns new energy such that
// f(t | s) = f(t) for all t. Does not change f(t) for t disjoint from s
// I.e., creates a set s whose members have zero marginal gain for all t
void ZeroMarginalSet(int n, std::vector<REAL>& energyTable, Assgn s);

// Updates f to f'(S) = f(S) + psi(S)
void AddLinear(int n, std::vector<REAL>& energyTable, const std::vector<REAL>& psi);

// Updates f to f'(S) = f(S) - psi1(S) - psi2(V\S)
void SubtractLinear(int n, std::vector<REAL>& energyTable, 
        const std::vector<REAL>& psi1, const std::vector<REAL>& psi2);

// Modifies an energy function to be >= 0, with f(0) = f(V) = 0
// energyTable is modified in place, must be submodular
// psi must be length n, gets filled so that 
//  f'(S) = f(S) + psi(S)
// where f' is the new energyTable, and f is the old one
void Normalize(int n, std::vector<REAL>& energyTable, std::vector<REAL>& psi);

bool CheckSubmodular(int n, const std::vector<REAL>& energyTable);
bool CheckUpperBoundInvariants(int n, const std::vector<REAL>& energyTable,
        const std::vector<REAL>& upperBound);

double DiffL1(const std::vector<REAL>& e1, const std::vector<REAL>& e2);
double DiffL2(const std::vector<REAL>& e1, const std::vector<REAL>& e2);
double DiffLInfty(const std::vector<REAL>& e1, const std::vector<REAL>& e2);

/********************** Implementation *************************/

static inline Assgn NextPerm(Assgn v) {
    Assgn t = v | (v - 1); // t gets v's least significant 0 bits set to 1
    // Next set to 1 the most significant bit to change,
    // set to 0 the least significant ones, and add the necessary 1 bits.
    return (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(v) + 1));
}

inline void UpperBoundCVPR14(int n, const std::vector<REAL>& origEnergy, std::vector<REAL>& energyTable) {
    ASSERT(n < 32);
    int max_assgn = 1 << n;
    for (int i = 0; i < max_assgn; ++i)
        energyTable[i] = origEnergy[i];
    std::vector<REAL> psi(max_assgn, 0);
    while (!CheckSubmodular(n, energyTable)) {
        // Reset psi
        for (auto& p : psi)
            p = 0;
        // Need to iterate over all k bit subsets in decreasing k
        for (int k = n-2; k >= 0; --k) {
            // Pattern to iterate over k bit subsets is: start with (1 << k) - 1
            // which has k bits set, and then increment each time with NextPerm
            // which gives the next k bit subset, until we get a subset with the 
            // n-th bit set (which will be >= max_assgn)
            // We also special case 0, which needs a different bound
            Assgn bound;
            if (k == 0) bound = 0;
            else bound = max_assgn - 1;
            Assgn s = (1 << k) - 1;
            do {
                for (int i = 0; i < n; ++i) {
                    Assgn s_i = s | (1 << i); // Set s + i
                    if (s_i == s) continue;
                    for (int j = i+1; j < n; ++j) {
                        Assgn s_j = s | (1 << j); // Set s + j
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
            s = (1 << (k+1)) - 1;
            do {
                energyTable[s] += psi[s];
                s = NextPerm(s);
            } while (s < bound);
            
        }
    }
}


inline void ChenUpperBound(int n, const std::vector<REAL>& origEnergy, std::vector<REAL>& energyTable) {
    ASSERT(n < 32);
    int max_assgn = 1 << n;
    for (int i = 0; i < max_assgn; ++i)
        energyTable[i] = origEnergy[i];
    std::vector<REAL> oldEnergy(energyTable);
    std::vector<REAL> diffEnergy(max_assgn, 0);
    int loopIterations = 0;
    std::vector<REAL> sumEnergy;
    while (!CheckSubmodular(n, energyTable)) {
        loopIterations++;
        REAL iterSumEnergy = 0;
        for (int i = 0; i < max_assgn; ++i)
            iterSumEnergy += energyTable[i];
        sumEnergy.push_back(iterSumEnergy);

        if (loopIterations > 1000) {
            std::cout << "Infinite upper bound loop\n";
            std::cout << "nVars = " << n << "\n";
            std::cout << "Energy: [";
            for (int i = 0; i < max_assgn; ++i)
                std::cout << origEnergy[i] << ", ";
            std::cout << "]\n";

            std::cout << "New energy: [";
            for (int i = 0; i < max_assgn; ++i)
                std::cout << energyTable[i] << ", ";
            std::cout << "]\n";

            std::cout << "Zero Values: ";
            for (int i = 0; i < max_assgn; ++i) {
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

        for (int i = 0; i < max_assgn; ++i)
            diffEnergy[i] = oldEnergy[i] - energyTable[i];
        for (int i = max_assgn - 2; i > 0; --i) {
            for (int k = 0; k < n; ++k) {
                if (i & (1 << k)) continue;
                int ik = i | (1 << k);
                bool t = false;
                for (int j = 0; j < n; ++j) {
                    if (i & (1 << j)) {
                        REAL tmp = diffEnergy[ik ^ (1 << j)];
                        if (tmp < diffEnergy[ik] / 2) {
                            t = true;
                            break;
                        }
                        if (diffEnergy[ik] - tmp > diffEnergy[i])
                            diffEnergy[i] = diffEnergy[ik] - tmp;
                    }
                }
                if (t) {
                    REAL tmp = (diffEnergy[i | (1 << k)] + 1) / 2;
                    if (tmp > diffEnergy[i])
                        diffEnergy[i] = tmp;
                }
            }
        }
        for (int i = 0; i < max_assgn; ++i) {
            energyTable[i] += diffEnergy[i];
        }
        std::copy(energyTable.begin(), energyTable.end(), oldEnergy.begin());
    }
}

inline REAL SubmodularLowerBound(int n, std::vector<REAL>& energyTable, bool early_finish) {
    ASSERT(n < 32);
    Assgn max_assgn = 1 << n;
    ASSERT(energyTable.size() == max_assgn);
    REAL max_diff = 0;

    // Need to iterate over all k bit subsets in increasing k
    for (int k = 1; k <= n; ++k) {
        bool changed = false;
        Assgn bound;
        if (k == 0) bound = 0;
        else bound = max_assgn - 1;
        Assgn s = (1 << k) - 1;
        do {
            REAL subtract_from_s = 0;
            for (int i = 0; i < n; ++i) {
                Assgn s_i = s ^ (1 << i); // Set s - i
                if (s_i >= s) continue;
                for (int j = i+1; j < n; ++j) {
                    Assgn s_j = s ^ (1 << j); // Set s - j
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
            changed |= (subtract_from_s > 0);
            max_diff = std::max(max_diff, subtract_from_s);
            s = NextPerm(s);
        } while (s < bound);
        if (early_finish && !changed)
            break;
    }
    return max_diff;
}

inline void ZeroMarginalSet(int n, std::vector<REAL>& energyTable, Assgn s) {
    Assgn base_set = (1 << n) - 1;
    Assgn not_s = base_set & (~s);
    for (Assgn t = 0; t <= base_set; ++t)
        energyTable[t] = energyTable[t & not_s];
}

inline void AddLinear(int n, std::vector<REAL>& energyTable, const std::vector<REAL>& psi) {
    Assgn max_assgn = 1 << n;
    ASSERT(max_assgn == energyTable.size());
    ASSERT(n == int(psi.size()));
    REAL sum = 0;
    Assgn last_gray = 0;
    for (Assgn a = 1; a < max_assgn; ++a) {
        Assgn gray = a ^ (a >> 1);
        Assgn diff = gray ^ last_gray;
        int changed_bit = __builtin_ctz(diff);
        if (gray & diff)
            sum += psi[changed_bit];
        else
            sum -= psi[changed_bit];
        energyTable[gray] += sum;
        last_gray = gray;
    }
}

inline void SubtractLinear(int n, std::vector<REAL>& energyTable, 
        const std::vector<REAL>& psi1, const std::vector<REAL>& psi2) {
    Assgn max_assgn = 1 << n;
    ASSERT(max_assgn == energyTable.size());
    ASSERT(n == int(psi1.size()));
    ASSERT(n == int(psi2.size()));
    REAL sum = 0;
    for (int i = 0; i < n; ++i)
        sum += psi2[i];
    energyTable[0] -= sum;
    Assgn last_gray = 0;
    for (Assgn a = 1; a < max_assgn; ++a) {
        Assgn gray = a ^ (a >> 1);
        Assgn diff = gray ^ last_gray;
        int changed_idx = __builtin_ctz(diff);
        if (gray & diff)
            sum += psi1[changed_idx] - psi2[changed_idx];
        else
            sum += psi2[changed_idx] - psi1[changed_idx];
        energyTable[gray] -= sum;
        last_gray = gray;
    }
}

inline void Normalize(int n, std::vector<REAL>& energyTable, std::vector<REAL>& psi) {
    Assgn max_assgn = 1 << n;
    ASSERT(max_assgn == energyTable.size());
    ASSERT(n == int(psi.size()));
    auto constTerm = energyTable[0];
    for (auto& e : energyTable)
        e -= constTerm;
    Assgn last_assgn = 0;
    Assgn this_assgn = 0;
    for (int i = 0; i < n; ++i) {
        this_assgn |= (1 << i);
        psi[i] = energyTable[last_assgn] - energyTable[this_assgn];
        last_assgn = this_assgn;
    }
    AddLinear(n, energyTable, psi);

    for (REAL e : energyTable)
        ASSERT(e >= 0);
    ASSERT(energyTable[0] == 0);
    ASSERT(energyTable[max_assgn-1] == 0);
}

inline bool CheckSubmodular(int n, const std::vector<REAL>& energyTable) {
    ASSERT(n < 32);
    Assgn max_assgn = 1 << n;
    ASSERT(energyTable.size() == max_assgn);

    for (Assgn s = 0; s < max_assgn; ++s) {
        for (int i = 0; i < n; ++i) {
            Assgn s_i = s | (1 << i);
            if (s_i == s) continue;
            for (int j = i+1; j < n; ++j) {
                Assgn s_j = s | (1 << j);
                if (s_j == s) continue;
                Assgn s_ij = s_i | s_j;

                REAL submodularity = energyTable[s] + energyTable[s_ij]
                    - energyTable[s_i] - energyTable[s_j];
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

inline bool CheckUpperBoundInvariants(int n, const std::vector<REAL>& energyTable,
        const std::vector<REAL>& upperBound) {
    int energy_len = energyTable.size();
    ASSERT(energy_len == int(upperBound.size()));
    REAL max_energy = std::numeric_limits<REAL>::min();
    for (int i = 0; i < energy_len; ++i) {
        if (energyTable[i] > upperBound[i])
            return false;
        max_energy = std::max(energyTable[i], max_energy);
    }
    for (int i = 0; i < n; ++i) {
        if (upperBound[1 << i] > max_energy)
            return false;
    }
    return CheckSubmodular(n, upperBound);
}

#endif
