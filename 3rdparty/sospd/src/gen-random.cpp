#include "litiv/3rdparty/sospd/gen-random.hpp"
#include <random>
#include <algorithm>
#include "litiv/3rdparty/sospd/higher-order-energy.hpp"
#include "litiv/3rdparty/sospd/submodular-ibfs.hpp"

template <typename REAL>
void GenRandomEnergyTable(std::vector<REAL>& energy_table, size_t k, REAL clique_range, std::mt19937& random_gen) {
    std::uniform_int_distribution<REAL> energy_dist(-clique_range, 0);
    const uint32_t num_assignments = 1 << k;
    for (auto& e : energy_table)
        e = 0;

    for (uint32_t subset = 1; subset < num_assignments; ++subset) {
        REAL subset_energy = energy_dist(random_gen);
        for (uint32_t assignment = 0; assignment < num_assignments; ++assignment) {
            if ((assignment & subset) == subset)
                energy_table[assignment] += subset_energy;
        }
    }
}

template <typename HigherOrder, typename REAL>
void GenRandom(HigherOrder& ho, 
        size_t n, 
        size_t k, 
        size_t m, 
        REAL clique_range, 
        REAL unary_mean,
        REAL unary_var,
        unsigned int seed)
{
    typedef typename HigherOrder::NodeId NodeId;
    std::mt19937 random_gen(seed); // Random number generator
    std::uniform_int_distribution<NodeId> node_gen(0, n-1);
    std::normal_distribution<double> unary_gen(unary_mean, unary_var);

    std::vector<REAL> energy_table(1 << k);
    ho.AddNode(n);
    for (size_t i = 0; i < m; ++i) {
        std::vector<NodeId> clique_nodes;
        while (clique_nodes.size() < k) {
            NodeId new_node = node_gen(random_gen);
            if (std::count(clique_nodes.begin(), clique_nodes.end(), new_node) == 0)
                clique_nodes.push_back(new_node);
        }
        GenRandomEnergyTable(energy_table, k, clique_range, random_gen);
        ho.AddClique(clique_nodes, energy_table);
    }

    for (size_t i = 0; i < n; ++i) {
        double unary_term = unary_gen(random_gen);
        ho.AddUnaryTerm(i, unary_term);
    }
}

// Explicit instantiations
#define GEN_RANDOM_INSTANTIATE(HO, R) template void GenRandom(HO&, size_t, size_t, size_t, R, R, R, unsigned int);

typedef HigherOrderEnergy<REAL, 4> HO;
GEN_RANDOM_INSTANTIATE(HO, REAL);
GEN_RANDOM_INSTANTIATE(SubmodularIBFS, REAL);

#undef GEN_RANDOM_INSTANTIATE
