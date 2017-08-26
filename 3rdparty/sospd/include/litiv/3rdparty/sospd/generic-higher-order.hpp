#ifndef _GENERIC_HIGHER_ORDER_HPP_
#define _GENERIC_HIGHER_ORDER_HPP_

#include "litiv/3rdparty/sospd/higher-order-energy.hpp"
#include "litiv/3rdparty/HOCR/HOCR.h"
#ifdef WITH_GRD
#include "PseudoBoolean.h"
#endif
#include "litiv/3rdparty/sospd/submodular-ibfs.hpp"
#include "litiv/3rdparty/sospd/submodular-functions.hpp"
#include <string>

enum class OptType {
    Fix,
    Fix_I, // FGBZ with QPBOI
    HOCR,
    GRD,
    GRD_Heur,
    PC,
    PC_I, // PairwiseCover with QPBOI
    PC_Grid,
    PC_Grid_I,
    Grad
};

inline std::string ToString(OptType ot) {
    switch (ot) {
        case OptType::Fix: return "fix";
        case OptType::Fix_I: return "fix-i";
        case OptType::HOCR: return "hocr";
        case OptType::GRD: return "grd";
        case OptType::GRD_Heur: return "grd-heur";
        case OptType::PC: return "pc";
        case OptType::PC_I: return "pc-i";
        case OptType::PC_Grid: return "pc-grid";
        case OptType::PC_Grid_I: return "pc-grid-i";
        case OptType::Grad: return "grad";
        default: return "unknown";
    }
}

//
// AddVars
//
template <typename Optimizer>
void AddVars(Optimizer& opt, size_t numVars) {
    opt.AddVars(numVars);
}

template <typename REAL, int D>
void AddVars(PBF<REAL, D>& opt, size_t numVars) {
    // noop
}

template <>
void AddVars(SubmodularIBFS& opt, size_t numVars) {
    opt.AddNode(numVars);
}


template <typename Optimizer, typename Energy>
void AddConstantTerm(Optimizer& opt, Energy r) {
    // noop
}

//
// AddUnaryTerm
//
template <typename Optimizer, typename Energy>
void AddUnaryTerm(Optimizer& opt, int v, Energy coeff) {
    opt.AddUnaryTerm(v, coeff);
}

template <typename REAL, int D>
void AddUnaryTerm(PBF<REAL, D>& opt, int v, REAL coeff) {
    opt.AddUnaryTerm(v, 0, coeff);
}


//
// AddClique
//
template <typename Optimizer, typename Energy>
void AddClique(Optimizer& opt, int d, const Energy *coeffs, const int *vars) {
    std::vector<int> vec_vars(vars, vars+d);
    std::vector<Energy> vec_coeffs(coeffs, coeffs+(1 << d));
    opt.AddClique(vec_vars, vec_coeffs);
}

template <typename REAL, int D>
void AddClique(PBF<REAL, D>& opt, int d, const REAL *coeffs, const int *vars) {
    std::vector<int> vec_vars(vars, vars+d);
    std::reverse(vec_vars.begin(), vec_vars.end());
    opt.AddHigherTerm(d, vec_vars.data(), const_cast<REAL*>(coeffs));
}

template <>
void AddClique(SubmodularIBFS& opt, int d, const REAL* coeffs, const int* vars) {
    auto varVec = std::vector<int>{vars, vars+d};
    auto energyTable = std::vector<REAL>{coeffs, coeffs+(1<<d)};
    opt.AddClique(varVec, energyTable);
}

template <typename Optimizer, typename Energy>
void AddClique(Optimizer& opt, const std::vector<int>& vars, const std::vector<Energy>& energyTable) {
    AddClique(opt, vars.size(), energyTable.data(), vars.data());
}

template <typename Opt, typename QR>
void ToQuadratic(Opt& opt, QR& qr) {
    opt.ToQuadratic(qr);
}

template <typename REAL, int D, typename QR>
void ToQuadratic(PBF<REAL, D>& opt, QR& qr) {
    PBF<REAL, 2> tmp_qr;
    opt.toQuadratic(tmp_qr);
    convert(qr, tmp_qr);
}

#ifdef WITH_GRD
template <typename REAL>
void AddVars(Petter::PseudoBoolean<REAL>& opt, size_t numVars) {
    // noop
}

template <typename REAL, typename PB_REAL>
void AddUnaryTerm(Petter::PseudoBoolean<PB_REAL>& opt, int v, REAL coeff) {
    opt.add_monomial(v, coeff);
}

template <typename REAL, typename PB_REAL>
void AddClique(Petter::PseudoBoolean<PB_REAL>& opt, int d, const REAL* coeffs, const int *vars) {
    ASSERT(d <= 4 && d >= 0);
    if (d == 0)
        return;
    else if (d == 1) {
        opt.add_clique(vars[0], coeffs[0], coeffs[1]);
    } else if (d == 2) {
        opt.add_clique(vars[1], vars[0], coeffs[0], coeffs[1], coeffs[2], coeffs[3]);
    } else if (d == 3) {
        std::vector<PB_REAL> vec_coeffs(coeffs, coeffs+(1 << d));
        opt.add_clique(vars[2], vars[1], vars[0], vec_coeffs);
    } else if (d == 4) {
        std::vector<PB_REAL> vec_coeffs(coeffs, coeffs+(1 << d));
        opt.add_clique(vars[3], vars[2], vars[1], vars[0], vec_coeffs);
    }
}
#endif // !WITH_GRD

#endif
