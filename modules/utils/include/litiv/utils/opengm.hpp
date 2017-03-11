
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2016 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

#pragma once

#include "litiv/utils/platform.hpp"
#include "litiv/utils/console.hpp"
#if !HAVE_OPENGM_EXTLIB
#error "OpenGM utils require extlib."
#endif //(!HAVE_OPENGM_EXTLIB)
#if !HAVE_OPENGM_EXTLIB_QPBO
#error "OpenGM utils require QPBO in extlib."
#endif //(!HAVE_OPENGM_EXTLIB_QPBO)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-qualifiers"
#pragma clang diagnostic ignored "-Wunused-local-typedefs"
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wpedantic"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wextra"
#elif (defined(__GNUC__) || defined(__GNUG__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wextra"
#elif defined(_MSC_VER)
#pragma warning(push,0)
#endif //defined(_MSC_VER)
#ifndef SYS_MEMORYINFO_ON
#define SYS_MEMORYINFO_ON
#endif //ndef(SYS_MEMORYINFO_ON)
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/graphicalmodel/space/static_simplediscretespace.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/visitors/visitors.hxx>
#include <opengm/inference/fix-fusion/higher-order-energy.hpp>
#include <opengm/inference/external/qpbo/QPBO.h>
#if HAVE_OPENGM_EXTLIB_FASTPD
#ifndef FASTPDENERGYVALUE
#define FASTPDENERGYVALUE float
#endif //ndef(FASTPDENERGYVALUE)
#ifndef FASTPDLABELVALUE
#define FASTPDLABELVALUE uint8_t
#endif //ndef(FASTPDLABELVALUE)
#include <opengm/inference/external/fastPD.hxx>
#endif //HAVE_OPENGM_EXTLIB_FASTPD
#if defined(_MSC_VER)
#pragma warning(pop)
#elif (defined(__GNUC__) || defined(__GNUG__))
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif //defined(__clang__)

namespace lv {

    namespace gm {

        template<typename GraphModelType>
        inline void printModelInfo(const GraphModelType& oGM) {
            size_t nMinVarLabelCount = SIZE_MAX, nMaxVarLabelCount = 0;
            for(size_t v=0; v<oGM.numberOfVariables(); ++v) {
                nMinVarLabelCount = std::min(oGM.numberOfLabels(v),nMinVarLabelCount);
                nMaxVarLabelCount = std::max(oGM.numberOfLabels(v),nMaxVarLabelCount);
            }
            std::map<size_t,size_t> mFactOrderHist;
            for(size_t f=0; f<oGM.numberOfFactors(); ++f)
                mFactOrderHist[oGM.operator[](f).numberOfVariables()] += 1;
            lvAssert_(std::accumulate(mFactOrderHist.begin(),mFactOrderHist.end(),size_t(0),[](const size_t n, const auto& p){return n+p.second;})==oGM.numberOfFactors(),"factor count mismatch");
            std::cout << "Model has " << oGM.numberOfVariables() << " variables (" << ((nMinVarLabelCount==nMaxVarLabelCount)?std::to_string(nMinVarLabelCount)+" labels each)":std::to_string(nMinVarLabelCount)+" labels min, "+std::to_string(nMaxVarLabelCount)+" labels max)") << std::endl;
            std::cout << "Model has " << oGM.numberOfFactors() << " factors;" << std::endl;
            for(const auto& oOrderBin : mFactOrderHist) {
                std::cout << "\t" << oOrderBin.second << " factors w/ order=" << oOrderBin.first << std::endl;
            }
        }

    } // namespace gm

} // namespace lv
