
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

#include "litiv/datasets.hpp"
#include "litiv/imgproc.hpp"
#include "litiv/utils/ConsoleUtils.hpp"
#if USE_BSDS500_BENCHMARK
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wwrite-strings"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"
#pragma clang diagnostic ignored "-Wformat="
#pragma clang diagnostic ignored "-Wparentheses"
#pragma clang diagnostic ignored "-Wformat-security"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-pedantic-errors"
#endif //__clang__
#if (defined(__GNUC__) || defined(__GNUG__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wwrite-strings"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wformat="
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wformat-security"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-pedantic-errors"
#endif //(defined(__GNUC__) || defined(__GNUG__))
#ifdef _MSC_VER
#pragma warning(push,0)
#endif //defined(_MSC_VER)
#include "litiv/3rdparty/BSDS500/csa.hpp"
#include "litiv/3rdparty/BSDS500/kofn.hpp"
#include "litiv/3rdparty/BSDS500/match.hpp"
#ifdef _MSC_VER
#pragma warning(pop)
#endif //defined(_MSC_VER)
#if (defined(__GNUC__) || defined(__GNUG__))
#pragma GCC diagnostic pop
#endif //(defined(__GNUC__) || defined(__GNUG__))
#ifdef __clang__
#pragma clang diagnostic pop
#endif //__clang__
#endif //USE_BSDS500_BENCHMARK

namespace litiv {

    struct BSDS500Counters { // edge detection counters for a single image
        BSDS500Counters(size_t nThresholdsBins) : // always skips zero threshold
            vnIndivTP(nThresholdsBins,0),
            vnIndivTPFN(nThresholdsBins,0),
            vnTotalTP(nThresholdsBins,0),
            vnTotalTPFP(nThresholdsBins,0),
            vnThresholds(PlatformUtils::linspace<uchar>(0,UCHAR_MAX,nThresholdsBins,false)) {
            CV_Assert(nThresholdsBins>0 && nThresholdsBins<=UCHAR_MAX);
        }
        std::vector<uint64_t> vnIndivTP; // one count per threshold
        std::vector<uint64_t> vnIndivTPFN; // one count per threshold
        std::vector<uint64_t> vnTotalTP; // one count per threshold
        std::vector<uint64_t> vnTotalTPFP; // one count per threshold
        std::vector<uchar> vnThresholds; // list of thresholds
        static bool isEqual(const BSDS500Counters& a, const BSDS500Counters& b) {
            return
                (a.vnThresholds==b.vnThresholds) &&
                (a.vnIndivTP==b.vnIndivTP) &&
                (a.vnIndivTPFN==b.vnIndivTPFN) &&
                (a.vnTotalTP==b.vnTotalTP) &&
                (a.vnTotalTPFP==b.vnTotalTPFP);
        }
    };

    struct BSDS500MetricsAccumulator : IMetricsAccumulator {
        virtual bool isEqual(const std::shared_ptr<const IMetricsAccumulator>& m) const override {
            const auto& m2 = dynamic_cast<const BSDS500MetricsAccumulator&>(*m.get());
            return
                (this->m_nThresholdBins==m2.m_nThresholdBins) &&
                (this->m_voMetricsBase.size()==m2.m_voMetricsBase.size()) &&
                std::equal(this->m_voMetricsBase.begin(),this->m_voMetricsBase.end(),m2.m_voMetricsBase.begin(),&BSDS500Counters::isEqual);
        }
        virtual std::shared_ptr<IMetricsAccumulator> accumulate(const std::shared_ptr<const IMetricsAccumulator>& m) override {
            const auto& m2 = dynamic_cast<const BSDS500MetricsAccumulator&>(*m.get());
            lvAssert(this->m_nThresholdBins==m2.m_nThresholdBins);
            this->m_voMetricsBase.insert(this->m_voMetricsBase.end(),m2.m_voMetricsBase.begin(),m2.m_voMetricsBase.end());
            return shared_from_this();
        }
        virtual void accumulate(const cv::Mat& oClassif, const cv::Mat& oGT, const cv::Mat& /*oROI*/) {
            if(oGT.empty())
                return;
            CV_Assert(oClassif.type()==CV_8UC1 && oGT.type()==CV_8UC1);
            CV_Assert(oClassif.isContinuous() && oGT.isContinuous());
            CV_Assert(oClassif.cols==oGT.cols && (oGT.rows%oClassif.rows)==0 && (oGT.rows/oClassif.rows)>=1);
            CV_Assert(oClassif.step.p[0]==oGT.step.p[0]);

            const double dMaxDist = DATASETS_BSDS500_EVAL_IMAGE_DIAG_RATIO_DIST*sqrt(double(oClassif.cols*oClassif.cols+oClassif.rows*oClassif.rows));
            const double dMaxDistSqr = dMaxDist*dMaxDist;
            const int nMaxDist = (int)ceil(dMaxDist);
            CV_Assert(dMaxDist>0 && nMaxDist>0);

            BSDS500Counters oMetricsBase(m_nThresholdBins);
            const std::vector<uchar> vuEvalUniqueVals = PlatformUtils::unique<uchar>(oClassif);
            cv::Mat oCurrSegmMask(oClassif.size(),CV_8UC1), oTmpSegmMask(oClassif.size(),CV_8UC1);
            cv::Mat oSegmTPAccumulator(oClassif.size(),CV_8UC1);
            size_t nNextEvalUniqueValIdx = 0;
            size_t nThresholdBinIdx = 0;
            while(nThresholdBinIdx<oMetricsBase.vnThresholds.size()) {
                cv::compare(oClassif,oMetricsBase.vnThresholds[nThresholdBinIdx],oTmpSegmMask,cv::CMP_GE);
                litiv::thinning(oTmpSegmMask,oCurrSegmMask);

    #if USE_BSDS500_BENCHMARK

                ///////////////////////////////////////////////////////
                // code below is adapted from match.cc::matchEdgeMaps()
                ///////////////////////////////////////////////////////

                const double dOutlierCost = 100*dMaxDist;
                CV_Assert(dOutlierCost>1);
                oSegmTPAccumulator = cv::Scalar_<uchar>(0);
                cv::Mat oGTAccumulator(oCurrSegmMask.size(),CV_8UC1,cv::Scalar_<uchar>(0));
                uint64_t nIndivTP = 0;
                uint64_t nGTPosCount = 0;

                static constexpr int multiplier = 100;
                static constexpr int degree = 6;
                static_assert(degree>0,"csa config bad; degree of outlier connections should be > 0");
                static_assert(multiplier>0,"csa config bad; floating-point weights to integers should be > 0");

                for(size_t nGTMaskIdx=0; nGTMaskIdx<size_t(oGT.rows/oCurrSegmMask.rows); ++nGTMaskIdx) {
                    cv::Mat oCurrGTSegmMask = oGT(cv::Rect(0,int(oCurrSegmMask.rows*nGTMaskIdx),oCurrSegmMask.cols,oCurrSegmMask.rows));
                    cv::Mat oMatchable_SEGM(oCurrSegmMask.size(),CV_8UC1,cv::Scalar_<uchar>(0));
                    cv::Mat oMatchable_GT(oCurrSegmMask.size(),CV_8UC1,cv::Scalar_<uchar>(0));
                    // Figure out which nodes are matchable, i.e. within maxDist
                    // of another node.
                    for(int i=0; i<oCurrSegmMask.rows; ++i) {
                        for(int j=0; j<oCurrSegmMask.cols; ++j) {
                            if(!oCurrGTSegmMask.at<uchar>(i,j)) continue;
                            for(int u=-nMaxDist; u<=nMaxDist; ++u) {
                                if(i+u<0) continue;
                                if(i+u>=oCurrSegmMask.rows) continue;
                                if(double(u)>dMaxDist) continue;
                                for(int v=-nMaxDist; v<=nMaxDist; ++v) {
                                    if(j+v<0) continue;
                                    if(j+v>=oCurrSegmMask.cols) continue;
                                    if(double(v)>dMaxDist) continue;
                                    const double dCurrDistSqr = u*u+v*v;
                                    if(dCurrDistSqr>dMaxDistSqr) continue;
                                    if(oCurrSegmMask.at<uchar>(i+u,j+v)) {
                                        oMatchable_SEGM.at<uchar>(i+u,j+v) = UCHAR_MAX;
                                        oMatchable_GT.at<uchar>(i,j) = UCHAR_MAX;
                                    }
                                }
                            }
                        }
                    }

                    int nNodeCount_SEGM=0, nNodeCount_GT=0;
                    std::vector<cv::Point2i> voNodeToPxLUT_SEGM,voNodeToPxLUT_GT;
                    cv::Mat oPxToNodeLUT_SEGM(oCurrSegmMask.size(),CV_32SC1,cv::Scalar_<int>(-1));
                    cv::Mat oPxToNodeLUT_GT(oCurrSegmMask.size(),CV_32SC1,cv::Scalar_<int>(-1));
                    // Count the number of nodes on each side of the match.
                    // Construct nodeID->pixel and pixel->nodeID maps.
                    // Node IDs range from [0,nNodeCount_SEGM) and [0,nNodeCount_GT).
                    for(int i=0; i<oCurrSegmMask.rows; ++i) {
                        for(int j=0; j<oCurrSegmMask.cols; ++j) {
                            cv::Point2i px(j,i);
                            if(oMatchable_SEGM.at<uchar>(px)) {
                                oPxToNodeLUT_SEGM.at<int>(px) = nNodeCount_SEGM;
                                voNodeToPxLUT_SEGM.push_back(px);
                                ++nNodeCount_SEGM;
                            }
                            if(oMatchable_GT.at<uchar>(px)) {
                                oPxToNodeLUT_GT.at<int>(px) = nNodeCount_GT;
                                voNodeToPxLUT_GT.push_back(px);
                                ++nNodeCount_GT;
                            }
                        }
                    }

                    struct Edge {
                        int nNodeIdx_SEGM;
                        int nNodeIdx_GT;
                        double dEdgeDist;
                    };
                    std::vector<Edge> voEdges;
                    // Construct the list of edges between pixels within maxDist.
                    for(int i=0; i<oCurrSegmMask.rows; ++i) {
                        for(int j=0; j<oCurrSegmMask.cols; ++j) {
                            if(!oMatchable_GT.at<uchar>(i,j)) continue;
                            for(int u=-nMaxDist; u<=nMaxDist; ++u) {
                                if(i+u<0) continue;
                                if(i+u>=oCurrSegmMask.rows) continue;
                                if(double(u)>dMaxDist) continue;
                                for(int v=-nMaxDist; v<=nMaxDist; ++v) {
                                    if(j+v<0) continue;
                                    if(j+v>=oCurrSegmMask.cols) continue;
                                    if(double(v)>dMaxDist) continue;
                                    if(!oMatchable_SEGM.at<uchar>(i+u,j+v)) continue;
                                    const double dCurrDistSqr = u*u+v*v;
                                    if(dCurrDistSqr>dMaxDistSqr) continue;
                                    Edge e;
                                    e.nNodeIdx_SEGM = oPxToNodeLUT_SEGM.at<int>(i+u,j+v);
                                    e.nNodeIdx_GT = oPxToNodeLUT_GT.at<int>(i,j);
                                    e.dEdgeDist = sqrt(dCurrDistSqr);
                                    CV_DbgAssert(e.nNodeIdx_SEGM>=0 && e.nNodeIdx_SEGM<nNodeCount_SEGM);
                                    CV_DbgAssert(e.nNodeIdx_GT>=0 && e.nNodeIdx_GT<nNodeCount_GT);
                                    voEdges.push_back(e);
                                }
                            }
                        }
                    }

                    // The cardinality of the match is n.
                    const int n = nNodeCount_SEGM+nNodeCount_GT;
                    const int nmin = std::min(nNodeCount_SEGM,nNodeCount_GT);
                    const int nmax = std::max(nNodeCount_SEGM,nNodeCount_GT);

                    // Compute the degree of various outlier connections.
                    const int degree_SEGM = std::max(0,std::min(degree,nNodeCount_SEGM-1)); // from map1
                    const int degree_GT = std::max(0,std::min(degree,nNodeCount_GT-1)); // from map2
                    const int degree_mix = std::min(degree,std::min(nNodeCount_SEGM,nNodeCount_GT)); // between outliers
                    const int dmax = std::max(degree_SEGM,std::max(degree_GT,degree_mix));

                    CV_DbgAssert(nNodeCount_SEGM==0 || (degree_SEGM>=0 && degree_SEGM<nNodeCount_SEGM));
                    CV_DbgAssert(nNodeCount_GT==0 || (degree_GT>=0 && degree_GT<nNodeCount_GT));
                    CV_DbgAssert(degree_mix>=0 && degree_mix<=nmin);

                    // Count the number of edges.
                    int m = 0;
                    m += (int)voEdges.size();         // real connections
                    m += degree_SEGM*nNodeCount_SEGM; // outlier connections
                    m += degree_GT*nNodeCount_GT;     // outlier connections
                    m += degree_mix*nmax;             // outlier-outlier connections
                    m += n;                           // high-cost perfect match overlay
                                                      // If the graph is empty, then there's nothing to do.
                    if(m>0) {
                        // Weight of outlier connections.
                        const int nOutlierWeight = (int)ceil(dOutlierCost*multiplier);
                        // Scratch array for outlier edges.
                        std::vector<int> vnOutliers(dmax);
                        // Construct the input graph for the assignment problem.
                        cv::Mat oGraph(m,3,CV_32SC1);
                        int nGraphIdx = 0;
                        // real edges
                        for(int a=0; a<(int)voEdges.size(); ++a) {
                            int nNodeIdx_SEGM = voEdges[a].nNodeIdx_SEGM;
                            int nNodeIdx_GT = voEdges[a].nNodeIdx_GT;
                            CV_DbgAssert(nNodeIdx_SEGM>=0 && nNodeIdx_SEGM<nNodeCount_SEGM);
                            CV_DbgAssert(nNodeIdx_GT>=0 && nNodeIdx_GT<nNodeCount_GT);
                            oGraph.at<int>(nGraphIdx,0) = nNodeIdx_SEGM;
                            oGraph.at<int>(nGraphIdx,1) = nNodeIdx_GT;
                            oGraph.at<int>(nGraphIdx,2) = (int)rint(voEdges[a].dEdgeDist*multiplier);
                            nGraphIdx++;
                        }
                        // outliers edges for map1, exclude diagonal
                        for(int nNodeIdx_SEGM=0; nNodeIdx_SEGM<nNodeCount_SEGM; ++nNodeIdx_SEGM) {
                            BSDS500::kOfN(degree_SEGM,nNodeCount_SEGM-1,vnOutliers.data());
                            for(int a=0; a<degree_SEGM; a++) {
                                int j = vnOutliers[a];
                                if(j>=nNodeIdx_SEGM) {j++;}
                                CV_DbgAssert(nNodeIdx_SEGM!=j);
                                CV_DbgAssert(j>=0 && j<nNodeCount_SEGM);
                                oGraph.at<int>(nGraphIdx,0) = nNodeIdx_SEGM;
                                oGraph.at<int>(nGraphIdx,1) = nNodeCount_GT+j;
                                oGraph.at<int>(nGraphIdx,2) = nOutlierWeight;
                                nGraphIdx++;
                            }
                        }
                        // outliers edges for map2, exclude diagonal
                        for(int nNodeIdx_GT = 0; nNodeIdx_GT<nNodeCount_GT; nNodeIdx_GT++) {
                            BSDS500::kOfN(degree_GT,nNodeCount_GT-1,vnOutliers.data());
                            for(int a = 0; a<degree_GT; a++) {
                                int i = vnOutliers[a];
                                if(i>=nNodeIdx_GT) {i++;}
                                CV_DbgAssert(i!=nNodeIdx_GT);
                                CV_DbgAssert(i>=0 && i<nNodeCount_GT);
                                oGraph.at<int>(nGraphIdx,0) = nNodeCount_SEGM+i;
                                oGraph.at<int>(nGraphIdx,1) = nNodeIdx_GT;
                                oGraph.at<int>(nGraphIdx,2) = nOutlierWeight;
                                nGraphIdx++;
                            }
                        }
                        // outlier-to-outlier edges
                        for(int i = 0; i<nmax; i++) {
                            BSDS500::kOfN(degree_mix,nmin,vnOutliers.data());
                            for(int a = 0; a<degree_mix; a++) {
                                const int j = vnOutliers[a];
                                CV_DbgAssert(j>=0 && j<nmin);
                                if(nNodeCount_SEGM<nNodeCount_GT) {
                                    CV_DbgAssert(i>=0 && i<nNodeCount_GT);
                                    CV_DbgAssert(j>=0 && j<nNodeCount_SEGM);
                                    oGraph.at<int>(nGraphIdx,0) = nNodeCount_SEGM+i;
                                    oGraph.at<int>(nGraphIdx,1) = nNodeCount_GT+j;
                                }
                                else {
                                    CV_DbgAssert(i>=0 && i<nNodeCount_SEGM);
                                    CV_DbgAssert(j>=0 && j<nNodeCount_GT);
                                    oGraph.at<int>(nGraphIdx,0) = nNodeCount_SEGM+j;
                                    oGraph.at<int>(nGraphIdx,1) = nNodeCount_GT+i;
                                }
                                oGraph.at<int>(nGraphIdx,2) = nOutlierWeight;
                                nGraphIdx++;
                            }
                        }
                        // perfect match overlay (diagonal)
                        for(int i = 0; i<nNodeCount_SEGM; i++) {
                            oGraph.at<int>(nGraphIdx,0) = i;
                            oGraph.at<int>(nGraphIdx,1) = nNodeCount_GT+i;
                            oGraph.at<int>(nGraphIdx,2) = nOutlierWeight*multiplier;
                            nGraphIdx++;
                        }
                        for(int i = 0; i<nNodeCount_GT; i++) {
                            oGraph.at<int>(nGraphIdx,0) = nNodeCount_SEGM+i;
                            oGraph.at<int>(nGraphIdx,1) = i;
                            oGraph.at<int>(nGraphIdx,2) = nOutlierWeight*multiplier;
                            nGraphIdx++;
                        }
                        CV_DbgAssert(nGraphIdx==m);

                        // Check all the edges, and set the values up for CSA.
                        for(int i = 0; i<m; i++) {
                            CV_DbgAssert(oGraph.at<int>(i,0)>=0 && oGraph.at<int>(i,0)<n);
                            CV_DbgAssert(oGraph.at<int>(i,1)>=0 && oGraph.at<int>(i,1)<n);
                            oGraph.at<int>(i,0) += 1;
                            oGraph.at<int>(i,1) += 1+n;
                        }

                        // Solve the assignment problem.
                        BSDS500::CSA oCSASolver(2*n,m,(int*)oGraph.data);
                        CV_Assert(oCSASolver.edges()==n);

                        cv::Mat oOutGraph(n,3,CV_32SC1);
                        for(int i = 0; i<n; i++) {
                            int a,b,c;
                            oCSASolver.edge(i,a,b,c);
                            oOutGraph.at<int>(i,0) = a-1;
                            oOutGraph.at<int>(i,1) = b-1-n;
                            oOutGraph.at<int>(i,2) = c;
                        }

                        // Check the solution.
                        // Count the number of high-cost edges from the perfect match
                        // overlay that were used in the match.
                        int nOverlayCount = 0;
                        for(int a = 0; a<n; a++) {
                            const int i = oOutGraph.at<int>(a,0);
                            const int j = oOutGraph.at<int>(a,1);
                            const int c = oOutGraph.at<int>(a,2);
                            CV_DbgAssert(i>=0 && i<n);
                            CV_DbgAssert(j>=0 && j<n);
                            CV_DbgAssert(c>=0);
                            // edge from high-cost perfect match overlay
                            if(c==nOutlierWeight*multiplier) {nOverlayCount++;}
                            // skip outlier edges
                            if(i>=nNodeCount_SEGM) {continue;}
                            if(j>=nNodeCount_GT) {continue;}
                            // for edges between real nodes, check the edge weight
                            CV_DbgAssert((int)rint(sqrt((voNodeToPxLUT_SEGM[i].x-voNodeToPxLUT_GT[j].x)*(voNodeToPxLUT_SEGM[i].x-voNodeToPxLUT_GT[j].x)+(voNodeToPxLUT_SEGM[i].y-voNodeToPxLUT_GT[j].y)*(voNodeToPxLUT_SEGM[i].y-voNodeToPxLUT_GT[j].y))*multiplier)==c);
                        }

                        // Print a warning if any of the edges from the perfect match overlay
                        // were used.  This should happen rarely.  If it happens frequently,
                        // then the outlier connectivity should be increased.
                        if(nOverlayCount>5) {
                            fprintf(stderr,"%s:%d: WARNING: The match includes %d outlier(s) from the perfect match overlay.\n",__FILE__,__LINE__,nOverlayCount);
                        }

                        // Compute match arrays.
                        for(int a = 0; a<n; a++) {
                            // node ids
                            const int i = oOutGraph.at<int>(a,0);
                            const int j = oOutGraph.at<int>(a,1);
                            // skip outlier edges
                            if(i>=nNodeCount_SEGM) {continue;}
                            if(j>=nNodeCount_GT) {continue;}
                            // for edges between real nodes, check the edge weight
                            const cv::Point2i oPx_SEGM = voNodeToPxLUT_SEGM[i];
                            const cv::Point2i oPx_GT = voNodeToPxLUT_GT[j];
                            // record edges
                            CV_Assert(oCurrSegmMask.at<uchar>(oPx_SEGM) && oCurrGTSegmMask.at<uchar>(oPx_GT));
                            oSegmTPAccumulator.at<uchar>(oPx_SEGM) = UCHAR_MAX;
                            ++nIndivTP;
                        }
                    }
                    nGTPosCount += cv::countNonZero(oCurrGTSegmMask);
                    oGTAccumulator |= oCurrGTSegmMask;
                }

    #else //(!USE_BSDS500_BENCHMARK)

                oSegmTPAccumulator = cv::Scalar_<uchar>(0); // accP |= ...
                uint64_t nIndivTP = 0; // cntR += ...
                uint64_t nGTPosCount = 0; // sumR += ...
                for(size_t nGTMaskIdx = 0; nGTMaskIdx<size_t(oGT.rows/oCurrSegmMask.rows); ++nGTMaskIdx) {
                    cv::Mat oCurrGTSegmMask = oGT(cv::Rect(0,int(oCurrSegmMask.rows*nGTMaskIdx),oCurrSegmMask.cols,oCurrSegmMask.rows));
                    for(int i = 0; i<oCurrSegmMask.rows; ++i) {
                        for(int j = 0; j<oCurrSegmMask.cols; ++j) {
                            if(!oCurrGTSegmMask.at<uchar>(i,j)) continue;
                            ++nGTPosCount;
                            bool bFoundMatch = false;
                            for(int u = -nMaxDist; u<=nMaxDist && !bFoundMatch; ++u) {
                                if(i+u<0) continue;
                                if(i+u>=oCurrSegmMask.rows) continue;
                                if(double(u)>dMaxDist) continue;
                                for(int v = -nMaxDist; v<=nMaxDist && !bFoundMatch; ++v) {
                                    if(j+v<0) continue;
                                    if(j+v>=oCurrSegmMask.cols) continue;
                                    if(double(v)>dMaxDist) continue;
                                    const double dCurrDistSqr = u*u+v*v;
                                    if(dCurrDistSqr>dMaxDistSqr) continue;
                                    if(oCurrSegmMask.at<uchar>(i+u,j+v)) {
                                        ++nIndivTP;
                                        oSegmTPAccumulator.at<uchar>(i+u,j+v) = UCHAR_MAX;
                                        bFoundMatch = true;
                                    }
                                }
                            }
                        }
                    }
                }

    #endif //(!USE_BSDS500_BENCHMARK)

                //re = TP / (TP + FN)
                CV_Assert(nGTPosCount>=nIndivTP);
                oMetricsBase.vnIndivTP[nThresholdBinIdx] = nIndivTP;
                oMetricsBase.vnIndivTPFN[nThresholdBinIdx] = nGTPosCount;

                //pr = TP / (TP + FP)
                uint64_t nSegmTPAccCount = uint64_t(cv::countNonZero(oSegmTPAccumulator));
                uint64_t nSegmPosCount = uint64_t(cv::countNonZero(oCurrSegmMask));
                CV_Assert(nSegmPosCount>=nSegmTPAccCount);
                oMetricsBase.vnTotalTP[nThresholdBinIdx] = nSegmTPAccCount;
                oMetricsBase.vnTotalTPFP[nThresholdBinIdx] = nSegmPosCount;
                while(nNextEvalUniqueValIdx+1<vuEvalUniqueVals.size() && vuEvalUniqueVals[nNextEvalUniqueValIdx]<=oMetricsBase.vnThresholds[nThresholdBinIdx])
                    ++nNextEvalUniqueValIdx;
                while(++nThresholdBinIdx<oMetricsBase.vnThresholds.size() && oMetricsBase.vnThresholds[nThresholdBinIdx]<=vuEvalUniqueVals[nNextEvalUniqueValIdx]) {
                    oMetricsBase.vnIndivTP[nThresholdBinIdx] = oMetricsBase.vnIndivTP[nThresholdBinIdx-1];
                    oMetricsBase.vnIndivTPFN[nThresholdBinIdx] = oMetricsBase.vnIndivTPFN[nThresholdBinIdx-1];
                    oMetricsBase.vnTotalTP[nThresholdBinIdx] = oMetricsBase.vnTotalTP[nThresholdBinIdx-1];
                    oMetricsBase.vnTotalTPFP[nThresholdBinIdx] = oMetricsBase.vnTotalTPFP[nThresholdBinIdx-1];
                }

                const float fCompltRatio = float(nThresholdBinIdx)/oMetricsBase.vnThresholds.size();
                litiv::updateConsoleProgressBar("BSDS500 eval:",fCompltRatio);
            }
            litiv::cleanConsoleRow();
            m_voMetricsBase.push_back(oMetricsBase);
        }
        static cv::Mat getColoredMask(const cv::Mat& oClassif, const cv::Mat& oGT, const cv::Mat& /*oROI*/) {
            if(oGT.empty()) {
                CV_Assert(!oClassif.empty() && oClassif.type()==CV_8UC1);
                cv::Mat oResult;
                cv::cvtColor(oClassif,oResult,cv::COLOR_GRAY2BGR);
                return oResult;
            }
            CV_Assert(oClassif.type()==CV_8UC1 && oGT.type()==CV_8UC1);
            CV_Assert(oClassif.cols==oGT.cols && (oGT.rows%oClassif.rows)==0 && (oGT.rows/oClassif.rows)>=1);
            CV_Assert(oClassif.step.p[0]==oGT.step.p[0]);
            const double dMaxDist = DATASETS_BSDS500_EVAL_IMAGE_DIAG_RATIO_DIST*sqrt(double(oClassif.cols*oClassif.cols+oClassif.rows*oClassif.rows));
            const int nMaxDist = (int)ceil(dMaxDist);
            CV_Assert(dMaxDist>0 && nMaxDist>0);
            cv::Mat oSegm_TP(oClassif.size(),CV_16UC1,cv::Scalar_<ushort>(0));
            cv::Mat oSegm_FN(oClassif.size(),CV_16UC1,cv::Scalar_<ushort>(0));
            cv::Mat oSegm_FP(oClassif.size(),CV_16UC1,cv::Scalar_<ushort>(0));
            const size_t nGTMaskCount = size_t(oGT.rows/oClassif.rows);
            for(size_t nGTMaskIdx=0; nGTMaskIdx<nGTMaskCount; ++nGTMaskIdx) {
                cv::Mat oCurrGTSegmMask = oGT(cv::Rect(0,int(oClassif.rows*nGTMaskIdx),oClassif.cols,oClassif.rows));
                cv::Mat oCurrGTSegmMask_dilated,oSegm_dilated;
                cv::Mat oDilateKernel(2*nMaxDist+1,2*nMaxDist+1,CV_8UC1,cv::Scalar_<uchar>(255));
                cv::dilate(oCurrGTSegmMask,oCurrGTSegmMask_dilated,oDilateKernel);
                cv::dilate(oClassif,oSegm_dilated,oDilateKernel);
                cv::add((oClassif&oCurrGTSegmMask_dilated),oSegm_TP,oSegm_TP,cv::noArray(),CV_16U);
                cv::add((oClassif&(oCurrGTSegmMask_dilated==0)),oSegm_FP,oSegm_FP,cv::noArray(),CV_16U);
                cv::add(((oSegm_dilated==0)&oCurrGTSegmMask),oSegm_FN,oSegm_FN,cv::noArray(),CV_16U);
            }
            cv::Mat oSegm_TP_byte, oSegm_FN_byte, oSegm_FP_byte;
            oSegm_TP.convertTo(oSegm_TP_byte,CV_8U,1.0/nGTMaskCount);
            oSegm_FN.convertTo(oSegm_FN_byte,CV_8U,1.0/nGTMaskCount);
            oSegm_FP.convertTo(oSegm_FP_byte,CV_8U,1.0/nGTMaskCount);
            cv::Mat oResult(oClassif.size(),CV_8UC3,cv::Scalar_<uchar>(0));
            const std::vector<int> vnMixPairs = {0,2, 1,0, 2,1};
            cv::mixChannels(std::vector<cv::Mat>{oSegm_FN_byte|oSegm_FP_byte,oSegm_FN_byte,oSegm_TP_byte},std::vector<cv::Mat>{oResult},vnMixPairs.data(),vnMixPairs.size()/2);
            return oResult;
        }
        static std::shared_ptr<BSDS500MetricsAccumulator> create(size_t nThresholdsBins) {
            struct MetricsAccumulatorWrapper : BSDS500MetricsAccumulator {
                MetricsAccumulatorWrapper(size_t nThresholdsBins) : BSDS500MetricsAccumulator(nThresholdsBins) {} // cant do 'using BaseCstr::BaseCstr;' since it keeps the access level
            };
            return std::make_shared<MetricsAccumulatorWrapper>(nThresholdsBins);
        }
        std::vector<BSDS500Counters> m_voMetricsBase; // one counter block per image
        const size_t m_nThresholdBins;
    protected:
        BSDS500MetricsAccumulator(size_t nThresholdBins) : m_nThresholdBins(nThresholdBins) {CV_Assert(m_nThresholdBins>0 && m_nThresholdBins<=UCHAR_MAX);}
    };
    using BSDS500MetricsAccumulatorPtr = std::shared_ptr<BSDS500MetricsAccumulator>;
    using BSDS500MetricsAccumulatorConstPtr = std::shared_ptr<const BSDS500MetricsAccumulator>;

    struct BSDS500Score { // edge detection score for a single threshold
        double dThreshold;
        double dRecall;
        double dPrecision;
        double dFMeasure;
    };

    inline BSDS500Score FindMaxFMeasure(const std::vector<uchar>& vnThresholds, const std::vector<double>& vdRecall, const std::vector<double>& vdPrecision) {
        CV_Assert(!vnThresholds.empty() && !vdRecall.empty() && !vdPrecision.empty());
        CV_Assert(vnThresholds.size()==vdRecall.size() && vdRecall.size()==vdPrecision.size());
        BSDS500Score oRes;
        oRes.dFMeasure = litiv::BinClassifMetricsCalculator::CalcFMeasure(vdRecall[0],vdPrecision[0]);
        oRes.dPrecision = vdPrecision[0];
        oRes.dRecall = vdRecall[0];
        oRes.dThreshold = double(vnThresholds[0])/UCHAR_MAX;
        for(size_t nThresholdIdx=1; nThresholdIdx<vnThresholds.size(); ++nThresholdIdx) {
            const size_t nInterpCount = 100;
            for(size_t nInterpIdx=0; nInterpIdx<=nInterpCount; ++nInterpIdx) {
                const double dLastInterp = double(nInterpCount-nInterpIdx)/nInterpCount;
                const double dCurrInterp = double(nInterpIdx)/nInterpCount;
                const double dInterpRecall = dLastInterp*vdRecall[nThresholdIdx-1] + dCurrInterp*vdRecall[nThresholdIdx];
                const double dInterpPrecision = dLastInterp*vdPrecision[nThresholdIdx-1] + dCurrInterp*vdPrecision[nThresholdIdx];
                const double dInterpFMeasure = litiv::BinClassifMetricsCalculator::CalcFMeasure(dInterpRecall,dInterpPrecision);
                if(dInterpFMeasure>oRes.dFMeasure) {
                    oRes.dThreshold = (dLastInterp*vnThresholds[nThresholdIdx-1] + dCurrInterp*vnThresholds[nThresholdIdx])/UCHAR_MAX;
                    oRes.dFMeasure = dInterpFMeasure;
                    oRes.dPrecision = dInterpPrecision;
                    oRes.dRecall = dInterpRecall;
                }
            }
        }
        return oRes;
    }

    inline BSDS500Score FindMaxFMeasure(const std::vector<BSDS500Score>& voScores) {
        CV_Assert(!voScores.empty());
        BSDS500Score oRes = voScores[0];
        for(size_t nScoreIdx=1; nScoreIdx<voScores.size(); ++nScoreIdx) {
            const size_t nInterpCount = 100;
            for(size_t nInterpIdx=0; nInterpIdx<=nInterpCount; ++nInterpIdx) {
                const double dLastInterp = double(nInterpCount-nInterpIdx)/nInterpCount;
                const double dCurrInterp = double(nInterpIdx)/nInterpCount;
                const double dInterpRecall = dLastInterp*voScores[nScoreIdx-1].dRecall + dCurrInterp*voScores[nScoreIdx].dRecall;
                const double dInterpPrecision = dLastInterp*voScores[nScoreIdx-1].dPrecision + dCurrInterp*voScores[nScoreIdx].dPrecision;
                const double dInterpFMeasure = litiv::BinClassifMetricsCalculator::CalcFMeasure(dInterpRecall,dInterpPrecision);
                if(dInterpFMeasure>oRes.dFMeasure) {
                    oRes.dThreshold = dLastInterp*voScores[nScoreIdx-1].dThreshold + dCurrInterp*voScores[nScoreIdx].dThreshold;
                    oRes.dFMeasure = dInterpFMeasure;
                    oRes.dPrecision = dInterpPrecision;
                    oRes.dRecall = dInterpRecall;
                }
            }
        }
        return oRes;
    }

    struct BSDS500MetricsCalculator : IMetricsCalculator {
        virtual IMetricsCalculatorPtr accumulate(const IMetricsCalculatorConstPtr& m) override {
            // this was not defined in the BSDS500 eval scripts (scores were only given per image set)
            const auto& m2 = dynamic_cast<const BSDS500MetricsCalculator&>(*m.get());
            lvAssert(this->m_nThresholdBins==m2.m_nThresholdBins);
            // note: weights are not used here (we simply add all image counters, and update scores)
            this->m_voMetricsBase.insert(this->m_voMetricsBase.begin(),m2.m_voMetricsBase.begin(),m2.m_voMetricsBase.end());
            updateScores();
            return shared_from_this();
        }
        static std::shared_ptr<BSDS500MetricsCalculator> create(const IMetricsAccumulatorConstPtr& m) {
            lvAssert(m.get());
            const auto& m2 = std::dynamic_pointer_cast<const BSDS500MetricsAccumulator>(m);
            lvAssert(m2.get());
            const BSDS500MetricsAccumulator& m3 = *m2.get();
            struct MetricsCalculatorWrapper : public BSDS500MetricsCalculator {
                MetricsCalculatorWrapper(const BSDS500MetricsAccumulator& m) : BSDS500MetricsCalculator(m) {} // cant do 'using BaseCstr::BaseCstr;' since it keeps the access level
            };
            return std::make_shared<MetricsCalculatorWrapper>(m3);
        }
        // high-level metrics for an entire image set
        std::vector<BSDS500Score> voBestImageScores; // one score per image (best threshold)
        std::vector<BSDS500Score> voThresholdScores; // one score per threshold (cumul images)
        BSDS500Score oBestScore; // best score for all thresholds
        double dMaxRecall;
        double dMaxPrecision;
        double dMaxFMeasure;
        double dAreaPR;
    protected:
        std::vector<BSDS500Counters> m_voMetricsBase; // one counter block per image (used for image set accumulation only)
        const size_t m_nThresholdBins;
        void updateScores() {
            BSDS500Counters oCumulMetricsBase(m_nThresholdBins);
            BSDS500Counters oMaxBinClassifMetricsAccumulator(1);
            const size_t nImageCount = m_voMetricsBase.size();
            voBestImageScores.resize(nImageCount);
            for(size_t nImageIdx = 0; nImageIdx<nImageCount; ++nImageIdx) {
                CV_DbgAssert(!m_voMetricsBase[nImageIdx].vnIndivTP.empty() && !m_voMetricsBase[nImageIdx].vnIndivTPFN.empty());
                CV_DbgAssert(!m_voMetricsBase[nImageIdx].vnTotalTP.empty() && !m_voMetricsBase[nImageIdx].vnTotalTPFP.empty());
                CV_DbgAssert(m_voMetricsBase[nImageIdx].vnIndivTP.size()==m_voMetricsBase[nImageIdx].vnIndivTPFN.size());
                CV_DbgAssert(m_voMetricsBase[nImageIdx].vnTotalTP.size()==m_voMetricsBase[nImageIdx].vnTotalTPFP.size());
                CV_DbgAssert(m_voMetricsBase[nImageIdx].vnIndivTP.size()==m_voMetricsBase[nImageIdx].vnTotalTP.size());
                CV_DbgAssert(m_voMetricsBase[nImageIdx].vnThresholds.size()==m_voMetricsBase[nImageIdx].vnTotalTP.size());
                CV_DbgAssert(nImageIdx==0 || m_voMetricsBase[nImageIdx].vnIndivTP.size()==m_voMetricsBase[nImageIdx-1].vnIndivTP.size());
                CV_DbgAssert(nImageIdx==0 || m_voMetricsBase[nImageIdx].vnThresholds==m_voMetricsBase[nImageIdx-1].vnThresholds);
                std::vector<BSDS500Score> voImageScore_PerThreshold(m_nThresholdBins);
                for(size_t nThresholdIdx = 0; nThresholdIdx<m_nThresholdBins; ++nThresholdIdx) {
                    voImageScore_PerThreshold[nThresholdIdx].dRecall = litiv::BinClassifMetricsCalculator::CalcRecall(m_voMetricsBase[nImageIdx].vnIndivTP[nThresholdIdx],m_voMetricsBase[nImageIdx].vnIndivTPFN[nThresholdIdx]);
                    voImageScore_PerThreshold[nThresholdIdx].dPrecision = litiv::BinClassifMetricsCalculator::CalcPrecision(m_voMetricsBase[nImageIdx].vnTotalTP[nThresholdIdx],m_voMetricsBase[nImageIdx].vnTotalTPFP[nThresholdIdx]);
                    voImageScore_PerThreshold[nThresholdIdx].dFMeasure = litiv::BinClassifMetricsCalculator::CalcFMeasure(voImageScore_PerThreshold[nThresholdIdx].dRecall,voImageScore_PerThreshold[nThresholdIdx].dPrecision);
                    voImageScore_PerThreshold[nThresholdIdx].dThreshold = double(m_voMetricsBase[nImageIdx].vnThresholds[nThresholdIdx])/UCHAR_MAX;
                    oCumulMetricsBase.vnIndivTP[nThresholdIdx] += m_voMetricsBase[nImageIdx].vnIndivTP[nThresholdIdx];
                    oCumulMetricsBase.vnIndivTPFN[nThresholdIdx] += m_voMetricsBase[nImageIdx].vnIndivTPFN[nThresholdIdx];
                    oCumulMetricsBase.vnTotalTP[nThresholdIdx] += m_voMetricsBase[nImageIdx].vnTotalTP[nThresholdIdx];
                    oCumulMetricsBase.vnTotalTPFP[nThresholdIdx] += m_voMetricsBase[nImageIdx].vnTotalTPFP[nThresholdIdx];
                }
                voBestImageScores[nImageIdx] = FindMaxFMeasure(voImageScore_PerThreshold);
                size_t nMaxFMeasureIdx = (size_t)std::distance(voImageScore_PerThreshold.begin(),std::max_element(voImageScore_PerThreshold.begin(),voImageScore_PerThreshold.end(),[](const BSDS500Score& n1, const BSDS500Score& n2){
                    return n1.dFMeasure<n2.dFMeasure;
                }));
                oMaxBinClassifMetricsAccumulator.vnIndivTP[0] += m_voMetricsBase[nImageIdx].vnIndivTP[nMaxFMeasureIdx];
                oMaxBinClassifMetricsAccumulator.vnIndivTPFN[0] += m_voMetricsBase[nImageIdx].vnIndivTPFN[nMaxFMeasureIdx];
                oMaxBinClassifMetricsAccumulator.vnTotalTP[0] += m_voMetricsBase[nImageIdx].vnTotalTP[nMaxFMeasureIdx];
                oMaxBinClassifMetricsAccumulator.vnTotalTPFP[0] += m_voMetricsBase[nImageIdx].vnTotalTPFP[nMaxFMeasureIdx];
            }
            // ^^^ voBestImageScores => eval_bdry_img.txt
            voThresholdScores.resize(m_nThresholdBins);
            for(size_t nThresholdIdx = 0; nThresholdIdx<oCumulMetricsBase.vnThresholds.size(); ++nThresholdIdx) {
                voThresholdScores[nThresholdIdx].dRecall = litiv::BinClassifMetricsCalculator::CalcRecall(oCumulMetricsBase.vnIndivTP[nThresholdIdx],oCumulMetricsBase.vnIndivTPFN[nThresholdIdx]);
                voThresholdScores[nThresholdIdx].dPrecision = litiv::BinClassifMetricsCalculator::CalcPrecision(oCumulMetricsBase.vnTotalTP[nThresholdIdx],oCumulMetricsBase.vnTotalTPFP[nThresholdIdx]);
                voThresholdScores[nThresholdIdx].dFMeasure = litiv::BinClassifMetricsCalculator::CalcFMeasure(voThresholdScores[nThresholdIdx].dRecall,voThresholdScores[nThresholdIdx].dPrecision);
                voThresholdScores[nThresholdIdx].dThreshold = double(oCumulMetricsBase.vnThresholds[nThresholdIdx])/UCHAR_MAX;
            }
            // ^^^ voThresholdScores => eval_bdry_thr.txt
            oBestScore = FindMaxFMeasure(voThresholdScores);
            dMaxRecall = litiv::BinClassifMetricsCalculator::CalcRecall(oMaxBinClassifMetricsAccumulator.vnIndivTP[0],oMaxBinClassifMetricsAccumulator.vnIndivTPFN[0]);
            dMaxPrecision = litiv::BinClassifMetricsCalculator::CalcPrecision(oMaxBinClassifMetricsAccumulator.vnTotalTP[0],oMaxBinClassifMetricsAccumulator.vnTotalTPFP[0]);
            dMaxFMeasure = litiv::BinClassifMetricsCalculator::CalcFMeasure(dMaxRecall,dMaxPrecision);
            dAreaPR = 0;
            std::vector<size_t> vnCumulRecallIdx_uniques = PlatformUtils::unique_indexes(voThresholdScores,[&](size_t n1, size_t n2) {
                    return voThresholdScores[n1].dRecall<voThresholdScores[n2].dRecall;
                },[&](size_t n1, size_t n2) {
                    return voThresholdScores[n1].dRecall==voThresholdScores[n2].dRecall;
            });
            if(vnCumulRecallIdx_uniques.size()>1) {
                std::vector<double> vdCumulRecall_uniques(vnCumulRecallIdx_uniques.size());
                std::vector<double> vdCumulPrecision_uniques(vnCumulRecallIdx_uniques.size());
                for(size_t n = 0; n<vnCumulRecallIdx_uniques.size(); ++n) {
                    vdCumulRecall_uniques[n] = voThresholdScores[vnCumulRecallIdx_uniques[n]].dRecall;
                    vdCumulPrecision_uniques[n] = voThresholdScores[vnCumulRecallIdx_uniques[n]].dPrecision;
                }
                const size_t nInterpReqIdxCount = 100;
                std::vector<double> vdInterpReqIdx(nInterpReqIdxCount+1);
                for(size_t n = 0; n<=nInterpReqIdxCount; ++n)
                    vdInterpReqIdx[n] = double(n)/nInterpReqIdxCount;
                std::vector<double> vdInterpVals = PlatformUtils::interp1(vdCumulRecall_uniques,vdCumulPrecision_uniques,vdInterpReqIdx);
                if(!vdInterpVals.empty())
                    for(size_t n = 0; n<=vdInterpVals.size(); ++n)
                        dAreaPR += vdInterpVals[n]*0.01;
            }
            // ^^^ oCumulScore,dMaxRecall,dMaxPrecision,dMaxFMeasure,dAreaPR => eval_bdry.txt
        }
        //! default contructor requires a base metrics counters, as otherwise, we may obtain NaN's
        BSDS500MetricsCalculator(const BSDS500MetricsAccumulator& m) :
                m_voMetricsBase(m.m_voMetricsBase), m_nThresholdBins(m.m_nThresholdBins) {
            updateScores();
        }
    };
    using BSDS500MetricsCalculatorPtr = std::shared_ptr<BSDS500MetricsCalculator>;
    using BSDS500MetricsCalculatorConstPtr = std::shared_ptr<const BSDS500MetricsCalculator>;

} //namespace litiv

void litiv::DatasetEvaluator_<litiv::eDatasetEval_BinaryClassifier,litiv::eDataset_BSDS500>::writeEvalReport() const {
    if(getBatches(false).empty() || !isUsingEvaluator()) {
        IDatasetEvaluator_<litiv::eDatasetEval_None>::writeEvalReport();
        return;
    }
    for(const auto& pGroupIter : getBatches(true))
        pGroupIter->shared_from_this_cast<const DataReporter_<eDatasetEval_BinaryClassifier,eDataset_BSDS500>>(true)->DataReporter_<eDatasetEval_BinaryClassifier,eDataset_BSDS500>::writeEvalReport();
    IMetricsCalculatorConstPtr pMetrics = getMetrics();
    lvAssert(pMetrics.get());
    const BSDS500MetricsCalculator& oMetrics = dynamic_cast<const BSDS500MetricsCalculator&>(*pMetrics.get());
    std::cout << CxxUtils::clampString(getName(),12) << " => MaxRcl=" << std::fixed << std::setprecision(4) << oMetrics.dMaxRecall << " MaxPrc=" << oMetrics.dMaxPrecision << " MaxFM=" << oMetrics.dMaxFMeasure << std::endl;
    std::cout << "                BestRcl=" << std::fixed << std::setprecision(4) << oMetrics.oBestScore.dRecall << " BestPrc=" << oMetrics.oBestScore.dPrecision << " BestFM=" << oMetrics.oBestScore.dFMeasure << "  (@ T=" << std::fixed << std::setprecision(4) << oMetrics.oBestScore.dThreshold << ")" << std::endl;
#if USE_BSDS500_BENCHMARK
    std::ofstream oMetricsOutput(getOutputPath()+"/overall_reimpl_eval.txt");
#else //(!USE_BSDS500_BENCHMARK)
    std::ofstream oMetricsOutput(getOutputPath()+"/overall_homemade_eval.txt");
#endif //(!USE_BSDS500_BENCHMARK)
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << std::fixed;
        oMetricsOutput << "BSDS500 edge detection evaluation report :\n\n";
        oMetricsOutput << "            ||   MaxRcl   |   MaxPrc   |    MaxFM   ||   BestRcl  |   BestPrc  |   BestFM   | @Threshold \n";
        oMetricsOutput << "------------||------------|------------|------------||------------|------------|------------|------------\n";
        size_t nOverallPacketCount = 0;
        double dOverallTimeElapsed = 0.0;
        for(const auto& pGroupIter : getBatches(true)) {
            oMetricsOutput << pGroupIter->shared_from_this_cast<const DataReporter_<eDatasetEval_BinaryClassifier,eDataset_BSDS500>>(true)->DataReporter_<eDatasetEval_BinaryClassifier,eDataset_BSDS500>::writeInlineEvalReport(0);
            nOverallPacketCount += pGroupIter->getTotPackets();
            dOverallTimeElapsed += pGroupIter->getProcessTime();
        }
        oMetricsOutput << "------------||------------|------------|------------||------------|------------|------------|------------\n";
        oMetricsOutput << "     overall||" <<
            std::setw(12) << oMetrics.dMaxRecall << "|" <<
            std::setw(12) << oMetrics.dMaxPrecision << "|" <<
            std::setw(12) << oMetrics.dMaxFMeasure << "||" <<
            std::setw(12) << oMetrics.oBestScore.dRecall << "|" <<
            std::setw(12) << oMetrics.oBestScore.dPrecision << "|" <<
            std::setw(12) << oMetrics.oBestScore.dFMeasure << "|" <<
            std::setw(12) << oMetrics.oBestScore.dThreshold << "\n";
        oMetricsOutput << "\nHz: " << nOverallPacketCount/dOverallTimeElapsed << "\n";
        oMetricsOutput << CxxUtils::getLogStamp();
    }
}

litiv::IMetricsAccumulatorConstPtr litiv::DatasetEvaluator_<litiv::eDatasetEval_BinaryClassifier,litiv::eDataset_BSDS500>::getMetricsBase() const {
    BSDS500MetricsAccumulatorPtr pMetricsBase = BSDS500MetricsAccumulator::create(DATASETS_BSDS500_EVAL_DEFAULT_THRESH_BINS);
    for(const auto& pBatch : getBatches(true))
        pMetricsBase->accumulate(dynamic_cast<const DataReporter_<eDatasetEval_BinaryClassifier,eDataset_BSDS500>&>(*pBatch).getMetricsBase());
    return pMetricsBase;
}

litiv::IMetricsCalculatorPtr litiv::DatasetEvaluator_<litiv::eDatasetEval_BinaryClassifier,litiv::eDataset_BSDS500>::getMetrics() const {
    return BSDS500MetricsCalculator::create(getMetricsBase());
}

litiv::IMetricsAccumulatorConstPtr litiv::DataReporter_<litiv::eDatasetEval_BinaryClassifier,litiv::eDataset_BSDS500>::getMetricsBase() const {
    lvAssert(isGroup()); // non-group specialization should override this method
    BSDS500MetricsAccumulatorPtr pMetricsBase = BSDS500MetricsAccumulator::create(DATASETS_BSDS500_EVAL_DEFAULT_THRESH_BINS);
    for(const auto& pBatch : getBatches(true))
        pMetricsBase->accumulate(dynamic_cast<const DataReporter_<eDatasetEval_BinaryClassifier,eDataset_BSDS500>&>(*pBatch).getMetricsBase());
    return pMetricsBase;
}

litiv::IMetricsCalculatorPtr litiv::DataReporter_<litiv::eDatasetEval_BinaryClassifier,litiv::eDataset_BSDS500>::getMetrics() const {
    return BSDS500MetricsCalculator::create(getMetricsBase());
}

void litiv::DataReporter_<litiv::eDatasetEval_BinaryClassifier,litiv::eDataset_BSDS500>::writeEvalReport() const {
    IDataReporter_<eDatasetEval_None>::writeEvalReport();
    if(!getTotPackets() || !getDatasetInfo()->isUsingEvaluator())
        return;
    else if(isGroup() && !isBare())
        for(const auto& pBatch : getBatches(true))
            pBatch->writeEvalReport();
    IMetricsCalculatorConstPtr pMetrics = getMetrics();
    lvAssert(pMetrics.get());
    const BSDS500MetricsCalculator& oMetrics = dynamic_cast<const BSDS500MetricsCalculator&>(*pMetrics.get());
    std::cout << "\t" << CxxUtils::clampString(std::string(size_t(!isGroup()),'>')+getName(),12) << " => MaxRcl=" << std::fixed << std::setprecision(4) << oMetrics.dMaxRecall << " MaxPrc=" << oMetrics.dMaxPrecision << " MaxFM=" << oMetrics.dMaxFMeasure << std::endl;
    std::cout << "\t" << "                BestRcl=" << std::fixed << std::setprecision(4) << oMetrics.oBestScore.dRecall << " BestPrc=" << oMetrics.oBestScore.dPrecision << " BestFM=" << oMetrics.oBestScore.dFMeasure << "  (@ T=" << std::fixed << std::setprecision(4) << oMetrics.oBestScore.dThreshold << ")" << std::endl;
#if USE_BSDS500_BENCHMARK
    const std::string sOutputPath = PlatformUtils::AddDirSlashIfMissing(getOutputPath())+"../"+getName()+"_reimpl_eval/";
#else //(!USE_BSDS500_BENCHMARK)
    const std::string sOutputPath = PlatformUtils::AddDirSlashIfMissing(getOutputPath())+"../"+getName()+"_homemade_eval/";
#endif //(!USE_BSDS500_BENCHMARK)
    PlatformUtils::CreateDirIfNotExist(sOutputPath);
    std::ofstream oImageScoresOutput(sOutputPath+"/eval_bdry_img.txt");
    if(oImageScoresOutput.is_open())
        for(size_t n=0; n<oMetrics.voBestImageScores.size(); ++n)
            oImageScoresOutput << cv::format("%10d %10g %10g %10g %10g\n",n+1,oMetrics.voBestImageScores[n].dThreshold,oMetrics.voBestImageScores[n].dRecall,oMetrics.voBestImageScores[n].dPrecision,oMetrics.voBestImageScores[n].dFMeasure);
    std::ofstream oThresholdMetricsOutput(sOutputPath+"/eval_bdry_thr.txt");
    if(oThresholdMetricsOutput.is_open())
        for(size_t n=0; n<oMetrics.voThresholdScores.size(); ++n)
            oThresholdMetricsOutput << cv::format("%10g %10g %10g %10g\n",oMetrics.voThresholdScores[n].dThreshold,oMetrics.voThresholdScores[n].dRecall,oMetrics.voThresholdScores[n].dPrecision,oMetrics.voThresholdScores[n].dFMeasure);
    std::ofstream oOverallMetricsOutput(sOutputPath+"/eval_bdry.txt");
    if(oOverallMetricsOutput.is_open())
        oOverallMetricsOutput << cv::format("%10g %10g %10g %10g %10g %10g %10g %10g\n",oMetrics.oBestScore.dThreshold,oMetrics.oBestScore.dRecall,oMetrics.oBestScore.dPrecision,oMetrics.oBestScore.dFMeasure,oMetrics.dMaxRecall,oMetrics.dMaxPrecision,oMetrics.dMaxFMeasure,oMetrics.dAreaPR);
}

std::string litiv::DataReporter_<litiv::eDatasetEval_BinaryClassifier,litiv::eDataset_BSDS500>::writeInlineEvalReport(size_t nIndentSize) const {
    if(!getTotPackets())
        return std::string();
    const size_t nCellSize = 12;
    std::stringstream ssStr;
    ssStr << std::fixed;
    if(isGroup() && !isBare())
        for(const auto& pBatch : getBatches(true))
            ssStr << pBatch->shared_from_this_cast<const DataReporter_<eDatasetEval_BinaryClassifier,eDataset_BSDS500>>(true)->DataReporter_<eDatasetEval_BinaryClassifier,eDataset_BSDS500>::writeInlineEvalReport(nIndentSize+1);
    IMetricsCalculatorConstPtr pMetrics = getMetrics();
    lvAssert(pMetrics.get());
    const BSDS500MetricsCalculator& oMetrics = dynamic_cast<const BSDS500MetricsCalculator&>(*pMetrics.get());
    ssStr << CxxUtils::clampString((std::string(nIndentSize,'>')+' '+getName()),nCellSize) << "||" <<
        std::setw(nCellSize) << oMetrics.dMaxRecall << "|" <<
        std::setw(nCellSize) << oMetrics.dMaxPrecision << "|" <<
        std::setw(nCellSize) << oMetrics.dMaxFMeasure << "||" <<
        std::setw(nCellSize) << oMetrics.oBestScore.dRecall << "|" <<
        std::setw(nCellSize) << oMetrics.oBestScore.dPrecision << "|" <<
        std::setw(nCellSize) << oMetrics.oBestScore.dFMeasure << "|" <<
        std::setw(nCellSize) << oMetrics.oBestScore.dThreshold << "\n";
    return ssStr.str();
}

litiv::IMetricsAccumulatorConstPtr litiv::DataEvaluator_<litiv::eDatasetEval_BinaryClassifier,litiv::eDataset_BSDS500,ParallelUtils::eNonParallel>::getMetricsBase() const {
    if(!m_pMetricsBase)
        return BSDS500MetricsAccumulator::create(DATASETS_BSDS500_EVAL_DEFAULT_THRESH_BINS);
    return m_pMetricsBase;
}

void litiv::DataEvaluator_<litiv::eDatasetEval_BinaryClassifier,litiv::eDataset_BSDS500,ParallelUtils::eNonParallel>::push(const cv::Mat& oClassif, size_t nIdx) {
    IDataConsumer_<eDatasetEval_BinaryClassifier>::push(oClassif,nIdx);
    if(getDatasetInfo()->isUsingEvaluator()) {
        auto pLoader = shared_from_this_cast<IDataLoader>(true);
        if(!m_pMetricsBase)
            m_pMetricsBase = BSDS500MetricsAccumulator::create(DATASETS_BSDS500_EVAL_DEFAULT_THRESH_BINS);
        m_pMetricsBase->accumulate(oClassif,pLoader->getGT(nIdx),pLoader->getInputROI(nIdx));
    }
}

cv::Mat litiv::DataEvaluator_<litiv::eDatasetEval_BinaryClassifier,litiv::eDataset_BSDS500,ParallelUtils::eNonParallel>::getColoredMask(const cv::Mat& oClassif, size_t nIdx) {
    auto pLoader = shared_from_this_cast<IDataLoader>(true);
    return BSDS500MetricsAccumulator::getColoredMask(oClassif,pLoader->getGT(nIdx),pLoader->getInputROI(nIdx));
}

void litiv::DataEvaluator_<litiv::eDatasetEval_BinaryClassifier,litiv::eDataset_BSDS500,ParallelUtils::eNonParallel>::resetMetrics() {
    m_pMetricsBase = BSDS500MetricsAccumulator::create(DATASETS_BSDS500_EVAL_DEFAULT_THRESH_BINS);
}
