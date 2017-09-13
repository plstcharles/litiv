
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

#include "litiv/imgproc/ForegroundStereoMatcher.hpp"

#if (STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF)
#if !HAVE_OPENGM_EXTLIB
#error "ForegroundStereoMatcher config requires OpenGM external lib w/ QPBO for inference."
#endif //!HAVE_OPENGM_EXTLIB
#if !HAVE_OPENGM_EXTLIB_QPBO
#error "ForegroundStereoMatcher config requires OpenGM external lib w/ QPBO for inference."
#endif //!HAVE_OPENGM_EXTLIB_QPBO
#endif //(STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF)
#if STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF
#if !HAVE_OPENGM_EXTLIB
#error "ForegroundStereoMatcher config requires OpenGM external lib w/ FastPD for inference."
#endif //!HAVE_OPENGM_EXTLIB
#if !HAVE_OPENGM_EXTLIB_FASTPD
#error "ForegroundStereoMatcher config requires OpenGM external lib w/ FastPD for inference."
#endif //!HAVE_OPENGM_EXTLIB_FASTPD
#endif //STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF
#if (STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF)
#if !HAVE_BOOST
#error "ForegroundStereoMatcher config requires boost due to 3rdparty sospd module for inference."
#endif //!HAVE_BOOST
#define STEREOSEGMATCH_CONFIG_USE_SOSPD_ALPHA_HEIGHTS_LABEL_ORDERING 0
#endif //(STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF)
#if (STEREOSEGMATCH_CONFIG_USE_DASCGF_AFFINITY+\
     STEREOSEGMATCH_CONFIG_USE_DASCRF_AFFINITY+\
     STEREOSEGMATCH_CONFIG_USE_LSS_AFFINITY+\
     STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY+\
     STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY/*+...*/\
    )!=1
#error "Must specify only one image affinity map computation approach to use."
#endif //(features config ...)!=1
#define STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY (STEREOSEGMATCH_CONFIG_USE_DASCGF_AFFINITY||STEREOSEGMATCH_CONFIG_USE_DASCRF_AFFINITY||STEREOSEGMATCH_CONFIG_USE_LSS_AFFINITY)
#if (STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF+\
     STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF+\
     STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF/*+...*/\
    )!=1
#error "Must specify only one stereo inference approach to use."
#endif //(stereo inf config ...)!=1
#if (STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF+\
     STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF/*+...*/\
    )!=1
#error "Must specify only one resegm inference approach to use."
#endif //(resegm inf config ...)!=1
#if STEREOSEGMATCH_CONFIG_USE_EPIPOLAR_CONN
#error "missing impl" // @@@@@
#endif //STEREOSEGMATCH_CONFIG_USE_EPIPOLAR_CONN

namespace {

    using InternalLabelType = StereoSegmMatcher::InternalLabelType;
    using OutputLabelType = StereoSegmMatcher::OutputLabelType;
    using AssocCountType = StereoSegmMatcher::AssocCountType;
    using AssocIdxType = StereoSegmMatcher::AssocIdxType;
    using ValueType =  StereoSegmMatcher::ValueType;
    using IndexType = StereoSegmMatcher::IndexType;
    template<typename T>
    using CamArray = StereoSegmMatcher::CamArray<T>;

    using ExplicitFunction = lv::gm::ExplicitViewFunction<ValueType,IndexType,InternalLabelType>; ///< shortcut for explicit view function
    using ExplicitAllocFunction = opengm::ExplicitFunction<ValueType,IndexType,InternalLabelType>; ///< shortcut for explicit allocated function
    using FunctionTypeList = opengm::meta::TypeListGenerator<ExplicitFunction,ExplicitAllocFunction/*,...*/>::type;  ///< list of all functions the models can use
    constexpr size_t s_nEpipolarConn = STEREOSEGMATCH_CONFIG_USE_EPIPOLAR_CONN?size_t(3):size_t(0); ///< epipolar connections count (i.e. order of stereo epipolar clique)
    constexpr size_t s_nEpipolarStride = STEREOSEGMATCH_HOENERGY_STEREO_STRIDE; ///< epipolar clique stride size (i.e. skipped connections; 1=fully connected)
    static_assert(s_nEpipolarStride>size_t(0),"stereo clique stride must be strictly positive");
    constexpr size_t s_nTemporalDepth = STEREOSEGMATCH_DEFAULT_TEMPORAL_DEPTH; ///< temporal depth level (i.e. connectivity layers for HO resegmentation cliques)
    constexpr size_t s_nTemporalConn = s_nTemporalDepth*StereoSegmMatcher::getCameraCount(); ///< temporal connections count (i.e. order of temporal cliques)
    constexpr size_t s_nTemporalLayers = s_nTemporalDepth+1; ///< number of temporal layers (i.e. images) to connect in resegm graphs
    constexpr size_t s_nTemporalStride = STEREOSEGMATCH_HOENERGY_RESEGM_STRIDE; ///< temporal clique stride size (i.e. skipped connections; 1=fully connected)
    static_assert(s_nTemporalStride>size_t(0),"resegm clique stride must be strictly positive");
    using PairwClique = lv::gm::Clique<size_t(2),ValueType,IndexType,InternalLabelType,ExplicitFunction>; ///< pairwise clique implementation wrapper
    using EpipolarClique = lv::gm::Clique<s_nEpipolarConn,ValueType,IndexType,InternalLabelType,ExplicitFunction>; ///< stereo epipolar line clique implementation wrapper
    using TemporalClique = lv::gm::Clique<s_nTemporalConn,ValueType,IndexType,InternalLabelType,ExplicitFunction>; ///< resegm temporal clique implementation wrapper
    using Clique = lv::gm::IClique<ValueType,IndexType,InternalLabelType,ExplicitFunction>; ///< general-use clique implementation wrapper for all terms
    using StereoSpaceType = opengm::SimpleDiscreteSpace<IndexType,InternalLabelType>; ///< shortcut for discrete stereo space type (simple = all nodes have the same # of labels)
    using ResegmSpaceType = opengm::StaticSimpleDiscreteSpace<2,IndexType,InternalLabelType>; ///< shortcut for discrete resegm space type (binary labels for fg/bg)
    using StereoModelType = opengm::GraphicalModel<ValueType,opengm::Adder,FunctionTypeList,StereoSpaceType>; ///< shortcut for stereo graphical model type
    using ResegmModelType = opengm::GraphicalModel<ValueType,opengm::Adder,FunctionTypeList,ResegmSpaceType>; ///< shortcut for resegm graphical model type
    static_assert(std::is_same<StereoModelType::FunctionIdentifier,ResegmModelType::FunctionIdentifier>::value,"mismatched function identifier for stereo/resegm graphs");
    using FuncIdentifType = StereoModelType::FunctionIdentifier; ///< shortcut for graph model function identifier type (for both stereo and resegm models)
    using FuncPairType = std::pair<FuncIdentifType,ExplicitFunction&>; ///< funcid-funcobj pair used as viewer to explicit data (for both stereo and resegm models)
    constexpr size_t s_nMaxOrder = lv::get_next_pow2((uint32_t)std::max(std::max(s_nTemporalConn,s_nEpipolarConn),StereoSegmMatcher::s_nInputArraySize)); ///< used to limit internal static assignment array sizes
    constexpr size_t s_nPairwOrients = size_t(2); ///< number of pairwise links owned by each node in the graph (2 = 1st order neighb connections)
    static_assert(s_nPairwOrients>size_t(0),"pairwise orientation count must be strictly positive");

    /// defines the indices of feature maps inside precalc packets (per camera head)
    enum FeatPackingList {
        FeatPackSize=18,
        FeatPackOffset=7,
        // absolute values for direct indexing
        FeatPack_LeftInitFGDist=0,
        FeatPack_LeftInitBGDist=1,
        FeatPack_LeftFGDist=2,
        FeatPack_LeftBGDist=3,
        FeatPack_LeftGradY=4,
        FeatPack_LeftGradX=5,
        FeatPack_LeftGradMag=6,
        FeatPack_RightInitFGDist=7,
        FeatPack_RightInitBGDist=8,
        FeatPack_RightFGDist=9,
        FeatPack_RightBGDist=10,
        FeatPack_RightGradY=11,
        FeatPack_RightGradX=12,
        FeatPack_RightGradMag=13,
        FeatPack_ImgSaliency=14,
        FeatPack_ShpSaliency=15,
        FeatPack_ImgAffinity=16,
        FeatPack_ShpAffinity=17,
        // relative values for cam-based indexing
        FeatPackOffset_InitFGDist=0,
        FeatPackOffset_InitBGDist=1,
        FeatPackOffset_FGDist=2,
        FeatPackOffset_BGDist=3,
        FeatPackOffset_GradY=4,
        FeatPackOffset_GradX=5,
        FeatPackOffset_GradMag=6,
    };

    /// basic info struct used for node-level graph model updates and data lookups
    struct NodeInfo {
        /// image grid coordinates associated with this node
        int nRowIdx,nColIdx;
        /// holds whether this is a valid graph node or not
        bool bValidGraphNode;
        /// if valid, this holds whether this node is near graph borders or not
        bool bNearBorders;
        /// node graph index used for model indexing
        size_t nGraphNodeIdx;
        /// id for this node's unary factor (not SIZE_MAX only if valid)
        size_t nUnaryFactID;
        /// pointer to this node's unary function (non-null only if valid)
        ExplicitFunction* pUnaryFunc;
        /// weights for this node's pairwise costs (constant post initialization)
        mutable std::array<float,s_nPairwOrients> afPairwWeights;
        /// array of pairwise cliques owned by this node as 1st member (evaluates to true only if valid)
        std::array<PairwClique,s_nPairwOrients> aPairwCliques;
        /// vector of pointers to all (valid) cliques owned by this node as 1st member (all must evaluate to true)
        std::vector<Clique*> vpCliques;
    };

    /// basic info struct used for node-level stereo graph model updates and data lookups
    struct StereoNodeInfo : NodeInfo {
        /// epipolar clique owned by this node as 1st member (evaluates to true only if valid)
        EpipolarClique oEpipolarClique;
    };

    /// basic info struct used for node-level resegm graph model updates and data lookups
    struct ResegmNodeInfo : NodeInfo {
        /// temporal depth index associated with this node (0 = current frame, positive values = past offset)
        int nTemporalDepth;
        /// temporal clique owned by this node as 1st member (evaluates to true only if valid)
        TemporalClique oTemporalClique;
    };

} // anonymous namespace

/// holds graph model data for both stereo and resegmentation models
struct StereoSegmMatcher::GraphModelData {
    /// default constructor; receives model construction data from algo constructor
    GraphModelData(const CamArray<cv::Mat>& aROIs, const std::vector<OutputLabelType>& vRealStereoLabels, size_t nStereoLabelStep, size_t nPrimaryCamIdx);
    /// (pre)calculates features required for model updates, and optionally returns them in packet format
    void calcFeatures(const MatArrayIn& aInputs, cv::Mat* pFeatsPacket=nullptr);
    /// sets a previously precalculated features packet to be used in the next model updates (do not modify it before that!)
    void setNextFeatures(const cv::Mat& oPackedFeats);
    /// performs the actual bi-model, bi-spectral inference
    opengm::InferenceTermination infer();
    /// translate an internal graph label to a real disparity offset label
    OutputLabelType getRealLabel(InternalLabelType nLabel) const;
    /// translate a real disparity offset label to an internal graph label
    InternalLabelType getInternalLabel(OutputLabelType nRealLabel) const;
    /// returns the offset value to use for indexing pixels in the other camera image
    int getOffsetValue(size_t nCamIdx, InternalLabelType nLabel) const;
    /// returns the offset column to use for indexing pixels in the other camera image
    int getOffsetColIdx(size_t nCamIdx, int nColIdx, InternalLabelType nLabel) const;
    /// returns the stereo associations count for a given graph node by row/col indices
    AssocCountType getAssocCount(int nRowIdx, int nColIdx) const;
    /// returns the cost of adding a stereo association for a given node coord set & origin column idx
    ValueType calcAddAssocCost(int nRowIdx, int nColIdx, InternalLabelType nLabel) const;
    /// returns the cost of removing a stereo association for a given node coord set & origin column idx
    ValueType calcRemoveAssocCost(int nRowIdx, int nColIdx, InternalLabelType nLabel) const;
    /// returns the total stereo association cost for all grid nodes
    ValueType calcTotalAssocCost() const;
    /// helper func to display segmentation maps
    cv::Mat getResegmMapDisplay(size_t nCamIdx) const;
    /// helper func to display scaled disparity maps
    cv::Mat getStereoDispMapDisplay(size_t nCamIdx) const;
    /// helper func to display scaled assoc count maps (for primary cam only)
    cv::Mat getAssocCountsMapDisplay() const;

    /// number of frame sets processed so far (used to toggle temporal links on/off)
    size_t m_nFramesProcessed;
    /// max move making iteration count allowed during inference
    size_t m_nMaxMoveIterCount;
    /// random seeds to use to initialize labeling/label-order arrays
    size_t m_nStereoLabelOrderRandomSeed,m_nStereoLabelingRandomSeed;
    /// contains the (internal) stereo label ordering to use for each iteration
    std::vector<InternalLabelType> m_vStereoLabelOrdering;
    /// holds the set of features to use during the next inference (mutable, as shape features will change during inference)
    mutable std::vector<cv::Mat> m_vNextFeats;
    /// contains the (internal) labelings of the stereo/resegm graphs (mutable for inference)
    mutable CamArray<cv::Mat_<InternalLabelType>> m_aStereoLabelings,m_aResegmLabelings;
    /// 2d map which contain how many associations a graph node possesses (mutable for inference)
    mutable cv::Mat_<AssocCountType> m_oAssocCounts;
    /// 3d map which list the associations (by idx) for each graph node (mutable for inference)
    mutable cv::Mat_<AssocIdxType> m_oAssocMap;
    /// 2d map which contains transient unary factor labeling costs for all stereo graph nodes (mutable for inference)
    mutable cv::Mat_<ValueType> m_oStereoUnaryCosts;
    /// 2d maps which contain transient unary factor labeling costs for all resegm graph nodes (mutable for inference)
    mutable CamArray<cv::Mat_<ValueType>> m_aResegmUnaryCosts;
    /// contains the ROIs used for grid setup passed in the constructor
    const CamArray<cv::Mat_<uchar>> m_aROIs;
    /// contains the predetermined (max) 2D grid size for the graph models
    const lv::MatSize m_oGridSize;
    /// contains all 'real' stereo labels, plus the 'occluded'/'dontcare' labels
    const std::vector<OutputLabelType> m_vStereoLabels;
    /// total number of (re)segmentation labels (could be constexpr, but messes up linking w/ gcc @@@@)
    const size_t m_nResegmLabels;
    /// number of 'real' (i.e. non-reserved) stereo disparity labels
    const size_t m_nRealStereoLabels;
    /// total number of stereo disparity labels (including reserved ones)
    const size_t m_nStereoLabels;
    /// contains the step size between stereo labels (i.e. the disparity granularity)
    const size_t m_nDispOffsetStep;
    /// contains the minimum disparity offset value
    const size_t m_nMinDispOffset;
    /// contains the maximum disparity offset value
    const size_t m_nMaxDispOffset;
    /// primary camera index for stereo disparity estimation base
    const size_t m_nPrimaryCamIdx;
    /// internal label used for 'dont care' labeling
    const InternalLabelType m_nDontCareLabelIdx;
    /// internal label used for 'occluded' labeling
    const InternalLabelType m_nOccludedLabelIdx;
    /// opengm stereo graph model object (for primary cam head)
    std::unique_ptr<StereoModelType> m_pStereoModel;
    /// opengm resegm graph model object (one per cam head)
    CamArray<std::unique_ptr<ResegmModelType>> m_apResegmModels;
    /// contains the eroded ROIs used for valid descriptor lookups
    CamArray<cv::Mat_<uchar>> m_aDescROIs;
    /// indices of valid nodes in the stereo graph (based on primary ROI)
    std::vector<size_t> m_vValidStereoLUTNodeIdxs;
    /// number of valid nodes/cliques in the stereo graph (based on primary ROI)
    size_t m_nValidStereoGraphNodes,m_nValidStereoGraphCliques;
    /// indices of valid nodes in the resegm graphs (based on each ROI)
    CamArray<std::vector<size_t>> m_avValidResegmLUTNodeIdxs;
    /// number of valid nodes/cliques in the resegm graphs (based on each ROI)
    CamArray<size_t> m_anValidResegmGraphNodes,m_anValidResegmGraphCliques;
    /// stereo model info lookup array
    std::vector<StereoNodeInfo> m_vStereoNodeInfos;
    /// resegm models info lookup array
    CamArray<std::vector<ResegmNodeInfo>> m_avResegmNodeInfos;
    /// graph model factor counts used for validation in debug mode
    size_t m_nStereoUnaryFactCount,m_nStereoPairwFactCount,m_nStereoEpipolarFactCount;
    /// graph model factor counts used for validation in debug mode
    CamArray<size_t> m_anResegmUnaryFactCounts,m_anResegmPairwFactCounts,m_anResegmTemporalFactCounts;
    /// stereo graph model unary/epipolar functions
    std::vector<FuncPairType> m_vStereoUnaryFuncs,m_vStereoEpipolarFuncs;
    /// stereo graph model pairwise functions arrays (for already-weighted lookups)
    std::array<std::vector<FuncPairType>,s_nPairwOrients> m_avStereoPairwFuncs;
    /// stereo model pairwise function id (for shared base lookups without weights)
    FuncIdentifType m_oStereoPairwFuncID_base;
    /// resegm graph model unary/temporal functions
    CamArray<std::vector<FuncPairType>> m_avResegmUnaryFuncs,m_avResegmTemporalFuncs;
    /// resegm graph model pairwise functions arrays (for already-weighted lookups)
    CamArray<std::array<std::vector<FuncPairType>,s_nPairwOrients>> m_aavResegmPairwFuncs;
    /// resegm model pairwise function ids (for shared base lookups without weights)
    CamArray<FuncIdentifType> m_aResegmPairwFuncIDs_base;
    /// functions data arrays (contiguous blocks for all factors)
    std::unique_ptr<ValueType[]> m_aStereoFuncsData,m_aResegmFuncsData;
    /// stereo models unary/pairw/epipolar functions base pointers
    ValueType *m_pStereoUnaryFuncsDataBase,*m_pStereoPairwFuncsDataBase,*m_pStereoEpipolarFuncsDataBase;
    /// resegm models unary/pairw/temporal functions base pointers
    CamArray<ValueType*> m_apResegmUnaryFuncsDataBase,m_apResegmPairwFuncsDataBase,m_apResegmTemporalFuncsDataBase;
    /// gmm fg/bg models used for intra-spectral visual-data-based segmentation (3ch)
    CamArray<lv::GMM<5,3>> m_aFGModels_3ch,m_aBGModels_3ch;
    /// gmm fg/bg models used for intra-spectral visual-data-based segmentation (1ch)
    CamArray<lv::GMM<3,1>> m_aFGModels_1ch,m_aBGModels_1ch;
    /// gmm label component maps used for model learning (kept here to avoid reallocs)
    CamArray<cv::Mat_<int>> m_aGMMCompAssignMap;
    /// cost lookup table for adding/removing/summing associations
    lv::AutoBuffer<ValueType,200> m_aAssocCostRealAddLUT,m_aAssocCostRealRemLUT,m_aAssocCostRealSumLUT;
    /// cost lookup table for approximately (worse case) adding/removing associations
    lv::AutoBuffer<ValueType,200> m_aAssocCostApproxAddLUT,m_aAssocCostApproxRemLUT;
    /// gradient factor lookup table for label similarity
    lv::LUT<uchar,float,256> m_aLabelSimCostGradFactLUT;

    /// holds the feature extractor to use on input images
#if STEREOSEGMATCH_CONFIG_USE_DASCGF_AFFINITY || STEREOSEGMATCH_CONFIG_USE_DASCRF_AFFINITY
    std::unique_ptr<DASC> m_pImgDescExtractor;
#elif STEREOSEGMATCH_CONFIG_USE_LSS_AFFINITY
    std::unique_ptr<LSS> m_pImgDescExtractor;
#elif STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY
    std::unique_ptr<MutualInfo> m_pImgDescExtractor; // although not really a 'descriptor' extractor...
#endif //STEREOSEGMATCH_CONFIG_USE_..._AFFINITY
    /// holds the feature extractor to use on input shapes
    std::unique_ptr<ShapeContext> m_pShpDescExtractor;
    /// defines the minimum grid border size based on the feature extractors used
    size_t m_nGridBorderSize;
    /// holds the last/next features packet info vector
    std::vector<lv::MatInfo> m_vExpectedFeatPackInfo,m_vNextFeatPackInfo;
    /// holds the set of (packed) features to use next
    cv::Mat m_oNextPackedFeats;
    /// holds the latest input image/mask matrices
    MatArrayIn m_aInputs;
    /// holds the latest output disp/mask matrices
    MatArrayOut m_aOutputs;
    /// defines whether the next model update should use precalc feats
    bool m_bUsePrecalcFeatsNext;
    /// defines whether the resegm graphs are fully built (i.e. with temporal links) or not
    bool m_bResegmGraphsFullyBuilt;
    /// used for debug only; passed from top-level algo when available
    lv::DisplayHelperPtr m_pDisplayHelper;

protected:
    /// adds a stereo association for a given node coord set & origin column idx
    void addAssoc(int nRowIdx, int nColIdx, InternalLabelType nLabel) const;
    /// removes a stereo association for a given node coord set & origin column idx
    void removeAssoc(int nRowIdx, int nColIdx, InternalLabelType nLabel) const;
    /// resets stereo graph labelings using init state parameters
    void resetStereoLabelings(size_t nCamIdx);
    /// builds a stereo graph model using ROI information
    void buildStereoModel();
    /// updates a stereo graph model using new feats data
    void updateStereoModel(bool bInit);
    /// builds a resegm graph model using ROI information
    void buildResegmModel(size_t nCamIdx, bool bUseTemporalLinks);
    /// updates a shape graph model using new feats data
    void updateResegmModel(size_t nCamIdx, bool bInit);
    /// calculates image features required for model updates using the provided input image array
    void calcImageFeatures(const CamArray<cv::Mat>& aInputImages);
    /// calculates shape features required for model updates using the provided input mask array
    void calcShapeFeatures(const CamArray<cv::Mat_<InternalLabelType>>& aInputMasks);
    /// calculates shape mask distance features required for model updates using the provided input mask & camera index
    void calcShapeDistFeatures(const cv::Mat_<InternalLabelType>& oInputMask, size_t nCamIdx);
    /// calculates a stereo unary move cost for a single graph node
    ValueType calcStereoUnaryMoveCost(size_t nGraphNodeIdx, InternalLabelType nOldLabel, InternalLabelType nNewLabel) const;
#if (STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF)
    /// fill internal temporary energy cost mats for the given stereo move operation
    void calcStereoMoveCosts(InternalLabelType nNewLabel) const;
#endif //(STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF)
#if (STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF)
    /// fill internal temporary energy cost mats for the given resegm move operation
    void calcResegmMoveCosts(size_t nCamIdx, InternalLabelType nNewLabel) const;
#endif //(STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF)
#if STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF
    /// runs sospd inference algorithm either to completion, or for a specific number of iterations
    // ... @@@ todo
    typedef IndexType VarId;
    typedef InternalLabelType Label;
    typedef std::vector<ValueType> LambdaAlpha;
    typedef std::vector<std::pair<IndexType,IndexType>> NodeNeighborList;
    typedef std::vector<NodeNeighborList> NodeCliqueList;
    bool InitialFusionLabeling();
    void PreEditDual(SubmodularIBFS<ValueType,VarId>& crf);
    bool UpdatePrimalDual(SubmodularIBFS<ValueType,VarId>& crf);
    void PostEditDual(SubmodularIBFS<ValueType,VarId>& crf);
    size_t __cam;
    size_t m_nStereoCliqueCount;
    InternalLabelType __alpha;
    ValueType& dualVariable(int alpha, VarId i, Label l) {return m_dual[alpha][i*m_nStereoLabels+l];}
    const ValueType& dualVariable(int alpha, VarId i, Label l) const {return m_dual[alpha][i*m_nStereoLabels+l];}
    ValueType& dualVariable(LambdaAlpha& lambdaAlpha,VarId i, Label l) {return lambdaAlpha[i*m_nStereoLabels+l];}
    const ValueType& dualVariable(LambdaAlpha& lambdaAlpha,VarId i, Label l) const {return lambdaAlpha[i*m_nStereoLabels+l];}
    //void HeightAlphaProposal();
    //void AlphaProposal();
    NodeCliqueList m_node_clique_list;
    cv::Mat_<ValueType> m_dual,m_heights;
    //std::vector<ValueType> m_heights;
    //ProposalCallback m_pc;
#endif //STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF
    /// holds stereo disparity graph inference algorithm interface (redirects for bi-model inference)
    std::unique_ptr<StereoGraphInference> m_pStereoInf;
    /// holds resegmentation graph inference algorithm interface (redirects for bi-model inference)
    CamArray<std::unique_ptr<ResegmGraphInference>> m_apResegmInfs;
};

/// algo interface for multi-label graph model inference
struct StereoSegmMatcher::StereoGraphInference : opengm::Inference<StereoModelType,opengm::Minimizer> {
    /// full constructor of the inference algorithm; the graphical model must have already been constructed prior to this call
    StereoGraphInference(size_t nCamIdx, GraphModelData& oData);
    /// returns the name of this inference method, for debugging/identification purposes
    virtual std::string name() const override;
    /// returns a copy of the internal const reference to the graphical model to solve
    virtual const StereoModelType& graphicalModel() const override;
    /// redirects inference to the bi-model implementation
    virtual opengm::InferenceTermination infer() override;
    /// sets an internal labeling starting point for the inference (the iterator must be valid over its 'nNodes' next values)
    virtual void setStartingPoint(typename std::vector<InternalLabelType>::const_iterator begin) override;
    /// translates and sets a labeling starting point for the inference (must be the same size as the internal graph model grid)
    virtual void setStartingPoint(const cv::Mat_<OutputLabelType>& oLabeling);
    /// returns the internal labeling solution, as determined by solving the graphical model inference problem
    virtual opengm::InferenceTermination arg(std::vector<InternalLabelType>& oLabeling, const size_t n=1) const override;
    /// returns the current real labeling solution, as determined by solving+translating the graphical model inference problem
    virtual void getOutput(cv::Mat_<OutputLabelType>& oLabeling) const;
    /// returns the energy of the current labeling solution
    virtual ValueType value() const override;
    /// ref to StereoSegmMatcher::m_pModelData
    GraphModelData& m_oData;
    /// camera head index targeted by the inference algo
    const size_t m_nPrimaryCamIdx;
};

/// algo interface for binary label graph model inference
struct StereoSegmMatcher::ResegmGraphInference : opengm::Inference<ResegmModelType,opengm::Minimizer> {
    /// full constructor of the inference algorithm; the graphical model must have already been constructed prior to this call
    ResegmGraphInference(size_t nCamIdx, GraphModelData& oData);
    /// returns the name of this inference method, for debugging/identification purposes
    virtual std::string name() const override;
    /// returns a copy of the internal const reference to the graphical model to solve
    virtual const ResegmModelType& graphicalModel() const override;
    /// redirects inference to the bi-model implementation
    virtual opengm::InferenceTermination infer() override;
    /// sets an internal labeling starting point for the inference (the iterator must be valid over its 'nNodes' next values)
    virtual void setStartingPoint(typename std::vector<InternalLabelType>::const_iterator begin) override;
    /// translates and sets a labeling starting point for the inference (must be the same size as the internal graph model grid)
    virtual void setStartingPoint(const cv::Mat_<OutputLabelType>& oLabeling);
    /// returns the internal labeling solution, as determined by solving the graphical model inference problem
    virtual opengm::InferenceTermination arg(std::vector<InternalLabelType>& oLabeling, const size_t n=1) const override;
    /// returns the current real labeling solution, as determined by solving+translating the graphical model inference problem
    virtual void getOutput(cv::Mat_<OutputLabelType>& oLabeling) const;
    /// returns the energy of the current labeling solution
    virtual ValueType value() const override;
    /// ref to StereoSegmMatcher::m_pModelData
    GraphModelData& m_oData;
    /// camera head index targeted by the inference algo
    const size_t m_nCamIdx;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////

StereoSegmMatcher::StereoSegmMatcher(size_t nMinDispOffset, size_t nMaxDispOffset) {
    static_assert(getInputStreamCount()==4 && getOutputStreamCount()==4 && getCameraCount()==2,"i/o stream must be two image-mask pairs");
    static_assert(getInputStreamCount()==InputPackSize && getOutputStreamCount()==OutputPackSize,"bad i/o internal enum mapping");
    lvDbgExceptionWatch;
    m_nDispStep = STEREOSEGMATCH_DEFAULT_DISPARITY_STEP;
    lvAssert_(m_nDispStep>0,"specified disparity offset step size must be strictly positive");
    if(nMaxDispOffset<nMinDispOffset)
        std::swap(nMaxDispOffset,nMinDispOffset);
    nMaxDispOffset -= (nMaxDispOffset-nMinDispOffset)%m_nDispStep;
    lvAssert_(nMaxDispOffset<size_t(s_nOccludedLabel),"using reserved disparity integer label value");
    lvAssert_((nMaxDispOffset-nMinDispOffset)/m_nDispStep>size_t(0),"disparity range must not be null");
    lvAssert_(((nMaxDispOffset-nMinDispOffset)%m_nDispStep)==0,"irregular disparity range label count with given step size");
    const size_t nMaxAllowedDispLabelCount = size_t(std::numeric_limits<InternalLabelType>::max()-2);
    const size_t nExpectedDispLabelCount = ((nMaxDispOffset-nMinDispOffset)/m_nDispStep)+1; // +1 since max label is included in the range
    lvAssert__(nMaxAllowedDispLabelCount>=nExpectedDispLabelCount,"internal stereo label type too small for given disparity range (max = %d)",(int)nMaxAllowedDispLabelCount);
    m_vStereoLabels = lv::make_range((OutputLabelType)nMinDispOffset,(OutputLabelType)nMaxDispOffset,(OutputLabelType)m_nDispStep);
    lvDbgAssert(nExpectedDispLabelCount==m_vStereoLabels.size());
    lvAssert_(m_vStereoLabels.size()>1,"graph must have at least two possible output labels, beyond reserved ones");
}

StereoSegmMatcher::~StereoSegmMatcher() {}

void StereoSegmMatcher::initialize(const std::array<cv::Mat,s_nCameraCount>& aROIs, size_t nPrimaryCamIdx) {
    static_assert(getCameraCount()==2,"bad static array size, mismatch with cam head count (hardcoded stuff below will break)");
    lvDbgExceptionWatch;
    lvAssert_(!aROIs[0].empty() && aROIs[0].total()>1 && aROIs[0].type()==CV_8UC1,"bad input ROI size/type");
    lvAssert_(lv::MatInfo(aROIs[0])==lv::MatInfo(aROIs[1]),"mismatched ROI size/type");
    lvAssert_(m_nDispStep>0,"specified disparity offset step size must be strictly positive");
    lvAssert_(m_vStereoLabels.size()>1,"graph must have at least two possible output labels, beyond reserved ones");
    lvAssert_(nPrimaryCamIdx<getCameraCount(),"primary camera idx is out of range");
    m_pModelData = std::make_unique<GraphModelData>(aROIs,m_vStereoLabels,m_nDispStep,nPrimaryCamIdx);
    if(m_pDisplayHelper)
        m_pModelData->m_pDisplayHelper = m_pDisplayHelper;
}

void StereoSegmMatcher::apply(const MatArrayIn& aInputs, MatArrayOut& aOutputs) {
    static_assert(s_nInputArraySize==4 && getCameraCount()==2,"lots of hardcoded indices below");
    lvDbgExceptionWatch;
    lvAssert_(m_pModelData,"model must be initialized first");
    for(size_t nInputIdx=0; nInputIdx<aInputs.size(); ++nInputIdx)
        lvAssert__(aInputs[nInputIdx].dims==2 && m_pModelData->m_oGridSize==aInputs[nInputIdx].size(),"input in array at index=%d had the wrong size",(int)nInputIdx);
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        const cv::Mat& oInputImg = aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img];
        const cv::Mat& oInputMask = aInputs[nCamIdx*InputPackOffset+InputPackOffset_Mask];
        lvAssert_(oInputImg.type()==CV_8UC1 || oInputImg.type()==CV_8UC3,"unexpected input image type");
        lvAssert_(oInputMask.type()==CV_8UC1,"unexpected input mask type");
        lvAssert_((cv::countNonZero(oInputMask==0)+cv::countNonZero(oInputMask==255))==(int)oInputMask.total(),"input mask must be binary (0 or 255 only)");
    }
    lvAssert_(!m_pModelData->m_bUsePrecalcFeatsNext || m_pModelData->m_vNextFeats.size()==FeatPackSize,"unexpected precalculated features vec size");
    for(size_t nInputIdx=0; nInputIdx<aInputs.size(); ++nInputIdx)
        aInputs[nInputIdx].copyTo(m_pModelData->m_aInputs[nInputIdx]);
    if(!m_pModelData->m_bUsePrecalcFeatsNext)
        m_pModelData->calcFeatures(aInputs);
    else
        m_pModelData->m_bUsePrecalcFeatsNext = false;
    m_pModelData->infer();
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        m_pModelData->m_aStereoLabelings[nCamIdx].copyTo(aOutputs[nCamIdx*OutputPackOffset+OutputPackOffset_Disp]);
        m_pModelData->m_aResegmLabelings[nCamIdx].copyTo(aOutputs[nCamIdx*OutputPackOffset+OutputPackOffset_Mask]);
    }
    for(size_t nOutputIdx=0; nOutputIdx<aOutputs.size(); ++nOutputIdx)
        aOutputs[nOutputIdx].copyTo(m_pModelData->m_aOutputs[nOutputIdx]); // copy for temporal analysis later
}

void StereoSegmMatcher::calcFeatures(const MatArrayIn& aInputs, cv::Mat* pFeatsPacket) {
    lvDbgExceptionWatch;
    lvAssert_(m_pModelData,"model must be initialized first");
    m_pModelData->calcFeatures(aInputs,pFeatsPacket);
}

void StereoSegmMatcher::setNextFeatures(const cv::Mat& oPackedFeats) {
    lvDbgExceptionWatch;
    lvAssert_(m_pModelData,"model must be initialized first");
    m_pModelData->setNextFeatures(oPackedFeats);
}

std::string StereoSegmMatcher::getFeatureExtractorName() const {
#if STEREOSEGMATCH_CONFIG_USE_DASCGF_AFFINITY
    return "sc-dasc-gf";
#elif STEREOSEGMATCH_CONFIG_USE_DASCRF_AFFINITY
    return "sc-dasc-rf";
#elif STEREOSEGMATCH_CONFIG_USE_LSS_AFFINITY
    return "sc-lss";
#elif STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY
    return "sc-mi";
#elif STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
    return "sc-ssqrdiff";
#endif //STEREOSEGMATCH_CONFIG_USE_..._AFFINITY
}

size_t StereoSegmMatcher::getMaxLabelCount() const {
    lvDbgExceptionWatch;
    lvAssert_(m_pModelData,"model must be initialized first");
    return m_pModelData->m_vStereoLabels.size();
}

const std::vector<StereoSegmMatcher::OutputLabelType>& StereoSegmMatcher::getLabels() const {
    lvDbgExceptionWatch;
    lvAssert_(m_pModelData,"model must be initialized first");
    return m_pModelData->m_vStereoLabels;
}

cv::Mat StereoSegmMatcher::getResegmMapDisplay(size_t nCamIdx) const {
    lvDbgExceptionWatch;
    lvAssert_(m_pModelData,"model must be initialized first");
    return m_pModelData->getResegmMapDisplay(nCamIdx);
}

cv::Mat StereoSegmMatcher::getStereoDispMapDisplay(size_t nCamIdx) const {
    lvDbgExceptionWatch;
    lvAssert_(m_pModelData,"model must be initialized first");
    return m_pModelData->getStereoDispMapDisplay(nCamIdx);
}

cv::Mat StereoSegmMatcher::getAssocCountsMapDisplay() const {
    lvDbgExceptionWatch;
    lvAssert_(m_pModelData,"model must be initialized first");
    return m_pModelData->getAssocCountsMapDisplay();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

StereoSegmMatcher::GraphModelData::GraphModelData(const CamArray<cv::Mat>& aROIs, const std::vector<OutputLabelType>& vRealStereoLabels, size_t nStereoLabelStep, size_t nPrimaryCamIdx) :
        m_nFramesProcessed(size_t(0)),
        m_nMaxMoveIterCount(STEREOSEGMATCH_DEFAULT_MAX_MOVE_ITER),
        m_nStereoLabelOrderRandomSeed(size_t(0)),
        m_nStereoLabelingRandomSeed(size_t(0)),
        m_aROIs(CamArray<cv::Mat_<uchar>>{aROIs[0]>0,aROIs[1]>0}),
        m_oGridSize(m_aROIs[0].size()),
        m_vStereoLabels(lv::concat<OutputLabelType>(vRealStereoLabels,std::vector<OutputLabelType>{s_nDontCareLabel,s_nOccludedLabel})),
        m_nResegmLabels(2),
        m_nRealStereoLabels(vRealStereoLabels.size()),
        m_nStereoLabels(vRealStereoLabels.size()+2),
        m_nDispOffsetStep(nStereoLabelStep),
        m_nMinDispOffset(size_t(m_vStereoLabels[0])),
        m_nMaxDispOffset(size_t(m_vStereoLabels.size()>3?m_vStereoLabels[m_vStereoLabels.size()-3]:m_vStereoLabels.back())),
        m_nPrimaryCamIdx(nPrimaryCamIdx),
        m_nDontCareLabelIdx(InternalLabelType(m_vStereoLabels.size()-2)),
        m_nOccludedLabelIdx(InternalLabelType(m_vStereoLabels.size()-1)),
        m_bUsePrecalcFeatsNext(false),m_bResegmGraphsFullyBuilt(false) {
    static_assert(getCameraCount()==2,"bad static array size, hardcoded stuff in constr init list and below will break");
    lvDbgExceptionWatch;
    lvAssert_(m_nMaxMoveIterCount>0,"max iter counts must be strictly positive");
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        lvAssert_(lv::MatInfo(m_aROIs[0])==lv::MatInfo(m_aROIs[nCamIdx]),"ROIs info must match");
        lvAssert_(cv::countNonZero(m_aROIs[nCamIdx]>0)>1,"ROIs must have at least two nodes");
    }
    lvAssert_(m_oGridSize.dims()==2 && m_oGridSize.total()>size_t(1),"graph grid must be 2D and have at least two nodes");
    lvAssert_(m_vStereoLabels.size()>3,"graph must have at least two possible output stereo labels, beyond reserved ones");
    lvAssert_(m_vStereoLabels.size()<=size_t(std::numeric_limits<InternalLabelType>::max()),"too many labels for internal type");
    lvDbgAssert(m_vStereoLabels[m_nDontCareLabelIdx]==s_nDontCareLabel && m_vStereoLabels[m_nOccludedLabelIdx]==s_nOccludedLabel);
    lvDbgAssert(std::min_element(m_vStereoLabels.begin(),m_vStereoLabels.begin()+m_vStereoLabels.size()-2)==m_vStereoLabels.begin() && m_vStereoLabels[0]>=OutputLabelType(0));
    lvDbgAssert(std::max_element(m_vStereoLabels.begin(),m_vStereoLabels.begin()+m_vStereoLabels.size()-2)==(m_vStereoLabels.begin()+m_vStereoLabels.size()-3));
    lvDbgAssert(std::equal(m_vStereoLabels.begin(),m_vStereoLabels.begin()+m_vStereoLabels.size()-2,lv::unique(m_vStereoLabels.begin(),m_vStereoLabels.end()).begin()+1));
    lvAssert_(m_nDispOffsetStep>0,"label step size must be positive");
    lvAssert_(m_oGridSize[1]>m_nMinDispOffset,"row length too small for smallest disp");
    lvAssert_(m_nMinDispOffset<m_nMaxDispOffset,"min/max disp offsets mismatch");
    lvAssert_(m_nPrimaryCamIdx<getCameraCount(),"bad primary camera index");
    lvDbgAssert_(std::numeric_limits<AssocCountType>::max()>m_oGridSize[1],"grid width is too large for association counter type");
#if STEREOSEGMATCH_CONFIG_USE_DASCGF_AFFINITY
    m_pImgDescExtractor = std::make_unique<DASC>(DASC_DEFAULT_GF_RADIUS,DASC_DEFAULT_GF_EPS,DASC_DEFAULT_GF_SUBSPL,DASC_DEFAULT_PREPROCESS);
    const cv::Size oDescWinSize = m_pImgDescExtractor->windowSize();
    m_nGridBorderSize = (size_t)std::max(m_pImgDescExtractor->borderSize(0),m_pImgDescExtractor->borderSize(1));
#elif STEREOSEGMATCH_CONFIG_USE_DASCRF_AFFINITY
    m_pImgDescExtractor = std::make_unique<DASC>(DASC_DEFAULT_RF_SIGMAS,DASC_DEFAULT_RF_SIGMAR,DASC_DEFAULT_RF_ITERS,DASC_DEFAULT_PREPROCESS);
    const cv::Size oDescWinSize = m_pImgDescExtractor->windowSize();
    m_nGridBorderSize = (size_t)std::max(m_pImgDescExtractor->borderSize(0),m_pImgDescExtractor->borderSize(1));
#elif STEREOSEGMATCH_CONFIG_USE_LSS_AFFINITY
    const int nLSSInnerRadius = 0;
    const int nLSSOuterRadius = (int)STEREOSEGMATCH_DEFAULT_LSSDESC_RAD;
    const int nLSSPatchSize = (int)STEREOSEGMATCH_DEFAULT_LSSDESC_PATCH;
    const int nLSSAngBins = (int)STEREOSEGMATCH_DEFAULT_LSSDESC_ANG_BINS;
    const int nLSSRadBins = (int)STEREOSEGMATCH_DEFAULT_LSSDESC_RAD_BINS;
    m_pImgDescExtractor = std::make_unique<LSS>(nLSSInnerRadius,nLSSOuterRadius,nLSSPatchSize,nLSSAngBins,nLSSRadBins);
    const cv::Size oDescWinSize = m_pImgDescExtractor->windowSize();
    m_nGridBorderSize = (size_t)std::max(m_pImgDescExtractor->borderSize(0),m_pImgDescExtractor->borderSize(1));
#elif STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY
    const int nWindowSize = int(STEREOSEGMATCH_DEFAULT_MI_WINDOW_RAD*2+1);
    const cv::Size oDescWinSize = cv::Size(nWindowSize,nWindowSize);
    m_nGridBorderSize = STEREOSEGMATCH_DEFAULT_MI_WINDOW_RAD;
#elif STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
    constexpr int nSSqrDiffKernelSize = int(STEREOSEGMATCH_DEFAULT_SSQDIFF_PATCH);
    const cv::Size oDescWinSize(nSSqrDiffKernelSize,nSSqrDiffKernelSize);
    m_nGridBorderSize = size_t(nSSqrDiffKernelSize/2);
#endif //STEREOSEGMATCH_CONFIG_USE_..._AFFINITY
    const size_t nShapeContextInnerRadius = 2;
    const size_t nShapeContextOuterRadius = STEREOSEGMATCH_DEFAULT_SCDESC_WIN_RAD;
    const size_t nShapeContextAngBins = STEREOSEGMATCH_DEFAULT_SCDESC_ANG_BINS;
    const size_t nShapeContextRadBins = STEREOSEGMATCH_DEFAULT_SCDESC_RAD_BINS;
    m_pShpDescExtractor = std::make_unique<ShapeContext>(nShapeContextInnerRadius,nShapeContextOuterRadius,nShapeContextAngBins,nShapeContextRadBins);
    lvAssert__(oDescWinSize.width<=(int)m_oGridSize[1] && oDescWinSize.height<=(int)m_oGridSize[0],"image is too small to compute descriptors with current pattern size -- need at least (%d,%d) and got (%d,%d)",oDescWinSize.width,oDescWinSize.height,(int)m_oGridSize[1],(int)m_oGridSize[0]);
    lvDbgAssert(m_nGridBorderSize<m_oGridSize[0] && m_nGridBorderSize<m_oGridSize[1]);
    lvDbgAssert(m_nGridBorderSize<(size_t)oDescWinSize.width && m_nGridBorderSize<(size_t)oDescWinSize.height);
    lvDbgAssert((size_t)std::max(m_pShpDescExtractor->borderSize(0),m_pShpDescExtractor->borderSize(1))<=m_nGridBorderSize);
    lvDbgAssert(m_aAssocCostRealAddLUT.size()==m_aAssocCostRealSumLUT.size() && m_aAssocCostRealRemLUT.size()==m_aAssocCostRealSumLUT.size());
    lvDbgAssert(m_aAssocCostApproxAddLUT.size()==m_aAssocCostRealAddLUT.size() && m_aAssocCostApproxRemLUT.size()==m_aAssocCostRealRemLUT.size());
    lvDbgAssert_(m_nMaxDispOffset+m_nDispOffsetStep<m_aAssocCostRealSumLUT.size(),"assoc cost lut size might not be large enough");
    lvDbgAssert(STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT<m_aAssocCostRealSumLUT.size());
    lvDbgAssert(STEREOSEGMATCH_UNIQUE_COST_INCR_REL(0)==0.0f);
    std::fill_n(m_aAssocCostRealAddLUT.begin(),STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT,cost_cast(0));
    std::fill_n(m_aAssocCostRealRemLUT.begin(),STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT,cost_cast(0));
    std::fill_n(m_aAssocCostRealSumLUT.begin(),STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT,cost_cast(0));
    for(size_t nIdx=STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT; nIdx<m_aAssocCostRealAddLUT.size(); ++nIdx) {
        m_aAssocCostRealAddLUT[nIdx] = cost_cast(STEREOSEGMATCH_UNIQUE_COST_INCR_REL(nIdx+1-STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT)*STEREOSEGMATCH_UNIQUE_COST_OVER_SCALE/m_nDispOffsetStep);
        m_aAssocCostRealRemLUT[nIdx] = -cost_cast(STEREOSEGMATCH_UNIQUE_COST_INCR_REL(nIdx-STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT)*STEREOSEGMATCH_UNIQUE_COST_OVER_SCALE/m_nDispOffsetStep);
        m_aAssocCostRealSumLUT[nIdx] = (nIdx==size_t(0)?cost_cast(0):(m_aAssocCostRealSumLUT[nIdx-1]+m_aAssocCostRealAddLUT[nIdx-1]));
    }
    for(size_t nIdx=0; nIdx<m_aAssocCostRealAddLUT.size(); ++nIdx) {
        // 'average' cost of removing one assoc from target pixel (not 'true' energy, but easy-to-optimize worse case)
        m_aAssocCostApproxRemLUT[nIdx] = (nIdx==size_t(0)?cost_cast(0):cost_cast(-1.0f*m_aAssocCostRealSumLUT[nIdx]/nIdx+0.5f));
        if(m_nDispOffsetStep==size_t(1))
            // if m_nDispOffsetStep==1, then use real cost of adding one assoc to target pixel
            // (target can only have one new assoc per iter, due to single label move)
            m_aAssocCostApproxAddLUT[nIdx] = m_aAssocCostRealAddLUT[nIdx];
        else {
            // otherwise, use average cost of adding 'm_nDispOffsetStep' assocs to target block
            // (i.e. the average cost of adding the max possible number of new assocs to a block per iter)
            m_aAssocCostApproxAddLUT[nIdx] = cost_cast(0);
            for(size_t nOffsetIdx=nIdx; nOffsetIdx<nIdx+m_nDispOffsetStep; ++nOffsetIdx)
                m_aAssocCostApproxAddLUT[nIdx] += m_aAssocCostRealAddLUT[std::min(nOffsetIdx,m_aAssocCostRealAddLUT.size()-1)];
            m_aAssocCostApproxAddLUT[nIdx] = cost_cast(float(m_aAssocCostApproxAddLUT[nIdx])/m_nDispOffsetStep+0.5f);
        }
    }
#if STEREOSEGMATCH_LBLSIM_USE_EXP_GRADPIVOT
    m_aLabelSimCostGradFactLUT.init(0,255,[](int nLocalGrad){
        return (float)std::exp(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE);
    });
#else //!STEREOSEGMATCH_LBLSIM_USE_EXP_GRADPIVOT
    m_aLabelSimCostGradFactLUT.init(0,255,[](int nLocalGrad){
        const float fGradPivotFact = 1.0f+(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/((nLocalGrad>=STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST)?(255-STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST):STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST));
        const float fGradScaleFact = STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE*fGradPivotFact*fGradPivotFact;
        lvDbgAssert(fGradScaleFact>=0.0f && fGradScaleFact<=4.0f*STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE);
        return fGradScaleFact;
    });
#endif //!STEREOSEGMATCH_LBLSIM_USE_EXP_GRADPIVOT
    lvDbgAssert(m_aLabelSimCostGradFactLUT.size()==size_t(256) && m_aLabelSimCostGradFactLUT.domain_offset_low()==0);
    lvDbgAssert(m_aLabelSimCostGradFactLUT.domain_index_step()==1.0 && m_aLabelSimCostGradFactLUT.domain_index_scale()==1.0);
    lvLog_(2,"\toutput disp labels:\n%s\n",lv::to_string(std::vector<OutputLabelType>(m_vStereoLabels.begin(),m_vStereoLabels.begin()+m_nRealStereoLabels)).c_str());
    const size_t nRows = m_oGridSize(0);
    const size_t nCols = m_oGridSize(1);
    static_assert(getCameraCount()==2,"bad static array size, hardcoded stuff below will break");
    size_t nTotValidNodes = 0;
    CamArray<size_t> anValidGraphNodes = {};
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        anValidGraphNodes[nCamIdx] = (size_t)cv::countNonZero(m_aROIs[nCamIdx]);
        lvAssert(anValidGraphNodes[nCamIdx]<m_oGridSize.total());
        nTotValidNodes += anValidGraphNodes[nCamIdx];
        cv::erode(m_aROIs[nCamIdx],m_aDescROIs[nCamIdx],cv::getStructuringElement(cv::MORPH_RECT,oDescWinSize),cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar_<uchar>::all(0));
    }
    const size_t nStereoUnaryFuncDataSize = anValidGraphNodes[m_nPrimaryCamIdx]*m_nStereoLabels;
    const size_t nStereoPairwFuncDataSize = anValidGraphNodes[m_nPrimaryCamIdx]*s_nPairwOrients*(m_nStereoLabels*m_nStereoLabels);
    const size_t nStereoEpipolarFuncDataSize = STEREOSEGMATCH_CONFIG_USE_EPIPOLAR_CONN?(anValidGraphNodes[m_nPrimaryCamIdx]*(int)std::pow((int)m_nStereoLabels,(int)s_nEpipolarConn)):size_t(0); // stride not taken into account here
    const size_t nStereoFuncDataSize = nStereoUnaryFuncDataSize+nStereoPairwFuncDataSize+nStereoEpipolarFuncDataSize;
    CamArray<size_t> anResegmUnaryFuncDataSize={},anResegmPairwFuncDataSize={},anResegmTemporalFuncDataSize={},anResegmFuncDataSize={};
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        anResegmUnaryFuncDataSize[nCamIdx] = anValidGraphNodes[nCamIdx]*m_nResegmLabels;
        anResegmPairwFuncDataSize[nCamIdx] = anValidGraphNodes[nCamIdx]*s_nPairwOrients*(m_nResegmLabels*m_nResegmLabels);
        anResegmTemporalFuncDataSize[nCamIdx] = STEREOSEGMATCH_DEFAULT_TEMPORAL_DEPTH?(anValidGraphNodes[nCamIdx]*(int)std::pow((int)m_nResegmLabels,(int)s_nTemporalConn)):size_t(0); // stride not taken into account here
        anResegmFuncDataSize[nCamIdx] = anResegmUnaryFuncDataSize[nCamIdx]+anResegmPairwFuncDataSize[nCamIdx]+anResegmTemporalFuncDataSize[nCamIdx];
    }
    const size_t nTotResegmUnaryFuncDataSize = std::accumulate(anResegmUnaryFuncDataSize.begin(),anResegmUnaryFuncDataSize.end(),size_t(0));
    const size_t nTotResegmPairwFuncDataSize = std::accumulate(anResegmPairwFuncDataSize.begin(),anResegmPairwFuncDataSize.end(),size_t(0));
    const size_t nTotResegmTemporalFuncDataSize = std::accumulate(anResegmTemporalFuncDataSize.begin(),anResegmTemporalFuncDataSize.end(),size_t(0));
    const size_t nTotResegmFuncDataSize = std::accumulate(anResegmFuncDataSize.begin(),anResegmFuncDataSize.end(),size_t(0));
    lvAssert(nTotResegmFuncDataSize==(nTotResegmUnaryFuncDataSize+nTotResegmPairwFuncDataSize+nTotResegmTemporalFuncDataSize));
    const size_t nModelSize = ((nStereoFuncDataSize+nTotResegmFuncDataSize)*sizeof(ValueType)/*+...externals unaccounted for, so x2*/*2);
    lvLog_(1,"Expecting total mem requirement < %zu mb\n\t(~%zu mb for stereo graph, ~%zu mb for resegm graphs)",nModelSize/1024/1024,nStereoFuncDataSize/1024/1024,nTotResegmFuncDataSize/1024/1024);
    lvAssert__(nModelSize<(CACHE_MAX_SIZE_GB*1024*1024*1024),"too many nodes/labels; model is unlikely to fit in memory (estimated: %zu mb)",nModelSize/1024/1024);
    lvLog(2,"Building stereo & resegm graph models...");
    lv::StopWatch oLocalTimer;
    const std::array<int,3> anAssocMapDims{int(m_oGridSize[0]),int((m_oGridSize[1]+m_nMaxDispOffset/*for oob usage*/)/m_nDispOffsetStep),int(m_nRealStereoLabels*m_nDispOffsetStep)};
    m_oAssocCounts.create(2,anAssocMapDims.data());
    m_oAssocMap.create(3,anAssocMapDims.data());
#if STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF
    m_oStereoUnaryCosts.create(m_oGridSize);
#elif STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF
    m_oStereoUnaryCosts.create(int(m_nStereoLabels),int(anValidGraphNodes[m_nPrimaryCamIdx])); // @@@ flip for optim?
#endif //STEREOSEGMATCH_CONFIG_USE_..._STEREO_INF
    const size_t nLayerSize = m_oGridSize.total();
    m_vValidStereoLUTNodeIdxs.reserve(anValidGraphNodes[m_nPrimaryCamIdx]);
    m_vStereoNodeInfos.resize(nLayerSize);
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        m_aStereoLabelings[nCamIdx].create(m_oGridSize);
        m_aResegmLabelings[nCamIdx].create(m_oGridSize);
        m_aGMMCompAssignMap[nCamIdx].create(m_oGridSize);
        m_aResegmUnaryCosts[nCamIdx].create(m_oGridSize);
        m_avValidResegmLUTNodeIdxs[nCamIdx].reserve(anValidGraphNodes[nCamIdx]*s_nTemporalLayers);
        m_avResegmNodeInfos[nCamIdx].resize(nLayerSize*s_nTemporalLayers);
    }
    m_nValidStereoGraphNodes = size_t(0);
    m_anValidResegmGraphNodes = CamArray<size_t>{};
    for(int nRowIdx = 0; nRowIdx<(int)nRows; ++nRowIdx) {
        for(int nColIdx = 0; nColIdx<(int)nCols; ++nColIdx) {
            // @@@@ prep cliques here? count?
            const size_t nLUTNodeIdx = nRowIdx*nCols+nColIdx;
            m_vStereoNodeInfos[nLUTNodeIdx].nRowIdx = nRowIdx;
            m_vStereoNodeInfos[nLUTNodeIdx].nColIdx = nColIdx;
            m_vStereoNodeInfos[nLUTNodeIdx].bValidGraphNode = m_aROIs[m_nPrimaryCamIdx](nRowIdx,nColIdx)>0;
            m_vStereoNodeInfos[nLUTNodeIdx].bNearBorders = m_aDescROIs[m_nPrimaryCamIdx](nRowIdx,nColIdx)==0;
            m_vStereoNodeInfos[nLUTNodeIdx].nUnaryFactID = SIZE_MAX;
            m_vStereoNodeInfos[nLUTNodeIdx].pUnaryFunc = nullptr;
            std::fill_n(m_vStereoNodeInfos[nLUTNodeIdx].afPairwWeights.begin(),s_nPairwOrients,0.0f);
            if(m_vStereoNodeInfos[nLUTNodeIdx].bValidGraphNode) {
                m_vStereoNodeInfos[nLUTNodeIdx].nGraphNodeIdx = m_nValidStereoGraphNodes++;
                m_vValidStereoLUTNodeIdxs.push_back(nLUTNodeIdx);
            }
            for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
                const bool bValidGraphNode = m_aROIs[nCamIdx](nRowIdx,nColIdx)>0;
                const bool bNearBorders = m_aDescROIs[nCamIdx](nRowIdx,nColIdx)==0;
                for(size_t nLayerIdx=0; nLayerIdx<s_nTemporalLayers; ++nLayerIdx) {
                    const size_t nLayeredLUTNodeIdx = nLUTNodeIdx+nLayerIdx*nLayerSize;
                    m_avResegmNodeInfos[nCamIdx][nLayeredLUTNodeIdx].nRowIdx = nRowIdx;
                    m_avResegmNodeInfos[nCamIdx][nLayeredLUTNodeIdx].nColIdx = nColIdx;
                    m_avResegmNodeInfos[nCamIdx][nLayeredLUTNodeIdx].bValidGraphNode = bValidGraphNode;
                    m_avResegmNodeInfos[nCamIdx][nLayeredLUTNodeIdx].bNearBorders = bNearBorders;
                    m_avResegmNodeInfos[nCamIdx][nLayeredLUTNodeIdx].nUnaryFactID = SIZE_MAX;
                    m_avResegmNodeInfos[nCamIdx][nLayeredLUTNodeIdx].pUnaryFunc = nullptr;
                    std::fill_n(m_avResegmNodeInfos[nCamIdx][nLayeredLUTNodeIdx].afPairwWeights.begin(),s_nPairwOrients,0.0f);
                    if(m_avResegmNodeInfos[nCamIdx][nLayeredLUTNodeIdx].bValidGraphNode) {
                        m_avResegmNodeInfos[nCamIdx][nLayeredLUTNodeIdx].nGraphNodeIdx = m_anValidResegmGraphNodes[nCamIdx]++;
                        m_avValidResegmLUTNodeIdxs[nCamIdx].push_back(nLayeredLUTNodeIdx);
                    }
                }
            }
        }
    }
    lvAssert(m_nValidStereoGraphNodes==anValidGraphNodes[m_nPrimaryCamIdx]);
    lvAssert(m_vValidStereoLUTNodeIdxs.size()==anValidGraphNodes[m_nPrimaryCamIdx]);
    m_aStereoFuncsData = std::make_unique<ValueType[]>(nStereoFuncDataSize);
    m_pStereoUnaryFuncsDataBase = m_aStereoFuncsData.get();
    m_pStereoPairwFuncsDataBase = m_pStereoUnaryFuncsDataBase+nStereoUnaryFuncDataSize;
    m_pStereoEpipolarFuncsDataBase = m_pStereoPairwFuncsDataBase+nStereoPairwFuncDataSize;
    lvAssert(std::accumulate(m_anValidResegmGraphNodes.begin(),m_anValidResegmGraphNodes.end(),size_t(0))==nTotValidNodes*s_nTemporalLayers);
    m_aResegmFuncsData = std::make_unique<ValueType[]>(nTotResegmFuncDataSize);
    ValueType* pLastResegmFuncsDataEnd = m_aResegmFuncsData.get();
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        lvAssert(m_anValidResegmGraphNodes[nCamIdx]==anValidGraphNodes[nCamIdx]*s_nTemporalLayers);
        lvAssert(m_avValidResegmLUTNodeIdxs[nCamIdx].size()==anValidGraphNodes[nCamIdx]*s_nTemporalLayers);
        m_apResegmUnaryFuncsDataBase[nCamIdx] = pLastResegmFuncsDataEnd;
        m_apResegmPairwFuncsDataBase[nCamIdx] = m_apResegmUnaryFuncsDataBase[nCamIdx]+anResegmUnaryFuncDataSize[nCamIdx];
        m_apResegmTemporalFuncsDataBase[nCamIdx] = m_apResegmPairwFuncsDataBase[nCamIdx]+anResegmPairwFuncDataSize[nCamIdx];
        pLastResegmFuncsDataEnd = STEREOSEGMATCH_DEFAULT_TEMPORAL_DEPTH?(m_apResegmTemporalFuncsDataBase[nCamIdx]+anResegmTemporalFuncDataSize[nCamIdx]):(m_apResegmTemporalFuncsDataBase[nCamIdx]);
    }
    buildStereoModel();
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx)
        buildResegmModel(nCamIdx,false);
    lvLog_(2,"Graph models built in %f second(s).\n",oLocalTimer.tock());
}

void StereoSegmMatcher::GraphModelData::resetStereoLabelings(size_t nCamIdx) {
    lvDbgExceptionWatch;
    lvDbgAssert_(nCamIdx<getCameraCount(),"bad input cam index");
    const bool bIsPrimaryCam = (nCamIdx==m_nPrimaryCamIdx);
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    lvIgnore(nRows); lvIgnore(nCols);
    std::fill(m_aStereoLabelings[nCamIdx].begin(),m_aStereoLabelings[nCamIdx].end(),m_nDontCareLabelIdx);
    lvDbgAssert(m_nValidStereoGraphNodes==m_vValidStereoLUTNodeIdxs.size());
    if(bIsPrimaryCam) {
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidStereoGraphNodes; ++nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_vValidStereoLUTNodeIdxs[nGraphNodeIdx];
            const StereoNodeInfo& oNode = m_vStereoNodeInfos[nLUTNodeIdx];
            lvDbgAssert(oNode.bValidGraphNode && oNode.nGraphNodeIdx==nGraphNodeIdx);
            lvDbgAssert(oNode.nUnaryFactID<m_pStereoModel->numberOfFactors());
            lvDbgAssert(m_pStereoModel->numberOfLabels(oNode.nUnaryFactID)==m_nStereoLabels);
            InternalLabelType nEvalLabel = m_aStereoLabelings[nCamIdx](oNode.nRowIdx,oNode.nColIdx) = 0;
            const ExplicitFunction& vUnaryStereoLUT = *oNode.pUnaryFunc;
            ValueType fOptimalEnergy = vUnaryStereoLUT(nEvalLabel);
            for(nEvalLabel=1; nEvalLabel<m_nStereoLabels; ++nEvalLabel) {
                const ValueType fCurrEnergy = vUnaryStereoLUT(nEvalLabel);
                if(fOptimalEnergy>fCurrEnergy) {
                    m_aStereoLabelings[nCamIdx](oNode.nRowIdx,oNode.nColIdx) = nEvalLabel;
                    fOptimalEnergy = fCurrEnergy;
                }
            }
        }
    }
    else {
        for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
            for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
                if(!m_aROIs[nCamIdx](nRowIdx,nColIdx))
                    continue;
                std::map<InternalLabelType,size_t> mWTALookupCounts;
                for(InternalLabelType nLookupLabel=0; nLookupLabel<m_nRealStereoLabels; ++nLookupLabel) {
                    const int nOffsetColIdx = getOffsetColIdx(nCamIdx,nColIdx,nLookupLabel);
                    if(nOffsetColIdx>=0 && nOffsetColIdx<nCols && m_aROIs[m_nPrimaryCamIdx](nRowIdx,nOffsetColIdx))
                        if(m_aStereoLabelings[m_nPrimaryCamIdx](nRowIdx,nOffsetColIdx)==nLookupLabel)
                            ++mWTALookupCounts[nLookupLabel];
                }
                auto pWTAPairIter = std::max_element(mWTALookupCounts.begin(),mWTALookupCounts.end(),[](const auto& p1, const auto& p2) {
                    return p1.second<p2.second;
                });
                if(pWTAPairIter!=mWTALookupCounts.end() && pWTAPairIter->second>size_t(0))
                    m_aStereoLabelings[nCamIdx](nRowIdx,nColIdx) = pWTAPairIter->first;
            }
        }
    }
    if(bIsPrimaryCam) {
        m_oAssocCounts = (AssocCountType)0;
        m_oAssocMap = (AssocIdxType)-1;
        std::vector<int> vLabelCounts(m_nStereoLabels,0);
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidStereoGraphNodes; ++nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_vValidStereoLUTNodeIdxs[nGraphNodeIdx];
            const StereoNodeInfo& oNode = m_vStereoNodeInfos[nLUTNodeIdx];
            lvDbgAssert(nLUTNodeIdx==(oNode.nRowIdx*m_oGridSize[1]+oNode.nColIdx));
            const InternalLabelType nLabel = ((InternalLabelType*)m_aStereoLabelings[nCamIdx].data)[nLUTNodeIdx];
            if(nLabel<m_nDontCareLabelIdx) // both special labels avoided here
                addAssoc(oNode.nRowIdx,oNode.nColIdx,nLabel);
            ++vLabelCounts[nLabel];
        }
        m_vStereoLabelOrdering = lv::sort_indices<InternalLabelType>(vLabelCounts,[&vLabelCounts](int a, int b){return vLabelCounts[a]>vLabelCounts[b];});
        lvDbgAssert(lv::unique(m_vStereoLabelOrdering.begin(),m_vStereoLabelOrdering.end())==lv::make_range(InternalLabelType(0),InternalLabelType(m_nStereoLabels-1)));
        // note: sospd might not follow this label order if using alpha heights strategy (reimpl to use same strat in every solver?) @@@
    }
    else {
        for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
            for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
                InternalLabelType& nCurrLabel = m_aStereoLabelings[nCamIdx](nRowIdx,nColIdx);
                if(nCurrLabel==m_nDontCareLabelIdx && m_aROIs[nCamIdx](nRowIdx,nColIdx)) {
                    for(int nOffset=0; nOffset<=(int)m_nMaxDispOffset; ++nOffset) {
                        const int nOffsetColIdx_pos = nColIdx+nOffset;
                        if(nOffsetColIdx_pos>=0 && nOffsetColIdx_pos<nCols && m_aROIs[nCamIdx](nRowIdx,nOffsetColIdx_pos)) {
                            const InternalLabelType& nNewLabel = m_aStereoLabelings[nCamIdx](nRowIdx,nOffsetColIdx_pos);
                            if(nNewLabel!=m_nDontCareLabelIdx) {
                                nCurrLabel = nNewLabel;
                                break;
                            }
                        }
                        const int nOffsetColIdx_neg = nColIdx-nOffset;
                        if(nOffsetColIdx_neg>=0 && nOffsetColIdx_neg<nCols && m_aROIs[nCamIdx](nRowIdx,nOffsetColIdx_neg)) {
                            const InternalLabelType& nNewLabel = m_aStereoLabelings[nCamIdx](nRowIdx,nOffsetColIdx_neg);
                            if(nNewLabel!=m_nDontCareLabelIdx) {
                                nCurrLabel = nNewLabel;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

void StereoSegmMatcher::GraphModelData::buildStereoModel() {
    lvDbgExceptionWatch;
    lvLog(2,"\tadding base functions to stereo graph...");
    const size_t nStereoMaxFactorsPerNode = /*unary*/1 + /*pairw*/s_nPairwOrients + /*ho*/STEREOSEGMATCH_CONFIG_USE_EPIPOLAR_CONN?size_t(1):size_t(0); // stride not taken into account here
    m_pStereoModel = std::make_unique<StereoModelType>(StereoSpaceType(m_nValidStereoGraphNodes,(InternalLabelType)m_nStereoLabels),nStereoMaxFactorsPerNode);
    m_pStereoModel->reserveFunctions<ExplicitFunction>(m_nValidStereoGraphNodes*nStereoMaxFactorsPerNode);
    const std::vector<size_t> aPairwStereoFuncDims(s_nPairwOrients,m_nStereoLabels);
    m_oStereoPairwFuncID_base = m_pStereoModel->addFunction(ExplicitAllocFunction(aPairwStereoFuncDims.begin(),aPairwStereoFuncDims.end()));
    ExplicitAllocFunction& oStereoBaseFunc = m_pStereoModel->getFunction<ExplicitAllocFunction>(m_oStereoPairwFuncID_base);
    lvDbgAssert(oStereoBaseFunc.size()==m_nStereoLabels*m_nStereoLabels);
    for(InternalLabelType nLabelIdx1=0; nLabelIdx1<m_nRealStereoLabels; ++nLabelIdx1) {
        for(InternalLabelType nLabelIdx2=0; nLabelIdx2<m_nRealStereoLabels; ++nLabelIdx2) {
            const OutputLabelType nRealLabel1 = getRealLabel(nLabelIdx1);
            const OutputLabelType nRealLabel2 = getRealLabel(nLabelIdx2);
            const int nRealLabelDiff = std::min(std::abs((int)nRealLabel1-(int)nRealLabel2),STEREOSEGMATCH_LBLSIM_STEREO_MAXDIFF_CST);
            oStereoBaseFunc(nLabelIdx1,nLabelIdx2) = cost_cast(nRealLabelDiff*nRealLabelDiff);
        }
    }
    for(size_t nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx) {
        // @@@@ outside-roi-dc to inside-roi-something is ok (0 cost)
        oStereoBaseFunc(m_nDontCareLabelIdx,nLabelIdx) = cost_cast(10000);
        oStereoBaseFunc(m_nOccludedLabelIdx,nLabelIdx) = cost_cast(10000); // @@@@ STEREOSEGMATCH_LBLSIM_COST_MAXOCCL scaled down if other label is high disp
        oStereoBaseFunc(nLabelIdx,m_nDontCareLabelIdx) = cost_cast(10000);
        oStereoBaseFunc(nLabelIdx,m_nOccludedLabelIdx) = cost_cast(10000); // @@@@ STEREOSEGMATCH_LBLSIM_COST_MAXOCCL scaled down if other label is high disp
    }
    oStereoBaseFunc(m_nDontCareLabelIdx,m_nDontCareLabelIdx) = cost_cast(0);
    oStereoBaseFunc(m_nOccludedLabelIdx,m_nOccludedLabelIdx) = cost_cast(0);
    lvLog(2,"\tadding unary factors to stereo graph...");
    m_nStereoUnaryFactCount = size_t(0);
    m_vStereoUnaryFuncs.clear();
    m_vStereoUnaryFuncs.reserve(m_nValidStereoGraphNodes);
    const std::array<size_t,1> aUnaryStereoFuncDims = {m_nStereoLabels};
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidStereoGraphNodes; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_vValidStereoLUTNodeIdxs[nGraphNodeIdx];
        StereoNodeInfo& oNode = m_vStereoNodeInfos[nLUTNodeIdx];
        oNode.vpCliques.clear();
        m_vStereoUnaryFuncs.push_back(m_pStereoModel->addFunctionWithRefReturn(ExplicitFunction()));
        FuncPairType& oStereoFunc = m_vStereoUnaryFuncs.back();
        lvDbgAssert((&m_pStereoModel->getFunction<ExplicitFunction>(oStereoFunc.first))==(&oStereoFunc.second));
        oStereoFunc.second.assign(aUnaryStereoFuncDims.begin(),aUnaryStereoFuncDims.end(),m_pStereoUnaryFuncsDataBase+(nGraphNodeIdx*m_nStereoLabels));
        lvDbgAssert(oStereoFunc.second.strides(0)==1); // expect no padding
        const std::array<size_t,1> aGraphNodeIndices = {nGraphNodeIdx};
        oNode.nUnaryFactID = m_pStereoModel->addFactorNonFinalized(oStereoFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
        lvDbgAssert(FuncIdentifType((*m_pStereoModel)[oNode.nUnaryFactID].functionIndex(),(*m_pStereoModel)[oNode.nUnaryFactID].functionType())==oStereoFunc.first);
        lvDbgAssert(m_pStereoModel->operator[](oNode.nUnaryFactID).numberOfVariables()==size_t(1));
        lvDbgAssert(oNode.nUnaryFactID==m_nStereoUnaryFactCount);
        oNode.pUnaryFunc = &oStereoFunc.second;
        ++m_nStereoUnaryFactCount;
    }
    lvLog(2,"\tadding pairwise factors to stereo graph...");
    m_nStereoPairwFactCount = size_t(0);
    for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
        m_avStereoPairwFuncs[nOrientIdx].clear();
        m_avStereoPairwFuncs[nOrientIdx].reserve(m_nValidStereoGraphNodes);
    }
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidStereoGraphNodes; ++nGraphNodeIdx) {
        const size_t nBaseLUTNodeIdx = m_vValidStereoLUTNodeIdxs[nGraphNodeIdx];
        NodeInfo& oBaseNode = m_vStereoNodeInfos[nBaseLUTNodeIdx];
        lvDbgAssert(oBaseNode.bValidGraphNode);
        oBaseNode.aPairwCliques = {}; // reset to default state
        const auto lPairwCliqueCreator = [&](size_t nOrientIdx, size_t nOffsetLUTNodeIdx) {
            const NodeInfo& oOffsetNode = m_vStereoNodeInfos[nOffsetLUTNodeIdx];
            if(oOffsetNode.bValidGraphNode) {
                m_avStereoPairwFuncs[nOrientIdx].push_back(m_pStereoModel->addFunctionWithRefReturn(ExplicitFunction()));
                FuncPairType& oStereoFunc = m_avStereoPairwFuncs[nOrientIdx].back();
                oStereoFunc.second.assign(aPairwStereoFuncDims.begin(),aPairwStereoFuncDims.end(),m_pStereoPairwFuncsDataBase+((nGraphNodeIdx*s_nPairwOrients+nOrientIdx)*m_nStereoLabels*m_nStereoLabels));
                lvDbgAssert(oStereoFunc.second.strides(0)==1 && oStereoFunc.second.strides(1)==m_nStereoLabels); // expect last-idx-major
                lvDbgAssert((&m_pStereoModel->getFunction<ExplicitFunction>(oStereoFunc.first))==(&oStereoFunc.second));
                PairwClique& oPairwClique = oBaseNode.aPairwCliques[nOrientIdx];
                oPairwClique.m_bValid = true;
                const std::array<size_t,2> aLUTNodeIndices = {nBaseLUTNodeIdx,nOffsetLUTNodeIdx};
                oPairwClique.m_anLUTNodeIdxs = aLUTNodeIndices;
                const std::array<size_t,2> aGraphNodeIndices = {nGraphNodeIdx,oOffsetNode.nGraphNodeIdx};
                oPairwClique.m_anGraphNodeIdxs = aGraphNodeIndices;
                oPairwClique.m_nGraphFactorId = m_pStereoModel->addFactorNonFinalized(oStereoFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
                lvDbgAssert(FuncIdentifType((*m_pStereoModel)[oPairwClique.m_nGraphFactorId].functionIndex(),(*m_pStereoModel)[oPairwClique.m_nGraphFactorId].functionType())==oStereoFunc.first);
                lvDbgAssert(m_pStereoModel->operator[](oPairwClique.m_nGraphFactorId).numberOfVariables()==size_t(2));
                lvDbgAssert(oPairwClique.m_nGraphFactorId==m_nStereoUnaryFactCount+m_nStereoPairwFactCount);
                oPairwClique.m_pGraphFunctionPtr = &oStereoFunc.second;
                oBaseNode.vpCliques.push_back(&oPairwClique);
                ++m_nStereoPairwFactCount;
            }
        };
        if((oBaseNode.nRowIdx+1)<(int)m_oGridSize[0]) { // vertical pair
            lvDbgAssert(int((oBaseNode.nRowIdx+1)*m_oGridSize[1]+oBaseNode.nColIdx)==int(nBaseLUTNodeIdx+m_oGridSize[1]));
            lPairwCliqueCreator(size_t(0),nBaseLUTNodeIdx+m_oGridSize[1]);
        }
        if((oBaseNode.nColIdx+1)<(int)m_oGridSize[1]) { // horizontal pair
            lvDbgAssert(int(oBaseNode.nRowIdx*m_oGridSize[1]+oBaseNode.nColIdx+1)==int(nBaseLUTNodeIdx+1));
            lPairwCliqueCreator(size_t(1),nBaseLUTNodeIdx+1);
        }
        static_assert(s_nPairwOrients==2,"missing some pairw instantiations here");
    }
    m_nStereoEpipolarFactCount = size_t(0);
    m_vStereoEpipolarFuncs.clear();
    if(STEREOSEGMATCH_CONFIG_USE_EPIPOLAR_CONN) {
        lvLog(2,"\tadding epipolar factors to stereo graph...");
        //m_vStereoEpipolarFuncs.reserve(m_nValidStereoGraphNodes); // stride not taken into account here
        //    oBaseNode.oEpipolarClique = {};
        //    ...
        //    oBaseNode.vpCliques.push_back(&oBaseNode.oEpipolarClique);
        lvAssert(false); // missing impl @@@
    }
    m_pStereoModel->finalize();
    m_pStereoInf = std::make_unique<StereoGraphInference>(m_nPrimaryCamIdx,*this);
    if(lv::getVerbosity()>=2) {
        lvCout << "Stereo model [" << m_nPrimaryCamIdx << "] :\n";
        lv::gm::printModelInfo(*m_pStereoModel);
        lvCout << "\n";
    }
}

void StereoSegmMatcher::GraphModelData::updateStereoModel(bool bInit) {
    static_assert(getCameraCount()==2,"bad static array size, hardcoded stuff below (incl xor) will break");
    lvDbgExceptionWatch;
    lvDbgAssert_(m_nPrimaryCamIdx<getCameraCount(),"bad primary cam index");
    lvDbgAssert(m_pStereoModel && m_pStereoModel->numberOfVariables()==m_nValidStereoGraphNodes);
    lvDbgAssert(m_nValidStereoGraphNodes==m_vValidStereoLUTNodeIdxs.size());
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    lvIgnore(nRows); lvIgnore(nCols);
    const cv::Mat_<float> oImgAffinity = m_vNextFeats[FeatPack_ImgAffinity];
    const cv::Mat_<float> oShpAffinity = m_vNextFeats[FeatPack_ShpAffinity];
    const cv::Mat_<float> oImgSaliency = m_vNextFeats[FeatPack_ImgSaliency];
    const cv::Mat_<float> oShpSaliency = m_vNextFeats[FeatPack_ShpSaliency];
    lvDbgAssert(oImgAffinity.dims==3 && oImgAffinity.size[0]==nRows && oImgAffinity.size[1]==nCols && oImgAffinity.size[2]==(int)m_nRealStereoLabels);
    lvDbgAssert(oShpAffinity.dims==3 && oShpAffinity.size[0]==nRows && oShpAffinity.size[1]==nCols && oShpAffinity.size[2]==(int)m_nRealStereoLabels);
    lvDbgAssert(oImgSaliency.dims==2 && oImgSaliency.size[0]==nRows && oImgSaliency.size[1]==nCols);
    lvDbgAssert(oShpSaliency.dims==2 && oShpSaliency.size[0]==nRows && oShpSaliency.size[1]==nCols);
    const cv::Mat_<uchar> oGradY = m_vNextFeats[FeatPackOffset*m_nPrimaryCamIdx+FeatPackOffset_GradY];
    const cv::Mat_<uchar> oGradX = m_vNextFeats[FeatPackOffset*m_nPrimaryCamIdx+FeatPackOffset_GradX];
    const cv::Mat_<uchar> oGradMag = m_vNextFeats[FeatPackOffset*m_nPrimaryCamIdx+FeatPackOffset_GradMag];
    lvDbgAssert(m_oGridSize==oGradY.size && m_oGridSize==oGradX.size && m_oGridSize==oGradMag.size);
    /*const int nMinGradThrs = STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-5;
    const int nMaxGradThrs = STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST+5;
    cv::imshow("oGradY",(oGradY>nMinGradThrs)&(oGradY<nMaxGradThrs));
    cv::imshow("oGradX",(oGradX>nMinGradThrs)&(oGradX<nMaxGradThrs));
    cv::waitKey(0);*/
    lvLog(4,"Updating stereo graph model energy terms based on new features...");
    lv::StopWatch oLocalTimer;
#if STEREOSEGMATCH_CONFIG_USE_PROGRESS_BARS
    lv::ProgressBarManager oProgressBarMgr("\tprogress:");
#endif //STEREOSEGMATCH_CONFIG_USE_PROGRESS_BARS
#if USING_OPENMP
    #pragma omp parallel for
#endif //USING_OPENMP
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidStereoGraphNodes; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_vValidStereoLUTNodeIdxs[nGraphNodeIdx];
        StereoNodeInfo& oNode = m_vStereoNodeInfos[nLUTNodeIdx];
        const int nRowIdx = oNode.nRowIdx;
        const int nColIdx = oNode.nColIdx;
        // update unary terms for each grid node
        lvDbgAssert(nLUTNodeIdx==(oNode.nRowIdx*m_oGridSize[1]+oNode.nColIdx));
        lvDbgAssert(oNode.nUnaryFactID!=SIZE_MAX && oNode.nUnaryFactID<m_nStereoUnaryFactCount && oNode.pUnaryFunc);
        lvDbgAssert(m_pStereoModel->operator[](oNode.nUnaryFactID).numberOfVariables()==size_t(1));
        ExplicitFunction& vUnaryStereoLUT = *oNode.pUnaryFunc;
        lvDbgAssert(vUnaryStereoLUT.dimension()==1 && vUnaryStereoLUT.size()==m_nStereoLabels);
        lvDbgAssert__(oImgSaliency(nRowIdx,nColIdx)>=-1e-6f && oImgSaliency(nRowIdx,nColIdx)<=1.0f+1e-6f,"fImgSaliency = %1.10f @ [%d,%d]",oImgSaliency(nRowIdx,nColIdx),nRowIdx,nColIdx);
        lvDbgAssert__(oShpSaliency(nRowIdx,nColIdx)>=-1e-6f && oShpSaliency(nRowIdx,nColIdx)<=1.0f+1e-6f,"fShpSaliency = %1.10f @ [%d,%d]",oShpSaliency(nRowIdx,nColIdx),nRowIdx,nColIdx);
        const float fImgSaliency = std::max(oImgSaliency(nRowIdx,nColIdx),0.0f);
        const float fShpSaliency = std::max(oShpSaliency(nRowIdx,nColIdx),0.0f);
        for(InternalLabelType nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx) {
            vUnaryStereoLUT(nLabelIdx) = cost_cast(0);
            const int nOffsetColIdx = getOffsetColIdx(m_nPrimaryCamIdx,nColIdx,nLabelIdx);
            if(nOffsetColIdx>=0 && nOffsetColIdx<nCols && m_aROIs[m_nPrimaryCamIdx^1/*@@@ cleanup for n-view?*/](nRowIdx,nOffsetColIdx)) {
                const float fImgAffinity = oImgAffinity(nRowIdx,nColIdx,nLabelIdx);
                const float fShpAffinity = oShpAffinity(nRowIdx,nColIdx,nLabelIdx);
                lvDbgAssert__(fImgAffinity>=0.0f,"fImgAffinity = %1.10f @ [%d,%d]",fImgAffinity,nRowIdx,nColIdx);
                lvDbgAssert__(fShpAffinity>=0.0f,"fShpAffinity = %1.10f @ [%d,%d]",fShpAffinity,nRowIdx,nColIdx);
                vUnaryStereoLUT(nLabelIdx) += cost_cast(fImgAffinity*fImgSaliency*STEREOSEGMATCH_IMGSIM_COST_DESC_SCALE);
                vUnaryStereoLUT(nLabelIdx) += cost_cast(fShpAffinity*fShpSaliency*STEREOSEGMATCH_SHPSIM_COST_DESC_SCALE);
                vUnaryStereoLUT(nLabelIdx) = std::min(vUnaryStereoLUT(nLabelIdx),STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
            }
            else {
                vUnaryStereoLUT(nLabelIdx) = STEREOSEGMATCH_UNARY_COST_OOB_CST;
            }
        }
        vUnaryStereoLUT(m_nDontCareLabelIdx) = cost_cast(10000); // @@@@ check roi, if dc set to 0, otherwise set to inf
        vUnaryStereoLUT(m_nOccludedLabelIdx) = cost_cast(10000);//STEREOSEGMATCH_IMGSIM_COST_OCCLUDED_CST;
        if(bInit) { // inter-spectral pairwise/epipolar term updates do not change w.r.t. segm or stereo updates
            ExplicitAllocFunction& vPairwiseStereoBaseFunc = m_pStereoModel->getFunction<ExplicitAllocFunction>(m_oStereoPairwFuncID_base);
            for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
                PairwClique& oPairwClique = oNode.aPairwCliques[nOrientIdx];
                if(oPairwClique) {
                    lvDbgAssert(oPairwClique.m_nGraphFactorId>=m_nStereoUnaryFactCount);
                    lvDbgAssert(oPairwClique.m_nGraphFactorId<m_nStereoUnaryFactCount+m_nStereoPairwFactCount);
                    lvDbgAssert(m_pStereoModel->operator[](oPairwClique.m_nGraphFactorId).numberOfVariables()==size_t(2));
                    lvDbgAssert(oPairwClique.m_anGraphNodeIdxs[0]!=SIZE_MAX && oPairwClique.m_anLUTNodeIdxs[0]!=SIZE_MAX);
                    lvDbgAssert(oPairwClique.m_anGraphNodeIdxs[1]!=SIZE_MAX && oPairwClique.m_anLUTNodeIdxs[1]!=SIZE_MAX);
                    lvDbgAssert(m_vStereoNodeInfos[oPairwClique.m_anLUTNodeIdxs[1]].bValidGraphNode && oPairwClique.m_pGraphFunctionPtr);
                    ExplicitFunction& vPairwiseStereoFunc = *oPairwClique.m_pGraphFunctionPtr;
                    lvDbgAssert(vPairwiseStereoFunc.dimension()==2 && vPairwiseStereoFunc.size()==m_nStereoLabels*m_nStereoLabels);
                    const int nLocalGrad = (int)((nOrientIdx==0)?oGradY:(nOrientIdx==1)?oGradX:oGradMag)(nRowIdx,nColIdx);
                    const float fGradScaleFact = m_aLabelSimCostGradFactLUT.eval_raw(nLocalGrad);
                    lvDbgAssert(fGradScaleFact==(float)std::exp(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE));
                    const float fPairwWeight = (float)(fGradScaleFact*STEREOSEGMATCH_LBLSIM_STEREO_SCALE_CST); // should be constant & uncapped for use in fastpd/bcd
                    oNode.afPairwWeights[nOrientIdx] = fPairwWeight;
                    // all stereo pairw functions are identical, but weighted differently (see base init in constructor)
                    for(InternalLabelType nLabelIdx1=0; nLabelIdx1<m_nStereoLabels; ++nLabelIdx1)
                        for(InternalLabelType nLabelIdx2=0; nLabelIdx2<m_nStereoLabels; ++nLabelIdx2)
                            vPairwiseStereoFunc(nLabelIdx1,nLabelIdx2) = cost_cast(vPairwiseStereoBaseFunc(nLabelIdx1,nLabelIdx2)*fPairwWeight);
                }
            }
            // @@@ add epipolar terms update here
            lvAssert(!STEREOSEGMATCH_CONFIG_USE_EPIPOLAR_CONN); // missing impl @@@
        }
    #if STEREOSEGMATCH_CONFIG_USE_PROGRESS_BARS
        if(lv::getVerbosity()>=3)
            oProgressBarMgr.update(float(nGraphNodeIdx)/m_anValidGraphNodes[nCamIdx]);
    #endif //STEREOSEGMATCH_CONFIG_USE_PROGRESS_BARS
    }
    lvLog_(4,"Stereo graph model energy terms update completed in %f second(s).",oLocalTimer.tock());
}

void StereoSegmMatcher::GraphModelData::buildResegmModel(size_t nCamIdx, bool bUseTemporalLinks) {
    lvLog(2,"\tadding base functions to resegm graph...");
    const size_t nResegmMaxFactorsPerNode = /*unary*/1 + /*pairw*/s_nPairwOrients + /*ho*/STEREOSEGMATCH_DEFAULT_TEMPORAL_DEPTH?size_t(1):size_t(0); // stride not taken into account here
    const std::vector<size_t> aPairwResegmFuncDims(s_nPairwOrients,m_nResegmLabels);
    m_apResegmModels[nCamIdx] = std::make_unique<ResegmModelType>(ResegmSpaceType(m_anValidResegmGraphNodes[nCamIdx]),nResegmMaxFactorsPerNode);
    m_apResegmModels[nCamIdx]->reserveFunctions<ExplicitFunction>(m_anValidResegmGraphNodes[nCamIdx]*nResegmMaxFactorsPerNode);
    m_aResegmPairwFuncIDs_base[nCamIdx] = m_apResegmModels[nCamIdx]->addFunction(ExplicitAllocFunction(aPairwResegmFuncDims.begin(),aPairwResegmFuncDims.end()));
    ExplicitAllocFunction& oResegmBaseFunc = m_apResegmModels[nCamIdx]->getFunction<ExplicitAllocFunction>(m_aResegmPairwFuncIDs_base[nCamIdx]);
    lvDbgAssert(oResegmBaseFunc.size()==m_nResegmLabels*m_nResegmLabels);
    for(InternalLabelType nLabelIdx1=0; nLabelIdx1<m_nResegmLabels; ++nLabelIdx1)
        for(InternalLabelType nLabelIdx2=0; nLabelIdx2<m_nResegmLabels; ++nLabelIdx2)
            oResegmBaseFunc(nLabelIdx1,nLabelIdx2) = cost_cast((nLabelIdx1^nLabelIdx2)*STEREOSEGMATCH_LBLSIM_RESEGM_SCALE_CST);
    lvLog(2,"\tadding unary factors to resegm graph...");
    m_anResegmUnaryFactCounts[nCamIdx] = size_t(0);
    m_avResegmUnaryFuncs[nCamIdx].clear();
    m_avResegmUnaryFuncs[nCamIdx].reserve(m_anValidResegmGraphNodes[nCamIdx]);
    const std::array<size_t,1> aUnaryResegmFuncDims = {m_nResegmLabels};
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidResegmGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_avValidResegmLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
        ResegmNodeInfo& oNode = m_avResegmNodeInfos[nCamIdx][nLUTNodeIdx];
        oNode.vpCliques.clear();
        m_avResegmUnaryFuncs[nCamIdx].push_back(m_apResegmModels[nCamIdx]->addFunctionWithRefReturn(ExplicitFunction()));
        FuncPairType& oResegmFunc = m_avResegmUnaryFuncs[nCamIdx].back();
        lvDbgAssert((&m_apResegmModels[nCamIdx]->getFunction<ExplicitFunction>(oResegmFunc.first))==(&oResegmFunc.second));
        oResegmFunc.second.assign(aUnaryResegmFuncDims.begin(),aUnaryResegmFuncDims.end(),m_apResegmUnaryFuncsDataBase[nCamIdx]+(nGraphNodeIdx*m_nResegmLabels));
        lvDbgAssert(oResegmFunc.second.strides(0)==1); // expect no padding
        const std::array<size_t,1> aGraphNodeIndices = {nGraphNodeIdx};
        oNode.nUnaryFactID = m_apResegmModels[nCamIdx]->addFactorNonFinalized(oResegmFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
        lvDbgAssert(FuncIdentifType((*m_apResegmModels[nCamIdx])[oNode.nUnaryFactID].functionIndex(),(*m_apResegmModels[nCamIdx])[oNode.nUnaryFactID].functionType())==oResegmFunc.first);
        lvDbgAssert(m_apResegmModels[nCamIdx]->operator[](oNode.nUnaryFactID).numberOfVariables()==size_t(1));
        lvDbgAssert(oNode.nUnaryFactID==m_anResegmUnaryFactCounts[nCamIdx]);
        oNode.pUnaryFunc = &oResegmFunc.second;
        ++m_anResegmUnaryFactCounts[nCamIdx];
    }
    lvLog(2,"\tadding pairwise factors to resegm graph...");
    m_anResegmPairwFactCounts[nCamIdx] = size_t(0);
    for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
        m_aavResegmPairwFuncs[nCamIdx][nOrientIdx].clear();
        m_aavResegmPairwFuncs[nCamIdx][nOrientIdx].reserve(m_anValidResegmGraphNodes[nCamIdx]);
    }
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidResegmGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
        const size_t nBaseLUTNodeIdx = m_avValidResegmLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
        ResegmNodeInfo& oBaseNode = m_avResegmNodeInfos[nCamIdx][nBaseLUTNodeIdx];
        lvDbgAssert(oBaseNode.bValidGraphNode);
        oBaseNode.aPairwCliques = {}; // reset to default state
        const auto lPairwCliqueCreator = [&](size_t nOrientIdx, size_t nOffsetLUTNodeIdx) {
            const NodeInfo& oOffsetNode = m_avResegmNodeInfos[nCamIdx][nOffsetLUTNodeIdx];
            if(oOffsetNode.bValidGraphNode) {
                m_aavResegmPairwFuncs[nCamIdx][nOrientIdx].push_back(m_apResegmModels[nCamIdx]->addFunctionWithRefReturn(ExplicitFunction()));
                FuncPairType& oResegmFunc = m_aavResegmPairwFuncs[nCamIdx][nOrientIdx].back();
                oResegmFunc.second.assign(aPairwResegmFuncDims.begin(),aPairwResegmFuncDims.end(),m_apResegmPairwFuncsDataBase[nCamIdx]+((nGraphNodeIdx*s_nPairwOrients+nOrientIdx)*m_nResegmLabels*m_nResegmLabels));
                lvDbgAssert(oResegmFunc.second.strides(0)==1 && oResegmFunc.second.strides(1)==m_nResegmLabels); // expect last-idx-major
                lvDbgAssert((&m_apResegmModels[nCamIdx]->getFunction<ExplicitFunction>(oResegmFunc.first))==(&oResegmFunc.second));
                PairwClique& oPairwClique = oBaseNode.aPairwCliques[nOrientIdx];
                oPairwClique.m_bValid = true;
                const std::array<size_t,2> aLUTNodeIndices = {nBaseLUTNodeIdx,nOffsetLUTNodeIdx};
                oPairwClique.m_anLUTNodeIdxs = aLUTNodeIndices;
                const std::array<size_t,2> aGraphNodeIndices = {nGraphNodeIdx,oOffsetNode.nGraphNodeIdx};
                oPairwClique.m_anGraphNodeIdxs = aGraphNodeIndices;
                oPairwClique.m_nGraphFactorId = m_apResegmModels[nCamIdx]->addFactorNonFinalized(oResegmFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
                lvDbgAssert(FuncIdentifType((*m_apResegmModels[nCamIdx])[oPairwClique.m_nGraphFactorId].functionIndex(),(*m_apResegmModels[nCamIdx])[oPairwClique.m_nGraphFactorId].functionType())==oResegmFunc.first);
                lvDbgAssert(m_apResegmModels[nCamIdx]->operator[](oPairwClique.m_nGraphFactorId).numberOfVariables()==size_t(2));
                lvDbgAssert(oPairwClique.m_nGraphFactorId==m_anResegmUnaryFactCounts[nCamIdx]+m_anResegmPairwFactCounts[nCamIdx]);
                oPairwClique.m_pGraphFunctionPtr = &oResegmFunc.second;
                oBaseNode.vpCliques.push_back(&oPairwClique);
                ++m_anResegmPairwFactCounts[nCamIdx];
            }
        };
        if((oBaseNode.nRowIdx+1)<(int)m_oGridSize[0]) { // vertical pair
            lvDbgAssert(int((oBaseNode.nRowIdx+1)*m_oGridSize[1]+oBaseNode.nColIdx+oBaseNode.nTemporalDepth*m_oGridSize.total())==int(nBaseLUTNodeIdx+m_oGridSize[1]));
            lPairwCliqueCreator(size_t(0),nBaseLUTNodeIdx+m_oGridSize[1]);
        }
        if((oBaseNode.nColIdx+1)<(int)m_oGridSize[1]) { // horizontal pair
            lvDbgAssert(int(oBaseNode.nRowIdx*m_oGridSize[1]+oBaseNode.nColIdx+1+oBaseNode.nTemporalDepth*m_oGridSize.total())==int(nBaseLUTNodeIdx+1));
            lPairwCliqueCreator(size_t(1),nBaseLUTNodeIdx+1);
        }
        static_assert(s_nPairwOrients==2,"missing some pairw instantiations here");
    }
    m_anResegmTemporalFactCounts[nCamIdx] = size_t(0);
    m_avResegmTemporalFuncs[nCamIdx].clear();
    if(STEREOSEGMATCH_DEFAULT_TEMPORAL_DEPTH && bUseTemporalLinks) {
        lvLog(2,"\tadding temporal factors to resegm graph...");
        const size_t nValidFirstLayerResegmGraphNodes = m_anValidResegmGraphNodes[nCamIdx]/s_nTemporalLayers;
        m_avResegmTemporalFuncs[nCamIdx].reserve(nValidFirstLayerResegmGraphNodes); // stride not taken into account here
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<nValidFirstLayerResegmGraphNodes; ++nGraphNodeIdx) {
            const size_t nBaseLUTNodeIdx = m_avValidResegmLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
            ResegmNodeInfo& oBaseNode = m_avResegmNodeInfos[nCamIdx][nBaseLUTNodeIdx];
            lvDbgAssert(oBaseNode.bValidGraphNode);
            oBaseNode.oTemporalClique = {};
            // @@@@@@@@@@@ TODO
            /*const NodeInfo& oOffsetNode = m_avResegmNodeInfos[nCamIdx][nOffsetLUTNodeIdx];
            if(oOffsetNode.bValidGraphNode) {
                m_aavResegmPairwFuncs[nCamIdx][nOrientIdx].push_back(m_apResegmModels[nCamIdx]->addFunctionWithRefReturn(ExplicitFunction()));
                FuncPairType& oResegmFunc = m_aavResegmPairwFuncs[nCamIdx][nOrientIdx].back();
                oResegmFunc.second.assign(aPairwResegmFuncDims.begin(),aPairwResegmFuncDims.end(),m_apResegmPairwFuncsDataBase[nCamIdx]+((nGraphNodeIdx*s_nPairwOrients+nOrientIdx)*m_nResegmLabels*m_nResegmLabels));
                lvDbgAssert(oResegmFunc.second.strides(0)==1 && oResegmFunc.second.strides(1)==m_nResegmLabels); // expect last-idx-major
                lvDbgAssert((&m_apResegmModels[nCamIdx]->getFunction<ExplicitFunction>(oResegmFunc.first))==(&oResegmFunc.second));
                PairwClique& oPairwClique = oBaseNode.aPairwCliques[nOrientIdx];
                oPairwClique.m_bValid = true;
                const std::array<size_t,2> aLUTNodeIndices = {nBaseLUTNodeIdx,nOffsetLUTNodeIdx};
                oPairwClique.m_anLUTNodeIdxs = aLUTNodeIndices;
                const std::array<size_t,2> aGraphNodeIndices = {nGraphNodeIdx,oOffsetNode.nGraphNodeIdx};
                oPairwClique.m_anGraphNodeIdxs = aGraphNodeIndices;
                oPairwClique.m_nGraphFactorId = m_apResegmModels[nCamIdx]->addFactorNonFinalized(oResegmFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
                lvDbgAssert(FuncIdentifType((*m_apResegmModels[nCamIdx])[oPairwClique.m_nGraphFactorId].functionIndex(),(*m_apResegmModels[nCamIdx])[oPairwClique.m_nGraphFactorId].functionType())==oResegmFunc.first);
                lvDbgAssert(m_apResegmModels[nCamIdx]->operator[](oPairwClique.m_nGraphFactorId).numberOfVariables()==size_t(2));
                lvDbgAssert(oPairwClique.m_nGraphFactorId==m_anResegmUnaryFactCounts[nCamIdx]+m_anResegmPairwFactCounts[nCamIdx]);
                oPairwClique.m_pGraphFunctionPtr = &oResegmFunc.second;

                oBaseNode.vpCliques.push_back(&oBaseNode.oTemporalClique);
                ++m_anResegmTemporalFactCounts[nCamIdx];
            }*/
        }
    }
    m_apResegmModels[nCamIdx]->finalize();
    m_apResegmInfs[nCamIdx] = std::make_unique<ResegmGraphInference>(nCamIdx,*this);
    if(lv::getVerbosity()>=2) {
        lvCout << "Resegm model [" << nCamIdx << "] :\n";
        lv::gm::printModelInfo(*m_apResegmModels[nCamIdx]);
        lvCout << "\n";
    }
}

void StereoSegmMatcher::GraphModelData::updateResegmModel(size_t nCamIdx, bool bInit) {
    static_assert(getCameraCount()==2,"bad static array size, hardcoded stuff below (incl xor) will break");
    lvDbgExceptionWatch;
    lvDbgAssert_(nCamIdx<getCameraCount(),"bad input cam index");
    lvDbgAssert(m_apResegmModels[nCamIdx] && m_apResegmModels[nCamIdx]->numberOfVariables()==m_anValidResegmGraphNodes[nCamIdx]);
    lvDbgAssert(m_oGridSize==m_aResegmLabelings[nCamIdx].size && m_oGridSize==m_aGMMCompAssignMap[nCamIdx].size);
    lvDbgAssert(m_anValidResegmGraphNodes[nCamIdx]==m_avValidResegmLUTNodeIdxs[nCamIdx].size());
    if(!m_bResegmGraphsFullyBuilt && m_nFramesProcessed>STEREOSEGMATCH_DEFAULT_TEMPORAL_DEPTH) {
        lvLog_(2,"Rebuilding resegm graph model [%d] w/ temporal links...",(int)nCamIdx);
        buildResegmModel(nCamIdx,true);
    }
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    lvIgnore(nRows); lvIgnore(nCols);
    const CamArray<cv::Mat_<float>> aInitFGDist = {m_vNextFeats[FeatPack_LeftInitFGDist],m_vNextFeats[FeatPack_RightInitFGDist]};
    const CamArray<cv::Mat_<float>> aInitBGDist = {m_vNextFeats[FeatPack_LeftInitBGDist],m_vNextFeats[FeatPack_RightInitBGDist]};
    lvDbgAssert(lv::MatInfo(aInitFGDist[0])==lv::MatInfo(aInitFGDist[1]) && m_oGridSize==aInitFGDist[0].size);
    lvDbgAssert(lv::MatInfo(aInitBGDist[0])==lv::MatInfo(aInitBGDist[1]) && m_oGridSize==aInitBGDist[0].size);
    const CamArray<cv::Mat_<float>> aFGDist = {m_vNextFeats[FeatPack_LeftFGDist],m_vNextFeats[FeatPack_RightFGDist]};
    const CamArray<cv::Mat_<float>> aBGDist = {m_vNextFeats[FeatPack_LeftBGDist],m_vNextFeats[FeatPack_RightBGDist]};
    lvDbgAssert(lv::MatInfo(aFGDist[0])==lv::MatInfo(aFGDist[1]) && m_oGridSize==aFGDist[0].size);
    lvDbgAssert(lv::MatInfo(aBGDist[0])==lv::MatInfo(aBGDist[1]) && m_oGridSize==aBGDist[0].size);
    const CamArray<cv::Mat_<uchar>> aGradY = {m_vNextFeats[FeatPack_LeftGradY],m_vNextFeats[FeatPack_RightGradY]};
    const CamArray<cv::Mat_<uchar>> aGradX = {m_vNextFeats[FeatPack_LeftGradX],m_vNextFeats[FeatPack_RightGradX]};
    const CamArray<cv::Mat_<uchar>> aGradMag = {m_vNextFeats[FeatPack_LeftGradMag],m_vNextFeats[FeatPack_RightGradMag]};
    lvDbgAssert(lv::MatInfo(aGradY[0])==lv::MatInfo(aGradY[1]) && m_oGridSize==aGradY[0].size);
    lvDbgAssert(lv::MatInfo(aGradX[0])==lv::MatInfo(aGradX[1]) && m_oGridSize==aGradX[0].size);
    lvDbgAssert(lv::MatInfo(aGradMag[0])==lv::MatInfo(aGradMag[1]) && m_oGridSize==aGradMag[0].size);
#if STEREOSEGMATCH_CONFIG_USE_GMM_LOCAL_BACKGR
    cv::Mat_<uchar> oGMMROI = (m_aResegmLabelings[nCamIdx]>0);
    cv::dilate(oGMMROI,oGMMROI,cv::getStructuringElement(cv::MORPH_RECT,cv::Size(75,75)));
    cv::bitwise_and(oGMMROI,m_aROIs[nCamIdx],oGMMROI);
#else //!STEREOSEGMATCH_CONFIG_USE_GMM_LOCAL_BACKGR
    const cv::Mat_<uchar>& oGMMROI = m_aROIs[nCamIdx];
#endif //!STEREOSEGMATCH_CONFIG_USE_GMM_LOCAL_BACKGR
    if(bInit) {
        const cv::Mat oInputImg = m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img];
        if(oInputImg.channels()==1) {
            lv::initGaussianMixtureParams(oInputImg,m_aResegmLabelings[nCamIdx],m_aBGModels_1ch[nCamIdx],m_aFGModels_1ch[nCamIdx],oGMMROI);
            lv::assignGaussianMixtureComponents(oInputImg,m_aResegmLabelings[nCamIdx],m_aGMMCompAssignMap[nCamIdx],m_aBGModels_1ch[nCamIdx],m_aFGModels_1ch[nCamIdx],oGMMROI);
        }
        else {// 3ch
            lv::initGaussianMixtureParams(oInputImg,m_aResegmLabelings[nCamIdx],m_aBGModels_3ch[nCamIdx],m_aFGModels_3ch[nCamIdx],oGMMROI);
            lv::assignGaussianMixtureComponents(oInputImg,m_aResegmLabelings[nCamIdx],m_aGMMCompAssignMap[nCamIdx],m_aBGModels_3ch[nCamIdx],m_aFGModels_3ch[nCamIdx],oGMMROI);
        }
    }
    else {
        const cv::Mat oInputImg = m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img];
        if(oInputImg.channels()==1) {
            lv::assignGaussianMixtureComponents(oInputImg,m_aResegmLabelings[nCamIdx],m_aGMMCompAssignMap[nCamIdx],m_aBGModels_1ch[nCamIdx],m_aFGModels_1ch[nCamIdx],oGMMROI);
            lv::learnGaussianMixtureParams(oInputImg,m_aResegmLabelings[nCamIdx],m_aGMMCompAssignMap[nCamIdx],m_aBGModels_1ch[nCamIdx],m_aFGModels_1ch[nCamIdx],oGMMROI);
        }
        else { // 3ch
            lv::assignGaussianMixtureComponents(oInputImg,m_aResegmLabelings[nCamIdx],m_aGMMCompAssignMap[nCamIdx],m_aBGModels_3ch[nCamIdx],m_aFGModels_3ch[nCamIdx],oGMMROI);
            lv::learnGaussianMixtureParams(oInputImg,m_aResegmLabelings[nCamIdx],m_aGMMCompAssignMap[nCamIdx],m_aBGModels_3ch[nCamIdx],m_aFGModels_3ch[nCamIdx],oGMMROI);
        }
    }
    if(lv::getVerbosity()>=4) {
        cv::Mat_<int> oClusterLabels = m_aGMMCompAssignMap[nCamIdx].clone();
        for(size_t nNodeIdx=0; nNodeIdx<m_aResegmLabelings[nCamIdx].total(); ++nNodeIdx)
            if(((InternalLabelType*)m_aResegmLabelings[nCamIdx].data)[nNodeIdx])
                ((int*)oClusterLabels.data)[nNodeIdx] += 1<<31;
        cv::Mat oClusterLabelsDisplay = lv::getUniqueColorMap(oClusterLabels);
        cv::imshow(std::string("gmm_clusters_")+std::to_string(nCamIdx),oClusterLabelsDisplay);
        cv::waitKey(1);
    }
    const float fInterSpectrScale = STEREOSEGMATCH_SHPDIST_INTERSPEC_SCALE;
    const float fInterSpectrRatioTot = 1.0f+fInterSpectrScale;
    const float fInitDistScale = STEREOSEGMATCH_SHPDIST_INITDIST_SCALE;
    const float fMaxDist = STEREOSEGMATCH_SHPDIST_PX_MAX_CST;
    lvLog_(4,"Updating resegm graph model[%d] energy terms based on new features...",(int)nCamIdx);
    lv::StopWatch oLocalTimer;
#if STEREOSEGMATCH_CONFIG_USE_PROGRESS_BARS
    lv::ProgressBarManager oProgressBarMgr("\tprogress:");
#endif //STEREOSEGMATCH_CONFIG_USE_PROGRESS_BARS
    const cv::Mat oInputImg = m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img];
    const bool bUsing3ChannelInput = oInputImg.channels()==3;
    /*cv::Mat_<double> oFGProbMap(m_oGridSize),oBGProbMap(m_oGridSize);
    oFGProbMap = 0.0; oBGProbMap = 0.0;
    double dMinFGProb,dMinBGProb;
    dMinFGProb = dMinBGProb = 9999999;
    double dMaxFGProb,dMaxBGProb;
    dMaxFGProb = dMaxBGProb = 0;*/
#if USING_OPENMP
    #pragma omp parallel for
#endif //USING_OPENMP
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidResegmGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_avValidResegmLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
        NodeInfo& oNode = m_avResegmNodeInfos[nCamIdx][nLUTNodeIdx];
        lvDbgAssert(oNode.nUnaryFactID<m_anResegmUnaryFactCounts[nCamIdx]);
        lvDbgAssert(m_apResegmModels[nCamIdx]->operator[](oNode.nUnaryFactID).numberOfVariables()==size_t(1));
        const int nRowIdx = oNode.nRowIdx;
        const int nColIdx = oNode.nColIdx;
        // update unary terms for each grid node
        lvDbgAssert(nLUTNodeIdx==(oNode.nRowIdx*m_oGridSize[1]+oNode.nColIdx));
        lvDbgAssert(oNode.nUnaryFactID!=SIZE_MAX && oNode.nUnaryFactID<m_anResegmUnaryFactCounts[nCamIdx] && oNode.pUnaryFunc);
        lvDbgAssert(m_pStereoModel->operator[](oNode.nUnaryFactID).numberOfVariables()==size_t(1));
        ExplicitFunction& vUnaryResegmLUT = *oNode.pUnaryFunc;
        lvDbgAssert(vUnaryResegmLUT.dimension()==1 && vUnaryResegmLUT.size()==m_nResegmLabels);
        const double dMinProbDensity = 1e-10;
        const double dMaxProbDensity = 1.0;
        const uchar* acInputColorSample = bUsing3ChannelInput?(oInputImg.data+nLUTNodeIdx*3):(oInputImg.data+nLUTNodeIdx);
        const float fInitFGDist = std::min(((float*)aInitFGDist[nCamIdx].data)[nLUTNodeIdx],fMaxDist);
        const float fCurrFGDist = std::min(((float*)aFGDist[nCamIdx].data)[nLUTNodeIdx],fMaxDist);
        const ValueType tFGDistUnaryCost = cost_cast((fCurrFGDist+fInitFGDist*fInitDistScale)*STEREOSEGMATCH_SHPDIST_COST_SCALE);
        lvDbgAssert(tFGDistUnaryCost>=cost_cast(0));
        const double dColorFGProb = std::min(std::max((bUsing3ChannelInput?m_aFGModels_3ch[nCamIdx](acInputColorSample):m_aFGModels_1ch[nCamIdx](acInputColorSample)),dMinProbDensity),dMaxProbDensity);
        /*((double*)oFGProbMap.data)[nLUTNodeIdx] = dColorFGProb;
        dMinFGProb = std::min(dMinFGProb,dColorFGProb);
        dMaxFGProb = std::max(dMaxFGProb,dColorFGProb);*/
        const ValueType tFGColorUnaryCost = cost_cast(-std::log2(dColorFGProb)*STEREOSEGMATCH_IMGSIM_COST_COLOR_SCALE);
        lvDbgAssert(tFGColorUnaryCost>=cost_cast(0));
        vUnaryResegmLUT(s_nForegroundLabelIdx) = std::min(tFGDistUnaryCost+tFGColorUnaryCost,STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
        lvDbgAssert(vUnaryResegmLUT(s_nForegroundLabelIdx)>=cost_cast(0));
        const float fInitBGDist = std::min(((float*)aInitBGDist[nCamIdx].data)[nLUTNodeIdx],fMaxDist);
        const float fCurrBGDist = std::min(((float*)aBGDist[nCamIdx].data)[nLUTNodeIdx],fMaxDist);
        const ValueType tBGDistUnaryCost = cost_cast((fCurrBGDist+fInitBGDist*fInitDistScale)*STEREOSEGMATCH_SHPDIST_COST_SCALE);
        lvDbgAssert(tBGDistUnaryCost>=cost_cast(0));
        const double dColorBGProb = std::min(std::max((bUsing3ChannelInput?m_aBGModels_3ch[nCamIdx](acInputColorSample):m_aBGModels_1ch[nCamIdx](acInputColorSample)),dMinProbDensity),dMaxProbDensity);
        /*((double*)oBGProbMap.data)[nLUTNodeIdx] = dColorBGProb;
        dMinBGProb = std::min(dMinBGProb,dColorBGProb);
        dMaxBGProb = std::max(dMaxBGProb,dColorBGProb);*/
        const ValueType tBGColorUnaryCost = cost_cast(-std::log2(dColorBGProb)*STEREOSEGMATCH_IMGSIM_COST_COLOR_SCALE);
        lvDbgAssert(tBGColorUnaryCost>=cost_cast(0));
        vUnaryResegmLUT(s_nBackgroundLabelIdx) = std::min(tBGDistUnaryCost+tBGColorUnaryCost,STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
        lvDbgAssert(vUnaryResegmLUT(s_nBackgroundLabelIdx)>=cost_cast(0));
        const InternalLabelType nStereoLabelIdx = ((InternalLabelType*)m_aStereoLabelings[nCamIdx].data)[nLUTNodeIdx];
        const int nOffsetColIdx = (nStereoLabelIdx<m_nRealStereoLabels)?getOffsetColIdx(nCamIdx,nColIdx,nStereoLabelIdx):INT_MAX;
        if(nOffsetColIdx>=0 && nOffsetColIdx<nCols && m_aROIs[nCamIdx^1](nRowIdx,nOffsetColIdx)) {
            const float fInitOffsetFGDist = std::min(aInitFGDist[nCamIdx^1](nRowIdx,nOffsetColIdx),fMaxDist);
            const float fCurrOffsetFGDist = std::min(aFGDist[nCamIdx^1](nRowIdx,nOffsetColIdx),fMaxDist);
            const ValueType tAddedFGDistUnaryCost = cost_cast((fCurrOffsetFGDist+fInitOffsetFGDist*fInitDistScale)*fInterSpectrScale*STEREOSEGMATCH_SHPDIST_COST_SCALE);
            vUnaryResegmLUT(s_nForegroundLabelIdx) = std::min(vUnaryResegmLUT(s_nForegroundLabelIdx)+tAddedFGDistUnaryCost,STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
            const float fInitOffsetBGDist = std::min(aInitBGDist[nCamIdx^1](nRowIdx,nOffsetColIdx),fMaxDist);
            const float fCurrOffsetBGDist = std::min(aBGDist[nCamIdx^1](nRowIdx,nOffsetColIdx),fMaxDist);
            const ValueType tAddedBGDistUnaryCost = cost_cast((fCurrOffsetBGDist+fInitOffsetBGDist*fInitDistScale)*fInterSpectrScale*STEREOSEGMATCH_SHPDIST_COST_SCALE);
            vUnaryResegmLUT(s_nBackgroundLabelIdx) = std::min(vUnaryResegmLUT(s_nBackgroundLabelIdx)+tAddedBGDistUnaryCost,STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
            for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
                PairwClique& oPairwClique = oNode.aPairwCliques[nOrientIdx];
                if(oPairwClique) {
                    lvDbgAssert(oPairwClique.m_nGraphFactorId>=m_anResegmUnaryFactCounts[nCamIdx]);
                    lvDbgAssert(oPairwClique.m_nGraphFactorId<m_anResegmUnaryFactCounts[nCamIdx]+m_anResegmPairwFactCounts[nCamIdx]);
                    lvDbgAssert(m_apResegmModels[nCamIdx]->operator[](oPairwClique.m_nGraphFactorId).numberOfVariables()==size_t(2));
                    lvDbgAssert(oPairwClique.m_anGraphNodeIdxs[0]!=SIZE_MAX && oPairwClique.m_anLUTNodeIdxs[0]!=SIZE_MAX);
                    lvDbgAssert(oPairwClique.m_anGraphNodeIdxs[1]!=SIZE_MAX && oPairwClique.m_anLUTNodeIdxs[1]!=SIZE_MAX);
                    lvDbgAssert(m_avResegmNodeInfos[nCamIdx][oPairwClique.m_anLUTNodeIdxs[1]].bValidGraphNode && oPairwClique.m_pGraphFunctionPtr);
                    ExplicitFunction& vPairwResegmLUT = *oPairwClique.m_pGraphFunctionPtr;
                    lvDbgAssert(vPairwResegmLUT.dimension()==2 && vPairwResegmLUT.size()==m_nResegmLabels*m_nResegmLabels);
                    const int nLocalGrad = (int)(((nOrientIdx==0)?aGradY[nCamIdx]:(nOrientIdx==1)?aGradX[nCamIdx]:aGradMag[nCamIdx])(nRowIdx,nColIdx));
                    const int nOffsetGrad = (int)(((nOrientIdx==0)?aGradY[nCamIdx^1]:(nOrientIdx==1)?aGradX[nCamIdx^1]:aGradMag[nCamIdx^1])(nRowIdx,nOffsetColIdx));
                    const float fLocalScaleFact = m_aLabelSimCostGradFactLUT.eval_raw(nLocalGrad);
                    const float fOffsetScaleFact = m_aLabelSimCostGradFactLUT.eval_raw(nOffsetGrad);
                    lvDbgAssert(fLocalScaleFact==(float)std::exp(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE));
                    lvDbgAssert(fOffsetScaleFact==(float)std::exp(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nOffsetGrad)/STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE));
                    //const float fScaleFact = std::min(fLocalScaleFact,fOffsetScaleFact);
                    //const float fScaleFact = fLocalScaleFact*fOffsetScaleFact;
                    const float fScaleFact = (fLocalScaleFact+fOffsetScaleFact*fInterSpectrScale)/fInterSpectrRatioTot;
                    for(InternalLabelType nLabelIdx1=0; nLabelIdx1<m_nResegmLabels; ++nLabelIdx1) {
                        for(InternalLabelType nLabelIdx2=0; nLabelIdx2<m_nResegmLabels; ++nLabelIdx2) {
                            vPairwResegmLUT(nLabelIdx1,nLabelIdx2) = cost_cast((nLabelIdx1^nLabelIdx2)*fScaleFact*STEREOSEGMATCH_LBLSIM_RESEGM_SCALE_CST);
                        }
                    }
                }
            }
        }
        else {
            vUnaryResegmLUT(s_nForegroundLabelIdx) = STEREOSEGMATCH_UNARY_COST_OOB_CST;
            vUnaryResegmLUT(s_nBackgroundLabelIdx) = cost_cast(0);
            for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
                PairwClique& oPairwClique = oNode.aPairwCliques[nOrientIdx];
                if(oPairwClique) {
                    lvDbgAssert(oPairwClique.m_nGraphFactorId>=m_anResegmUnaryFactCounts[nCamIdx]);
                    lvDbgAssert(oPairwClique.m_nGraphFactorId<m_anResegmUnaryFactCounts[nCamIdx]+m_anResegmPairwFactCounts[nCamIdx]);
                    lvDbgAssert(m_apResegmModels[nCamIdx]->operator[](oPairwClique.m_nGraphFactorId).numberOfVariables()==size_t(2));
                    lvDbgAssert(oPairwClique.m_anGraphNodeIdxs[0]!=SIZE_MAX && oPairwClique.m_anLUTNodeIdxs[0]!=SIZE_MAX);
                    lvDbgAssert(oPairwClique.m_anGraphNodeIdxs[1]!=SIZE_MAX && oPairwClique.m_anLUTNodeIdxs[1]!=SIZE_MAX);
                    lvDbgAssert(m_avResegmNodeInfos[nCamIdx][oPairwClique.m_anLUTNodeIdxs[1]].bValidGraphNode && oPairwClique.m_pGraphFunctionPtr);
                    ExplicitFunction& vPairwResegmLUT = *oPairwClique.m_pGraphFunctionPtr;
                    lvDbgAssert(vPairwResegmLUT.dimension()==2 && vPairwResegmLUT.size()==m_nResegmLabels*m_nResegmLabels);
                    const int nLocalGrad = (int)(((nOrientIdx==0)?aGradY[nCamIdx]:(nOrientIdx==1)?aGradX[nCamIdx]:aGradMag[nCamIdx])(nRowIdx,nColIdx));
                    const float fLocalScaleFact = m_aLabelSimCostGradFactLUT.eval_raw(nLocalGrad);
                    lvDbgAssert(fLocalScaleFact==(float)std::exp(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE));
                    for(InternalLabelType nLabelIdx1=0; nLabelIdx1<m_nResegmLabels; ++nLabelIdx1) {
                        for(InternalLabelType nLabelIdx2=0; nLabelIdx2<m_nResegmLabels; ++nLabelIdx2) {
                            vPairwResegmLUT(nLabelIdx1,nLabelIdx2) = cost_cast((nLabelIdx1^nLabelIdx2)*fLocalScaleFact*STEREOSEGMATCH_LBLSIM_RESEGM_SCALE_CST);
                        }
                    }
                }
            }
        }
        if(bInit) {
            // @@@ add temporal terms update here
            lvAssert(!STEREOSEGMATCH_DEFAULT_TEMPORAL_DEPTH); // missing impl @@@
        }
    #if STEREOSEGMATCH_CONFIG_USE_PROGRESS_BARS
        if(lv::getVerbosity()>=3)
            oProgressBarMgr.update(float(nGraphNodeIdx)/m_anValidGraphNodes[nCamIdx]);
    #endif //STEREOSEGMATCH_CONFIG_USE_PROGRESS_BARS
    }
    /*cv::imshow(std::string("oFGProbMap_")+std::to_string(nCamIdx),oFGProbMap);
    lvCout << " fg : min=" << dMinFGProb << ", max=" << dMaxFGProb << std::endl;
    cv::imshow(std::string("oBGProbMap_")+std::to_string(nCamIdx),oBGProbMap);
    lvCout << " bg : min=" << dMinBGProb << ", max=" << dMaxBGProb << std::endl;
    cv::waitKey(0);*/
    lvLog_(4,"Resegm graph model[%d] energy terms update completed in %f second(s).",(int)nCamIdx,oLocalTimer.tock());
}

void StereoSegmMatcher::GraphModelData::calcFeatures(const MatArrayIn& aInputs, cv::Mat* pFeatsPacket) {
    static_assert(s_nInputArraySize==4 && getCameraCount()==2,"lots of hardcoded indices below");
    lvDbgExceptionWatch;
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        const cv::Mat& oInputImg = aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img];
        lvAssert__(oInputImg.dims==2 && m_oGridSize==oInputImg.size(),"input image in array at index=%d had the wrong size",(int)nCamIdx);
        lvAssert_(oInputImg.type()==CV_8UC1 || oInputImg.type()==CV_8UC3,"unexpected input image type");
        const cv::Mat& oInputMask = aInputs[nCamIdx*InputPackOffset+InputPackOffset_Mask];
        lvAssert__(oInputMask.dims==2 && m_oGridSize==oInputMask.size(),"input mask in array at index=%d had the wrong size",(int)nCamIdx);
        lvAssert_(oInputMask.type()==CV_8UC1,"unexpected input mask type");
    }
    m_vNextFeats.resize(FeatPackSize);
    calcImageFeatures(CamArray<cv::Mat>{aInputs[InputPack_LeftImg],aInputs[InputPack_RightImg]});
    calcShapeFeatures(CamArray<cv::Mat_<InternalLabelType>>{aInputs[InputPack_LeftMask],aInputs[InputPack_RightMask]});
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_FGDist].copyTo(m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_InitFGDist]);
        m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_BGDist].copyTo(m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_InitBGDist]);
    }
    for(cv::Mat& oFeatMap : m_vNextFeats)
        lvAssert_(oFeatMap.isContinuous(),"internal func used non-continuous data block for feature maps");
    if(pFeatsPacket)
        *pFeatsPacket = lv::packData(m_vNextFeats,&m_vNextFeatPackInfo);
    else {
        m_vNextFeatPackInfo.resize(m_vNextFeats.size());
        for(size_t nFeatMapIdx=0; nFeatMapIdx<m_vNextFeats.size(); ++nFeatMapIdx)
            m_vNextFeatPackInfo[nFeatMapIdx] = lv::MatInfo(m_vNextFeats[nFeatMapIdx]);
    }
    if(m_vExpectedFeatPackInfo.empty())
        m_vExpectedFeatPackInfo = m_vNextFeatPackInfo;
    lvAssert_(m_vNextFeatPackInfo==m_vExpectedFeatPackInfo,"packed features info mismatch (should stay constant for all inputs)");
}

void StereoSegmMatcher::GraphModelData::calcImageFeatures(const CamArray<cv::Mat>& aInputImages) {
    static_assert(getCameraCount()==2,"bad input image array size");
    lvDbgExceptionWatch;
    for(size_t nInputIdx=0; nInputIdx<aInputImages.size(); ++nInputIdx) {
        lvDbgAssert__(aInputImages[nInputIdx].dims==2 && m_oGridSize==aInputImages[nInputIdx].size(),"input at index=%d had the wrong size",(int)nInputIdx);
        lvDbgAssert_(aInputImages[nInputIdx].type()==CV_8UC1 || aInputImages[nInputIdx].type()==CV_8UC3,"unexpected input image type");
    }
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    lvLog(3,"Calculating image features maps...");
    const int nWinRadius = (int)m_nGridBorderSize;
    const int nWinSize = nWinRadius*2+1;
    CamArray<cv::Mat> aEnlargedInput;
#if STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
    lvIgnore(nWinSize);
    CamArray<cv::Mat_<float>> aEnlargedDescs,aDescs;
    const int nPatchSize = STEREOSEGMATCH_DEFAULT_DESC_PATCH_SIZE;
#else //!STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
    CamArray<cv::Mat_<uchar>> aEnlargedROIs;
    const int nPatchSize = nWinSize;
#endif //STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
    lvAssert_((nPatchSize%2)==1,"patch sizes must be odd");
    lv::StopWatch oLocalTimer;
#if USING_OPENMP
    //#pragma omp parallel for num_threads(getCameraCount())
#endif //USING_OPENMP
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        cv::copyMakeBorder(aInputImages[nCamIdx],aEnlargedInput[nCamIdx],nWinRadius,nWinRadius,nWinRadius,nWinRadius,cv::BORDER_DEFAULT);
    #if STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY
        if(aEnlargedInput[nCamIdx].channels()==3)
            cv::cvtColor(aEnlargedInput[nCamIdx],aEnlargedInput[nCamIdx],cv::COLOR_BGR2GRAY);
        cv::copyMakeBorder(m_aROIs[nCamIdx],aEnlargedROIs[nCamIdx],nWinRadius,nWinRadius,nWinRadius,nWinRadius,cv::BORDER_CONSTANT,cv::Scalar(0));
    #elif STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
        if(aEnlargedInput[nCamIdx].channels()==3)
            cv::cvtColor(aEnlargedInput[nCamIdx],aEnlargedInput[nCamIdx],cv::COLOR_BGR2GRAY);
        aEnlargedInput[nCamIdx].convertTo(aEnlargedInput[nCamIdx],CV_64F,(1.0/UCHAR_MAX)/nWinSize);
        aEnlargedInput[nCamIdx] -= cv::mean(aEnlargedInput[nCamIdx])[0];
        cv::copyMakeBorder(m_aROIs[nCamIdx],aEnlargedROIs[nCamIdx],nWinRadius,nWinRadius,nWinRadius,nWinRadius,cv::BORDER_CONSTANT,cv::Scalar(0));
    #elif STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
        lvLog_(3,"\tcam[%d] image descriptors...",(int)nCamIdx);
        m_pImgDescExtractor->compute2(aEnlargedInput[nCamIdx],aEnlargedDescs[nCamIdx]);
        lvDbgAssert(aEnlargedDescs[nCamIdx].dims==3 && aEnlargedDescs[nCamIdx].size[0]==nRows+nWinRadius*2 && aEnlargedDescs[nCamIdx].size[1]==nCols+nWinRadius*2);
        std::vector<cv::Range> vRanges(size_t(3),cv::Range::all());
        vRanges[0] = cv::Range(nWinRadius,nRows+nWinRadius);
        vRanges[1] = cv::Range(nWinRadius,nCols+nWinRadius);
        aEnlargedDescs[nCamIdx](vRanges.data()).copyTo(aDescs[nCamIdx]); // copy to avoid bugs when reshaping non-continuous data
        lvDbgAssert(aDescs[nCamIdx].dims==3 && aDescs[nCamIdx].size[0]==nRows && aDescs[nCamIdx].size[1]==nCols);
        lvDbgAssert(std::equal(aDescs[nCamIdx].ptr<float>(0,0),aDescs[nCamIdx].ptr<float>(0,0)+aDescs[nCamIdx].size[2],aEnlargedDescs[nCamIdx].ptr<float>(nWinRadius,nWinRadius)));
    #if STEREOSEGMATCH_CONFIG_USE_ROOT_SIFT_DESCS
        const size_t nDescSize = size_t(aDescs[nCamIdx].size[2]);
        for(size_t nDescIdx=0; nDescIdx<aDescs[nCamIdx].total(); nDescIdx+=nDescSize)
            lv::rootSIFT(((float*)aDescs[nCamIdx].data)+nDescIdx,nDescSize);
    #endif //STEREOSEGMATCH_CONFIG_USE_ROOT_SIFT_DESCS
    #endif //STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
        lvLog_(3,"\tcam[%d] image gradient magnitudes...",(int)nCamIdx);
        cv::Mat oBlurredInput;
        cv::GaussianBlur(aInputImages[nCamIdx],oBlurredInput,cv::Size(3,3),0);
        cv::Mat oBlurredGrayInput;
        if(oBlurredInput.channels()==3)
            cv::cvtColor(oBlurredInput,oBlurredGrayInput,cv::COLOR_BGR2GRAY);
        else
            oBlurredGrayInput = oBlurredInput;
        cv::Mat oGradInput_X,oGradInput_Y;
        cv::Sobel(oBlurredGrayInput,oGradInput_Y,CV_16S,0,1,STEREOSEGMATCH_DEFAULT_GRAD_KERNEL_SIZE);
        cv::Mat& oGradY = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_GradY];
        cv::normalize(cv::abs(oGradInput_Y),oGradY,255,0,cv::NORM_MINMAX,CV_8U);
        cv::Sobel(oBlurredGrayInput,oGradInput_X,CV_16S,1,0,STEREOSEGMATCH_DEFAULT_GRAD_KERNEL_SIZE);
        cv::Mat& oGradX = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_GradX];
        cv::normalize(cv::abs(oGradInput_X),oGradX,255,0,cv::NORM_MINMAX,CV_8U);
        cv::Mat& oGradMag = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_GradMag];
        cv::addWeighted(oGradY,0.5,oGradX,0.5,0,oGradMag);
        /*cv::imshow("gradm_full",oGradMag);
        cv::imshow("gradm_0.5piv",oGradMag>STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST/2);
        cv::imshow("gradm_1.0piv",oGradMag>STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST);
        cv::imshow("gradm_2.0piv",oGradMag>STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST*2);
        cv::imshow("gradm_100",oGradMag>100);
        cv::imshow("gradm_150",oGradMag>150);
        cv::waitKey(0);*/
    }
    lvLog_(3,"Image features maps computed in %f second(s).",oLocalTimer.tock());
    lvLog(3,"Calculating image affinity map...");
    const std::array<int,3> anAffinityMapDims = {nRows,nCols,(int)m_nRealStereoLabels};
    m_vNextFeats[FeatPack_ImgAffinity].create(3,anAffinityMapDims.data(),CV_32FC1);
    cv::Mat_<float> oAffinity = m_vNextFeats[FeatPack_ImgAffinity];
    std::vector<int> vDisparityOffsets;
    for(InternalLabelType nLabelIdx = 0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx)
        vDisparityOffsets.push_back(getOffsetValue(0,nLabelIdx));
    // note: we only create the dense affinity map for 1st cam here; affinity for 2nd cam will be deduced from it
#if STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
    lv::computeDescriptorAffinity(aDescs[0],aDescs[1],nPatchSize,oAffinity,vDisparityOffsets,lv::AffinityDist_L2,m_aROIs[0],m_aROIs[1]);
#elif STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY
    lv::computeImageAffinity(aEnlargedInput[0],aEnlargedInput[1],nWinSize,oAffinity,vDisparityOffsets,lv::AffinityDist_MI,aEnlargedROIs[0],aEnlargedROIs[1]);
#elif STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
    lv::computeImageAffinity(aEnlargedInput[0],aEnlargedInput[1],nWinSize,oAffinity,vDisparityOffsets,lv::AffinityDist_SSD,aEnlargedROIs[0],aEnlargedROIs[1]);
#endif //STEREOSEGMATCH_CONFIG_USE_..._AFFINITY
    lvDbgAssert(lv::MatInfo(oAffinity)==lv::MatInfo(lv::MatSize(3,anAffinityMapDims.data()),CV_32FC1));
    lvDbgAssert(m_vNextFeats[FeatPack_ImgAffinity].data==oAffinity.data);
    lvLog_(3,"Image affinity map computed in %f second(s).",oLocalTimer.tock());
    lvLog(3,"Calculating image saliency map...");
    m_vNextFeats[FeatPack_ImgSaliency].create(2,anAffinityMapDims.data(),CV_32FC1);
    cv::Mat_<float> oSaliency = m_vNextFeats[FeatPack_ImgSaliency];
    oSaliency = 0.0f; // default value for OOB pixels
    std::vector<float> vValidAffinityVals;
    vValidAffinityVals.reserve(m_nRealStereoLabels);
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidStereoGraphNodes; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_vValidStereoLUTNodeIdxs[nGraphNodeIdx];
        const StereoNodeInfo& oNode = m_vStereoNodeInfos[nLUTNodeIdx];
        const int nRowIdx = oNode.nRowIdx;
        const int nColIdx = oNode.nColIdx;
        lvDbgAssert(oNode.bValidGraphNode && m_aROIs[m_nPrimaryCamIdx](nRowIdx,nColIdx)>0);
        vValidAffinityVals.resize(0);
        const float* pAffinityPtr = oAffinity.ptr<float>(nRowIdx,nColIdx);
        std::copy_if(pAffinityPtr,pAffinityPtr+m_nRealStereoLabels,std::back_inserter(vValidAffinityVals),[](float v){return v>=0.0f;});
        const float fCurrDistSparseness = vValidAffinityVals.size()>1?(float)lv::sparseness(vValidAffinityVals.data(),vValidAffinityVals.size()):0.0f;
#if STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
        const float fCurrDescSparseness = (float)lv::sparseness(aDescs[m_nPrimaryCamIdx].ptr<float>(nRowIdx,nColIdx),size_t(aDescs[m_nPrimaryCamIdx].size[2]));
        oSaliency.at<float>(nRowIdx,nColIdx) = std::max(fCurrDescSparseness,fCurrDistSparseness);
#else //!STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
        oSaliency.at<float>(nRowIdx,nColIdx) = fCurrDistSparseness;
#endif //!STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
    }
    cv::normalize(oSaliency,oSaliency,1,0,cv::NORM_MINMAX,-1,m_aROIs[m_nPrimaryCamIdx]);
    lvDbgExec( // cv::normalize leftover fp errors are sometimes awful; need to 0-max when using map
        for(int nRowIdx=0; nRowIdx<oSaliency.rows; ++nRowIdx)
            for(int nColIdx=0; nColIdx<oSaliency.cols; ++nColIdx)
                lvDbgAssert((oSaliency.at<float>(nRowIdx,nColIdx)>=-1e-6f && oSaliency.at<float>(nRowIdx,nColIdx)<=1.0f+1e-6f) || m_aROIs[m_nPrimaryCamIdx](nRowIdx,nColIdx)==0);
    );
#if STEREOSEGMATCH_CONFIG_USE_SALIENT_MAP_BORDR
    cv::multiply(oSaliency,cv::Mat_<float>(oSaliency.size(),1.0f).setTo(0.5f,m_aDescROIs[m_nPrimaryCamIdx]==0),oSaliency);
#endif //STEREOSEGMATCH_CONFIG_USE_SALIENT_MAP_BORDR
    if(lv::getVerbosity()>=4) {
        cv::imshow("oSaliency_img",oSaliency);
        cv::waitKey(1);
    }
    lvLog_(3,"Image saliency map computed in %f second(s).",oLocalTimer.tock());
    /*if(m_pDisplayHelper) {
        std::vector<std::pair<cv::Mat,std::string>> vAffMaps;
        for(InternalLabelType nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx)
            vAffMaps.push_back(std::pair<cv::Mat,std::string>(lv::squeeze(lv::getSubMat(oAffinity,2,(int)nLabelIdx)),std::string()));
        m_pDisplayHelper->displayAlbumAndWaitKey(vAffMaps);
    }*/
}

void StereoSegmMatcher::GraphModelData::calcShapeFeatures(const CamArray<cv::Mat_<InternalLabelType>>& aInputMasks) {
    static_assert(getCameraCount()==2,"bad input mask array size");
    lvDbgExceptionWatch;
    for(size_t nInputIdx=0; nInputIdx<aInputMasks.size(); ++nInputIdx) {
        lvDbgAssert__(aInputMasks[nInputIdx].dims==2 && m_oGridSize==aInputMasks[nInputIdx].size(),"input at index=%d had the wrong size",(int)nInputIdx);
        lvDbgAssert_(aInputMasks[nInputIdx].type()==CV_8UC1,"unexpected input mask type");
    }
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    lvLog(3,"Calculating shape features maps...");
    CamArray<cv::Mat_<float>> aDescs;
    const int nPatchSize = STEREOSEGMATCH_DEFAULT_DESC_PATCH_SIZE;
    lvAssert_((nPatchSize%2)==1,"patch sizes must be odd");
    lv::StopWatch oLocalTimer;
#if USING_OPENMP
    //#pragma omp parallel for num_threads(getCameraCount())
#endif //USING_OPENMP
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        const cv::Mat& oInputMask = aInputMasks[nCamIdx];
        // @@@@@ use swap mask to prevent recalc of identical descriptors? (swap + radius based on desc)
        // LOW PRIORITY (segm will swap a lot every iter, might cover pretty much everything in segmented ROIs)
        // convert 8ui mask to keypoint list, then pass to compute2 (keeps desc struct/old values for untouched kps)
        // also base 8ui mask w.r.t fg dist? (2x radius = too far)
        // @@@@ use also for aff maps? or too fast to care?
        lvLog_(3,"\tcam[%d] shape descriptors...",(int)nCamIdx);
        m_pShpDescExtractor->compute2(oInputMask,aDescs[nCamIdx]);
        lvDbgAssert(aDescs[nCamIdx].dims==3 && aDescs[nCamIdx].size[0]==nRows && aDescs[nCamIdx].size[1]==nCols);
    #if STEREOSEGMATCH_CONFIG_USE_ROOT_SIFT_DESCS
        const size_t nDescSize = size_t(aDescs[nCamIdx].size[2]);
        for(size_t nDescIdx=0; nDescIdx<aDescs[nCamIdx].total(); nDescIdx+=nDescSize)
            lv::rootSIFT(((float*)aDescs[nCamIdx].data)+nDescIdx,nDescSize);
    #endif //STEREOSEGMATCH_CONFIG_USE_ROOT_SIFT_DESCS
        lvLog_(3,"\tcam[%d] shape distance fields...",(int)nCamIdx);
        calcShapeDistFeatures(aInputMasks[nCamIdx],nCamIdx);
    }
    lvLog_(3,"Shape features maps computed in %f second(s).",oLocalTimer.tock());
    lvLog(3,"Calculating shape affinity map...");
    const std::array<int,3> anAffinityMapDims = {nRows,nCols,(int)m_nRealStereoLabels};
    m_vNextFeats[FeatPack_ShpAffinity].create(3,anAffinityMapDims.data(),CV_32FC1);
    cv::Mat_<float> oAffinity = m_vNextFeats[FeatPack_ShpAffinity];
    std::vector<int> vDisparityOffsets;
    for(InternalLabelType nLabelIdx = 0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx)
        vDisparityOffsets.push_back(getOffsetValue(0,nLabelIdx));
#if STEREOSEGMATCH_CONFIG_USE_SHAPE_EMD_AFFIN
    lv::computeDescriptorAffinity(aDescs[0],aDescs[1],nPatchSize,oAffinity,vDisparityOffsets,lv::AffinityDist_EMD,m_aROIs[0],m_aROIs[1],m_pShpDescExtractor->getEMDCostMap());
#else //!STEREOSEGMATCH_CONFIG_USE_SHAPE_EMD_AFFIN
    lv::computeDescriptorAffinity(aDescs[0],aDescs[1],nPatchSize,oAffinity,vDisparityOffsets,lv::AffinityDist_L2,m_aROIs[0],m_aROIs[1]);
#endif //!STEREOSEGMATCH_CONFIG_USE_SHAPE_EMD_AFFIN
    lvDbgAssert(lv::MatInfo(oAffinity)==lv::MatInfo(lv::MatSize(3,anAffinityMapDims.data()),CV_32FC1));
    lvDbgAssert(m_vNextFeats[FeatPack_ShpAffinity].data==oAffinity.data);
    lvLog_(3,"Shape affinity map computed in %f second(s).",oLocalTimer.tock());
    lvLog(3,"Calculating shape saliency map...");
    m_vNextFeats[FeatPack_ShpSaliency].create(2,anAffinityMapDims.data(),CV_32FC1);
    cv::Mat_<float> oSaliency = m_vNextFeats[FeatPack_ShpSaliency];
    oSaliency = 0.0f; // default value for OOB pixels
    std::vector<float> vValidAffinityVals;
    vValidAffinityVals.reserve(m_nRealStereoLabels);
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidStereoGraphNodes; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_vValidStereoLUTNodeIdxs[nGraphNodeIdx];
        const StereoNodeInfo& oNode = m_vStereoNodeInfos[nLUTNodeIdx];
        lvDbgAssert(oNode.bValidGraphNode);
        const int nRowIdx = oNode.nRowIdx;
        const int nColIdx = oNode.nColIdx;
        vValidAffinityVals.resize(0);
        const float* pAffinityPtr = oAffinity.ptr<float>(nRowIdx,nColIdx);
        std::copy_if(pAffinityPtr,pAffinityPtr+m_nRealStereoLabels,std::back_inserter(vValidAffinityVals),[](float v){return v>=0.0f;});
        const float fCurrDistSparseness = vValidAffinityVals.size()>1?(float)lv::sparseness(vValidAffinityVals.data(),vValidAffinityVals.size()):0.0f;
        const float fCurrDescSparseness = (float)lv::sparseness(aDescs[m_nPrimaryCamIdx].ptr<float>(nRowIdx,nColIdx),size_t(aDescs[m_nPrimaryCamIdx].size[2]));
        oSaliency.at<float>(nRowIdx,nColIdx) = std::max(fCurrDescSparseness,fCurrDistSparseness);
    #if STEREOSEGMATCH_DEFAULT_SALIENT_SHP_RAD>0
        const cv::Mat& oFGDist = m_vNextFeats[m_nPrimaryCamIdx*FeatPackOffset+FeatPackOffset_FGDist];
        const float fCurrFGDist = oFGDist.at<float>(nRowIdx,nColIdx);
        oSaliency.at<float>(nRowIdx,nColIdx) *= std::max(1-fCurrFGDist/STEREOSEGMATCH_DEFAULT_SALIENT_SHP_RAD,0.0f);
    #endif //STEREOSEGMATCH_DEFAULT_SALIENT_SHP_RAD>0
    }
    cv::normalize(oSaliency,oSaliency,1,0,cv::NORM_MINMAX,-1,m_aROIs[m_nPrimaryCamIdx]);
    lvDbgExec( // cv::normalize leftover fp errors are sometimes awful; need to 0-max when using map
        for(int nRowIdx=0; nRowIdx<oSaliency.rows; ++nRowIdx)
            for(int nColIdx=0; nColIdx<oSaliency.cols; ++nColIdx)
                lvDbgAssert((oSaliency.at<float>(nRowIdx,nColIdx)>=-1e-6f && oSaliency.at<float>(nRowIdx,nColIdx)<=1.0f+1e-6f) || m_aROIs[m_nPrimaryCamIdx](nRowIdx,nColIdx)==0);
    );
#if STEREOSEGMATCH_CONFIG_USE_SALIENT_MAP_BORDR
    cv::multiply(oSaliency,cv::Mat_<float>(oSaliency.size(),1.0f).setTo(0.5f,m_aDescROIs[m_nPrimaryCamIdx]==0),oSaliency);
#endif //STEREOSEGMATCH_CONFIG_USE_SALIENT_MAP_BORDR
    if(lv::getVerbosity()>=4) {
        cv::imshow("oSaliency_shp",oSaliency);
        cv::waitKey(1);
    }
    lvLog_(3,"Shape saliency map computed in %f second(s).",oLocalTimer.tock());
    /*if(m_pDisplayHelper) {
        std::vector<std::pair<cv::Mat,std::string>> vAffMaps;
        for(InternalLabelType nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx)
            vAffMaps.push_back(std::pair<cv::Mat,std::string>(lv::squeeze(lv::getSubMat(oAffinity,2,(int)nLabelIdx)),std::string()));
        m_pDisplayHelper->displayAlbumAndWaitKey(vAffMaps);
    }*/
}

void StereoSegmMatcher::GraphModelData::calcShapeDistFeatures(const cv::Mat_<InternalLabelType>& oInputMask, size_t nCamIdx) {
    lvDbgExceptionWatch;
    lvDbgAssert_(nCamIdx<getCameraCount(),"bad input cam index");
    lvDbgAssert_(oInputMask.dims==2 && m_oGridSize==oInputMask.size(),"input had the wrong size");
    lvDbgAssert_(oInputMask.type()==CV_8UC1,"unexpected input mask type");
    cv::Mat& oFGDist = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_FGDist];
    cv::distanceTransform(oInputMask==0,oFGDist,cv::DIST_L2,cv::DIST_MASK_PRECISE,CV_32F);
    cv::exp(STEREOSEGMATCH_DEFAULT_DISTTRANSF_SCALE*oFGDist,oFGDist);
    //lvPrint(cv::Mat_<float>(oFGDist(cv::Rect(0,128,256,1))));
    cv::divide(1.0,oFGDist,oFGDist);
    oFGDist -= 1.0f;
    cv::min(oFGDist,1000.0f,oFGDist);
    //lvPrint(cv::Mat_<float>(oFGDist(cv::Rect(0,128,256,1))));
    cv::Mat& oBGDist = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_BGDist];
    cv::distanceTransform(oInputMask>0,oBGDist,cv::DIST_L2,cv::DIST_MASK_PRECISE,CV_32F);
    cv::exp(STEREOSEGMATCH_DEFAULT_DISTTRANSF_SCALE*oBGDist,oBGDist);
    //lvPrint(cv::Mat_<float>(oBGSim(cv::Rect(0,128,256,1))));
    cv::divide(1.0,oBGDist,oBGDist);
    oBGDist -= 1.0f;
    cv::min(oBGDist,1000.0f,oBGDist);
    //lvPrint(cv::Mat_<float>(oBGDist(cv::Rect(0,128,256,1))));
}

void StereoSegmMatcher::GraphModelData::setNextFeatures(const cv::Mat& oPackedFeats) {
    lvDbgExceptionWatch;
    lvAssert_(!oPackedFeats.empty() && oPackedFeats.isContinuous(),"features packet must be non-empty and continuous");
    if(m_vExpectedFeatPackInfo.empty()) {
        m_vExpectedFeatPackInfo.resize(FeatPackSize);
        for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_InitFGDist] = lv::MatInfo(m_oGridSize,CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_InitBGDist] = lv::MatInfo(m_oGridSize,CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_FGDist] = lv::MatInfo(m_oGridSize,CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_BGDist] = lv::MatInfo(m_oGridSize,CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_GradY] = lv::MatInfo(m_oGridSize,CV_8UC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_GradX] = lv::MatInfo(m_oGridSize,CV_8UC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_GradMag] = lv::MatInfo(m_oGridSize,CV_8UC1);
        }
        m_vExpectedFeatPackInfo[FeatPack_ImgSaliency] = lv::MatInfo(m_oGridSize,CV_32FC1);
        m_vExpectedFeatPackInfo[FeatPack_ShpSaliency] = lv::MatInfo(m_oGridSize,CV_32FC1);
        const int nRows = (int)m_oGridSize(0);
        const int nCols = (int)m_oGridSize(1);
        m_vExpectedFeatPackInfo[FeatPack_ImgAffinity] = lv::MatInfo(std::array<int,3>{nRows,nCols,(int)m_nRealStereoLabels},CV_32FC1);
        m_vExpectedFeatPackInfo[FeatPack_ShpAffinity] = lv::MatInfo(std::array<int,3>{nRows,nCols,(int)m_nRealStereoLabels},CV_32FC1);
    }
    m_oNextPackedFeats = oPackedFeats; // get pointer to data instead of copy; provider should not overwrite until next 'apply' call!
    m_vNextFeats = lv::unpackData(m_oNextPackedFeats,m_vExpectedFeatPackInfo);
    for(cv::Mat& oFeatMap : m_vNextFeats)
        lvAssert_(oFeatMap.isContinuous(),"internal func used non-continuous data block for feature maps");
    m_bUsePrecalcFeatsNext = true;
}

inline StereoSegmMatcher::OutputLabelType StereoSegmMatcher::GraphModelData::getRealLabel(InternalLabelType nLabel) const {
    lvDbgExceptionWatch;
    lvDbgAssert(nLabel<m_vStereoLabels.size());
    return m_vStereoLabels[nLabel];
}

inline StereoSegmMatcher::InternalLabelType StereoSegmMatcher::GraphModelData::getInternalLabel(OutputLabelType nRealLabel) const {
    lvDbgExceptionWatch;
    lvDbgAssert(nRealLabel==s_nOccludedLabel || nRealLabel==s_nDontCareLabel);
    lvDbgAssert(nRealLabel>=(OutputLabelType)m_nMinDispOffset && nRealLabel<=(OutputLabelType)m_nMaxDispOffset);
    lvDbgAssert(((nRealLabel-m_nMinDispOffset)%m_nDispOffsetStep)==0);
    return (InternalLabelType)std::distance(m_vStereoLabels.begin(),std::find(m_vStereoLabels.begin(),m_vStereoLabels.end(),nRealLabel));
}

inline int StereoSegmMatcher::GraphModelData::getOffsetValue(size_t nCamIdx, InternalLabelType nLabel) const {
    static_assert(getCameraCount()==2,"bad hardcoded offset sign");
    lvDbgExceptionWatch;
    lvDbgAssert(nCamIdx==size_t(0) || nCamIdx==size_t(1));
    lvDbgAssert(nLabel<m_nRealStereoLabels);
    const OutputLabelType nRealLabel = getRealLabel(nLabel);
    lvDbgAssert((int)nRealLabel>=(int)m_nMinDispOffset && (int)nRealLabel<=(int)m_nMaxDispOffset);
    return (nCamIdx==size_t(0))?(-nRealLabel):(nRealLabel);
}

inline int StereoSegmMatcher::GraphModelData::getOffsetColIdx(size_t nCamIdx, int nColIdx, InternalLabelType nLabel) const {
    lvDbgExceptionWatch;
    lvDbgAssert(nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    return nColIdx+getOffsetValue(nCamIdx,nLabel);
}

inline StereoSegmMatcher::AssocCountType StereoSegmMatcher::GraphModelData::getAssocCount(int nRowIdx, int nColIdx) const {
    lvDbgExceptionWatch;
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0]);
    lvDbgAssert(m_nPrimaryCamIdx==1 || (nColIdx>=-(int)m_nMaxDispOffset && nColIdx<(int)m_oGridSize[1]));
    lvDbgAssert(m_nPrimaryCamIdx==0 || (nColIdx>=0 && nColIdx<int(m_oGridSize[1]+m_nMaxDispOffset)));
    lvDbgAssert(m_oAssocCounts.dims==2 && !m_oAssocCounts.empty() && m_oAssocCounts.isContinuous());
    const size_t nMapOffset = ((m_nPrimaryCamIdx==size_t(1))?size_t(0):m_nMaxDispOffset);
    return ((AssocCountType*)m_oAssocCounts.data)[nRowIdx*m_oAssocCounts.cols + (nColIdx+nMapOffset)/m_nDispOffsetStep];
}

inline void StereoSegmMatcher::GraphModelData::addAssoc(int nRowIdx, int nColIdx, InternalLabelType nLabel) const {
    static_assert(getCameraCount()==2,"bad hardcoded assoc range check");
    lvDbgExceptionWatch;
    lvDbgAssert(nLabel<m_nDontCareLabelIdx);
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0] && nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    const int nAssocColIdx = getOffsetColIdx(m_nPrimaryCamIdx,nColIdx,nLabel);
    lvDbgAssert(m_nPrimaryCamIdx==1 || (nAssocColIdx<=int(nColIdx-m_nMinDispOffset) && nAssocColIdx>=int(nColIdx-m_nMaxDispOffset)));
    lvDbgAssert(m_nPrimaryCamIdx==0 || (nAssocColIdx>=int(nColIdx+m_nMinDispOffset) && nAssocColIdx<=int(nColIdx+m_nMaxDispOffset)));
    const size_t nMapOffset = ((m_nPrimaryCamIdx==size_t(1))?size_t(0):m_nMaxDispOffset);
    lvDbgAssert(m_oAssocMap.dims==3 && !m_oAssocMap.empty() && m_oAssocMap.isContinuous());
    AssocIdxType* const pAssocList = ((AssocIdxType*)m_oAssocMap.data)+(nRowIdx*m_oAssocMap.size[1] + (nAssocColIdx+nMapOffset)/m_nDispOffsetStep)*m_oAssocMap.size[2];
    lvDbgAssert(pAssocList==m_oAssocMap.ptr<AssocIdxType>(nRowIdx,(nAssocColIdx+nMapOffset)/m_nDispOffsetStep));
    const size_t nListOffset = size_t(nLabel)*m_nDispOffsetStep + nColIdx%m_nDispOffsetStep;
    lvDbgAssert((int)nListOffset<m_oAssocMap.size[2]);
    lvDbgAssert(pAssocList[nListOffset]==AssocIdxType(-1));
    pAssocList[nListOffset] = AssocIdxType(nColIdx);
    lvDbgAssert(m_oAssocCounts.dims==2 && !m_oAssocCounts.empty() && m_oAssocCounts.isContinuous());
    AssocCountType& nAssocCount = ((AssocCountType*)m_oAssocCounts.data)[nRowIdx*m_oAssocCounts.cols + (nAssocColIdx+nMapOffset)/m_nDispOffsetStep];
    lvDbgAssert((&nAssocCount)==&m_oAssocCounts(nRowIdx,int((nAssocColIdx+nMapOffset)/m_nDispOffsetStep)));
    lvDbgAssert(nAssocCount<std::numeric_limits<AssocCountType>::max());
    ++nAssocCount;
}

inline void StereoSegmMatcher::GraphModelData::removeAssoc(int nRowIdx, int nColIdx, InternalLabelType nLabel) const {
    static_assert(getCameraCount()==2,"bad hardcoded assoc range check");
    lvDbgExceptionWatch;
    lvDbgAssert(nLabel<m_nDontCareLabelIdx);
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0] && nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    const int nAssocColIdx = getOffsetColIdx(m_nPrimaryCamIdx,nColIdx,nLabel);
    lvDbgAssert(m_nPrimaryCamIdx==1 || (nAssocColIdx<=int(nColIdx-m_nMinDispOffset) && nAssocColIdx>=int(nColIdx-m_nMaxDispOffset)));
    lvDbgAssert(m_nPrimaryCamIdx==0 || (nAssocColIdx>=int(nColIdx+m_nMinDispOffset) && nAssocColIdx<=int(nColIdx+m_nMaxDispOffset)));
    const size_t nMapOffset = ((m_nPrimaryCamIdx==size_t(1))?size_t(0):m_nMaxDispOffset);
    lvDbgAssert(m_oAssocMap.dims==3 && !m_oAssocMap.empty() && m_oAssocMap.isContinuous());
    AssocIdxType* const pAssocList = ((AssocIdxType*)m_oAssocMap.data)+(nRowIdx*m_oAssocMap.size[1] + (nAssocColIdx+nMapOffset)/m_nDispOffsetStep)*m_oAssocMap.size[2];
    lvDbgAssert(pAssocList==m_oAssocMap.ptr<AssocIdxType>(nRowIdx,(nAssocColIdx+nMapOffset)/m_nDispOffsetStep));
    const size_t nListOffset = size_t(nLabel)*m_nDispOffsetStep + nColIdx%m_nDispOffsetStep;
    lvDbgAssert((int)nListOffset<m_oAssocMap.size[2]);
    lvDbgAssert(pAssocList[nListOffset]==AssocIdxType(nColIdx));
    pAssocList[nListOffset] = AssocIdxType(-1);
    lvDbgAssert(m_oAssocCounts.dims==2 && !m_oAssocCounts.empty() && m_oAssocCounts.isContinuous());
    AssocCountType& nAssocCount = ((AssocCountType*)m_oAssocCounts.data)[nRowIdx*m_oAssocCounts.cols + (nAssocColIdx+nMapOffset)/m_nDispOffsetStep];
    lvDbgAssert((&nAssocCount)==&m_oAssocCounts(nRowIdx,int((nAssocColIdx+nMapOffset)/m_nDispOffsetStep)));
    lvDbgAssert(nAssocCount>AssocCountType(0));
    --nAssocCount;
}

inline StereoSegmMatcher::ValueType StereoSegmMatcher::GraphModelData::calcAddAssocCost(int nRowIdx, int nColIdx, InternalLabelType nLabel) const {
    static_assert(getCameraCount()==2,"bad hardcoded assoc range check");
    lvDbgExceptionWatch;
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0] && nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    if(nLabel<m_nDontCareLabelIdx) {
        const int nAssocColIdx = getOffsetColIdx(m_nPrimaryCamIdx,nColIdx,nLabel);
        lvDbgAssert(m_nPrimaryCamIdx==1 || (nAssocColIdx<=int(nColIdx-m_nMinDispOffset) && nAssocColIdx>=int(nColIdx-m_nMaxDispOffset)));
        lvDbgAssert(m_nPrimaryCamIdx==0 || (nAssocColIdx>=int(nColIdx+m_nMinDispOffset) && nAssocColIdx<=int(nColIdx+m_nMaxDispOffset)));
        const AssocCountType nAssocCount = getAssocCount(nRowIdx,nAssocColIdx);
        lvDbgAssert(nAssocCount<m_aAssocCostApproxAddLUT.size());
        return m_aAssocCostApproxAddLUT[nAssocCount];
    }
    return cost_cast(100000); // @@@@ dirty
}

inline StereoSegmMatcher::ValueType StereoSegmMatcher::GraphModelData::calcRemoveAssocCost(int nRowIdx, int nColIdx, InternalLabelType nLabel) const {
    static_assert(getCameraCount()==2,"bad hardcoded assoc range check");
    lvDbgExceptionWatch;
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0] && nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    if(nLabel<m_nDontCareLabelIdx) {
        const int nAssocColIdx = getOffsetColIdx(m_nPrimaryCamIdx,nColIdx,nLabel);
        lvDbgAssert(m_nPrimaryCamIdx==1 || (nAssocColIdx<=int(nColIdx-m_nMinDispOffset) && nAssocColIdx>=int(nColIdx-m_nMaxDispOffset)));
        lvDbgAssert(m_nPrimaryCamIdx==0 || (nAssocColIdx>=int(nColIdx+m_nMinDispOffset) && nAssocColIdx<=int(nColIdx+m_nMaxDispOffset)));
        const AssocCountType nAssocCount = getAssocCount(nRowIdx,nAssocColIdx);
        lvDbgAssert(nAssocCount>0); // cannot be zero, must have at least an association in order to remove it
        lvDbgAssert(nAssocCount<m_aAssocCostApproxAddLUT.size());
        return m_aAssocCostApproxRemLUT[nAssocCount];
    }
    return -cost_cast(100000); // @@@@ dirty
}

StereoSegmMatcher::ValueType StereoSegmMatcher::GraphModelData::calcTotalAssocCost() const {
    lvDbgExceptionWatch;
    lvDbgAssert(m_oGridSize.total()==m_vStereoNodeInfos.size());
    ValueType tEnergy = cost_cast(0);
    const int nColIdxStart = ((m_nPrimaryCamIdx==size_t(1))?int(m_nMinDispOffset):-int(m_nMaxDispOffset));
    const int nColIdxEnd = ((m_nPrimaryCamIdx==size_t(1))?int(m_oGridSize[1]+m_nMaxDispOffset):int(m_oGridSize[1]-m_nMinDispOffset));
    for(int nRowIdx=0; nRowIdx<(int)m_oGridSize[0]; ++nRowIdx)
        for(int nColIdx=nColIdxStart; nColIdx<nColIdxEnd; nColIdx+=m_nDispOffsetStep)
            tEnergy += m_aAssocCostRealSumLUT[getAssocCount(nRowIdx,nColIdx)];
    lvDbgAssert(tEnergy>=cost_cast(0));
    return tEnergy;
}

inline StereoSegmMatcher::ValueType StereoSegmMatcher::GraphModelData::calcStereoUnaryMoveCost(size_t nGraphNodeIdx, InternalLabelType nOldLabel, InternalLabelType nNewLabel) const {
    lvDbgExceptionWatch;
    lvDbgAssert(nGraphNodeIdx<m_nValidStereoGraphNodes);
    const size_t nLUTNodeIdx = m_vValidStereoLUTNodeIdxs[nGraphNodeIdx];
    const StereoNodeInfo& oNode = m_vStereoNodeInfos[nLUTNodeIdx];
    lvDbgAssert(oNode.bValidGraphNode);
    if(nOldLabel!=nNewLabel) {
        lvDbgAssert(nOldLabel<m_nStereoLabels && nNewLabel<m_nStereoLabels);
        const ValueType tAssocEnergyCost = calcRemoveAssocCost(oNode.nRowIdx,oNode.nColIdx,nOldLabel)+calcAddAssocCost(oNode.nRowIdx,oNode.nColIdx,nNewLabel);
        const ExplicitFunction& vUnaryStereoLUT = *oNode.pUnaryFunc;
        const ValueType tUnaryEnergyInit = vUnaryStereoLUT(nOldLabel);
        const ValueType tUnaryEnergyModif = vUnaryStereoLUT(nNewLabel);
        return tAssocEnergyCost+tUnaryEnergyModif-tUnaryEnergyInit;
    }
    else
        return cost_cast(0);
}

#if (STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF)

void StereoSegmMatcher::GraphModelData::calcStereoMoveCosts(InternalLabelType nNewLabel) const {
    lvDbgExceptionWatch;
    lvDbgAssert(m_oGridSize.total()==m_vStereoNodeInfos.size() && m_oGridSize.total()>1 && m_oGridSize==m_oStereoUnaryCosts.size);
    const InternalLabelType* pInitLabeling = ((InternalLabelType*)m_aStereoLabelings[m_nPrimaryCamIdx].data);
    // @@@@@ openmp here?
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidStereoGraphNodes; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_vValidStereoLUTNodeIdxs[nGraphNodeIdx];
        const StereoNodeInfo& oNode = m_vStereoNodeInfos[nLUTNodeIdx];
        const InternalLabelType& nInitLabel = pInitLabeling[nLUTNodeIdx];
        lvIgnore(oNode);
        lvDbgAssert(oNode.bValidGraphNode);
        lvDbgAssert(&nInitLabel==&m_aStereoLabelings[m_nPrimaryCamIdx](oNode.nRowIdx,oNode.nColIdx));
        ValueType& tUnaryCost = ((ValueType*)m_oStereoUnaryCosts.data)[nLUTNodeIdx];
        lvDbgAssert(&tUnaryCost==&m_oStereoUnaryCosts(m_vStereoNodeInfos[nLUTNodeIdx].nRowIdx,m_vStereoNodeInfos[nLUTNodeIdx].nColIdx));
    #if STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF
        tUnaryCost = calcStereoUnaryMoveCost(nGraphNodeIdx,nInitLabel,nNewLabel);
    #elif STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF
        tUnaryCost = -calcStereoUnaryMoveCost(nGraphNodeIdx,nInitLabel,nNewLabel);
        for(const auto& p : m_node_clique_list[nGraphNodeIdx]) {
            const int nStereoCliqueIdx = (int)p.first;
            const int nStereoCliqueDimIdx = (int)p.second;
            const ValueType tInitCliqueCost = m_dual(nStereoCliqueIdx,int(nStereoCliqueDimIdx*m_nStereoLabels+nInitLabel));
            const ValueType tNewCliqueCost = m_dual(nStereoCliqueIdx,int(nStereoCliqueDimIdx*m_nStereoLabels+nNewLabel));
            tUnaryCost += tInitCliqueCost-tNewCliqueCost;
        }
    #endif //STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF
    }
}

#endif //(STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF)

#if STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF

bool StereoSegmMatcher::GraphModelData::InitialFusionLabeling() {
    const InternalLabelType* pLabeling = ((InternalLabelType*)m_aStereoLabelings[__cam].data);
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[__cam]; ++nGraphNodeIdx) {
        const size_t& nLUTNodeIdx = m_avValidLUTNodeIdxs[__cam][nGraphNodeIdx];
        if(pLabeling[nLUTNodeIdx]!=__alpha)
            return true;
    }
    return false;
}

void StereoSegmMatcher::GraphModelData::PreEditDual(SubmodularIBFS<ValueType,VarId>& crf) {
    auto& fixedVars = crf.Params().fixedVars;
    fixedVars.resize(m_anValidGraphNodes[__cam]);
    const InternalLabelType* pLabeling = ((InternalLabelType*)m_aStereoLabelings[__cam].data);
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[__cam]; ++nGraphNodeIdx) {
        const size_t& nLUTNodeIdx = m_avValidLUTNodeIdxs[__cam][nGraphNodeIdx];
        fixedVars[nGraphNodeIdx] = (pLabeling[nLUTNodeIdx]==__alpha);
    }
    // Allocate all the buffers we need in one place, resize as necessary
    Label label_buf[32];
    std::vector<Label> current_labels;
    std::vector<Label> fusion_labels;
    std::vector<ValueType> psi;
    std::vector<ValueType> current_lambda;
    std::vector<ValueType> fusion_lambda;

    auto& crf_cliques = crf.Graph().GetCliques();
    lvDbgAssert(crf_cliques.size() == m_nStereoCliqueCount);
    int clique_index = 0;
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[__cam]; ++nGraphNodeIdx) {
        const size_t& nLUTNodeIdx = m_avValidLUTNodeIdxs[__cam][nGraphNodeIdx];
        const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
        lvDbgAssert(oNode.abValidGraphNode[__cam]);
        for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
            const PairwClique& oStereoClique = oNode.aaStereoPairwCliques[__cam][nOrientIdx];
            if(oStereoClique) {
                constexpr size_t nCliqueSize = oStereoClique.s_nCliqueSize;
                ValueType* lambda_a = m_dual.ptr<ValueType>(clique_index);
                auto& crf_c = crf_cliques[clique_index];
                lvDbgAssert(nCliqueSize == crf_c.Size());
                std::vector<ValueType>& energy_table = crf_c.EnergyTable();
                sospd::Assgn max_assgn = sospd::Assgn(size_t(1)<<nCliqueSize);
                lvDbgAssert(energy_table.size() == max_assgn);
                psi.resize(nCliqueSize);
                current_labels.resize(nCliqueSize);
                fusion_labels.resize(nCliqueSize);
                current_lambda.resize(nCliqueSize);
                fusion_lambda.resize(nCliqueSize);
                for(size_t i = 0; i < nCliqueSize; ++i) {
                    current_labels[i] = pLabeling[m_avValidLUTNodeIdxs[__cam][crf_c.Nodes()[i]]];
                    fusion_labels[i] = __alpha;
                    current_lambda[i] = lambda_a[i*m_nStereoLabels+current_labels[i]];
                    fusion_lambda[i] = lambda_a[i*m_nStereoLabels+fusion_labels[i]];
                }
                // compute costs of all fusion assignments
                const ExplicitFunction& vPairwStereoLUT = *oStereoClique.m_pGraphFunctionPtr;
                sospd::Assgn last_gray = 0;
                for(size_t i_idx = 0; i_idx < nCliqueSize; ++i_idx)
                    label_buf[i_idx] = current_labels[i_idx];
                energy_table[0] = vPairwStereoLUT(label_buf);
                for(sospd::Assgn a = 1; a < max_assgn; ++a) {
                    sospd::Assgn gray = a ^ (a >> 1);
                    sospd::Assgn diff = gray ^ last_gray;
                    int changed_idx = __builtin_ctz(diff);
                    if (diff & gray)
                        label_buf[changed_idx] = fusion_labels[changed_idx];
                    else
                        label_buf[changed_idx] = current_labels[changed_idx];
                    last_gray = gray;
                    energy_table[gray] = vPairwStereoLUT(label_buf);
                }
                // compute the residual function: g(S) - lambda_fusion(S) - lambda_current(C\S)
                sospd::SubtractLinear(nCliqueSize,energy_table,fusion_lambda,current_lambda);
                lvDbgAssert(energy_table[0] == 0); // check tightness of current labeling
                ++clique_index;
            }
        }
    }
}

bool StereoSegmMatcher::GraphModelData::UpdatePrimalDual(SubmodularIBFS<ValueType,VarId>& crf) {
    bool ret = false;
    crf.ClearUnaries();
    crf.AddConstantTerm(-crf.GetConstantTerm());
    const InternalLabelType* pLabeling = ((InternalLabelType*)m_aStereoLabelings[__cam].data);
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[__cam]; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[__cam][nGraphNodeIdx];
        const InternalLabelType& nInitLabel = pLabeling[nLUTNodeIdx];
        lvDbgAssert(&nInitLabel==&m_aStereoLabelings[__cam](m_vNodeInfos[nLUTNodeIdx].nRowIdx,m_vNodeInfos[nLUTNodeIdx].nColIdx));
        const ValueType& tUnaryCost = ((ValueType*)m_aStereoUnaryCosts[__cam].data)[nLUTNodeIdx];
        lvDbgAssert(&tUnaryCost==&m_aStereoUnaryCosts[__cam](m_vNodeInfos[nLUTNodeIdx].nRowIdx,m_vNodeInfos[nLUTNodeIdx].nColIdx));
        if (tUnaryCost>cost_cast(0))
            crf.AddUnaryTerm((int)nGraphNodeIdx, tUnaryCost, 0);
        else
            crf.AddUnaryTerm((int)nGraphNodeIdx, 0, -tUnaryCost);
    }
    crf.Solve();
    cv::Mat_<InternalLabelType>& oCurrStereoLabeling = m_aStereoLabelings[__cam];
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[__cam]; ++nGraphNodeIdx) {
        const size_t& nLUTNodeIdx = m_avValidLUTNodeIdxs[__cam][nGraphNodeIdx];
        const int& nRowIdx = m_vNodeInfos[nLUTNodeIdx].nRowIdx;
        const int& nColIdx = m_vNodeInfos[nLUTNodeIdx].nColIdx;
        const int nMoveLabel = crf.GetLabel((int)nGraphNodeIdx);
        lvDbgAssert(nMoveLabel==0 || nMoveLabel==1 || nMoveLabel<0);
        if(nMoveLabel==1) { // node label changed to alpha
            const InternalLabelType nOldLabel = oCurrStereoLabeling(nRowIdx,nColIdx);
            if(nOldLabel!=__alpha) {
                if(nOldLabel<m_nDontCareLabelIdx)
                    removeAssoc(__cam,nRowIdx,nColIdx,nOldLabel);
                oCurrStereoLabeling(nRowIdx,nColIdx) = __alpha;
                if(__alpha<m_nDontCareLabelIdx)
                    addAssoc(__cam,nRowIdx,nColIdx,__alpha);
                ret = true;
            }
        }
    }
    const auto& clique = crf.Graph().GetCliques();
    size_t nCliqueIdx = 0;
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[__cam]; ++nGraphNodeIdx) {
        const size_t& nLUTNodeIdx = m_avValidLUTNodeIdxs[__cam][nGraphNodeIdx];
        const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
        lvDbgAssert(oNode.abValidGraphNode[__cam]);
        for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
            const PairwClique& oStereoClique = oNode.aaStereoPairwCliques[__cam][nOrientIdx];
            if(oStereoClique) {
                auto& crf_c = clique[nCliqueIdx];
                const std::vector<ValueType>& phiCi = crf_c.AlphaCi();
                for (size_t j = 0; j < phiCi.size(); ++j) {
                    m_dual((int)nCliqueIdx,(int)(j*m_nStereoLabels+__alpha)) += phiCi[j];
                    m_heights((int)crf_c.Nodes()[j],(int)__alpha) += phiCi[j];
                }
                ++nCliqueIdx;
            }
        }
    }
    return ret;
}

void StereoSegmMatcher::GraphModelData::PostEditDual(SubmodularIBFS<ValueType,VarId>& crf/*temp for clique nodes & dbg*/) {
    Label labelBuf[32];
    int clique_index = 0;
    const auto& clique = crf.Graph().GetCliques();
    const InternalLabelType* pLabeling = ((InternalLabelType*)m_aStereoLabelings[__cam].data);
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[__cam]; ++nGraphNodeIdx) {
        const size_t& nLUTNodeIdx = m_avValidLUTNodeIdxs[__cam][nGraphNodeIdx];
        const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
        lvDbgAssert(oNode.abValidGraphNode[__cam]);
        for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
            const PairwClique& oStereoClique = oNode.aaStereoPairwCliques[__cam][nOrientIdx];
            if(oStereoClique) {
                auto& crf_c = clique[clique_index];
                int k = (int)crf_c.Nodes().size();
                ASSERT(k < 32);
                ValueType lambdaSum = 0;
                for (int i = 0; i < k; ++i) {
                    labelBuf[i] = pLabeling[m_avValidLUTNodeIdxs[__cam][crf_c.Nodes()[i]]];
                    lambdaSum += m_dual(clique_index,int(i*m_nStereoLabels+labelBuf[i]));
                }
                const ExplicitFunction& vPairwStereoLUT = *oStereoClique.m_pGraphFunctionPtr;
                ValueType energy = vPairwStereoLUT(labelBuf);
                ValueType correction = energy - lambdaSum;
                if (correction > 0) {
                    std::cout << "Bad clique in PostEditDual!\t Id:" << clique_index << "\n";
                    std::cout << "Correction: " << correction << "\tenergy: " << energy << "\tlambdaSum " << lambdaSum << "\n";
                    const auto& c = crf.Graph().GetCliques()[clique_index];
                    std::cout << "EnergyTable: ";
                    for (const auto& e : c.EnergyTable())
                        std::cout << e << ", ";
                    std::cout << "\n";
                }
                ASSERT(correction <= 0);
                ValueType avg = correction / k;
                int remainder = correction % k;
                if (remainder < 0) {
                    avg -= 1;
                    remainder += k;
                }
                for (int i = 0; i < k; ++i) {
                    auto& lambda_ail = m_dual(clique_index,int(i*m_nStereoLabels+labelBuf[i]));
                    m_heights((int)crf_c.Nodes()[i],(int)labelBuf[i]) -= lambda_ail;
                    lambda_ail += avg;
                    if (i < remainder)
                        lambda_ail += 1;
                    m_heights((int)crf_c.Nodes()[i],(int)labelBuf[i]) += lambda_ail;
                }
                ++clique_index;
            }
        }
    }
}

#endif //STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF

#if (STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF)

void StereoSegmMatcher::GraphModelData::calcResegmMoveCosts(size_t nCamIdx, InternalLabelType nNewLabel) const {
    lvDbgExceptionWatch;
    lvDbgAssert(m_oGridSize.total()==m_avResegmNodeInfos[nCamIdx].size()*s_nTemporalLayers && m_oGridSize.total()>1);
    lvDbgAssert(m_oGridSize==m_aResegmUnaryCosts[nCamIdx].size);
    const InternalLabelType* pLabeling = ((InternalLabelType*)m_aResegmLabelings[nCamIdx].data);
    // @@@@@ openmp here?
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidResegmGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_avValidResegmLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
        const NodeInfo& oNode = m_avResegmNodeInfos[nCamIdx][nLUTNodeIdx];
        const InternalLabelType& nInitLabel = pLabeling[nLUTNodeIdx];
        lvDbgAssert(&nInitLabel==&m_aResegmLabelings[nCamIdx](oNode.nRowIdx,oNode.nColIdx));
        lvDbgAssert(nInitLabel==s_nForegroundLabelIdx || nInitLabel==s_nBackgroundLabelIdx);
        ValueType& tUnaryCost = ((ValueType*)m_aResegmUnaryCosts[nCamIdx].data)[nLUTNodeIdx];
        lvDbgAssert(&tUnaryCost==&m_aResegmUnaryCosts[nCamIdx](oNode.nRowIdx,oNode.nColIdx));
        if(nInitLabel!=nNewLabel) {
            const ExplicitFunction& vUnaryResegmLUT = *oNode.pUnaryFunc;
            const ValueType tEnergyInit = vUnaryResegmLUT(nInitLabel);
            const ValueType tEnergyModif = vUnaryResegmLUT(nNewLabel);
            tUnaryCost = tEnergyModif-tEnergyInit;
        }
        else
            tUnaryCost = cost_cast(0);
    }
}

#endif //(STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF)

opengm::InferenceTermination StereoSegmMatcher::GraphModelData::infer() {
    static_assert(s_nInputArraySize==4 && getCameraCount()==2,"hardcoded indices below will break");
    lvDbgExceptionWatch;
    if(lv::getVerbosity()>=3) {
        cv::Mat oTargetImg = m_aInputs[m_nPrimaryCamIdx*InputPackOffset+InputPackOffset_Img].clone();
        if(oTargetImg.channels()==1)
            cv::cvtColor(oTargetImg,oTargetImg,cv::COLOR_GRAY2BGR);
        cv::Mat oTargetMask = m_aInputs[m_nPrimaryCamIdx*InputPackOffset+InputPackOffset_Mask].clone();
        cv::cvtColor(oTargetMask,oTargetMask,cv::COLOR_GRAY2BGR);
        oTargetMask &= cv::Vec3b(255,0,0);
        cv::imshow("primary input",(oTargetImg+oTargetMask)/2);
        if(getCameraCount()==2) {
            const size_t nSecondaryCamIdx = m_nPrimaryCamIdx^1;
            cv::Mat oOtherImg = m_aInputs[(nSecondaryCamIdx)*InputPackOffset+InputPackOffset_Img].clone();
            if(oOtherImg.channels()==1)
                cv::cvtColor(oOtherImg,oOtherImg,cv::COLOR_GRAY2BGR);
            cv::Mat oOtherMask = m_aInputs[(nSecondaryCamIdx)*InputPackOffset+InputPackOffset_Mask].clone();
            cv::cvtColor(oOtherMask,oOtherMask,cv::COLOR_GRAY2BGR);
            oOtherMask &= cv::Vec3b(255,0,0);
            cv::imshow("other input",(oOtherImg+oOtherMask)/2);
        }
        cv::waitKey(1);
    }
    updateStereoModel(true);
    resetStereoLabelings(m_nPrimaryCamIdx);
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        cv::Mat(((m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Mask]>0)&m_aROIs[nCamIdx])&s_nForegroundLabelIdx).copyTo(m_aResegmLabelings[nCamIdx]);
        updateResegmModel(nCamIdx,true);
        lvDbgAssert(m_oGridSize.dims()==2 && m_oGridSize==m_aStereoLabelings[nCamIdx].size && m_oGridSize==m_aResegmLabelings[nCamIdx].size);
        lvDbgAssert(m_oGridSize.total()==m_avResegmNodeInfos[nCamIdx].size());
        lvDbgAssert(m_anValidResegmGraphNodes[nCamIdx]==m_avValidResegmLUTNodeIdxs[nCamIdx].size());
    }
    lvLog_(2,"Running inference for primary camera idx=%d...",(int)m_nPrimaryCamIdx);
#if (STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF)
    using HOEReducer = HigherOrderEnergy<ValueType,s_nMaxOrder>;
#endif //(STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF)
#if STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF
    //calcStereoCosts(m_nPrimaryCamIdx);
    //const size_t nStereoPairwCount = m_nValidStereoGraphNodes*s_nPairwOrients; (max, not actual) // numPairs_
    // pairs_ : lookup node, 1st pairw node idx + 2nd pairw node idx
    // distance_[2ndlabel*nlabels + 1stlabel] : m_aaStereoPairwFuncIDs_base[nCamIdx][nOrientIdx](1stlabel,2ndlabel)
    // weights_ : lookup node, each orient has its weight
    /*@@@ todo fastpd setup
    setLabelCosts();
    getNumPairs();
    setPairs();
    setDistance();
    setWeights();
    pdInference_ = new fastPDLib::CV_Fast_PD(
            gm_.numberOfVariables(),
            gm_.numberOfLabels(0),
            labelCosts_,
            numPairs_,
            pairs_,
            distance_,
            parameter_.numberOfIterations_,
            weights_
    );*/
    // @@@@ see if maxflow used in fastpd can be replaced by https://github.com/gerddie/maxflow?
#elif STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF
    constexpr int nMaxStereoEdgesPerNode = (s_nPairwOrients/*+...@@@*/);
    kolmogorov::qpbo::QPBO<ValueType> oStereoMinimizer((int)m_nValidStereoGraphNodes,(int)m_nValidStereoGraphNodes*nMaxStereoEdgesPerNode);
    HOEReducer oStereoReducer;
    size_t nOrderingIdx = 0;
#elif STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF
    static_assert(std::is_integral<StereoSegmMatcher::ValueType>::value,"sospd height weight redistr requires integer type"); // @@@ could rewrite for float type
    //calcStereoCosts(m_nPrimaryCamIdx);
    constexpr bool bUseHeightAlphaExp = STEREOSEGMATCH_CONFIG_USE_SOSPD_ALPHA_HEIGHTS_LABEL_ORDERING;
    lvAssert_(!bUseHeightAlphaExp,"missing impl"); // @@@@
    size_t nOrderingIdx = 0;
    // setup graph/dual/cliquelist
    /*
        if (m_iter == I(0)) {
            SetupGraph(m_ibfs);
            InitialLabeling();
            InitialDual();
            InitialNodeCliqueList();
        }
     */
    SubmodularIBFS<ValueType,VarId> crf; //oStereoMinimizer @@@@
    __cam = m_nPrimaryCamIdx;
    {
        m_nStereoCliqueCount = size_t(0);
        crf.AddNode((int)nGraphNodes);
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidStereoGraphNodes; ++nGraphNodeIdx) {
            const size_t& nLUTNodeIdx = m_vValidStereoLUTNodeIdxs[nGraphNodeIdx];
            const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
            lvDbgAssert(oNode.abValidGraphNode[m_nPrimaryCamIdx]);
            for(size_t nCliqueIdx=0; nCliqueIdx<oNode.vpCliques.size(); ++nCliqueIdx) {
                const Clique& oClique = *oNode.vpCliques[nCliqueIdx];
                lvDbgAssert(oClique);
                const IndexType nCliqueSize = oClique.getSize();
                lvDbgAssert(nCliqueSize<=s_nMaxOrder);
                const IndexType* aLUTNodeIdxs = oClique.getLUTNodeIter();
                lvDbgAssert(aLUTNodeIdxs[0]==nLUTNodeIdx && oNode.abValidGraphNode[nPrimaryCamIdx]);
                crf.AddClique(
                    std::vector<IndexType>(aLUTNodeIdxs,aLUTNodeIdxs+nCliqueSize),
                    std::vector<ValueType>(size_t(1<<nCliqueSize),cost_cast(0))
                );
                ++m_nStereoCliqueCount;
            }
        }
        m_heights.create((int)m_nValidStereoGraphNodes,(int)m_nStereoLabels);
        m_heights = cost_cast(0);
        m_dual.create((int)m_nStereoCliqueCount,(int)(s_nMaxOrder*m_nStereoLabels));
        m_dual = cost_cast(0);
        m_node_clique_list.clear();
        m_node_clique_list.resize(m_nValidStereoGraphNodes);
        std::array<InternalLabelType,s_nMaxOrder> aLabelingBuffer;
        const InternalLabelType* pLabeling = ((InternalLabelType*)m_aStereoLabelings[nPrimaryCamIdx].data);
        const size_t nLabelMapSize = m_aStereoLabelings[nPrimaryCamIdx].total();
        size_t nStereoCliqueIdx = 0;
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidStereoGraphNodes; ++nGraphNodeIdx) {
            const size_t& nLUTNodeIdx = m_avValidLUTNodeIdxs[nPrimaryCamIdx][nGraphNodeIdx];
            const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
            const ExplicitFunction& vUnaryStereoLUT = *oNode.apStereoUnaryFuncs[nPrimaryCamIdx];
            for(size_t nStereoLabel=0; nStereoLabel<m_nStereoLabels; ++nStereoLabel)
                m_heights((int)nGraphNodeIdx,(int)nStereoLabel) += vUnaryStereoLUT(nStereoLabel);
            for(size_t nCliqueIdx=0; nCliqueIdx<oNode.avpStereoCliques.size(); ++nCliqueIdx) {
                const Clique& oClique = *oNode.avpStereoCliques[nPrimaryCamIdx][nCliqueIdx];
                const IndexType nCliqueSize = oClique.getSize();
                const IndexType* aLUTNodeIdxs = oClique.getLUTNodeIter();
                for(IndexType nDimIdx=0; nDimIdx<nCliqueSize; ++nDimIdx) {
                    const IndexType nOffsetLUTNodeIdx = aLUTNodeIdxs[nDimIdx];
                    lvDbgAssert(nOffsetLUTNodeIdx!=SIZE_MAX && nOffsetLUTNodeIdx<nLabelMapSize);
                    aLabelingBuffer[nDimIdx] = pLabeling[nOffsetLUTNodeIdx];
                }
                const ExplicitFunction* pvEnergyLUT = oClique.getFunctionPtr();
                lvDbgAssert(pvEnergyLUT);
                const ExplicitFunction& vEnergyLUT = *pvEnergyLUT;
                const ValueType tCurrCost = vEnergyLUT(aLabelingBuffer.data());
                lvDbgAssert(tCurrCost>=cost_cast(0));
                ValueType* pLambdas = m_dual.ptr<ValueType>((int)nStereoCliqueIdx);
                ValueType tAvgCost = tCurrCost/nCliqueSize;
                const int tRemainderCost = int(tCurrCost)%int(nCliqueSize);
                const IndexType* aGraphNodeIdxs = oClique.getGraphNodeIter();
                lvDbgAssert(aGraphNodeIdxs[0]==nGraphNodeIdx);
                for(IndexType nDimIdx=0; nDimIdx<nCliqueSize; ++nDimIdx) {
                    const IndexType nOffsetGraphNodeIdx = aGraphNodeIdxs[nDimIdx];
                    lvDbgAssert(nOffsetGraphNodeIdx!=SIZE_MAX && nOffsetGraphNodeIdx<m_nValidStereoGraphNodes);
                    ValueType& tLambda = pLambdas[nDimIdx*m_nStereoLabels+aLabelingBuffer[nDimIdx]];
                    tLambda = tAvgCost;
                    if(int(nDimIdx)<tRemainderCost)
                        tLambda += cost_cast(1);
                    m_heights((int)nOffsetGraphNodeIdx,(int)aLabelingBuffer[nDimIdx]) += tLambda;
                    m_node_clique_list[nOffsetGraphNodeIdx].push_back(std::make_pair(nStereoCliqueIdx,nDimIdx));
                }
                ++nStereoCliqueIdx;
            }
        }
    }
#endif //STEREOSEGMATCH_CONFIG_USE_..._STEREO_INF
#if STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF
    constexpr int nMaxResegmEdgesPerNode = int(s_nPairwOrients+s_nTemporalConn);
    std::vector<std::unique_ptr<kolmogorov::qpbo::QPBO<ValueType>>> apResegmMinimizers;
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx)
        apResegmMinimizers.push_back(std::make_unique<kolmogorov::qpbo::QPBO<ValueType>>((int)m_anValidResegmGraphNodes[nCamIdx],(int)m_anValidResegmGraphNodes[nCamIdx]*nMaxResegmEdgesPerNode));
    CamArray<HOEReducer> aResegmReducers;
#elif STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF
    @@@ todo sospd resegm setup
#endif //STEREOSEGMATCH_CONFIG_USE_..._RESEGM_INF
    size_t nMoveIter=0, nConsecUnchangedLabels=0;
    lvDbgAssert(m_vStereoLabelOrdering.size()==m_vStereoLabels.size());
    lv::StopWatch oLocalTimer;
    ValueType tLastStereoEnergy = m_pStereoInf->value();
    ValueType tLastResegmEnergyTotal = std::numeric_limits<ValueType>::max();
    CamArray<ValueType> atLastResegmEnergies = {std::numeric_limits<ValueType>::max(),std::numeric_limits<ValueType>::max()};
    cv::Mat_<InternalLabelType>& oCurrStereoLabeling = m_aStereoLabelings[m_nPrimaryCamIdx];
    bool bJustUpdatedSegm = false;
    while(++nMoveIter<=m_nMaxMoveIterCount && nConsecUnchangedLabels<m_nStereoLabels) {
        const bool bDisableStereoCliques = (STEREOSEGMATCH_CONFIG_USE_UNARY_ONLY_FIRST)&&(nMoveIter<=m_nStereoLabels);
    #if STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF

        // fastpd only works with shared+scaled pairwise costs, and no higher order terms
        opengm::external::FastPD<StereoModelType> oStereoMinimizer2(*m_apStereoModels[nPrimaryCamIdx],opengm::external::FastPD<StereoModelType>::Parameter());
        oStereoMinimizer2.infer();
        std::vector<InternalLabelType> outputlabels;
        oStereoMinimizer2.arg(outputlabels);
        lvAssert(outputlabels.size()==m_anValidGraphNodes[nPrimaryCamIdx]);

        //pdInference_->run(); @@@ just one iter at a time --- split and use while above
        // @@@ nullify stereo pairw costs?

        size_t nChangedStereoLabels = 0;
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nPrimaryCamIdx]; ++nGraphNodeIdx) {
            const size_t& nLUTNodeIdx = m_avValidLUTNodeIdxs[nPrimaryCamIdx][nGraphNodeIdx];
            const int& nRowIdx = m_vNodeInfos[nLUTNodeIdx].nRowIdx;
            const int& nColIdx = m_vNodeInfos[nLUTNodeIdx].nColIdx;
            const InternalLabelType nOldLabel = oCurrStereoLabeling(nRowIdx,nColIdx);
            //const InternalLabelType nNewLabel = pdInference_->_pinfo[nGraphNodeIdx].label;
            const InternalLabelType nNewLabel = outputlabels[nGraphNodeIdx];
            if(nOldLabel!=nNewLabel) {
                if(nOldLabel<m_nDontCareLabelIdx)
                    removeAssoc(nPrimaryCamIdx,nRowIdx,nColIdx,nOldLabel);
                oCurrStereoLabeling(nRowIdx,nColIdx) = nNewLabel;
                if(nNewLabel<m_nDontCareLabelIdx)
                    addAssoc(nPrimaryCamIdx,nRowIdx,nColIdx,nNewLabel);
                ++nChangedStereoLabels;
            }
        }
        nMoveIter += m_nRealStereoLabels;
        nConsecUnchangedLabels = (nChangedStereoLabels>0)?0:nConsecUnchangedLabels+m_nRealStereoLabels;
        const bool bResegmNext = true;
    #elif STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF
        // each iter below is a fusion move based on A. Fix's energy minimization method for higher-order MRFs
        // see "A Graph Cut Algorithm for Higher-order Markov Random Fields" in ICCV2011 for more info (doi = 10.1109/ICCV.2011.6126347)
        // (note: this approach is very generic, and not very well adapted to a dynamic MRF problem!)
        const InternalLabelType nStereoAlphaLabel = m_vStereoLabelOrdering[nOrderingIdx];
        calcStereoMoveCosts(nStereoAlphaLabel);
        oStereoReducer.Clear();
        oStereoReducer.AddVars((int)m_nValidStereoGraphNodes);
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidStereoGraphNodes; ++nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_vValidStereoLUTNodeIdxs[nGraphNodeIdx];
            const StereoNodeInfo& oNode = m_vStereoNodeInfos[nLUTNodeIdx];
            if(oNode.nUnaryFactID!=SIZE_MAX) {
                const ValueType& tUnaryCost = ((ValueType*)m_oStereoUnaryCosts.data)[nLUTNodeIdx];
                lvDbgAssert(&tUnaryCost==&m_oStereoUnaryCosts(oNode.nRowIdx,oNode.nColIdx));
                oStereoReducer.AddUnaryTerm((int)nGraphNodeIdx,tUnaryCost);
            }
            if(!bDisableStereoCliques) {
                for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
                    const PairwClique& oStereoClique = oNode.aPairwCliques[nOrientIdx];
                    if(oStereoClique)
                        lv::gm::factorReducer<s_nMaxOrder,ValueType>(m_pStereoModel->operator[](oStereoClique.m_nGraphFactorId),2,oStereoReducer,nStereoAlphaLabel,m_vValidStereoLUTNodeIdxs.data(),(InternalLabelType*)oCurrStereoLabeling.data);
                }
                // @@@@@ add higher o facts here (3-conn on epi lines?)
                if(STEREOSEGMATCH_CONFIG_USE_EPIPOLAR_CONN)
                    lvAssert(false); // @@@@
            }
        }
        oStereoMinimizer.Reset();
        oStereoReducer.ToQuadratic(oStereoMinimizer);
        oStereoMinimizer.Solve();
        oStereoMinimizer.ComputeWeakPersistencies(); // @@@@ check if any good
        size_t nChangedStereoLabels = 0;
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidStereoGraphNodes; ++nGraphNodeIdx) {
            const size_t& nLUTNodeIdx = m_vValidStereoLUTNodeIdxs[nGraphNodeIdx];
            const int& nRowIdx = m_vStereoNodeInfos[nLUTNodeIdx].nRowIdx;
            const int& nColIdx = m_vStereoNodeInfos[nLUTNodeIdx].nColIdx;
            const int nMoveLabel = oStereoMinimizer.GetLabel((int)nGraphNodeIdx);
            lvDbgAssert(nMoveLabel==0 || nMoveLabel==1 || nMoveLabel<0);
            if(nMoveLabel==1) { // node label changed to alpha
                const InternalLabelType nOldLabel = oCurrStereoLabeling(nRowIdx,nColIdx);
                if(nOldLabel<m_nDontCareLabelIdx)
                    removeAssoc(nRowIdx,nColIdx,nOldLabel);
                oCurrStereoLabeling(nRowIdx,nColIdx) = nStereoAlphaLabel;
                if(nStereoAlphaLabel<m_nDontCareLabelIdx)
                    addAssoc(nRowIdx,nColIdx,nStereoAlphaLabel);
                ++nChangedStereoLabels;
            }
        }
        ++nOrderingIdx %= m_nStereoLabels;
        nConsecUnchangedLabels = (nChangedStereoLabels>0)?0:nConsecUnchangedLabels+1;
        const bool bResegmNext = (nMoveIter%STEREOSEGMATCH_DEFAULT_ITER_PER_RESEGM)==0;
    #elif STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF

        // @@@ use bDisableStereoCliques?
        const InternalLabelType nStereoAlphaLabel = m_vStereoLabelOrdering[nOrderingIdx];
        __alpha = nStereoAlphaLabel;
        calcStereoMoveCosts(nPrimaryCamIdx,nStereoAlphaLabel);
        bool bGotLabelChange = false;
        if(InitialFusionLabeling()) {
            PreEditDual(crf);
            bGotLabelChange = UpdatePrimalDual(crf);
            PostEditDual(crf);
        }
        lvIgnore(oCurrStereoLabeling);
        ++nOrderingIdx %= m_nStereoLabels;
        nConsecUnchangedLabels = bGotLabelChange?0:nConsecUnchangedLabels+1;
        const bool bResegmNext = (nMoveIter%STEREOSEGMATCH_DEFAULT_ITER_PER_RESEGM)==0;
    #endif //STEREOSEGMATCH_CONFIG_USE_..._STEREO_INF
        if(lv::getVerbosity()>=3) {
            cv::Mat oCurrLabelingDisplay = getStereoDispMapDisplay(m_nPrimaryCamIdx);
            if(oCurrLabelingDisplay.size().area()<640*480)
                cv::resize(oCurrLabelingDisplay,oCurrLabelingDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
            cv::imshow(std::string("disp-")+std::to_string(m_nPrimaryCamIdx),oCurrLabelingDisplay);
            cv::waitKey(1);
        }
        const ValueType tCurrStereoEnergy = m_pStereoInf->value();
        lvDbgAssert(tCurrStereoEnergy>=cost_cast(0));
        std::stringstream ssStereoEnergyDiff;
        if((tCurrStereoEnergy-tLastStereoEnergy)==cost_cast(0))
            ssStereoEnergyDiff << "null";
        else
            ssStereoEnergyDiff << std::showpos << tCurrStereoEnergy-tLastStereoEnergy;
    #if STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF
        // no control on label w/ fastpd (could decompose algo later on...) @@@
        lvLog_(2,"\t\tdisp      e = %d      (delta=%s)      [iter=%d]",(int)tCurrStereoEnergy,ssStereoEnergyDiff.str().c_str(),(int)nMoveIter);
    #else //!STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF
        lvLog_(2,"\t\tdisp [+label:%d]   e = %d   (delta=%s)      [iter=%d]",(int)nStereoAlphaLabel,(int)tCurrStereoEnergy,ssStereoEnergyDiff.str().c_str(),(int)nMoveIter);
    #endif //!STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF
        if(bDisableStereoCliques)
            lvLog(2,"\t\t\t(disabling clique costs)");
        else if(bJustUpdatedSegm)
            lvLog(2,"\t\t\t(just updated segmentation)");
 //       else
//            lvAssert_(tLastStereoEnergy>=tCurrStereoEnergy,"stereo energy not minimizing!");
        tLastStereoEnergy = tCurrStereoEnergy;
        bJustUpdatedSegm = false;
        if(bResegmNext) {
            for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
                if(nCamIdx!=m_nPrimaryCamIdx) {
                    resetStereoLabelings(nCamIdx);
                    if(lv::getVerbosity()>=3) {
                        cv::Mat oCurrLabelingDisplay = getStereoDispMapDisplay(nCamIdx);
                        if(oCurrLabelingDisplay.size().area()<640*480)
                            cv::resize(oCurrLabelingDisplay,oCurrLabelingDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
                        cv::imshow(std::string("disp-")+std::to_string(nCamIdx),oCurrLabelingDisplay);
                        cv::waitKey(1);
                    }
                }
            }
            size_t nTotChangedResegmLabelings = 0;
            for(size_t nResegmLoopIdx=0; nResegmLoopIdx<STEREOSEGMATCH_DEFAULT_RESEGM_PER_LOOP; ++nResegmLoopIdx) {
                for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
                    cv::Mat_<InternalLabelType>& oCurrResegmLabeling = m_aResegmLabelings[nCamIdx];
                    for(InternalLabelType nResegmAlphaLabel : {s_nForegroundLabelIdx,s_nBackgroundLabelIdx}) {
                        updateResegmModel(nCamIdx,false);
                    #if STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF
                        calcResegmMoveCosts(nCamIdx,nResegmAlphaLabel);
                        aResegmReducers[nCamIdx].Clear();
                        aResegmReducers[nCamIdx].AddVars((int)m_anValidResegmGraphNodes[nCamIdx]);
                        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidResegmGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
                            const size_t nLUTNodeIdx = m_avValidResegmLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
                            const ResegmNodeInfo& oNode = m_avResegmNodeInfos[nCamIdx][nLUTNodeIdx];
                            if(oNode.nUnaryFactID!=SIZE_MAX) {
                                const ValueType& tUnaryCost = ((ValueType*)m_aResegmUnaryCosts[nCamIdx].data)[nLUTNodeIdx];
                                lvDbgAssert(&tUnaryCost==&m_aResegmUnaryCosts[nCamIdx](oNode.nRowIdx,oNode.nColIdx));
                                aResegmReducers[nCamIdx].AddUnaryTerm((int)nGraphNodeIdx,tUnaryCost);
                            }
                            for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
                                const PairwClique& oResegmClique = oNode.aPairwCliques[nOrientIdx];
                                if(oResegmClique)
                                    lv::gm::factorReducer<s_nMaxOrder,ValueType>(m_apResegmModels[nCamIdx]->operator[](oResegmClique.m_nGraphFactorId),2,aResegmReducers[nCamIdx],nResegmAlphaLabel,m_avValidResegmLUTNodeIdxs[nCamIdx].data(),(InternalLabelType*)oCurrResegmLabeling.data);
                            }
                            // @@@@@ add higher o facts here (3/4-conn spatiotemporal?)
                            if(STEREOSEGMATCH_DEFAULT_TEMPORAL_DEPTH)
                                lvAssert(false); // @@@@
                        }
                        kolmogorov::qpbo::QPBO<ValueType>& oResegmMinimizer = *apResegmMinimizers[nCamIdx].get();
                        oResegmMinimizer.Reset();
                        aResegmReducers[nCamIdx].ToQuadratic(oResegmMinimizer);
                        oResegmMinimizer.Solve();
                        oResegmMinimizer.ComputeWeakPersistencies(); // @@@@ check if any good
                        size_t nChangedResegmLabelings = 0;
                        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidResegmGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
                            const size_t& nLUTNodeIdx = m_avValidResegmLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
                            const int& nRowIdx = m_avResegmNodeInfos[nCamIdx][nLUTNodeIdx].nRowIdx;
                            const int& nColIdx = m_avResegmNodeInfos[nCamIdx][nLUTNodeIdx].nColIdx;
                            const int nMoveLabel = oResegmMinimizer.GetLabel((int)nGraphNodeIdx);
                            lvDbgAssert(nMoveLabel==0 || nMoveLabel==1 || nMoveLabel<0);
                            if(nMoveLabel==1) { // node label changed to alpha
                                oCurrResegmLabeling(nRowIdx,nColIdx) = nResegmAlphaLabel;
                                ++nChangedResegmLabelings;
                            }
                        }
                    #elif STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF
                        @@@ todo solve 1x iter w/ sospd
                    #endif //STEREOSEGMATCH_CONFIG_USE_..._RESEGM_INF
                        if(lv::getVerbosity()>=3) {
                            cv::Mat oCurrLabelingDisplay = getResegmMapDisplay(nCamIdx);
                            if(oCurrLabelingDisplay.size().area()<640*480)
                                cv::resize(oCurrLabelingDisplay,oCurrLabelingDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
                            cv::imshow(std::string("segm-")+std::to_string(nCamIdx),oCurrLabelingDisplay);
                            cv::waitKey(1);
                        }
                        const ValueType tCurrResegmEnergy = m_apResegmInfs[nCamIdx]->value();
                        lvDbgAssert(tCurrResegmEnergy>=cost_cast(0));
                        std::stringstream ssResegmEnergyDiff;
                        if((tCurrResegmEnergy-atLastResegmEnergies[nCamIdx])==cost_cast(0))
                            ssResegmEnergyDiff << "null";
                        else
                            ssResegmEnergyDiff << std::showpos << tCurrResegmEnergy-atLastResegmEnergies[nCamIdx];
                        lvLog_(2,"\t\tsegm [%d][+%s]   e = %d   (delta=%s)      [iter=%d]",(int)nCamIdx,(nResegmAlphaLabel==s_nForegroundLabelIdx?"fg":"bg"),(int)tCurrResegmEnergy,ssResegmEnergyDiff.str().c_str(),(int)nMoveIter);
                        if(nChangedResegmLabelings) {
                            calcShapeDistFeatures(m_aResegmLabelings[nCamIdx],nCamIdx);
                            nTotChangedResegmLabelings += nChangedResegmLabelings;
                        }
                        atLastResegmEnergies[nCamIdx] = tCurrResegmEnergy;
                    }
                }
            }
            const ValueType tCurrResegmEnergyTotal = std::accumulate(atLastResegmEnergies.begin(),atLastResegmEnergies.end(),cost_cast(0));
            std::stringstream ssResegmEnergyDiff;
            if((tCurrResegmEnergyTotal-tLastResegmEnergyTotal)==cost_cast(0))
                ssResegmEnergyDiff << "null";
            else
                ssResegmEnergyDiff << std::showpos << tCurrResegmEnergyTotal-tLastResegmEnergyTotal;
            lvLog_(2,"\t\tsegm overall   e = %d   (delta=%s)      [iter=%d]",(int)tCurrResegmEnergyTotal,ssResegmEnergyDiff.str().c_str(),(int)nMoveIter);
            if(nTotChangedResegmLabelings) {
                calcShapeFeatures(m_aResegmLabelings);
                updateStereoModel(false);
                bJustUpdatedSegm = true;
                nConsecUnchangedLabels = 0;
            }
            tLastResegmEnergyTotal = tCurrResegmEnergyTotal;
        }
    }
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx)
        if(nCamIdx!=m_nPrimaryCamIdx)
            resetStereoLabelings(nCamIdx);
    lvLog_(2,"Inference for primary camera idx=%d completed in %f second(s).",(int)m_nPrimaryCamIdx,oLocalTimer.tock());
    if(lv::getVerbosity()>=4)
        cv::waitKey(0);
    ++m_nFramesProcessed;
    return opengm::InferenceTermination::NORMAL;
}

cv::Mat StereoSegmMatcher::GraphModelData::getResegmMapDisplay(size_t nCamIdx) const {
    lvDbgExceptionWatch;
    lvAssert_(nCamIdx<getCameraCount(),"camera index out of range");
    lvAssert(!m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img].empty());
    lvAssert(!m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Mask].empty());
    lvAssert(m_oGridSize==m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img].size);
    lvAssert(m_oGridSize==m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Mask].size);
    lvAssert(!m_aResegmLabelings[nCamIdx].empty() && m_oGridSize==m_aResegmLabelings[nCamIdx].size);
    cv::Mat oOutput(m_oGridSize(),CV_8UC3);
    for(int nRowIdx=0; nRowIdx<m_aResegmLabelings[nCamIdx].rows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<m_aResegmLabelings[nCamIdx].cols; ++nColIdx) {
            const InternalLabelType nCurrLabel = m_aResegmLabelings[nCamIdx](nRowIdx,nColIdx);
            const uchar nInitLabel = m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Mask].at<uchar>(nRowIdx,nColIdx);
            if(nCurrLabel==s_nForegroundLabelIdx && nInitLabel>0)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(127,127,127);
            else if(nCurrLabel==s_nBackgroundLabelIdx && nInitLabel>0)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(0,0,127);
            else if(nCurrLabel==s_nForegroundLabelIdx && nInitLabel==0)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(0,127,0);
            else
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(0,0,0);
        }
    }
    cv::Mat oInputDisplay = m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img];
    if(oInputDisplay.channels()==1)
        cv::cvtColor(oInputDisplay,oInputDisplay,cv::COLOR_GRAY2BGR);
    oOutput = (oOutput+oInputDisplay)/2;
    return oOutput;
}

cv::Mat StereoSegmMatcher::GraphModelData::getStereoDispMapDisplay(size_t nCamIdx) const {
    lvDbgExceptionWatch;
    lvAssert(m_nMaxDispOffset>m_nMinDispOffset);
    lvAssert(!m_aStereoLabelings[nCamIdx].empty() && m_oGridSize==m_aStereoLabelings[nCamIdx].size);
    const float fRescaleFact = float(UCHAR_MAX)/(m_nMaxDispOffset-m_nMinDispOffset+1);
    cv::Mat oOutput(m_oGridSize(),CV_8UC3);
    for(int nRowIdx=0; nRowIdx<m_aStereoLabelings[nCamIdx].rows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<m_aStereoLabelings[nCamIdx].cols; ++nColIdx) {
            const size_t nLUTNodeIdx = nRowIdx*m_oGridSize[1]+nColIdx;
            const StereoNodeInfo& oNode = m_vStereoNodeInfos[nLUTNodeIdx];
            const OutputLabelType nRealLabel = getRealLabel(m_aStereoLabelings[nCamIdx](nRowIdx,nColIdx));
            if(nRealLabel==s_nDontCareLabel)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(0,128,128);
            else if(nRealLabel==s_nOccludedLabel)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(128,0,128);
            else {
                const uchar nIntensity = uchar((nRealLabel-m_nMinDispOffset)*fRescaleFact);
                if(oNode.bNearBorders)
                    oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(0,0,nIntensity);
                //else if( has invalid offset desc at max disp )
                //    oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(uchar(nIntensity/2),uchar(nIntensity/2),nIntensity);
                else
                    oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b::all(nIntensity);
            }
        }
    }
    return oOutput;
}

cv::Mat StereoSegmMatcher::GraphModelData::getAssocCountsMapDisplay() const {
    lvDbgExceptionWatch;
    lvAssert(m_nMaxDispOffset>m_nMinDispOffset);
    lvAssert(!m_oAssocCounts.empty() && m_oAssocCounts.rows==int(m_oGridSize(0)));
    lvAssert(m_oAssocCounts.cols==int((m_oGridSize(1)+m_nMaxDispOffset)/m_nDispOffsetStep));
    double dMax;
    cv::minMaxIdx(m_oAssocCounts,nullptr,&dMax);
    const float fRescaleFact = float(UCHAR_MAX)/(int(dMax)+1);
    cv::Mat oOutput(int(m_oGridSize(0)),int(m_oGridSize(1)+m_nMaxDispOffset),CV_8UC3);
    const int nColIdxStart = ((m_nPrimaryCamIdx==size_t(1))?0:-int(m_nMaxDispOffset));
    const int nColIdxEnd = ((m_nPrimaryCamIdx==size_t(1))?int(m_oGridSize(1)+m_nMaxDispOffset):int(m_oGridSize(1)));
    for(int nRowIdx=0; nRowIdx<int(m_oGridSize(0)); ++nRowIdx) {
        for(int nColIdx=nColIdxStart; nColIdx<nColIdxEnd; ++nColIdx) {
            const AssocCountType nCount = getAssocCount(nRowIdx,nColIdx);
            if(nColIdx<0 || nColIdx>=int(m_oGridSize(1)))
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx-nColIdxStart) = cv::Vec3b(0,0,uchar(nCount*fRescaleFact));
            else
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx-nColIdxStart) = cv::Vec3b::all(uchar(nCount*fRescaleFact));
        }
    }
    return oOutput;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

StereoSegmMatcher::StereoGraphInference::StereoGraphInference(size_t nCamIdx, GraphModelData& oData) :
        m_oData(oData),m_nPrimaryCamIdx(nCamIdx) {
    lvDbgExceptionWatch;
    lvAssert_(m_nPrimaryCamIdx<getCameraCount(),"camera index out of range");
    lvAssert_(m_oData.m_pStereoModel,"invalid graph");
    const StereoModelType& oGM = *m_oData.m_pStereoModel;
    lvAssert_(oGM.numberOfFactors()>0,"invalid graph");
    for(size_t nFactIdx=0; nFactIdx<oGM.numberOfFactors(); ++nFactIdx)
        lvDbgAssert__(oGM[nFactIdx].numberOfVariables()<=s_nMaxOrder,"graph had some factors of order %d (max allowed is %d)",(int)oGM[nFactIdx].numberOfVariables(),(int)s_nMaxOrder);
    lvDbgAssert_(oGM.numberOfVariables()>0 && oGM.numberOfVariables()<=(IndexType)m_oData.m_oGridSize.total(),"graph node count must match grid size");
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<oGM.numberOfVariables(); ++nGraphNodeIdx)
        lvDbgAssert_(oGM.numberOfLabels(nGraphNodeIdx)==m_oData.m_vStereoLabels.size(),"graph nodes must all have the same number of labels");
}

std::string StereoSegmMatcher::StereoGraphInference::name() const {
    return std::string("litiv-stereo-matcher");
}

const StereoModelType& StereoSegmMatcher::StereoGraphInference::graphicalModel() const {
    lvDbgExceptionWatch;
    lvDbgAssert(m_oData.m_pStereoModel);
    return *m_oData.m_pStereoModel;
}

opengm::InferenceTermination StereoSegmMatcher::StereoGraphInference::infer() {
    lvDbgExceptionWatch;
    lvDbgAssert(m_oData.m_nPrimaryCamIdx==m_nPrimaryCamIdx);
    return m_oData.infer();
}

void StereoSegmMatcher::StereoGraphInference::setStartingPoint(typename std::vector<InternalLabelType>::const_iterator begin) {
    lvDbgExceptionWatch;
    lvDbgAssert_(lv::filter_out(lv::unique(begin,begin+m_oData.m_oGridSize.total()),lv::make_range(InternalLabelType(0),InternalLabelType(m_oData.m_vStereoLabels.size()-1))).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvDbgAssert_(m_oData.m_aStereoLabelings[m_nPrimaryCamIdx].isContinuous() && m_oData.m_aStereoLabelings[m_nPrimaryCamIdx].total()==m_oData.m_oGridSize.total(),"unexpected internal labeling size");
    std::copy_n(begin,m_oData.m_oGridSize.total(),m_oData.m_aStereoLabelings[m_nPrimaryCamIdx].begin());
}

void StereoSegmMatcher::StereoGraphInference::setStartingPoint(const cv::Mat_<OutputLabelType>& oLabeling) {
    lvDbgExceptionWatch;
    lvDbgAssert_(lv::filter_out(lv::unique(oLabeling.begin(),oLabeling.end()),m_oData.m_vStereoLabels).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvAssert_(m_oData.m_oGridSize==oLabeling.size && oLabeling.isContinuous(),"provided labeling must fit grid size & be continuous");
    std::transform(oLabeling.begin(),oLabeling.end(),m_oData.m_aStereoLabelings[m_nPrimaryCamIdx].begin(),[&](const OutputLabelType& nRealLabel){return m_oData.getInternalLabel(nRealLabel);});
}

opengm::InferenceTermination StereoSegmMatcher::StereoGraphInference::arg(std::vector<InternalLabelType>& oLabeling, const size_t n) const {
    lvDbgExceptionWatch;
    if(n==1) {
        lvDbgAssert_(m_oData.m_aStereoLabelings[m_nPrimaryCamIdx].total()==m_oData.m_oGridSize.total(),"mismatch between internal graph label count and labeling mat size");
        oLabeling.resize(m_oData.m_aStereoLabelings[m_nPrimaryCamIdx].total());
        std::copy(m_oData.m_aStereoLabelings[m_nPrimaryCamIdx].begin(),m_oData.m_aStereoLabelings[m_nPrimaryCamIdx].end(),oLabeling.begin()); // the labels returned here are NOT the 'real' ones!
        return opengm::InferenceTermination::NORMAL;
    }
    return opengm::InferenceTermination::UNKNOWN;
}

void StereoSegmMatcher::StereoGraphInference::getOutput(cv::Mat_<OutputLabelType>& oLabeling) const {
    lvDbgExceptionWatch;
    lvDbgAssert_(m_oData.m_aStereoLabelings[m_nPrimaryCamIdx].isContinuous() && m_oData.m_aStereoLabelings[m_nPrimaryCamIdx].total()==m_oData.m_oGridSize.total(),"unexpected internal labeling size");
    oLabeling.create(m_oData.m_aStereoLabelings[m_nPrimaryCamIdx].size());
    lvDbgAssert_(oLabeling.isContinuous(),"provided matrix must be continuous for in-place label transform");
    std::transform(m_oData.m_aStereoLabelings[m_nPrimaryCamIdx].begin(),m_oData.m_aStereoLabelings[m_nPrimaryCamIdx].end(),oLabeling.begin(),[&](const InternalLabelType& nLabel){return m_oData.getRealLabel(nLabel);});
}

StereoSegmMatcher::ValueType StereoSegmMatcher::StereoGraphInference::value() const {
    lvDbgExceptionWatch;
    lvDbgAssert_(m_oData.m_oGridSize.dims()==2 && m_oData.m_oGridSize==m_oData.m_aStereoLabelings[m_nPrimaryCamIdx].size,"output labeling must be a 2d grid");
    const ValueType tTotAssocCost = m_oData.calcTotalAssocCost();
    struct GraphNodeLabelIter {
        GraphNodeLabelIter(size_t nCamIdx, const GraphModelData& oData) : m_oData(oData),m_nPrimaryCamIdx(nCamIdx) {}
        InternalLabelType operator[](size_t nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_oData.m_vValidStereoLUTNodeIdxs[nGraphNodeIdx];
            return ((InternalLabelType*)m_oData.m_aStereoLabelings[m_nPrimaryCamIdx].data)[nLUTNodeIdx];
        }
        const GraphModelData& m_oData;
        const size_t m_nPrimaryCamIdx;
    } oLabelIter(m_nPrimaryCamIdx,m_oData);
    const ValueType tTotStereoLabelCost = m_oData.m_pStereoModel->evaluate(oLabelIter);
    return tTotAssocCost+tTotStereoLabelCost;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

StereoSegmMatcher::ResegmGraphInference::ResegmGraphInference(size_t nCamIdx, GraphModelData& oData) :
        m_oData(oData),m_nCamIdx(nCamIdx) {
    lvDbgExceptionWatch;
    lvAssert_(m_nCamIdx<getCameraCount(),"camera index out of range");
    lvAssert_(m_oData.m_apResegmModels[m_nCamIdx],"invalid graph");
    const ResegmModelType& oGM = *m_oData.m_apResegmModels[m_nCamIdx];
    lvAssert_(oGM.numberOfFactors()>0,"invalid graph");
    for(size_t nFactIdx=0; nFactIdx<oGM.numberOfFactors(); ++nFactIdx)
        lvDbgAssert__(oGM[nFactIdx].numberOfVariables()<=s_nMaxOrder,"graph had some factors of order %d (max allowed is %d)",(int)oGM[nFactIdx].numberOfVariables(),(int)s_nMaxOrder);
    lvDbgAssert_(oGM.numberOfVariables()>0 && oGM.numberOfVariables()<=(IndexType)m_oData.m_oGridSize.total(),"graph node count must match grid size");
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<oGM.numberOfVariables(); ++nGraphNodeIdx)
        lvDbgAssert_(oGM.numberOfLabels(nGraphNodeIdx)==size_t(2),"graph nodes must all have the same number of labels");
}

std::string StereoSegmMatcher::ResegmGraphInference::name() const {
    return std::string("litiv-segm-matcher");
}

const ResegmModelType& StereoSegmMatcher::ResegmGraphInference::graphicalModel() const {
    lvDbgExceptionWatch;
    lvDbgAssert(m_oData.m_apResegmModels[m_nCamIdx]);
    return *m_oData.m_apResegmModels[m_nCamIdx];
}

opengm::InferenceTermination StereoSegmMatcher::ResegmGraphInference::infer() {
    lvDbgExceptionWatch;
    return m_oData.infer();
}

void StereoSegmMatcher::ResegmGraphInference::setStartingPoint(typename std::vector<InternalLabelType>::const_iterator begin) {
    lvDbgExceptionWatch;
    lvDbgAssert_(lv::filter_out(lv::unique(begin,begin+m_oData.m_oGridSize.total()),std::vector<InternalLabelType>{s_nBackgroundLabelIdx,s_nForegroundLabelIdx}).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvDbgAssert_(m_oData.m_aResegmLabelings[m_nCamIdx].isContinuous() && m_oData.m_aResegmLabelings[m_nCamIdx].total()==m_oData.m_oGridSize.total(),"unexpected internal labeling size");
    std::copy_n(begin,m_oData.m_oGridSize.total(),m_oData.m_aResegmLabelings[m_nCamIdx].begin());
}

void StereoSegmMatcher::ResegmGraphInference::setStartingPoint(const cv::Mat_<OutputLabelType>& oLabeling) {
    lvDbgExceptionWatch;
    lvDbgAssert_(lv::filter_out(lv::unique(oLabeling.begin(),oLabeling.end()),std::vector<OutputLabelType>{s_nBackgroundLabel,s_nForegroundLabel}).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvAssert_(m_oData.m_oGridSize==oLabeling.size && oLabeling.isContinuous(),"provided labeling must fit grid size & be continuous");
    std::transform(oLabeling.begin(),oLabeling.end(),m_oData.m_aResegmLabelings[m_nCamIdx].begin(),[&](const OutputLabelType& nRealLabel){return nRealLabel?s_nForegroundLabelIdx:s_nBackgroundLabelIdx;});
}

opengm::InferenceTermination StereoSegmMatcher::ResegmGraphInference::arg(std::vector<InternalLabelType>& oLabeling, const size_t n) const {
    lvDbgExceptionWatch;
    if(n==1) {
        lvDbgAssert_(m_oData.m_aResegmLabelings[m_nCamIdx].total()==m_oData.m_oGridSize.total(),"mismatch between internal graph label count and labeling mat size");
        oLabeling.resize(m_oData.m_aResegmLabelings[m_nCamIdx].total());
        std::copy(m_oData.m_aResegmLabelings[m_nCamIdx].begin(),m_oData.m_aResegmLabelings[m_nCamIdx].end(),oLabeling.begin()); // the labels returned here are NOT the 'real' ones (same values, different type)
        return opengm::InferenceTermination::NORMAL;
    }
    return opengm::InferenceTermination::UNKNOWN;
}

void StereoSegmMatcher::ResegmGraphInference::getOutput(cv::Mat_<OutputLabelType>& oLabeling) const {
    lvDbgExceptionWatch;
    lvDbgAssert_(m_oData.m_aResegmLabelings[m_nCamIdx].isContinuous() && m_oData.m_aResegmLabelings[m_nCamIdx].total()==m_oData.m_oGridSize.total(),"unexpected internal labeling size");
    oLabeling.create(m_oData.m_aResegmLabelings[m_nCamIdx].size());
    lvDbgAssert_(oLabeling.isContinuous(),"provided matrix must be continuous for in-place label transform");
    std::transform(m_oData.m_aResegmLabelings[m_nCamIdx].begin(),m_oData.m_aResegmLabelings[m_nCamIdx].end(),oLabeling.begin(),[&](const InternalLabelType& nLabel){return nLabel?s_nForegroundLabel:s_nBackgroundLabel;});
}

StereoSegmMatcher::ValueType StereoSegmMatcher::ResegmGraphInference::value() const {
    lvDbgExceptionWatch;
    lvDbgAssert_(m_oData.m_oGridSize.dims()==2 && m_oData.m_oGridSize==m_oData.m_aResegmLabelings[m_nCamIdx].size,"output labeling must be a 2d grid");
    struct GraphNodeLabelIter {
        GraphNodeLabelIter(size_t nCamIdx, const GraphModelData& oData) : m_oData(oData),m_nCamIdx(nCamIdx) {}
        InternalLabelType operator[](size_t nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_oData.m_avValidResegmLUTNodeIdxs[m_nCamIdx][nGraphNodeIdx];
            return ((InternalLabelType*)m_oData.m_aResegmLabelings[m_nCamIdx].data)[nLUTNodeIdx];
        }
        const GraphModelData& m_oData;
        const size_t m_nCamIdx;
    } oLabelIter(m_nCamIdx,m_oData);
    const ValueType tTotResegmLabelCost = m_oData.m_apResegmModels[m_nCamIdx]->evaluate(oLabelIter);
    return tTotResegmLabelCost;
}