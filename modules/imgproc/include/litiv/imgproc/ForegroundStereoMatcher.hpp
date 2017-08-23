
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

#define OPENGM_ENABLE_FAST_DEBUG_MAT_OPS 1

#include "litiv/utils/opengm.hpp"
#include "litiv/features2d.hpp"
#include "litiv/imgproc.hpp"

// config options
#define STEREOSEGMATCH_CONFIG_USE_DASCGF_AFFINITY   1
#define STEREOSEGMATCH_CONFIG_USE_DASCRF_AFFINITY   0
#define STEREOSEGMATCH_CONFIG_USE_LSS_AFFINITY      0
#define STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY       0
#define STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY  0
#define STEREOSEGMATCH_CONFIG_USE_SHAPE_EMD_AFFIN   0
#define STEREOSEGMATCH_CONFIG_USE_UNARY_ONLY_FIRST  1
#define STEREOSEGMATCH_CONFIG_USE_SALIENT_MAP_BORDR 1
#define STEREOSEGMATCH_CONFIG_USE_ROOT_SIFT_DESCS   0
#define STEREOSEGMATCH_CONFIG_USE_GMM_LOCAL_BACKGR  1
#define STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF   0
#define STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF 1
#define STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF  0
#define STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF   1
#define STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF  0
#define STEREOSEGMATCH_CONFIG_USE_PROGRESS_BARS     0

// default param values
#define STEREOSEGMATCH_DEFAULT_DISPARITY_STEP       (size_t(1))
#define STEREOSEGMATCH_DEFAULT_MAX_MOVE_ITER        (size_t(300))
#define STEREOSEGMATCH_DEFAULT_SCDESC_WIN_RAD       (size_t(40))
#define STEREOSEGMATCH_DEFAULT_SCDESC_RAD_BINS      (size_t(3))
#define STEREOSEGMATCH_DEFAULT_SCDESC_ANG_BINS      (size_t(10))
#define STEREOSEGMATCH_DEFAULT_LSSDESC_RAD          (size_t(40))
#define STEREOSEGMATCH_DEFAULT_LSSDESC_PATCH        (size_t(7))
#define STEREOSEGMATCH_DEFAULT_LSSDESC_RAD_BINS     (size_t(3))
#define STEREOSEGMATCH_DEFAULT_LSSDESC_ANG_BINS     (size_t(10))
#define STEREOSEGMATCH_DEFAULT_SSQDIFF_PATCH        (size_t(7))
#define STEREOSEGMATCH_DEFAULT_MI_WINDOW_RAD        (size_t(12))
#define STEREOSEGMATCH_DEFAULT_GRAD_KERNEL_SIZE     (int(1))
#define STEREOSEGMATCH_DEFAULT_DISTTRANSF_SCALE     (-0.1f)
#define STEREOSEGMATCH_DEFAULT_ITER_PER_RESEGM      ((m_nStereoLabels*3)/2)
#define STEREOSEGMATCH_DEFAULT_RESEGM_PER_LOOP      (3)
#define STEREOSEGMATCH_DEFAULT_SALIENT_SHP_RAD      (3)
#define STEREOSEGMATCH_DEFAULT_DESC_PATCH_HEIGHT    (15)
#define STEREOSEGMATCH_DEFAULT_DESC_PATCH_WIDTH     (15)

// unary costs params
#define STEREOSEGMATCH_UNARY_COST_OOB_CST           (ValueType(5000))
#define STEREOSEGMATCH_UNARY_COST_OCCLUDED_CST      (ValueType(2000))
#define STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST      (ValueType(10000))
#define STEREOSEGMATCH_IMGSIM_COST_COLOR_SCALE      (40)
#define STEREOSEGMATCH_IMGSIM_COST_DESC_SCALE       (400)
#define STEREOSEGMATCH_SHPSIM_COST_DESC_SCALE       (400)
#define STEREOSEGMATCH_UNIQUE_COST_OVER_SCALE       (400)
#define STEREOSEGMATCH_SHPDIST_COST_SCALE           (400)
#define STEREOSEGMATCH_SHPDIST_PX_MAX_CST           (10.0f)
#define STEREOSEGMATCH_SHPDIST_INTERSPEC_SCALE      (0.50f)
#define STEREOSEGMATCH_SHPDIST_INITDIST_SCALE       (0.00f)
// pairwise costs params
#define STEREOSEGMATCH_LBLSIM_COST_MAXOCCL          (ValueType(5000))
#define STEREOSEGMATCH_LBLSIM_COST_MAXTRUNC_CST     (ValueType(5000))
#define STEREOSEGMATCH_LBLSIM_RESEGM_SCALE_CST      (400)
#define STEREOSEGMATCH_LBLSIM_STEREO_SCALE_CST      (1.f)
#define STEREOSEGMATCH_LBLSIM_STEREO_MAXDIFF_CST    (10)
#define STEREOSEGMATCH_LBLSIM_USE_EXP_GRADPIVOT     (1)
#if STEREOSEGMATCH_LBLSIM_USE_EXP_GRADPIVOT
#define STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE    (32)
#define STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST    (32)
#else //!STEREOSEGMATCH_LBLSIM_USE_EXP_GRADPIVOT
#define STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE    (10)
#define STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST    (32)
#endif //!STEREOSEGMATCH_LBLSIM_USE_EXP_GRADPIVOT
// higher order costs params
// ...

// hardcoded term relations
#define STEREOSEGMATCH_UNIQUE_COST_INCR_REL(n)      (float((n)*3)/((n)+2))
#define STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT       (2)

/// this stereo matcher assumes both input images are rectified, and have the same size;
/// it also expects four inputs (image0,mask0,image1,mask1), and provides 4 outputs (disp0,mask0,disp1,mask1)
struct StereoSegmMatcher : ICosegmentor<int32_t,4> {
    // @@@ add template for input array size? (output size & graph struct/max order can be deduced from it)
    using InternalLabelType = uint8_t; ///< type used for internal labeling (disparity + fg/bg)
    using OutputLabelType = int32_t; ///< type used in returned labelings (i.e. output of 'apply')
    using AssocCountType = uint16_t; ///< type used for stereo association counting in cv::Mat_'s
    using AssocIdxType = int16_t; ///< type used for stereo association idx listing in cv::Mat_'s
    using ValueType =  double; ///< type used for factor values (@@@@ could be integer? retest speed later?)
    using IndexType = size_t; ///< type used for node indexing (note: pretty much hardcoded everywhere in impl below)
    using ExplicitFunction = lv::gm::ExplicitViewFunction<ValueType,IndexType,InternalLabelType>; ///< shortcut for explicit view function
    using ExplicitAllocFunction = opengm::ExplicitFunction<ValueType,IndexType,InternalLabelType>; ///< shortcut for explicit allocated function
    using FunctionTypeList = opengm::meta::TypeListGenerator<ExplicitFunction,ExplicitAllocFunction/*,...*/>::type;  ///< list of all functions the models can use
    using PairwClique = lv::gm::Clique<size_t(2),ValueType,IndexType,InternalLabelType,ExplicitFunction>; ///< pairwise clique implementation wrapper
    using StereoSpaceType = opengm::SimpleDiscreteSpace<IndexType,InternalLabelType>; ///< shortcut for discrete stereo space type (simple = all nodes have the same # of labels)
    using ResegmSpaceType = opengm::StaticSimpleDiscreteSpace<2,IndexType,InternalLabelType>; ///< shortcut for discrete resegm space type (binary labels for fg/bg)
    using StereoModelType = opengm::GraphicalModel<ValueType,opengm::Adder,FunctionTypeList,StereoSpaceType>; ///< shortcut for stereo graphical model type
    using ResegmModelType = opengm::GraphicalModel<ValueType,opengm::Adder,FunctionTypeList,ResegmSpaceType>; ///< shortcut for resegm graphical model type
    static_assert(std::is_same<StereoModelType::FunctionIdentifier,ResegmModelType::FunctionIdentifier>::value,"mismatched function identifier for stereo/resegm graphs");
    using FuncIdentifType = StereoModelType::FunctionIdentifier; ///< shortcut for graph model function identifier type (for both stereo and resegm models)
    using FuncPairType = std::pair<FuncIdentifType,ExplicitFunction&>; ///< funcid-funcobj pair used as viewer to explicit data (for both stereo and resegm models)
    using ICosegmentor<OutputLabelType,s_nInputArraySize,s_nOutputArraySize>::apply; ///< helps avoid 'no matching function' issues for apply overloads
    template<typename T> using CamArray = std::array<T,getInputStreamCount()/2>; ///< shortcut typename for variables and members that are assigned to each camera head
    static constexpr size_t getCameraCount() {return getInputStreamCount()/2;} ///< returns the expected input camera head count
    static constexpr size_t s_nCameraCount = getInputStreamCount()/2; ///< holds the expected input camera head count
    static constexpr OutputLabelType s_nDontCareLabel = std::numeric_limits<OutputLabelType>::min(); ///< real label value reserved for 'dont care' pixels
    static constexpr OutputLabelType s_nOccludedLabel = std::numeric_limits<OutputLabelType>::max(); ///< real label value reserved for 'occluded' pixels
    static constexpr OutputLabelType s_nForegroundLabel = OutputLabelType(std::numeric_limits<InternalLabelType>::max()); ///< real label value reserved for foreground pixels
    static constexpr OutputLabelType s_nBackgroundLabel = OutputLabelType(0); ///< real label value reserved for background pixels
    static constexpr InternalLabelType s_nForegroundLabelIdx = InternalLabelType(1); ///< internal label value used for 'foreground' labeling
    static constexpr InternalLabelType s_nBackgroundLabelIdx = InternalLabelType(0); ///< internal label value used for 'background' labeling
    static constexpr size_t s_nMaxOrder = /*@@@@*/s_nInputArraySize; ///< used to limit internal static assignment array sizes
    static constexpr size_t s_nMaxCliqueAssign = size_t(1)<<s_nMaxOrder; ///< used to limit internal static assignment array sizes
    static constexpr size_t s_nPairwOrients = size_t(2); ///< number of pairwise links owned by each node in the graph (2 = 1st order neighb connections)
    static constexpr OutputLabelType getStereoDontCareLabel() {return s_nDontCareLabel;} ///< returns the output stereo label used to represent 'dont care' pixels
    static constexpr OutputLabelType getStereoOccludedLabel() {return s_nOccludedLabel;} ///< returns the output stereo label used to represent 'occluded' pixels
    static_assert(std::is_integral<IndexType>::value,"Graph index type must be integral");
    static_assert(std::is_integral<InternalLabelType>::value,"Graph internal label type must be integral");
    static_assert(size_t(std::numeric_limits<IndexType>::max())>=size_t(std::numeric_limits<InternalLabelType>::max()),"Graph index type max value must be greater than internal label type max value");
    struct StereoGraphInference;
    struct ResegmGraphInference;

    /// defines the indices of provided matrices inside the input array
    enum InputPackingList {
        InputPackSize=4,
        InputPackOffset=2,
        // absolute values for direct indexing
        InputPack_LeftImg=0,
        InputPack_LeftMask=1,
        InputPack_RightImg=2,
        InputPack_RightMask=3,
        // relative values for cam-based indexing
        InputPackOffset_Img=0,
        InputPackOffset_Mask=1,
    };

    /// defines the indices of provided matrices inside the input array
    enum OutputPackingList {
        OutputPackSize=4,
        OutputPackOffset=2,
        // absolute values for direct indexing
        OutputPack_LeftDisp=0,
        OutputPack_LeftMask=1,
        OutputPack_RightDisp=2,
        OutputPack_RightMask=3,
        // relative values for cam-based indexing
        OutputPackOffset_Disp=0,
        OutputPackOffset_Mask=1,
    };

    /// full stereo graph matcher constructor; only takes parameters to ready graphical model base initialization
    StereoSegmMatcher(size_t nMinDispOffset, size_t nMaxDispOffset);
    /// stereo graph matcher initialization function; will allocate & initialize graph model using provided ROI data (one ROI per camera head)
    virtual void initialize(const std::array<cv::Mat,s_nCameraCount>& aROIs);
    /// stereo matcher function; solves the graph model to find pixel-level matches on epipolar lines in the masked input images, and returns disparity maps + masks
    virtual void apply(const MatArrayIn& aInputs, MatArrayOut& aOutputs) override;
    /// (pre)calculates initial features required for model updates, and optionally returns them in packet format for archiving
    virtual void calcFeatures(const MatArrayIn& aInputs, cv::Mat* pFeatsPacket=nullptr);
    /// sets a previously precalculated initial features packet to be used in the next 'apply' call (do not modify its data before that!)
    virtual void setNextFeatures(const cv::Mat& oPackedFeats);
    /// returns the (friendly) name of the input image feature extractor that will be used internally
    virtual std::string getFeatureExtractorName() const;
    /// returns the (maximum) number of stereo disparity labels used in the output masks
    virtual size_t getMaxLabelCount() const override;
    /// returns the list of (real) stereo disparity labels used in the output masks
    virtual const std::vector<OutputLabelType>& getLabels() const override;
    /// helper func to display segmentation maps
    cv::Mat getResegmMapDisplay(size_t nCamIdx) const;
    /// helper func to display scaled disparity maps
    cv::Mat getStereoDispMapDisplay(size_t nCamIdx) const;
    /// helper func to display scaled assoc count maps
    cv::Mat getAssocCountsMapDisplay(size_t nCamIdx) const;

    /// holds graph model data for both stereo and resegmentation models
    struct GraphModelData {
        /// basic info struct used for node-level graph model updates and data lookups
        struct NodeInfo {
            /// image grid coordinates associated this node
            int nRowIdx,nColIdx;
            /// holds whether this is a valid node (in-graph) or not
            CamArray<bool> abValidGraphNode;
            /// if valid, this holds whether this node is near graph borders or not
            CamArray<bool> abNearGraphBorders;
            /// node graph index used for model indexing
            CamArray<size_t> anGraphNodeIdxs;

            /// id for this node's unary factors (not SIZE_MAX only if valid)
            CamArray<size_t> anStereoUnaryFactIDs,anResegmUnaryFactIDs;
            /// pointer to this node's unary functions (non-null only if valid)
            CamArray<ExplicitFunction*> apStereoUnaryFuncs,apResegmUnaryFuncs;

            /// weights for this node's stereo pairwise costs (constant post-init)
            mutable CamArray<std::array<float,s_nPairwOrients>> aafStereoPairwWeights;
            /// array of pairwise cliques owned by this node as 1st member
            CamArray<std::array<PairwClique,s_nPairwOrients>> aaStereoPairwCliques,aaResegmPairwCliques;

            // @@@@@ add higher o facts/funcptrs here
        };
        /// default constructor; receives model construction data from algo constructor
        GraphModelData(const CamArray<cv::Mat>& aROIs, const std::vector<OutputLabelType>& vRealStereoLabels, size_t nStereoLabelStep);
        /// (pre)calculates features required for model updates, and optionally returns them in packet format
        void calcFeatures(const MatArrayIn& aInputs, cv::Mat* pFeatsPacket=nullptr);
        /// sets a previously precalculated features packet to be used in the next model updates (do not modify it before that!)
        void setNextFeatures(const cv::Mat& oPackedFeats);
        /// performs the actual bi-model, bi-spectral inference
        opengm::InferenceTermination infer(size_t nPrimaryCamIdx=0);
        /// translate an internal graph label to a real disparity offset label
        OutputLabelType getRealLabel(InternalLabelType nLabel) const;
        /// translate a real disparity offset label to an internal graph label
        InternalLabelType getInternalLabel(OutputLabelType nRealLabel) const;
        /// returns the offset column to use for indexing pixels in the other camera image
        int getOffsetColIdx(size_t nCamIdx, int nColIdx, InternalLabelType nLabel) const;
        /// returns the stereo associations count for a given graph node by row/col indices
        AssocCountType getAssocCount(size_t nCamIdx, int nRowIdx, int nColIdx) const;
        /// returns the cost of adding a stereo association for a given node coord set & origin column idx
        ValueType calcAddAssocCost(size_t nCamIdx, int nRowIdx, int nColIdx, InternalLabelType nLabel) const;
        /// returns the cost of removing a stereo association for a given node coord set & origin column idx
        ValueType calcRemoveAssocCost(size_t nCamIdx, int nRowIdx, int nColIdx, InternalLabelType nLabel) const;
        /// returns the total stereo association cost for all grid nodes
        ValueType calcTotalAssocCost(size_t nCamIdx) const;
        /// helper func to display segmentation maps
        cv::Mat getResegmMapDisplay(size_t nCamIdx) const;
        /// helper func to display scaled disparity maps
        cv::Mat getStereoDispMapDisplay(size_t nCamIdx) const;
        /// helper func to display scaled assoc count maps
        cv::Mat getAssocCountsMapDisplay(size_t nCamIdx) const;

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
        /// 2d maps which contain how many associations a graph node possesses (mutable for inference)
        mutable CamArray<cv::Mat_<AssocCountType>> m_aAssocCounts;
        /// 3d maps which list the associations (by idx) for each graph node (mutable for inference)
        mutable CamArray<cv::Mat_<AssocIdxType>> m_aAssocMaps;
        /// 2d maps which contain transient unary factor labeling costs for all graph nodes (mutable for inference)
        mutable CamArray<cv::Mat_<ValueType>> m_aStereoUnaryCosts,m_aResegmUnaryCosts;
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
        /// internal label used for 'dont care' labeling
        const InternalLabelType m_nDontCareLabelIdx;
        /// internal label used for 'occluded' labeling
        const InternalLabelType m_nOccludedLabelIdx;
        /// opengm stereo graph model object (one per cam head)
        CamArray<std::unique_ptr<StereoModelType>> m_apStereoModels;
        /// opengm resegm graph model object (one per cam head)
        CamArray<std::unique_ptr<ResegmModelType>> m_apResegmModels;
        /// contains the eroded ROIs used for valid descriptor lookups
        CamArray<cv::Mat_<uchar>> m_aDescROIs;
        /// indices of valid nodes in the graph (based on each ROI)
        CamArray<std::vector<size_t>> m_avValidLUTNodeIdxs;
        /// number of valid nodes in the graph (based on each ROI)
        CamArray<size_t> m_anValidGraphNodes;
        /// total number of valid graph nodes, across cameras
        size_t m_nTotValidGraphNodes;
        /// model info lookup array
        std::vector<NodeInfo> m_vNodeInfos;
        /// graph model factor counts used for validation in debug mode
        CamArray<size_t> m_anUnaryFactCounts,m_anPairwFactCounts;
        /// graph model unary functions
        CamArray<std::vector<FuncPairType>> m_avStereoUnaryFuncs,m_avResegmUnaryFuncs;
        /// graph model pairwise functions arrays (for already-weighted lookups)
        CamArray<std::array<std::vector<FuncPairType>,s_nPairwOrients>> m_aavStereoPairwFuncs,m_aavResegmPairwFuncs;
        /// stereo model pairwise function ids (for shared base lookups without weights)
        CamArray<std::array<FuncIdentifType,s_nPairwOrients>> m_aaStereoPairwFuncIDs_base;
        /// functions data arrays (contiguous blocks for all factors)
        CamArray<std::unique_ptr<ValueType[]>> m_aaStereoFuncsData,m_aaResegmFuncsData;
        /// stereo/resegm models unary functions base pointers
        CamArray<ValueType*> m_apStereoUnaryFuncsDataBase,m_apResegmUnaryFuncsDataBase;
        /// stereo/resegm models pairwise functions base pointers
        CamArray<ValueType*> m_apStereoPairwFuncsDataBase,m_apResegmPairwFuncsDataBase;
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
        /// used for debug only; passed from top-level algo when available
        lv::DisplayHelperPtr m_pDisplayHelper;
        /// defines the indices of feature maps inside precalc packets (per camera head)
        enum FeatPackingList {
            FeatPackSize=20,
            FeatPackOffset=9,
            // absolute values for direct indexing
            FeatPack_LeftInitFGDist=0,
            FeatPack_LeftInitBGDist=1,
            FeatPack_LeftFGDist=2,
            FeatPack_LeftBGDist=3,
            FeatPack_LeftGradY=4,
            FeatPack_LeftGradX=5,
            FeatPack_LeftGradMag=6,
            FeatPack_LeftImgSaliency=7,
            FeatPack_LeftShpSaliency=8,
            FeatPack_RightInitFGDist=9,
            FeatPack_RightInitBGDist=10,
            FeatPack_RightFGDist=11,
            FeatPack_RightBGDist=12,
            FeatPack_RightGradY=13,
            FeatPack_RightGradX=14,
            FeatPack_RightGradMag=15,
            FeatPack_RightImgSaliency=16,
            FeatPack_RightShpSaliency=17,
            FeatPack_ImgAffinity=18,
            FeatPack_ShpAffinity=19,
            // relative values for cam-based indexing
            FeatPackOffset_InitFGDist=0,
            FeatPackOffset_InitBGDist=1,
            FeatPackOffset_FGDist=2,
            FeatPackOffset_BGDist=3,
            FeatPackOffset_GradY=4,
            FeatPackOffset_GradX=5,
            FeatPackOffset_GradMag=6,
            FeatPackOffset_ImgSaliency=7,
            FeatPackOffset_ShpSaliency=8,
        };

    protected:
        /// adds a stereo association for a given node coord set & origin column idx
        void addAssoc(size_t nCamIdx, int nRowIdx, int nColIdx, InternalLabelType nLabel) const;
        /// removes a stereo association for a given node coord set & origin column idx
        void removeAssoc(size_t nCamIdx, int nRowIdx, int nColIdx, InternalLabelType nLabel) const;
        /// resets stereo graph labelings using init state parameters
        void resetStereoLabelings(size_t nCamIdx, bool bIsPrimaryCam);
        /// updates a stereo graph model using new feats data
        void updateStereoModel(size_t nCamIdx, bool bInit);
        /// updates a shape graph model using new feats data
        void updateResegmModel(size_t nCamIdx, bool bInit);
        /// calculates image features required for model updates using the provided input image array
        void calcImageFeatures(const CamArray<cv::Mat>& aInputImages);
        /// calculates shape features required for model updates using the provided input mask array
        void calcShapeFeatures(const CamArray<cv::Mat_<InternalLabelType>>& aInputMasks);
        /// calculates shape mask distance features required for model updates using the provided input mask & camera index
        void calcShapeDistFeatures(const cv::Mat_<InternalLabelType>& oInputMask, size_t nCamIdx);
        /// calculates a stereo unary move cost for a single graph node
        ValueType calcStereoUnaryMoveCost(size_t nCamIdx, size_t nGraphNodeIdx, InternalLabelType nOldLabel, InternalLabelType nNewLabel) const;
    #if STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF
        /// fill internal temporary energy cost mats for the given stereo move operation
        void calcStereoMoveCosts(size_t nCamIdx, InternalLabelType nNewLabel) const;
    #endif //STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF
    #if STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF
        /// fill internal temporary energy cost mats for the given resegm move operation
        void calcResegmMoveCosts(size_t nCamIdx, InternalLabelType nNewLabel) const;
    #endif //STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF
    #if (STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF)
        /// fill internal temporary energy cost mats for all stereo label move operations
        void calcStereoCosts(size_t nCamIdx) const;
    #endif //(STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF)
    #if STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF
        /// fill internal temporary energy cost mats for all resegm label move operations
        void calcResegmCosts(size_t nCamIdx) const;
    #endif //STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF
    #if STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF
        /// runs sospd inference algorithm either to completion, or for a specific number of iterations
        typedef int VarId;
        typedef size_t Label;
        typedef std::vector<REAL> LambdaAlpha;
        typedef std::vector<std::pair<size_t, size_t>> NodeNeighborList;
        typedef std::vector<NodeNeighborList> NodeCliqueList;
        REAL ComputeHeightDiff(VarId i, Label l1, Label l2);
        void SetupAlphaEnergy(SubmodularIBFS& crf);
        bool InitialFusionLabeling();
        void PreEditDual(SubmodularIBFS& crf);
        bool UpdatePrimalDual(SubmodularIBFS& crf);
        void PostEditDual(SubmodularIBFS& crf);
        size_t __cam;
        size_t m_nStereoCliqueCount;
        InternalLabelType __alpha;
        REAL& Height(VarId i, Label l) {
            return m_heights[i*m_nStereoLabels+l];
        }
        REAL& dualVariable(int alpha, VarId i, Label l) {
            return m_dual[alpha][i*m_nStereoLabels+l];
        }
        REAL& dualVariable(LambdaAlpha& lambdaAlpha,VarId i, Label l) {
            return lambdaAlpha[i*m_nStereoLabels+l];
        }
        LambdaAlpha& lambdaAlpha(int alpha){
            return m_dual[alpha];
        }
        //void HeightAlphaProposal();
        //void AlphaProposal();
        NodeCliqueList m_node_clique_list;
        // @@@ FIXME(afix) change way m_dual is stored. Put lambda_alpha as separate REAL* for each clique, indexed by i, l.
        std::vector<LambdaAlpha> m_dual;
        std::vector<REAL> m_heights;
        //ProposalCallback m_pc;
    #endif //STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF
        /// holds stereo disparity graph inference algorithm interface (redirects for bi-model inference)
        CamArray<std::unique_ptr<StereoGraphInference>> m_apStereoInfs;
        /// holds resegmentation graph inference algorithm interface (redirects for bi-model inference)
        CamArray<std::unique_ptr<ResegmGraphInference>> m_apResegmInfs;
    };

    /// algo interface for multi-label graph model inference
    struct StereoGraphInference : opengm::Inference<StereoModelType,opengm::Minimizer> {
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
        const size_t m_nCamIdx;
    };

    /// algo interface for binary label graph model inference
    struct ResegmGraphInference : opengm::Inference<ResegmModelType,opengm::Minimizer> {
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

protected:
    /// disparity label step size (will be passed to model constr)
    size_t m_nDispStep;
    /// output disparity label set (will be passed to model constr)
    std::vector<OutputLabelType> m_vStereoLabels;
    /// holds bimodel data & inference algo impls
    std::unique_ptr<GraphModelData> m_pModelData;
    /*/// converts a floating point value to the model's value type, rounding if necessary
    template<typename TVal>
    static inline std::enable_if_t<std::is_floating_point<TVal>::value,ValueType> cost_cast(TVal val) {return (ValueType)std::round(val);}
    /// converts an integral value to the model's value type
    template<typename TVal>
    static inline std::enable_if_t<std::is_integral<TVal>::value,ValueType> cost_cast(TVal val) {return (ValueType)val;}*/
    template<typename TVal>
    static inline ValueType cost_cast(TVal val) {return (ValueType)val;}
};

#ifdef __LITIV_FGSTEREOM_HPP__
#error "bad inline header config"
#endif //def(__LITIV_FGSTEREOM_HPP__)
#define __LITIV_FGSTEREOM_HPP__
#include "litiv/imgproc/ForegroundStereoMatcher.inl.hpp"
#undef __LITIV_FGSTEREOM_HPP__