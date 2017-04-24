
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

#include "litiv/utils/opengm.hpp"
#include "litiv/features2d.hpp"
#include "litiv/imgproc.hpp"

// config toggles/options
#define STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL        0
#define STEREOSEGMATCH_CONFIG_USE_DASCGF_FEATS      1
#define STEREOSEGMATCH_CONFIG_USE_DASCRF_FEATS      0
#define STEREOSEGMATCH_CONFIG_USE_LSS_FEATS         0
#define STEREOSEGMATCH_CONFIG_USE_SHAPE_EMD_SIM     0

// default param values
#define STEREOSEGMATCH_DEFAULT_MAX_MOVE_ITER        (size_t(1000))
#define STEREOSEGMATCH_DEFAULT_SHAPEDESC_RAD        (size_t(30))
#define STEREOSEGMATCH_DEFAULT_GRAD_KERNEL_SIZE     (int(7))
#define STEREOSEGMATCH_DEFAULT_DISTTRANSF_SCALE     (-0.1f)
#define STEREOSEGMATCH_DEFAULT_ITER_PER_RESEGM      ((nStereoLabels*3)/2)

// unary costs params
#define STEREOSEGMATCH_UNARY_COST_OOB_CST           (ValueType(1000))
#define STEREOSEGMATCH_UNARY_COST_OCCLUDED_CST      (ValueType(2000))
#define STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST      (ValueType(4000))
#define STEREOSEGMATCH_IMGSIM_COST_DESC_SCALE       (2500)
#define STEREOSEGMATCH_IMGSIM_COST_RAW_SCALE        (2.0f/STEREOSEGMATCH_IMGSIM_COST_DESC_SCALE)
#define STEREOSEGMATCH_SHPSIM_COST_DESC_SCALE       (2500)
#define STEREOSEGMATCH_UNIQUE_COST_OVER_SCALE       (500)
#define STEREOSEGMATCH_SHPDIST_COST_SCALE           (1000)
// pairwise costs params
#define STEREOSEGMATCH_LBLSIM_COST_MAXOCCL          (ValueType(5000))
#define STEREOSEGMATCH_LBLSIM_COST_MAXTRUNC         (ValueType(5000))
#define STEREOSEGMATCH_LBLSIM_COST_MAXDIFF_CST      (10)
#define STEREOSEGMATCH_LBLSIM_USE_EXP_GRADPIVOT     (1)
#if STEREOSEGMATCH_LBLSIM_USE_EXP_GRADPIVOT
#define STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE    (4)
#define STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST    (16)
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
    using ValueType = int; ///< type used for factor values (@@@@ could be integer? retest speed later?)
    using IndexType = size_t; ///< type used for node indexing (note: pretty much hardcoded everywhere in impl below)
#if STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
    using ExplicitFunction = opengm::ExplicitFunction<ValueType,IndexType,InternalLabelType>; ///< shortcut for explicit function
#else //!STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
    using ExplicitFunction = lv::gm::ExplicitViewFunction<ValueType,IndexType,InternalLabelType>; ///< shortcut for explicit view function
#endif //!STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
    using FunctionTypeList = opengm::meta::TypeListGenerator<ExplicitFunction/*,...*/>::type;  ///< list of all functions the models can use
    using StereoSpaceType = opengm::SimpleDiscreteSpace<IndexType,InternalLabelType>; ///< shortcut for discrete stereo space type (simple = all nodes have the same # of labels)
    using ResegmSpaceType = opengm::StaticSimpleDiscreteSpace<2,IndexType,InternalLabelType>; ///< shortcut for discrete resegm space type (binary labels for fg/bg)
    using StereoModelType = opengm::GraphicalModel<ValueType,opengm::Adder,FunctionTypeList,StereoSpaceType>; ///< shortcut for stereo graphical model type
    using ResegmModelType = opengm::GraphicalModel<ValueType,opengm::Adder,FunctionTypeList,ResegmSpaceType>; ///< shortcut for resegm graphical model type
    using StereoFuncID = StereoModelType::FunctionIdentifier; ///< shortcut for stereo model function identifier type
    using ResegmFuncID = ResegmModelType::FunctionIdentifier; ///< shortcut for resegm model function identifier type
    using StereoFunc = std::pair<StereoFuncID,ExplicitFunction&>; ///< stereo funcid-funcobj pair used as viewer to explicit data
    using ResegmFunc = std::pair<ResegmFuncID,ExplicitFunction&>; ///< stereo funcid-funcobj pair used as viewer to explicit data
    using ICosegmentor<OutputLabelType,s_nInputArraySize,s_nOutputArraySize>::apply; ///< helps avoid 'no matching function' issues for apply overloads
    static constexpr OutputLabelType s_nDontCareLabel = std::numeric_limits<OutputLabelType>::min(); ///< real label value reserved for 'dont care' pixels
    static constexpr OutputLabelType s_nOccludedLabel = std::numeric_limits<OutputLabelType>::max(); ///< real label value reserved for 'occluded' pixels
    static constexpr OutputLabelType s_nForegroundLabel = OutputLabelType(std::numeric_limits<InternalLabelType>::max()); ///< real label value reserved for foreground pixels
    static constexpr OutputLabelType s_nBackgroundLabel = OutputLabelType(0); ///< real label value reserved for background pixels
    static constexpr InternalLabelType s_nForegroundLabelIdx = InternalLabelType(1); ///< internal label value used for 'foreground' labeling
    static constexpr InternalLabelType s_nBackgroundLabelIdx = InternalLabelType(0); ///< internal label value used for 'background' labeling
    static constexpr size_t s_nMaxOrder = /*@@@@*/s_nInputArraySize; ///< used to limit internal static assignment array sizes
    static constexpr size_t s_nMaxCliqueAssign = size_t(1)<<s_nMaxOrder; ///< used to limit internal static assignment array sizes
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

    /// full stereo graph matcher constructor; relies on provided parameters to build graphical model base
    StereoSegmMatcher(const cv::Size& oImageSize, size_t nMinDispOffset, size_t nMaxDispOffset, size_t nDispStep=1);
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
    /// returns the output stereo label used to represent 'dont care' pixels
    static constexpr OutputLabelType getStereoDontCareLabel() {return s_nDontCareLabel;}
    /// returns the output stereo label used to represent 'occluded' pixels
    static constexpr OutputLabelType getStereoOccludedLabel() {return s_nOccludedLabel;}
    /// returns the expected input stereo head count
    static constexpr size_t getCameraCount() {return getInputStreamCount()/2;}

    /// holds graph model data for both stereo and resegmentation models
    struct GraphModelData {
        /// list of boostrap labeling types allowed for graph initialization
        enum LabelInitType {
            LabelInit_Default, ///< initializes labeling matrices with zeros
            LabelInit_Random, ///< initializes labeling matrices with random values via MT19937
            LabelInit_LocalOptim, ///< initializes labeling matrices with local optimas by checking low order factors only
            LabelInit_Explicit, ///< initializes labeling matrices with user-provided values
        };
        /// list of label order types allowed for iterative move making
        enum LabelOrderType {
            LabelOrder_Default, ///< uses ascending label ordering
            LabelOrder_Random, ///< uses random (but non-repeating) label ordering
            LabelOrder_Explicit, ///< uses user-provided label ordering
        };
        /// basic info struct used for node-level graph model updates and data lookups
        struct NodeInfo {
            /// image grid coordinates associated this node model's index
            int nRowIdx,nColIdx;
            /// id for this node's unary factors
            size_t nStereoUnaryFactID,nResegmUnaryFactID;
            /// pointer to this node's stereo unary function
            StereoFunc* pStereoUnaryFunc;
            /// pointer to this node's resegm unary function
            ResegmFunc* pResegmUnaryFunc;
            /// ids for this node's two (pairwise) connected neighboring nodes
            std::array<size_t,2> anPairwNodeIdxs;
            /// ids for this node's two pairwise factors
            std::array<size_t,2> anStereoPairwFactIDs,anResegmPairwFactIDs;
            /// pointer to this node's two stereo pairwise functions
            std::array<StereoFunc*,2> apStereoPairwFuncs;
            /// pointer to this node's two stereo pairwise functions
            std::array<StereoFunc*,2> apResegmPairwFuncs;
            // @@@@@ add higher o facts/funcptrs here
        };

        /// default constructor; receives model construction data from algo constructor
        GraphModelData(const cv::Size& oImageSize, std::vector<OutputLabelType>&& vStereoLabels, size_t nStereoLabelStep);
        /// (pre)calculates features required for model updates, and optionally returns them in packet format
        void calcFeatures(const MatArrayIn& aInputs, cv::Mat* pFeatsPacket=nullptr);
        /// sets a previously precalculated features packet to be used in the next model updates (do not modify it before that!)
        void setNextFeatures(const cv::Mat& oPackedFeats);
        /// performs the actual bi-model inference
        opengm::InferenceTermination infer();

        /// translate an internal graph label to a real disparity offset label
        OutputLabelType getRealLabel(InternalLabelType nLabel) const;
        /// translate a real disparity offset label to an internal graph label
        InternalLabelType getInternalLabel(OutputLabelType nRealLabel) const;
        /// returns the stereo associations count for a given graph node by row/col indices
        AssocCountType getAssocCount(int nRowIdx, int nColIdx) const;
        /// returns the cost of adding a stereo association for a given node coord set & origin column idx
        ValueType calcAddAssocCost(int nRowIdx, int nColIdx, InternalLabelType nLabel) const;
        /// returns the cost of removing a stereo association for a given node coord set & origin column idx
        ValueType calcRemoveAssocCost(int nRowIdx, int nColIdx, InternalLabelType nLabel) const;
        /// returns the total stereo association cost for all grid nodes
        ValueType calcTotalAssocCost() const;

        /// max move making iteration count allowed during inference
        size_t m_nMaxMoveIterCount;
        /// boostrap labeling type to use for stereo graph initialization
        LabelInitType m_eStereoLabelInitType;
        /// label order type to use for iterative stereo move making
        LabelOrderType m_eStereoLabelOrderType;
        /// random seeds to use to initialize labeling/label-order arrays
        size_t m_nStereoLabelOrderRandomSeed,m_nStereoLabelingRandomSeed;
        /// contains the (internal) stereo label ordering to use for each iteration
        std::vector<InternalLabelType> m_vStereoLabelOrdering;
        /// contains the (internal) labeling of the stereo/resegm graphs (mutable for inference)
        mutable cv::Mat_<InternalLabelType> m_oStereoLabeling,m_oResegmLabeling;
        /// 2d map which contains how many associations a node possesses (mutable for inference)
        mutable cv::Mat_<AssocCountType> m_oAssocCounts;
        /// 3d map which lists the associations (by idx) for each node in the graph (mutable for inference)
        mutable cv::Mat_<AssocIdxType> m_oAssocMap;
        /// 2d map which contains transient unary factor energy costs for all graph nodes (mutable for inference)
        mutable cv::Mat_<ValueType> m_oAssocCosts,m_oStereoUnaryCosts,m_oResegmUnaryCosts;
        /// 2d map which contains transient pairwise factor energy costs for all graph nodes (mutable for inference, for debug only)
        mutable cv::Mat_<ValueType> m_oStereoPairwCosts,m_oResegmPairwCosts;
        /// holds the set of features to use during the next inference (mutable, as shape features will change during inference)
        mutable std::vector<cv::Mat> m_vNextFeats;
        /// contains the predetermined (max) 2D grid size for the graph models
        const lv::MatSize m_oGridSize;
        /// contains all 'real' stereo labels, plus the 'occluded'/'dontcare' labels
        const std::vector<OutputLabelType> m_vStereoLabels;
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
        /// opengm stereo graph model object
        std::unique_ptr<StereoModelType> m_pStereoModel;
        /// opengm resegm graph model object
        std::unique_ptr<ResegmModelType> m_pResegmModel;
        /// model info lookup array
        std::vector<NodeInfo> m_vNodeInfos;
        /// stereo model unary functions
        std::vector<StereoFunc> m_vStereoUnaryFuncs;
        /// stereo model pairwise functions array
        std::array<std::vector<StereoFunc>,2> m_avStereoPairwFuncs;
        /// resegm model unary functions
        std::vector<ResegmFunc> m_vResegmUnaryFuncs;
        /// resegm model pairwise functions array
        std::array<std::vector<ResegmFunc>,2> m_avResegmPairwFuncs;
        /// functions data arrays (contiguous blocks)
        std::unique_ptr<ValueType[]> m_aStereoFuncsData,m_aResegmFuncsData;
        /// stereo/resegm models unary functions base pointer
        ValueType* m_pStereoUnaryFuncsDataBase,*m_pResegmUnaryFuncsDataBase;
        /// stereo/resegm models pairwise functions base pointer
        ValueType* m_pStereoPairwFuncsDataBase,*m_pResegmPairwFuncsDataBase;
        /// lookup table for added association cost w.r.t. prior count
        lv::AutoBuffer<ValueType,200> m_aAssocCostAddLUT;
        /// lookup table for removed association cost w.r.t. prior count
        lv::AutoBuffer<ValueType,200> m_aAssocCostRemLUT;
        /// lookup table for total association cost w.r.t. prior count
        lv::AutoBuffer<ValueType,200> m_aAssocCostSumLUT;
        /// lookup table for label similarity cost gradient factor
        lv::LUT<uchar,float,256> m_aLabelSimCostGradFactLUT;

        /// holds the feature extractor to use on input images
#if STEREOSEGMATCH_CONFIG_USE_DASCGF_FEATS || STEREOSEGMATCH_CONFIG_USE_DASCRF_FEATS
        std::unique_ptr<DASC> m_pImgDescExtractor;
#elif STEREOSEGMATCH_CONFIG_USE_LSS_FEATS
        std::unique_ptr<LSS> m_pImgDescExtractor;
#endif //STEREOSEGMATCH_CONFIG_USE_..._FEATS
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
        /// defines the indices of feature maps inside precalc packets (per camera head)
        enum FeatPackingList {
            FeatPackSize=20,
            FeatPackOffset=9,
            // absolute values for direct indexing
            FeatPack_LeftImgDescs=0,
            FeatPack_LeftShpDescs=1,
            FeatPack_LeftInitFGDist=2,
            FeatPack_LeftInitBGDist=3,
            FeatPack_LeftFGDist=4,
            FeatPack_LeftBGDist=5,
            FeatPack_LeftGradY=6,
            FeatPack_LeftGradX=7,
            FeatPack_LeftGradMag=8,
            FeatPack_RightImgDescs=9,
            FeatPack_RightShpDescs=10,
            FeatPack_RightInitFGDist=11,
            FeatPack_RightInitBGDist=12,
            FeatPack_RightFGDist=13,
            FeatPack_RightBGDist=14,
            FeatPack_RightGradY=15,
            FeatPack_RightGradX=16,
            FeatPack_RightGradMag=17,
            FeatPack_ImgAffinity=18,
            FeatPack_ShpAffinity=19,
            // relative values for cam-based indexing
            FeatPackOffset_ImgDescs=0,
            FeatPackOffset_ShpDescs=1,
            FeatPackOffset_InitFGDist=2,
            FeatPackOffset_InitBGDist=3,
            FeatPackOffset_FGDist=4,
            FeatPackOffset_BGDist=5,
            FeatPackOffset_GradY=6,
            FeatPackOffset_GradX=7,
            FeatPackOffset_GradMag=8,
        };

    protected:
        /// adds a stereo association for a given node coord set & origin column idx
        void addAssoc(int nRowIdx, int nColIdx, InternalLabelType nLabel) const;
        /// removes a stereo association for a given node coord set & origin column idx
        void removeAssoc(int nRowIdx, int nColIdx, InternalLabelType nLabel) const;
        /// resets the stereo graph labelings using init state parameters
        void resetStereoLabelings();
        /// updates stereo model using new feats data
        void updateStereoModel(bool bInit);
        /// updates shape model using new feats data
        void updateResegmModel(bool bInit);
        /// calculates image features required for model updates using the provided input image array
        void calcImageFeatures(const std::array<cv::Mat,InputPackSize/InputPackOffset>& aInputImages, bool bInit);
        /// calculates shape features required for model updates using the provided input mask array
        void calcShapeFeatures(const std::array<cv::Mat,InputPackSize/InputPackOffset>& aInputMasks, bool bInit);
        /// fill internal temporary energy cost mats for the given stereo move operation
        void calcStereoMoveCosts(InternalLabelType nNewLabel) const;
        /// fill internal temporary energy cost mats for the given resegm move operation
        void calcResegmMoveCosts(InternalLabelType nNewLabel) const;
        /// holds stereo disparity graph inference algorithm impl (redirects for bi-model inference)
        std::unique_ptr<StereoGraphInference> m_pStereoInf;
        /// holds resegmentation graph inference algorithm impl (redirects for bi-model inference)
        std::unique_ptr<ResegmGraphInference> m_pResegmInf;
    };

    /// algo interface for multi-label graph model inference
    struct StereoGraphInference : opengm::Inference<StereoModelType,opengm::Minimizer> {
        /// full constructor of the inference algorithm; the graphical model must have already been constructed prior to this call
        StereoGraphInference(GraphModelData& oData);
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
    };

    /// algo interface for binary label graph model inference
    struct ResegmGraphInference : opengm::Inference<ResegmModelType,opengm::Minimizer> {
        /// full constructor of the inference algorithm; the graphical model must have already been constructed prior to this call
        ResegmGraphInference(GraphModelData& oData);
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
    };

protected:
    /// expected size for all input images
    const cv::Size m_oImageSize;
    /// holds bimodel data & inference algo impls
    std::unique_ptr<GraphModelData> m_pModelData;
    /// helper func to display scaled disparity maps
    static cv::Mat getStereoMapDisplay(const GraphModelData& oData);
    /// helper func to display scaled assoc count maps
    static cv::Mat getAssocCountsMapDisplay(const GraphModelData& oData);
};

#define __LITIV_FGSTEREOM_HPP__
#include "litiv/imgproc/ForegroundStereoMatcher.inl.hpp"
#undef __LITIV_FGSTEREOM_HPP__