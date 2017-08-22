
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2017 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

#include "litiv/utils/opencv.hpp"
#include "litiv/utils/math.hpp"
#include <opencv2/features2d.hpp>

#define SHAPECONTEXT_DEFAULT_ANG_BINS    (12)
#define SHAPECONTEXT_DEFAULT_RAD_BINS    (5)
#define SHAPECONTEXT_DEFAULT_INNER_RAD   (0.1)
#define SHAPECONTEXT_DEFAULT_OUTER_RAD   (1.0)
#define sHAPECONTEXT_DEFAULT_ROT_INVAR   (false)
#define SHAPECONTEXT_DEFAULT_NORM_BINS   (true)
#define SHAPECONTEXT_DEFAULT_USE_NZ_INIT (true)

/**
    Shape Context (SC) feature extractor

    (inspired from OpenCV's implementation in the 'shape' module)

    For more details on the different parameters, see S. Belongie, J. Malik and J. Puzicha, "Shape
    Matching and Object Recognition Using Shape Contexts", in IEEE TPAMI2002.

*/
class ShapeContext : public cv::DescriptorExtractor {
public:
    /// constructor for absolute description space (i.e. using absolute radii values)
    explicit ShapeContext(size_t nInnerRadius,
                          size_t nOuterRadius,
                          size_t nAngularBins=SHAPECONTEXT_DEFAULT_ANG_BINS,
                          size_t nRadialBins=SHAPECONTEXT_DEFAULT_RAD_BINS,
                          bool bRotationInvariant=sHAPECONTEXT_DEFAULT_ROT_INVAR,
                          bool bNormalizeBins=SHAPECONTEXT_DEFAULT_NORM_BINS,
                          bool bUseNonZeroInit=SHAPECONTEXT_DEFAULT_USE_NZ_INIT);
    /// constructor for mean-normalized description space (i.e. using relative radii values)
    explicit ShapeContext(double dRelativeInnerRadius/*=SHAPECONTEXT_DEFAULT_INNER_RAD*/,
                          double dRelativeOuterRadius/*=SHAPECONTEXT_DEFAULT_OUTER_RAD*/,
                          size_t nAngularBins=SHAPECONTEXT_DEFAULT_ANG_BINS,
                          size_t nRadialBins=SHAPECONTEXT_DEFAULT_RAD_BINS,
                          bool bRotationInvariant=sHAPECONTEXT_DEFAULT_ROT_INVAR,
                          bool bNormalizeBins=SHAPECONTEXT_DEFAULT_NORM_BINS,
                          bool bUseNonZeroInit=SHAPECONTEXT_DEFAULT_USE_NZ_INIT);
    /// loads extractor params from the specified file node @@@@ not impl
    virtual void read(const cv::FileNode&) override;
    /// writes extractor params to the specified file storage @@@@ not impl
    virtual void write(cv::FileStorage&) const override;
    /// returns the window size that will be used around each keypoint (null here; defined at extraction time)
    virtual cv::Size windowSize() const;
    /// returns the border size required around each keypoint in x or y direction (null here; defined at extraction time)
    virtual int borderSize(int nDim=0) const; // typically equal to windowSize().width/2
    /// returns the expected dense descriptor matrix output info, for a given input matrix size/type
    virtual lv::MatInfo getOutputInfo(const lv::MatInfo& oInputInfo) const;
    /// returns the current descriptor size, in bytes (overrides cv::DescriptorExtractor's)
    virtual int descriptorSize() const override;
    /// returns the current descriptor data type (overrides cv::DescriptorExtractor's)
    virtual int descriptorType() const override;
    /// returns the default norm type to use with this descriptor (overrides cv::DescriptorExtractor's)
    virtual int defaultNorm() const override;
    /// return true if detector object is empty (overrides cv::DescriptorExtractor's)
    virtual bool empty() const override;
    /// sets whether cuda should be used internally (if possible) or not
    virtual bool setUseCUDA(bool bVal, int nDeviceID=0);
    /// sets the block size to use in the cuda kernel (might speed up compute when using many kpts)
    virtual bool setBlockSize(size_t nThreadCount=size_t(cv::cuda::DeviceInfo().warpSize()));

    /// returns whether descriptor bin arrays will be 0-1 normalized before returning or not
    bool isNormalizingBins() const;
    /// returns whether descriptor bins will be initialized with (small) nonzero values or not
    bool isNonZeroInitBins() const;
    /// returns the cv::ContourApproximationModes detection strategy to use when finding contours in binary images
    int chainDetectMethod() const;

    /// similar to DescriptorExtractor::compute(const cv::Mat& image, ...), but in this case, the descriptors matrix has the same shape as the input matrix, and all image points are described (note: descriptors close to borders will be invalid)
    void compute2(const cv::Mat& oImage, cv::Mat& oDescMap);
    /// similar to DescriptorExtractor::compute(const cv::Mat& image, ...), but in this case, the descriptors matrix has the same shape as the input matrix, and all image points are described (note: descriptors close to borders will be invalid)
    void compute2(const cv::Mat& oImage, cv::Mat_<float>& oDescMap);
    /// similar to DescriptorExtractor::compute(const cv::Mat& image, ...), but in this case, the descriptors matrix has the same shape as the input matrix
    void compute2(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat_<float>& oDescMap);
    /// batch version of LBSP::compute2(const cv::Mat& image, ...)
    void compute2(const std::vector<cv::Mat>& voImageCollection, std::vector<cv::Mat_<float>>& voDescMapCollection);
    /// batch version of LBSP::compute2(const cv::Mat& image, ...)
    void compute2(const std::vector<cv::Mat>& voImageCollection, std::vector<std::vector<cv::KeyPoint> >& vvoPointCollection, std::vector<cv::Mat_<float>>& voDescMapCollection);

    /// utility function, used to reshape a descriptors matrix to its input image size (assumes fully-dense keypoints over input)
    void reshapeDesc(cv::Size oSize, cv::Mat& oDescriptors) const;
    /// utility function, used to filter out bad keypoints that would trigger out of bounds error because they're too close to the image border
    void validateKeyPoints(std::vector<cv::KeyPoint>& voKeypoints, cv::Size oImgSize) const;
    /// utility function, used to filter out bad pixels in a ROI that would trigger out of bounds error because they're too close to the image border
    void validateROI(cv::Mat& oROI) const;
    /// utility function, used to calculate the (C)EMD-L1 distance between two individual descriptors
    inline double calcDistance_EMD(const float* aDescriptor1, const float* aDescriptor2) const {
        lvDbgAssert_(!std::all_of(aDescriptor1,aDescriptor1+m_nDescSize,[](float v){
            lvDbgAssert(v>=0.0f);
            return v==0.0f;
        }),"opencv emd cannot handle null descriptors");
        lvDbgAssert_(!std::all_of(aDescriptor2,aDescriptor2+m_nDescSize,[](float v){
            lvDbgAssert(v>=0.0f);
            return v==0.0f;
        }),"opencv emd cannot handle null descriptors");
        const cv::Mat_<float> oDesc1(m_nDescSize,1,const_cast<float*>(aDescriptor1));
        const cv::Mat_<float> oDesc2(m_nDescSize,1,const_cast<float*>(aDescriptor2));
        return cv::EMD(oDesc1,oDesc2,-1,m_oEMDCostMap);
    }
    /// utility function, used to calculate the (C)EMD-L1 distance between two individual descriptors
    inline double calcDistance_EMD(const cv::Mat_<float>& oDescriptor1, const cv::Mat_<float>& oDescriptor2) const {
        lvAssert_(oDescriptor1.dims==oDescriptor2.dims && oDescriptor1.size==oDescriptor2.size,"descriptor mat sizes mismatch");
        lvAssert_(oDescriptor1.dims==2 || oDescriptor1.dims==3,"unexpected descriptor matrix dim count");
        lvAssert_(oDescriptor1.dims!=2 || oDescriptor1.total()==size_t(m_nRadialBins*m_nAngularBins),"unexpected descriptor size");
        lvAssert_(oDescriptor1.dims!=3 || (oDescriptor1.size[0]==1 && oDescriptor1.size[1]==1 && oDescriptor1.size[2]==m_nRadialBins*m_nAngularBins),"unexpected descriptor size");
        return calcDistance_EMD(oDescriptor1.ptr<float>(0),oDescriptor2.ptr<float>(0));
    }
    /// utility function, used to calculate the L2 distance between two individual descriptors
    inline double calcDistance_L2(const float* aDescriptor1, const float* aDescriptor2) const {
        const cv::Mat_<float> oDesc1(1,m_nRadialBins*m_nAngularBins,const_cast<float*>(aDescriptor1));
        const cv::Mat_<float> oDesc2(1,m_nRadialBins*m_nAngularBins,const_cast<float*>(aDescriptor2));
        return cv::norm(oDesc1,oDesc2,cv::NORM_L2);
    }
    /// utility function, used to calculate the L2 distance between two individual descriptors
    inline double calcDistance_L2(const cv::Mat_<float>& oDescriptor1, const cv::Mat_<float>& oDescriptor2) const {
        lvAssert_(oDescriptor1.dims==oDescriptor2.dims && oDescriptor1.size==oDescriptor2.size,"descriptor mat sizes mismatch");
        lvAssert_(oDescriptor1.dims==2 || oDescriptor1.dims==3,"unexpected descriptor matrix dim count");
        lvAssert_(oDescriptor1.dims!=2 || oDescriptor1.total()==size_t(m_nRadialBins*m_nAngularBins),"unexpected descriptor size");
        lvAssert_(oDescriptor1.dims!=3 || (oDescriptor1.size[0]==1 && oDescriptor1.size[1]==1 && oDescriptor1.size[2]==m_nRadialBins*m_nAngularBins),"unexpected descriptor size");
        return calcDistance_L2(oDescriptor1.ptr<float>(0),oDescriptor2.ptr<float>(0));
    }

protected:
    /// hides default keypoint detection impl (this class is a descriptor extractor only)
    using cv::DescriptorExtractor::detect;
    /// classic 'compute' implementation, based on DescriptorExtractor's arguments & expected output
    virtual void detectAndCompute(cv::InputArray oImage, cv::InputArray oMask, std::vector<cv::KeyPoint>& voKeypoints, cv::OutputArray oDescriptors, bool bUseProvidedKeypoints=false) override;
    /// number of angular bins to use
    const int m_nAngularBins;
    /// number of radial bins to use
    const int m_nRadialBins;
    /// total bin size of each descriptor
    const int m_nDescSize;
    /// absolute inner radius of the descriptor (in 'image' space)
    const int m_nInnerRadius;
    /// absolute outer radius of the descriptor (in 'image' space)
    const int m_nOuterRadius;
    /// relative inner radius of the descriptor (in 'mean-normalized' space)
    const double m_dInnerRadius;
    /// relative outer radius of the descriptor (in 'mean-normalized' space)
    const double m_dOuterRadius;
    /// defines whether descriptors will be made via 'mean-normalized' or 'image' space radii
    const bool m_bUseRelativeSpace;
    /// defines whether descriptors will be made rotation-invariant or not
    const bool m_bRotationInvariant;
    /// defines whether descriptor bins will be 0-1 normalized or not
    const bool m_bNormalizeBins;
    /// defines whether descriptor bins should be initialized with (small) nonzero values or not
    const bool m_bNonZeroInitBins;

private:

    /// generates radius limits mask using internal parameters
    void scdesc_generate_radmask();
    /// generates angle limits mask using internal parameters
    void scdesc_generate_angmask();
    /// generates EMD distance cost map using internal parameters
    void scdesc_generate_emdmask();
    /// fills contour point map using provided binary image
    void scdesc_fill_contours(const cv::Mat& oImage);
    /// fills mean-normalized dist map & angle map using internal contour/key points
    void scdesc_fill_maps(double dMeanDist=-1.0);
    /// fills descriptor using internal maps
    void scdesc_fill_desc(cv::Mat_<float>& oDescriptors, bool bGenDescMap);
    /// fills descriptor without using internal maps (only for absolute descs w/o rot inv)
    void scdesc_fill_desc_direct(cv::Mat_<float>& oDescriptors, bool bGenDescMap);
    /// descriptor normalisation approach impl
    void scdesc_norm(cv::Mat_<float>& oDescriptors) const;

    // helper variables for internal impl (helps avoid continuous mem realloc)
    std::vector<double> m_vAngularLimits,m_vRadialLimits;
    cv::Mat_<double> m_oDistMap,m_oAngMap;
    cv::Mat_<float> m_oEMDCostMap;
    cv::Mat_<int> m_oAbsDescLUMap;
#if HAVE_CUDA
    cv::cuda::GpuMat m_oDescriptors_dev;
    cv::cuda::GpuMat m_oKeyPts_dev,m_oContourPts_dev;
    cv::cuda::GpuMat m_oDistMask_dev,m_oDescLUMap_dev;
    unsigned long long m_pDescLUMap_tex;
#endif //HAVE_CUDA
    cv::Mat_<cv::Point2f> m_oKeyPts,m_oContourPts;
    cv::Mat_<uchar> m_oBinMask,m_oDistMask,m_oDilateKernel;
    cv::Size m_oCurrImageSize;
    size_t m_nBlockSize;
    bool m_bUsingFullKeyPtMap;
    bool m_bUseCUDA;
};