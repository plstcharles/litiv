
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

#pragma once

#include "litiv/utils/simd.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <unordered_set>
#include <map>

#ifndef CV_MAT_COND_DEPTH_TYPE
#define CV_MAT_COND_DEPTH_TYPE(cvtype_flag,depth_flag,depth_type,depth_alt) \
    std::conditional_t<CV_MAT_DEPTH(cvtype_flag)==depth_flag,depth_type,depth_alt>
#endif //ndef(CV_MAT_COND_DEPTH_TYPE)

#ifndef CV_MAT_DEPTH_BYTES
#define CV_MAT_DEPTH_BYTES(cvtype_flag) \
    (size_t(CV_MAT_DEPTH(cvtype_flag)<=1?1:CV_MAT_DEPTH(cvtype_flag)<4?2:CV_MAT_DEPTH(cvtype_flag)<6?4:8))
#endif //ndef(CV_MAT_DEPTH_BYTES)

namespace lv {

    struct DisplayHelper;
    using DisplayHelperPtr = std::shared_ptr<DisplayHelper>;

    /// returns a 16-byte aligned matrix allocator for SSE(1/2/3/4.1/4.2) support (should never be modified, despite non-const!)
    cv::MatAllocator* getMatAllocator16a();
    /// returns a 16-byte aligned matrix allocator for AVX(1/2) support (should never be modified, despite non-const!)
    cv::MatAllocator* getMatAllocator32a();

    /// returns whether a given memory address is aligned to the templated step or not
    template<size_t nByteAlign, typename Taddr>
    bool isAligned(Taddr pData) {
        static_assert(std::is_pointer<Taddr>::value,"need to pass pointer type");
        static_assert(nByteAlign>0,"byte align must be positive value");
        return (uintptr_t(pData)%nByteAlign)==0;
    }

    /// mat type helper struct which provides basic static traits info on ocv matrix element types
    template<int nCVType>
    struct MatType_ {
        /// holds the typename for underlying mat data
        typedef std::enable_if_t<(CV_MAT_DEPTH(nCVType)>=0 && CV_MAT_DEPTH(nCVType)<=6),
            CV_MAT_COND_DEPTH_TYPE(nCVType,0/**/,uchar,
                CV_MAT_COND_DEPTH_TYPE(nCVType,1,char,
                    CV_MAT_COND_DEPTH_TYPE(nCVType,2,ushort,
                        CV_MAT_COND_DEPTH_TYPE(nCVType,3,short,
                            CV_MAT_COND_DEPTH_TYPE(nCVType,4,int,
                                CV_MAT_COND_DEPTH_TYPE(nCVType,5,float,
                                    CV_MAT_COND_DEPTH_TYPE(nCVType,6,double,void)))))))> base_type;
        /// returns the channel count for the underlying mat data (i.e. smallest implicit mat dim)
        static constexpr int channels() {return CV_MAT_CN(nCVType);}
        /// returns the ocv depth id (NOT A BYTE COUNT!) for the underlying mat data
        static constexpr int depth() {return CV_MAT_DEPTH(nCVType);}
        /// returns the byte depth for the underlying mat data
        static constexpr size_t depthBytes() {return CV_MAT_DEPTH_BYTES(nCVType);}
        /// returns the element size (in bytes) for the underlying mat data
        static constexpr size_t elemSize() {return CV_MAT_DEPTH_BYTES(nCVType)*size_t(CV_MAT_CN(nCVType));}
        /// returns the internal opencv mat type argument
        static constexpr int type() {return nCVType;}
        /// returns the internal opencv mat type argument
        int operator()() const {return nCVType;}
    };

    /// mat type helper struct which provides basic dynamic info on ocv matrix element types
    struct MatType {
        /// default constructor; assigns type as CV_8UC1 internally
        MatType() : m_nCVType(CV_8UC1) {}
        /// cv type-based constructor; also validates m_nCVType for possible ocv mat configs
        MatType(int nCVType) : m_nCVType(nCVType) {
            lvAssert__((CV_MAT_DEPTH(m_nCVType)>=0 && CV_MAT_DEPTH(m_nCVType)<=6),"bad ocv type depth (type=%d, depth=%d)",m_nCVType,CV_MAT_DEPTH(m_nCVType));
            lvAssert__((CV_MAT_CN(m_nCVType)>0 && CV_MAT_CN(m_nCVType)<=4),"bad ocv type channels (type=%d, channels=%d)",m_nCVType,CV_MAT_CN(m_nCVType));
        }
        /// assignment operator for new opencv matrix data type
        MatType& operator=(int nCVType) {
            lvAssert__((CV_MAT_DEPTH(nCVType)>=0 && CV_MAT_DEPTH(nCVType)<=6),"bad ocv type depth (type=%d, depth=%d)",nCVType,CV_MAT_DEPTH(nCVType));
            lvAssert__((CV_MAT_CN(nCVType)>0 && CV_MAT_CN(nCVType)<=4),"bad ocv type channels (type=%d, channels=%d)",nCVType,CV_MAT_CN(nCVType));
            m_nCVType = nCVType;
            return *this;
        }
        /// returns the channel count for the underlying mat data (i.e. smallest implicit mat dim)
        int channels() const {return CV_MAT_CN(m_nCVType);}
        /// returns the ocv depth id (NOT A BYTE COUNT!) for the underlying mat data
        int depth() const {return CV_MAT_DEPTH(m_nCVType);}
        /// returns the byte depth for the underlying mat data
        size_t depthBytes() const {return CV_MAT_DEPTH_BYTES(m_nCVType);}
        /// returns the element size (in bytes) for the underlying mat data
        size_t elemSize() const {return CV_MAT_DEPTH_BYTES(m_nCVType)*size_t(CV_MAT_CN(m_nCVType));}
        /// returns the internal opencv mat type argument
        int type() const {return m_nCVType;}
        /// returns the internal opencv mat type argument
        int operator()() const {return m_nCVType;}
        /// is-equal test operator for other MatType structs
        bool operator==(const MatType& o) const {return m_nCVType==o.m_nCVType;}
        /// is-not-equal test operator for other MatType structs
        bool operator!=(const MatType& o) const {return !(*this==o);}
        /// returns whether the given typename is compatible with the internal opencv mat type (does not consider channels by default)
        template<typename T, bool bCheckWithChannels=false>
        bool isTypeCompat() const {
            const int nChannels = channels();
            const int nDepth = depth();
            if(bCheckWithChannels && nChannels>1) {
                if(nChannels==2)
                    return
                        (nDepth==CV_8U && std::is_same<T,cv::Vec<uint8_t,2>>::value) ||
                        (nDepth==CV_8S && std::is_same<T,cv::Vec<int8_t,2>>::value) ||
                        (nDepth==CV_16U && std::is_same<T,cv::Vec<uint16_t,2>>::value) ||
                        (nDepth==CV_16S && std::is_same<T,cv::Vec<int16_t,2>>::value) ||
                        (nDepth==CV_32S && std::is_same<T,cv::Vec<int32_t,2>>::value) ||
                        (nDepth==CV_32F && std::is_same<T,cv::Vec<float,2>>::value) ||
                        (nDepth==CV_64F && std::is_same<T,cv::Vec<double,2>>::value);
                else if(nChannels==3)
                    return
                        (nDepth==CV_8U && std::is_same<T,cv::Vec<uint8_t,3>>::value) ||
                        (nDepth==CV_8S && std::is_same<T,cv::Vec<int8_t,3>>::value) ||
                        (nDepth==CV_16U && std::is_same<T,cv::Vec<uint16_t,3>>::value) ||
                        (nDepth==CV_16S && std::is_same<T,cv::Vec<int16_t,3>>::value) ||
                        (nDepth==CV_32S && std::is_same<T,cv::Vec<int32_t,3>>::value) ||
                        (nDepth==CV_32F && std::is_same<T,cv::Vec<float,3>>::value) ||
                        (nDepth==CV_64F && std::is_same<T,cv::Vec<double,3>>::value);
                else //if(nChannels==4)
                    return
                        (nDepth==CV_8U && std::is_same<T,cv::Vec<uint8_t,4>>::value) ||
                        (nDepth==CV_8S && std::is_same<T,cv::Vec<int8_t,4>>::value) ||
                        (nDepth==CV_16U && std::is_same<T,cv::Vec<uint16_t,4>>::value) ||
                        (nDepth==CV_16S && std::is_same<T,cv::Vec<int16_t,4>>::value) ||
                        (nDepth==CV_32S && std::is_same<T,cv::Vec<int32_t,4>>::value) ||
                        (nDepth==CV_32F && std::is_same<T,cv::Vec<float,4>>::value) ||
                        (nDepth==CV_64F && std::is_same<T,cv::Vec<double,4>>::value);
            }
            return
                (nDepth==CV_8U && std::is_same<T,uint8_t>::value) ||
                (nDepth==CV_8S && std::is_same<T,int8_t>::value) ||
                (nDepth==CV_16U && std::is_same<T,uint16_t>::value) ||
                (nDepth==CV_16S && std::is_same<T,int16_t>::value) ||
                (nDepth==CV_32S && std::is_same<T,int32_t>::value) ||
                (nDepth==CV_32F && std::is_same<T,float>::value) ||
                (nDepth==CV_64F && std::is_same<T,double>::value);
        }
    private:
        /// holds the internal opencv mat type argument
        int m_nCVType;
    };

    /// mat dim size helper which provides easy-to-use and safe conversions from cv::Size and cv::MatSize
    /// (note: internally using 'major' dimension == first in vec)
    template<typename Tinteger>
    struct MatSize_ {
        /// default constructor; initializes internal config as zero-dim (empty)
        MatSize_() :
                m_vSizes{Tinteger(0)},m_aSizes(nullptr) {}
        /// cv::MatSize-based constructor
        MatSize_(const cv::MatSize& oSize) :
                m_vSizes(cvtSizes(oSize.p)),m_aSizes(m_vSizes[0]>Tinteger(0)?m_vSizes.data()+1:nullptr) {}
        /// cv::Size-based constructor
        MatSize_(const cv::Size& oSize) : // row-major by default, like opencv
                m_vSizes{Tinteger(2),Tinteger(oSize.height),Tinteger(oSize.width)},m_aSizes(m_vSizes.data()+1) {
            lvAssert_(oSize.width>=0 && oSize.height>=0,"sizes must be null or positive values");
        }
        /// array-based constructor
        template<size_t nDims, typename Tinteger2>
        MatSize_(const std::array<Tinteger2,nDims>& aSizes) :
                m_vSizes(cvtSizes((Tinteger2)nDims,aSizes.data())),m_aSizes(m_vSizes[0]>Tinteger(0)?m_vSizes.data()+1:nullptr) {}
        /// vector-based constructor
        template<typename Tinteger2>
        MatSize_(const std::vector<Tinteger2>& vSizes) :
                m_vSizes(cvtSizes((Tinteger2)vSizes.size(),vSizes.data())),m_aSizes(m_vSizes[0]>Tinteger(0)?m_vSizes.data()+1:nullptr) {}
        /// initlist-based constructor
        template<typename Tinteger2>
        MatSize_(const std::initializer_list<Tinteger2>& aSizes) :
                m_vSizes(cvtSizes((Tinteger2)aSizes.size(),aSizes.begin())),m_aSizes(m_vSizes[0]>Tinteger(0)?m_vSizes.data()+1:nullptr) {}
        /// explicit dims constructor; initializes internal config by casting all provided args
        template<typename... Tintegers>
        MatSize_(Tintegers... anSizes) :
                m_vSizes{Tinteger(sizeof...(anSizes)),Tinteger(anSizes)...},m_aSizes(m_vSizes.data()+1) {
            static_assert(lv::static_reduce(std::array<bool,sizeof...(Tintegers)>{(std::is_integral<Tintegers>::value)...},lv::static_reduce_and),"all given args should be integral");
            for(int nDimIdx=0; nDimIdx<dims(); ++nDimIdx)
                lvAssert_(m_aSizes[nDimIdx]>=Tinteger(0),"sizes must be null or positive values");
        }
        /// copy constructor for similar struct
        template<typename Tinteger2>
        MatSize_(const MatSize_<Tinteger2>& oSize) :
                m_vSizes(cvtSizes(oSize.m_vSizes.data()+1)),m_aSizes(m_vSizes[0]>Tinteger(0)?m_vSizes.data()+1:nullptr) {}
        /// returns the dimension count
        const Tinteger& dims() const {
            lvDbgAssert(!m_vSizes.empty()); // all constructors should guarantee proper sizing
            return m_vSizes[0];
        }
        /// returns the size at a given dimension index
        template<typename Tindex>
        const Tinteger& size(Tindex nDimIdx) const {
            static_assert(std::is_integral<Tindex>::value,"need an integer type for dimensions/size indexing");
            lvDbgAssert__((Tinteger)nDimIdx<dims(),"dimension index is out of bounds (dims=%d)",(int)dims());
            return m_aSizes[nDimIdx];
        }
        /// returns the (modifiable) size at a given dimension index
        template<typename Tindex>
        Tinteger& size(Tindex nDimIdx) {
            static_assert(std::is_integral<Tindex>::value,"need an integer type for dimensions/size indexing");
            lvDbgAssert__((Tinteger)nDimIdx<dims(),"dimension index is out of bounds (dims=%d)",(int)dims());
            return m_aSizes[nDimIdx];
        }
        /// returns the (unmodifiable) size array converted to int, for inline cv mat creation (non-trivial)
        const int* sizes() const {
            m_vSizesExt.resize(m_vSizes.size());
            for(size_t nIdx=0; nIdx<m_vSizes.size(); ++nIdx)
                m_vSizesExt[nIdx] = (int)m_vSizes[nIdx];
            return m_vSizesExt.data()+1;
        }
        /// returns the size at a given dimension index
        template<typename Tindex>
        const Tinteger& operator[](Tindex nDimIdx) const {
            return size(nDimIdx);
        }
        /// returns the (modifiable) size at a given dimension index
        template<typename Tindex>
        Tinteger& operator[](Tindex nDimIdx) {
            return size(nDimIdx);
        }
        /// returns the size at a given dimension index
        template<typename Tindex>
        const Tinteger& operator()(Tindex nDimIdx) const {
            return size(nDimIdx);
        }
        /// returns the (modifiable) size at a given dimension index
        template<typename Tindex>
        Tinteger& operator()(Tindex nDimIdx) {
            return size(nDimIdx);
        }
        /// implicit conversion op to raw 'Tinteger' size lookup array
        operator const Tinteger*() const {
            return m_aSizes;
        }
        /// implicit conversion op to cv::MatSize (non-trivial)
        operator cv::MatSize() const {
            m_vSizesExt.resize(m_vSizes.size());
            for(size_t nIdx=0; nIdx<m_vSizes.size(); ++nIdx)
                m_vSizesExt[nIdx] = (int)m_vSizes[nIdx];
            return cv::MatSize(m_vSizesExt.data()+1);
        }
        /// implicit conversion op to cv::Size (fails if this object contains more than two dimensions)
        operator cv::Size() const {
            lvAssert__(dims()<=Tinteger(2),"cannot fit dim count (%d) into cv::Size object",(int)dims());
            return (dims()==Tinteger(0))?cv::Size():(dims()==Tinteger(1))?cv::Size((int)size(0),1):cv::Size((int)size(1),(int)size(0));
        }
        /// is-equal test operator for cv::Size structs
        bool operator==(const cv::Size& oSize) const {
            return dims()<=Tinteger(2) && oSize==this->operator cv::Size();
        }
        /// is-not-equal test operator for cv::Size structs
        bool operator!=(const cv::Size& oSize) const {
            return !(*this==oSize);
        }
        /// is-equal test operator for cv::MatSize structs
        bool operator==(const cv::MatSize& oSize) const {
            const Tinteger nDims = dims();
            if(nDims!=(Tinteger)oSize[-1])
                return false;
            else if(nDims==Tinteger(2))
                return size(0)==(Tinteger)oSize[0] && size(1)==(Tinteger)oSize[1];
            for(Tinteger nDimIdx=0; nDimIdx<nDims; ++nDimIdx)
                if(size(nDimIdx)!=(Tinteger)oSize[nDimIdx])
                    return false;
            return true;
        }
        /// is-not-equal test operator for cv::MatSize structs
        bool operator!=(const cv::MatSize& oSize) const {
            return !(*this==oSize);
        }
        /// is-equal test operator for other MatSize_ structs
        template<typename Tinteger2>
        bool operator==(const MatSize_<Tinteger2>& oSize) const {
            if(empty() && oSize.empty())
                return true;
            const Tinteger nDims = dims();
            if(nDims!=(Tinteger)oSize.dims())
                return false;
            else if(nDims==Tinteger(2))
                return size(0)==(Tinteger)oSize.size(0) && size(1)==(Tinteger)oSize.size(1);
            for(Tinteger nDimIdx=0; nDimIdx<nDims; ++nDimIdx)
                if(size(nDimIdx)!=(Tinteger)oSize.size(nDimIdx))
                    return false;
            return true;
        }
        /// is-not-equal test operator for other MatSize_ structs
        template<typename Tinteger2>
        bool operator!=(const MatSize_<Tinteger2>& oSize) const {
            return !(*this==oSize);
        }
        /// assignment operator for different templated integer struct
        template<typename Tinteger2>
        MatSize_& operator=(const MatSize_<Tinteger2>& oSize) {
            m_vSizes = cvtSizes(oSize.m_vSizes.data()+1);
            m_aSizes = m_vSizes[0]>Tinteger(0)?m_vSizes.data()+1:nullptr;
            return *this;
        }
        /// returns the number of elements in the current configuration (i.e. by multiplying all dim sizes)
        size_t total() const {
            if(dims()==Tinteger(0))
                return size_t(0);
            size_t nElem = size_t(size(0));
            for(Tinteger nDimIdx=1; nDimIdx<dims(); ++nDimIdx)
                nElem *= size_t(size(nDimIdx));
            return nElem;
        }
        /// returns whether the current configuration contains elements or not (i.e. if all dim sizes are non-null)
        bool empty() const {
            return total()==size_t(0);
        }
    protected:
        static_assert(std::is_integral<Tinteger>::value,"need an integer type for dimensions/size indexing");
        template<typename Tinteger2>
        static std::vector<Tinteger> cvtSizes(const Tinteger2* aSizes) {
            lvAssert__((int)aSizes[-1]>=0,"negative MatSize dim count (got '%d')",(int)aSizes[-1]);
            std::vector<Tinteger> vOutput((int)aSizes[-1]+1);
            vOutput[0] = (Tinteger)aSizes[-1];
            for(int nDimIdx=0; nDimIdx<(int)aSizes[-1]; ++nDimIdx) {
                lvAssert_((int)aSizes[nDimIdx]>=0,"MatSize sizes must be null or positive values");
                vOutput[nDimIdx+1] = (Tinteger)aSizes[nDimIdx];
            }
            return vOutput;
        }
        template<typename Tinteger2>
        static std::vector<Tinteger> cvtSizes(Tinteger2 nDims, const Tinteger2* aSizes) {
            lvAssert__((int)nDims>=0,"negative MatSize dim count (got '%d')",(int)nDims);
            std::vector<Tinteger> vOutput((int)nDims+1);
            vOutput[0] = (Tinteger)nDims;
            for(int nDimIdx=0; nDimIdx<(int)nDims; ++nDimIdx) {
                lvAssert_((int)aSizes[nDimIdx]>=0,"MatSize sizes must be null or positive values");
                vOutput[nDimIdx+1] = (Tinteger)aSizes[nDimIdx];
            }
            return vOutput;
        }
    private:
        template<typename Tinteger2>
        friend struct MatSize_;
        mutable std::vector<int> m_vSizesExt; // needed for cast to MatSize/int* only
        std::vector<Tinteger> m_vSizes;
        Tinteger* m_aSizes;
    };

    /// mat dim size helper struct defaults to size_t for dim/size indexing
    using MatSize = MatSize_<size_t>;

    /// ostream-friendly overload for MatSize (ADL will allow usage from this namespace)
    template<typename Tinteger>
    std::ostream& operator<<(std::ostream& os, const MatSize_<Tinteger>& oSize) {
        if(oSize.dims()==Tinteger(0))
            return os << "0-d:[]<empty>";
        os << (int)oSize.dims() << "-d:[";
        for(Tinteger nDimIdx=0; nDimIdx<oSize.dims(); ++nDimIdx)
            os << (int)oSize[nDimIdx] << (nDimIdx<oSize.dims()-1?",":"");
        return os << "]" << (oSize.total()>0?"":"<empty>");
    }

    /// simplified cv::Mat header info container for matrix preallocation
    struct MatInfo {
        /// contains info about the layout of the matrix's elements
        MatSize size;
        /// contains info about the type of the matrix's elements
        MatType type;
        /// is-equal test operator for other MatInfo structs
        bool operator==(const MatInfo& o) const {return size==o.size && type==o.type;}
        /// is-not-equal test operator for other MatInfo structs
        bool operator!=(const MatInfo& o) const {return !(*this==o);}
    };

    /// helper function to zero-init sparse and non-sparse matrices (sparse mat overload)
    template<typename T>
    inline void zeroMat(cv::SparseMat_<T>& oMat) {
        oMat.clear();
    }

    /// helper function to zero-init sparse and non-sparse matrices (regular mat specialization)
    template<typename T>
    inline void zeroMat(cv::Mat_<T>& oMat) {
        oMat = T();
    }

    /// helper function to fetch iterator location from sparse and non-sparse matrices (sparse iter overload)
    template<size_t nDims>
    inline const int* getIterPos(const cv::SparseMatConstIterator& pIter, std::array<int,nDims>& /*anIndices_TempStorage*/) {
        static_assert(nDims>0,"matrix dimension count too small");
        lvDbgAssert(pIter.m->dims()==int(nDims));
        return pIter.node()->idx; // not using idx_opt
    }

    /// helper function to fetch iterator location from sparse and non-sparse matrices (regular iter overload)
    template<size_t nDims>
    inline const int* getIterPos(const cv::MatConstIterator& pIter, std::array<int,nDims>& anIndices_TempStorage) {
        static_assert(nDims>1,"matrix dimension count too small");
        lvDbgAssert(pIter.m->dims==int(nDims));
        pIter.pos(anIndices_TempStorage.data());
        return anIndices_TempStorage.data();
    }

    /// helper function to fetch references from sparse and non-sparse matrices (sparse mat overload)
    template<typename T>
    inline T& getElem(cv::SparseMat_<T>& oMat, const int* idx) {
        return oMat.ref(idx);
    }

    /// helper function to fetch references from sparse and non-sparse matrices (regular mat overload)
    template<typename T>
    inline T& getElem(cv::Mat_<T>& oMat, const int* idx) {
        return oMat(idx);
    }

    /// helper function to count valid/allocated elements in sparse and non-sparse matrices (sparse mat overload)
    template<typename T>
    inline size_t getElemCount(const cv::SparseMat_<T>& oMat) {
        return oMat.nzcount();
    }

    /// helper function to count valid/allocated elements in sparse and non-sparse matrices (regular mat overload)
    template<typename T>
    inline size_t getElemCount(const cv::Mat_<T>& oMat) {
        return oMat.total();
    }

    /// returns pixel coordinates clamped to the given image & border size
    inline void clampImageCoords(int& nSampleCoord_X, int& nSampleCoord_Y, const int nBorderSize, const cv::Size& oImageSize) {
        lvDbgAssert_(nBorderSize>=0,"border size cannot be negative");
        lvDbgAssert_(oImageSize.area()>=0,"image size cannot be negative");
        if(nSampleCoord_X<nBorderSize)
            nSampleCoord_X = nBorderSize;
        else if(nSampleCoord_X>=oImageSize.width-nBorderSize)
            nSampleCoord_X = oImageSize.width-nBorderSize-1;
        if(nSampleCoord_Y<nBorderSize)
            nSampleCoord_Y = nBorderSize;
        else if(nSampleCoord_Y>=oImageSize.height-nBorderSize)
            nSampleCoord_Y = oImageSize.height-nBorderSize-1;
    }

    /// returns a random init/sampling position for the specified pixel position, given a predefined kernel; also guards against out-of-bounds values via image/border size check
    template<int nKernelHeight, int nKernelWidth>
    inline void getRandSamplePosition(const std::array<std::array<int,nKernelWidth>,nKernelHeight>& anSamplesInitPattern,
                                      const int nSamplesInitPatternTot, int& nSampleCoord_X, int& nSampleCoord_Y,
                                      const int nOrigCoord_X, const int nOrigCoord_Y, const int nBorderSize, const cv::Size& oImageSize) {
        static_assert(nKernelWidth>0 && nKernelHeight>0,"invalid init pattern array size");
        lvDbgAssert_(nSamplesInitPatternTot>0,"pattern max count must be positive");
        int r = 1+rand()%nSamplesInitPatternTot;
        for(nSampleCoord_Y=0; nSampleCoord_Y<nKernelHeight; ++nSampleCoord_Y) {
            for(nSampleCoord_X=0; nSampleCoord_X<nKernelWidth; ++nSampleCoord_X) {
                r -= anSamplesInitPattern[nSampleCoord_Y][nSampleCoord_X];
                if(r<=0)
                    goto stop;
            }
        }
        stop:
        nSampleCoord_X += nOrigCoord_X-nKernelWidth/2;
        nSampleCoord_Y += nOrigCoord_Y-nKernelHeight/2;
        clampImageCoords(nSampleCoord_X,nSampleCoord_Y,nBorderSize,oImageSize);
    }

    /// returns a random init/sampling position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
    inline void getRandSamplePosition_3x3_std1(int& nSampleCoord_X, int& nSampleCoord_Y, const int nOrigCoord_X, const int nOrigCoord_Y, const int nBorderSize, const cv::Size& oImageSize) {
        // based on 'floor(fspecial('gaussian',3,1)*256)'
        static_assert(sizeof(std::array<int,3>)==sizeof(int)*3,"bad std::array stl impl");
        static const int s_nSamplesInitPatternTot = 256;
        static const std::array<std::array<int,3>,3> s_anSamplesInitPattern ={
                std::array<int,3>{19,32,19,},
                std::array<int,3>{32,52,32,},
                std::array<int,3>{19,32,19,},
        };
        getRandSamplePosition<3,3>(s_anSamplesInitPattern,s_nSamplesInitPatternTot,nSampleCoord_X,nSampleCoord_Y,nOrigCoord_X,nOrigCoord_Y,nBorderSize,oImageSize);
    }

    /// returns a random init/sampling position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
    inline void getRandSamplePosition_7x7_std2(int& nSampleCoord_X, int& nSampleCoord_Y, const int nOrigCoord_X, const int nOrigCoord_Y, const int nBorderSize, const cv::Size& oImageSize) {
        // based on 'floor(fspecial('gaussian',7,2)*512)'
        static_assert(sizeof(std::array<int,7>)==sizeof(int)*7,"bad std::array stl impl");
        static const int s_nSamplesInitPatternTot = 512;
        static const std::array<std::array<int,7>,7> s_anSamplesInitPattern ={
                std::array<int,7>{ 2, 4, 6, 7, 6, 4, 2,},
                std::array<int,7>{ 4, 8,12,14,12, 8, 4,},
                std::array<int,7>{ 6,12,21,25,21,12, 6,},
                std::array<int,7>{ 7,14,25,28,25,14, 7,},
                std::array<int,7>{ 6,12,21,25,21,12, 6,},
                std::array<int,7>{ 4, 8,12,14,12, 8, 4,},
                std::array<int,7>{ 2, 4, 6, 7, 6, 4, 2,},
        };
        getRandSamplePosition<7,7>(s_anSamplesInitPattern,s_nSamplesInitPatternTot,nSampleCoord_X,nSampleCoord_Y,nOrigCoord_X,nOrigCoord_Y,nBorderSize,oImageSize);
    }

    /// returns a random neighbor position for the specified pixel position, given a predefined neighborhood; also guards against out-of-bounds values via image/border size check.
    template<int nNeighborCount>
    inline void getRandNeighborPosition(const std::array<std::array<int,2>,nNeighborCount>& anNeighborPattern,
                                        int& nNeighborCoord_X, int& nNeighborCoord_Y,
                                        const int nOrigCoord_X, const int nOrigCoord_Y,
                                        const int nBorderSize, const cv::Size& oImageSize) {
        static_assert(nNeighborCount>0,"invalid input neighbor pattern array size");
        int r = rand()%nNeighborCount;
        nNeighborCoord_X = nOrigCoord_X+anNeighborPattern[r][0];
        nNeighborCoord_Y = nOrigCoord_Y+anNeighborPattern[r][1];
        clampImageCoords(nNeighborCoord_X,nNeighborCoord_Y,nBorderSize,oImageSize);
    }

    /// returns a random neighbor position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
    inline void getRandNeighborPosition_3x3(int& nNeighborCoord_X, int& nNeighborCoord_Y, const int nOrigCoord_X, const int nOrigCoord_Y, const int nBorderSize, const cv::Size& oImageSize) {
        typedef std::array<int,2> Nb;
        static const std::array<std::array<int,2>,8> s_anNeighborPattern ={
                Nb{-1, 1},Nb{0, 1},Nb{1, 1},
                Nb{-1, 0},         Nb{1, 0},
                Nb{-1,-1},Nb{0,-1},Nb{1,-1},
        };
        getRandNeighborPosition<8>(s_anNeighborPattern,nNeighborCoord_X,nNeighborCoord_Y,nOrigCoord_X,nOrigCoord_Y,nBorderSize,oImageSize);
    }

    /// returns a random neighbor position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
    inline void getRandNeighborPosition_5x5(int& nNeighborCoord_X, int& nNeighborCoord_Y, const int nOrigCoord_X, const int nOrigCoord_Y, const int nBorderSize, const cv::Size& oImageSize) {
        typedef std::array<int,2> Nb;
        static const std::array<std::array<int,2>,24> s_anNeighborPattern ={
                Nb{-2, 2},Nb{-1, 2},Nb{0, 2},Nb{1, 2},Nb{2, 2},
                Nb{-2, 1},Nb{-1, 1},Nb{0, 1},Nb{1, 1},Nb{2, 1},
                Nb{-2, 0},Nb{-1, 0},         Nb{1, 0},Nb{2, 0},
                Nb{-2,-1},Nb{-1,-1},Nb{0,-1},Nb{1,-1},Nb{2,-1},
                Nb{-2,-2},Nb{-1,-2},Nb{0,-2},Nb{1,-2},Nb{2,-2},
        };
        getRandNeighborPosition<24>(s_anNeighborPattern,nNeighborCoord_X,nNeighborCoord_Y,nOrigCoord_X,nOrigCoord_Y,nBorderSize,oImageSize);
    }

    /// writes a given text string on an image using the original cv::putText
    inline void putText(cv::Mat& oImg, const std::string& sText, const cv::Scalar& vColor, bool bBottom=false, const cv::Point2i& oOffset=cv::Point2i(4,15), int nThickness=2, double dScale=1.2) {
        cv::putText(oImg,sText,cv::Point(oOffset.x,bBottom?(oImg.rows-oOffset.y):oOffset.y),cv::FONT_HERSHEY_PLAIN,dScale,vColor,nThickness,cv::LINE_AA);
    }

    /// prints the content of a matrix to the given stream with constant output element size
    template<typename T>
    inline void print(const cv::Mat_<T>& oMat, std::ostream& os=std::cout) {
        lvAssert_(oMat.dims==2,"function currently only defined for 2d mats; split dims and call for 2d slices");
        if(oMat.empty() || oMat.size().area()==0) {
            os << "   <empty>" << std::endl;
            return;
        }
        const size_t nMaxMetaColWidth = (size_t)std::max(lv::digit_count(oMat.cols),lv::digit_count(oMat.rows));
        double dMin,dMax;
        cv::minMaxIdx(oMat,&dMin,&dMax);
        const T tMin = (T)dMin;
        const T tMax = (T)dMax;
        constexpr bool bIsFloat = !std::is_integral<T>::value;
        using PrintType = std::conditional_t<bIsFloat,float,int64_t>;
        const bool bIsNormalized = tMax<=T(1) && tMin>=T(0); // useful for floats only
        const bool bHasNegative = int64_t(tMin)<int64_t(0);
        const size_t nMaxColWidth = size_t(bIsFloat?(bIsNormalized?6:(std::max(lv::digit_count((int64_t)tMin),lv::digit_count((int64_t)tMax))+5+int(bHasNegative!=0))):(std::max(lv::digit_count(tMin),lv::digit_count(tMax))+int(bHasNegative!=0)));
        const std::string sFormat = bIsFloat?(bIsNormalized?std::string("%6.4f"):((bHasNegative?std::string("%+"):std::string("%"))+std::to_string(nMaxColWidth)+std::string(".4f"))):((bHasNegative?std::string("%+"):std::string("%"))+std::to_string(nMaxColWidth)+std::string(PRId64));
        const std::string sMetaFormat = std::string("%")+std::to_string(nMaxMetaColWidth)+"i";
        const std::string sSpacer = "  ";
        const auto lPrinter = [&](const T& v) {os << sSpacer << lv::putf(sFormat.c_str(),(PrintType)v);};
        os << std::endl << std::string("   ")+std::string(nMaxMetaColWidth,' ')+std::string("x=");
        for(int nColIdx=0; nColIdx<oMat.cols; ++nColIdx)
            os << sSpacer << lv::clampString(lv::putf(sMetaFormat.c_str(),nColIdx),nMaxColWidth);
        os << std::endl << std::endl;
        for(int nRowIdx=0; nRowIdx<oMat.rows; ++nRowIdx) {
            os << " y=" << lv::putf(sMetaFormat.c_str(),nRowIdx) << sSpacer;
            for(int nColIdx=0; nColIdx<oMat.cols; ++nColIdx)
                lPrinter(oMat.template at<T>(nRowIdx,nColIdx));
            os << std::endl;
        }
        os << std::endl;
    }

    /// removes all keypoints from voKPs which fall on null values (or outside the bounds) of oROI
    inline void validateKeyPoints(const cv::Mat& oROI, std::vector<cv::KeyPoint>& voKPs) {
        if(oROI.empty())
            return;
        lvAssert_(oROI.type()==CV_8UC1,"input ROI must be of type 8UC1");
        std::vector<cv::KeyPoint> voNewKPs;
        voNewKPs.reserve(voKPs.size());
        for(size_t k=0; k<voKPs.size(); ++k) {
            if(voKPs[k].pt.x>=0 && voKPs[k].pt.x<oROI.cols && voKPs[k].pt.y>=0 && voKPs[k].pt.y<oROI.rows && oROI.at<uchar>(voKPs[k].pt)>0)
                voNewKPs.push_back(voKPs[k]);
        }
        voKPs = voNewKPs;
    }

    /// returns the vector of all sorted unique values contained in a templated matrix
    template<typename T>
    inline std::vector<T> unique(const cv::Mat_<T>& oMat) {
        if(oMat.empty())
            return std::vector<T>();
        const std::set<T> mMap(oMat.begin(),oMat.end());
        return std::vector<T>(mMap.begin(),mMap.end());
    }

    /// returns whether the two matrices are equal or not
    template<typename T>
    inline bool isEqual(const cv::Mat& a, const cv::Mat& b) {
        if(a.empty() && b.empty())
            return true;
        if(a.dims!=b.dims || a.size!=b.size || a.type()!=b.type())
            return false;
        lvDbgAssert(a.total()*a.elemSize()==b.total()*b.elemSize());
        if(a.isContinuous() && b.isContinuous())
            return std::equal((T*)a.data,(T*)(a.data+a.total()*a.elemSize()),(T*)b.data);
        else {
            lvAssert_(a.dims==2,"undefined for non-continuous, multi-dim matrices");
            for(int nRowIdx=0; nRowIdx<a.rows; ++nRowIdx)
                for(int nColIdx=0; nColIdx<a.cols; ++nColIdx)
                    if(a.at<T>(nRowIdx,nColIdx)!=b.at<T>(nRowIdx,nColIdx))
                        return false;
            return true;
        }
    }

    /// returns whether the two matrices are nearly equal or not, given a maximum allowed error
    template<typename T, typename Teps>
    inline bool isNearlyEqual(const cv::Mat& a, const cv::Mat& b, Teps eps) {
        if(a.empty() && b.empty())
            return true;
        if(a.dims!=b.dims || a.size!=b.size || a.type()!=b.type())
            return false;
        lvDbgAssert(a.total()*a.elemSize()==b.total()*b.elemSize());
        const double dEps = double(eps);
        if(a.isContinuous() && b.isContinuous())
            return std::equal((T*)a.data,(T*)(a.data+a.total()*a.elemSize()),(T*)b.data,[&dEps](const T& _a, const T& _b){
                return std::abs(double(_a)-double(_b))<=dEps;
            });
        else {
            lvAssert_(a.dims==2,"undefined for non-continuous, multi-dim matrices");
            for(int nRowIdx=0; nRowIdx<a.rows; ++nRowIdx)
                for(int nColIdx=0; nColIdx<a.cols; ++nColIdx)
                    if(std::abs(double(a.at<T>(nRowIdx,nColIdx))-double(b.at<T>(nRowIdx,nColIdx)))>dEps)
                        return false;
            return true;
        }
    }

    /// converts a single HSL triplet (0-360 hue, 0-1 sat & lightness) into an 8-bit RGB triplet
    inline cv::Vec3b getBGRFromHSL(float fHue, float fSaturation, float fLightness) {
        // this function is not intended for fast conversions; use OpenCV's cvtColor for large-scale stuff
        lvDbgAssert__(fHue>=0.0f && fHue<360.0f,"bad input hue range (fHue=%f)",fHue);
        lvDbgAssert__(fSaturation>=0.0f && fSaturation<=1.0f,"bad input saturation range (fSaturation=%f)",fSaturation);
        lvDbgAssert__(fLightness>=0.0f && fLightness<=1.0f,"bad input lightness range (fLightness=%f)",fLightness);
        if(fSaturation==0.0f)
            return cv::Vec3b::all(cv::saturate_cast<uchar>(std::round(fLightness*255)));
        if(fLightness==0.0f)
            return cv::Vec3b::all(0);
        if(fLightness==1.0f)
            return cv::Vec3b::all(255);
        const auto lH2RGB = [&](float p, float q, float t) {
            if(t<0.0f)
                t += 1;
            if(t>1.0f)
                t -= 1;
            if(t<1.0f/6)
                return p + (q - p) * 6.0f * t;
            if(t<1.0f/2)
                return q;
            if(t<2.0f/3)
                return p + (q - p) * (2.0f/3 - t) * 6.0f;
            return p;
        };
        const float q = (fLightness<0.5f)?fLightness*(1+fSaturation):fLightness+fSaturation-fLightness*fSaturation;
        const float p = 2.0f*fLightness-q;
        const float h = fHue/360.0f;
        return cv::Vec3b(cv::saturate_cast<uchar>(std::round(lH2RGB(p,q,h-1.0f/3)*255)),cv::saturate_cast<uchar>(std::round(lH2RGB(p,q,h)*255)),cv::saturate_cast<uchar>(std::round(lH2RGB(p,q,h+1.0f/3)*255)));
    }

    /// converts a single HSL triplet (0-360 hue, 0-1 sat & lightness) into an 8-bit RGB triplet
    inline cv::Vec3b getBGRFromHSL(const cv::Vec3f& vHSL) {
        return getBGRFromHSL(vHSL[0],vHSL[1],vHSL[2]);
    }

    /// converts a single 8-bit RGB triplet into an HSL triplet (0-360 hue, 0-1 sat & lightness)
    inline cv::Vec3f getHSLFromBGR(const cv::Vec3b& vBGR) {
        // this function is not intended for fast conversions; use OpenCV's cvtColor for large-scale stuff
        const float r = vBGR[2]/255.0f, g=vBGR[1]/255.0f, b=vBGR[0]/255.0f;
        const float fMaxChroma = std::max(r,std::max(g,b));
        const float fMinChroma = std::min(r,std::min(g,b));
        const float fLightness = (fMaxChroma+fMinChroma)/2.0f;
        if(fMaxChroma==fMinChroma)
            return cv::Vec3f(0.0f,0.0f,fLightness);
        const float fDiffChroma = fMaxChroma-fMinChroma;
        const float fSaturation = std::max(0.0f,std::min(fDiffChroma/(1.0f-std::abs(2.0f*fLightness-1.0f)),1.0f));
        const float fHue = (fMaxChroma==r?(((g-b)/fDiffChroma)+(g<b?6.0f:0.0f)):(fMaxChroma==g?((b-r)/fDiffChroma+2.0f):(r-g)/fDiffChroma+4.0f))*60.0f;
        return cv::Vec3f(fHue,fSaturation,fLightness);
    }

    /// converts a 3-ch RGB image into a packed ushort YCbCr image where Cb and Cr are quantified to 4 bits each
    inline void cvtBGRToPackedYCbCr(const cv::Mat_<cv::Vec3b>& oInput, cv::Mat_<ushort>& oOutput) {
        lvAssert_(oInput.dims==2,"function only defined for 2-dims, 3-ch matrices");
        if(oInput.empty()) {
            oOutput.release();
            return;
        }
        cv::Mat_<cv::Vec3b> oInput_YCrCb;
        cv::cvtColor(oInput,oInput_YCrCb,cv::COLOR_BGR2YCrCb);
        std::vector<cv::Mat_<uchar>> voInputs(3);
        if(!oOutput.empty() && oOutput.allocator!=getMatAllocator16a())
            oOutput.release();
        oOutput.allocator = voInputs[0].allocator = voInputs[1].allocator = voInputs[2].allocator = getMatAllocator16a();
        cv::split(oInput_YCrCb,voInputs);
        lvDbgAssert(voInputs.size()==size_t(3) && voInputs[0].size==oInput.size && voInputs[1].size==oInput.size && voInputs[2].size==oInput.size);
        oOutput.create(oInput.dims,oInput.size.operator const int*());
        for(int nRowIdx=0; nRowIdx<oInput.rows; ++nRowIdx) {
            const uchar* pYRow = voInputs[0].ptr<uchar>(nRowIdx), *pCrRow = voInputs[1].ptr<uchar>(nRowIdx), *pCbRow = voInputs[2].ptr<uchar>(nRowIdx);
            ushort* pOutputRow = oOutput.ptr<ushort>(nRowIdx);
            lvDbgAssert(isAligned<16>(pYRow) && isAligned<16>(pCrRow) && isAligned<16>(pCbRow) && isAligned<16>(pOutputRow));
            int nColIdx = 0;
        #if HAVE_SSE2
            for(; nColIdx<=oInput.cols-16; nColIdx+=16) {
                lvDbgAssert(isAligned<16>(pYRow+nColIdx) && isAligned<16>(pCrRow+nColIdx) && isAligned<16>(pCbRow+nColIdx));
                lvDbgAssert(isAligned<16>(pOutputRow+nColIdx) && isAligned<16>(pOutputRow+nColIdx+8));
                const __m128i aYVals_8ui = _mm_load_si128((__m128i*)(pYRow+nColIdx));
                const __m128i aYVals_lo = lv::unpack_8ui_to_16ui<true>(aYVals_8ui);
                const __m128i aYVals_hi = lv::unpack_8ui_to_16ui<false>(aYVals_8ui);
                const __m128i aCrVals_8ui = _mm_load_si128((__m128i*)(pCrRow+nColIdx));
                const __m128i aCrVals_lo = _mm_slli_epi16(_mm_srli_epi16(lv::unpack_8ui_to_16ui<true>(aCrVals_8ui),4),8);
                const __m128i aCrVals_hi = _mm_slli_epi16(_mm_srli_epi16(lv::unpack_8ui_to_16ui<false>(aCrVals_8ui),4),8);
                const __m128i aCbVals_8ui = _mm_load_si128((__m128i*)(pCbRow+nColIdx));
                const __m128i aCbVals_lo = _mm_slli_epi16(_mm_srli_epi16(lv::unpack_8ui_to_16ui<true>(aCbVals_8ui),4),12);
                const __m128i aCbVals_hi = _mm_slli_epi16(_mm_srli_epi16(lv::unpack_8ui_to_16ui<false>(aCbVals_8ui),4),12);
                _mm_store_si128((__m128i*)(pOutputRow+nColIdx),_mm_or_si128(aYVals_lo,_mm_or_si128(aCrVals_lo,aCbVals_lo)));
                _mm_store_si128((__m128i*)(pOutputRow+nColIdx+8),_mm_or_si128(aYVals_hi,_mm_or_si128(aCrVals_hi,aCbVals_hi)));
            }
        #endif //HAVE_SSE2
            for(; nColIdx<oInput.cols; ++nColIdx)
                pOutputRow[nColIdx] = ushort(pYRow[nColIdx]+((pCrRow[nColIdx]>>4)<<8)+((pCbRow[nColIdx]>>4)<<12));
        }
    }

    /// converts a 3-ch RGB image into a packed ushort YCbCr image where Cb and Cr are quantified to 4 bits each (inline version)
    inline cv::Mat_<ushort> cvtBGRToPackedYCbCr(const cv::Mat_<cv::Vec3b>& oInput) {
        cv::Mat_<ushort> oOutput;
        cvtBGRToPackedYCbCr(oInput,oOutput);
        return oOutput;
    }

    /// converts a packed ushort YCbCr image where Cb and Cr are quantified to 4 bits each into a 3-ch RGB image
    inline void cvtPackedYCbCrToBGR(const cv::Mat_<ushort>& oInput, cv::Mat_<cv::Vec3b>& oOutput) {
        lvAssert_(oInput.dims==2,"function only defined for 2-dims matrices");
        if(oInput.empty()) {
            oOutput.release();
            return;
        }
        std::vector<cv::Mat_<uchar>> voOutputs(3);
        if(!oOutput.empty() && oOutput.allocator!=getMatAllocator16a())
            oOutput.release();
        oOutput.allocator = voOutputs[0].allocator = voOutputs[1].allocator = voOutputs[2].allocator = getMatAllocator16a();
        oOutput.create(oInput.size()); voOutputs[0].create(oInput.size()); voOutputs[1].create(oInput.size()); voOutputs[2].create(oInput.size());
        for(int nRowIdx=0; nRowIdx<oInput.rows; ++nRowIdx) {
            const ushort* pPackedRow = oInput.ptr<ushort>(nRowIdx);
            const bool bPackedRowAligned = isAligned<16>(pPackedRow);
            uchar* pYRow = voOutputs[0].ptr<uchar>(nRowIdx), *pCrRow = voOutputs[1].ptr<uchar>(nRowIdx), *pCbRow = voOutputs[2].ptr<uchar>(nRowIdx);
            lvDbgAssert(isAligned<16>(pYRow) && isAligned<16>(pCrRow) && isAligned<16>(pCbRow));
            int nColIdx = 0;
        #if HAVE_SSE2
            const __m128i aCrMask = _mm_set1_epi16(short(15));
            const __m128i aYMask = _mm_set1_epi16(short(255));
            for(; nColIdx<=oInput.cols-16; nColIdx+=16) {
                lvDbgAssert(!bPackedRowAligned || (isAligned<16>(pPackedRow+nColIdx) && isAligned<16>(pPackedRow+nColIdx+8)));
                const __m128i aPackedVals_lo = bPackedRowAligned?_mm_load_si128((__m128i*)(pPackedRow+nColIdx)):_mm_loadu_si128((__m128i*)(pPackedRow+nColIdx));
                const __m128i aPackedVals_hi = bPackedRowAligned?_mm_load_si128((__m128i*)(pPackedRow+nColIdx+8)):_mm_loadu_si128((__m128i*)(pPackedRow+nColIdx+8));
                const __m128i aYVals = _mm_packus_epi16(_mm_and_si128(aPackedVals_lo,aYMask),_mm_and_si128(aPackedVals_hi,aYMask));
                _mm_store_si128((__m128i*)(pYRow+nColIdx),aYVals);
                const __m128i aCrVals = _mm_packus_epi16(_mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(aPackedVals_lo,8),aCrMask),4),_mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(aPackedVals_hi,8),aCrMask),4));
                _mm_store_si128((__m128i*)(pCrRow+nColIdx),aCrVals);
                const __m128i aCbVals = _mm_packus_epi16(_mm_slli_epi16(_mm_srli_epi16(aPackedVals_lo,12),4),_mm_slli_epi16(_mm_srli_epi16(aPackedVals_hi,12),4));
                _mm_store_si128((__m128i*)(pCbRow+nColIdx),aCbVals);
            }
        #endif //HAVE_SSE2
            for(; nColIdx<oInput.cols; ++nColIdx) {
                pYRow[nColIdx] = uchar(pPackedRow[nColIdx]);
                pCrRow[nColIdx] = uchar(((pPackedRow[nColIdx]>>8)&15)<<4);
                pCbRow[nColIdx] = uchar((pPackedRow[nColIdx]>>12)<<4);
            }
        }
        cv::Mat_<cv::Vec3b> oOutput_YCrCb;
        cv::merge(voOutputs,oOutput_YCrCb);
        cv::cvtColor(oOutput_YCrCb,oOutput,cv::COLOR_YCrCb2BGR);
    }

    /// converts a packed ushort YCbCr image where Cb and Cr are quantified to 4 bits each into a 3-ch RGB image (inline version)
    inline cv::Mat_<cv::Vec3b> cvtPackedYCbCrToBGR(const cv::Mat_<ushort>& oInput) {
        cv::Mat_<cv::Vec3b> oOutput;
        cvtPackedYCbCrToBGR(oInput,oOutput);
        return oOutput;
    }

    /// returns a 8uc3 color map such that all equal values in the given matrix are assigned the same unique color in the map
    template<typename T>
    inline cv::Mat getUniqueColorMap(const cv::Mat_<T>& m, std::map<T,cv::Vec3b>* pmColorMap=nullptr) {
        static_assert(std::is_integral<T>::value,"function only defined for integer maps");
        lvAssert_(m.dims==2,"function currently only defined for 2d mats; split dims and call for 2d slices");
        if(m.empty())
            return cv::Mat();
        const std::vector<T> vUniques = lv::unique(m);
        const size_t nColors = vUniques.size();
        if(nColors<=1)
            return cv::Mat(m.size(),CV_8UC3,cv::Scalar::all(255));
        lvAssert_(nColors<720,"too many uniques for internal multi-slice HSL model");
        const size_t nMaxAng = 45;
        //const float fMinSat = 0.33f, fMaxSat = 1.0f;
        const float fAvgLight = 0.50f, fVarLight = 0.25f;
        const size_t nDivs = size_t(std::ceil(std::log2(nColors)));
        std::vector<cv::Vec3b> vColors(nColors);
        size_t nColorIdx = 0;
        for(size_t nDivIdx=0; nDivIdx<nDivs && nColorIdx<nColors; ++nDivIdx) {
            const size_t nSampleCount = std::min(std::max(nColors/(size_t(1)<<(nDivIdx+1)),(360/nMaxAng)-1),nColors-nColorIdx);
            const float fCurrSat = 1.0f; //const float fCurrSat = fMaxSat-((fMaxSat-fMinSat)/nDivs)*nDivIdx;
            const float fCurrLight = fAvgLight + int(nDivIdx>0)*(((nDivIdx%2)?-fVarLight:fVarLight)/((std::max(nDivIdx,size_t(1))+1)/2));
            std::unordered_set<ushort> mDivAngSet;
            ushort nCurrAng = ushort(rand())%nMaxAng;
            for(size_t nSampleIdx=0; nSampleIdx<nSampleCount; ++nSampleIdx) {
                lvDbgAssert(mDivAngSet.size()<360);
                while(mDivAngSet.count(nCurrAng))
                    ++nCurrAng %= 360;
                mDivAngSet.insert(nCurrAng);
                vColors[nColorIdx++] = lv::getBGRFromHSL(float(nCurrAng),fCurrSat,fCurrLight);
                nCurrAng = (nCurrAng+360/nSampleCount)%360;
            }
        }
        std::random_device oRandDev;
        std::default_random_engine oRandEng(oRandDev());
        std::shuffle(vColors.begin(),vColors.end(),oRandEng);
        std::map<T,cv::Vec3b> mColorMap;
        for(nColorIdx=0; nColorIdx<nColors; ++nColorIdx)
            mColorMap[vUniques[nColorIdx]] = vColors[nColorIdx];
        cv::Mat oOutputMap(m.size(),CV_8UC3);
        for(size_t nElemIdx=0; nElemIdx<m.total(); ++nElemIdx)
            oOutputMap.at<cv::Vec3b>((int)nElemIdx) = mColorMap[m((int)nElemIdx)];
        if(pmColorMap)
            std::swap(*pmColorMap,mColorMap);
        return oOutputMap;
    }

    /// helper struct for image display & callback management (must be created via DisplayHelper::create due to enable_shared_from_this interface)
    struct DisplayHelper : lv::enable_shared_from_this<DisplayHelper> {
        /// displayed window title (specified on creation)
        const std::string m_sDisplayName;
        /// displayed window maximum size (specified on creation)
        const cv::Size m_oMaxDisplaySize;
        /// general-use file storage tied to the display helper (will be closed & printed on helper destruction)
        cv::FileStorage m_oFS;
        /// public mutex that should be always used if m_oLatestMouseEvent is accessed externally
        std::mutex m_oEventMutex;
        /// raw-interpreted callback data structure
        struct CallbackData {
            cv::Point2i oPosition,oInternalPosition;
            cv::Size oTileSize,oDisplaySize;
            int nEvent,nFlags;
        } m_oLatestMouseEvent;
        /// by default, comes with a filestorage algorithms can use for debug
        static DisplayHelperPtr create(const std::string& sDisplayName,
                                       const std::string& sDebugFSDirPath="./",
                                       const cv::Size& oMaxSize=cv::Size(1920,1080),
                                       int nWindowFlags=cv::WINDOW_AUTOSIZE);
        /// will reformat the given image, print the index and mouse cursor point on it, and show it
        void display(const cv::Mat& oImage, size_t nIdx);
        /// will reformat the given images, print the index and mouse cursor point on them, and show them horizontally concatenated
        void display(const cv::Mat& oInputImg, const cv::Mat& oDebugImg, const cv::Mat& oOutputImg, size_t nIdx);
        /// will reformat the given images, print their names and mouse cursor point on them, and show them based on row-col ordering
        void display(const std::vector<std::vector<std::pair<cv::Mat,std::string>>>& vvImageNamePairs, const cv::Size& oSuggestedTileSize);
        /// sets the provided external function to be called when mouse events are captured for the displayed window
        void setMouseCallback(std::function<void(const CallbackData&)> lCallback);
        /// sets whether the waitKey call should block and wait for a key press or allow timeouts and return without one
        void setContinuousUpdates(bool b);
        /// calls cv::waitKey (blocking for a key press if m_bContinuousUpdates is false) and returns the cv::waitKey result
        int waitKey(int nDefaultSleepDelay=1);
        /// desctructor automatically closes its window
        ~DisplayHelper();
    protected:
        /// should always be constructor via static 'create' member due to enable_shared_from_this interface
        DisplayHelper(const std::string& sDisplayName, const std::string& sDebugFSDirPath, const cv::Size& oMaxSize, int nWindowFlags);
        /// local entrypoint for opencv mouse callbacks
        void onMouseEventCallback(int nEvent, int x, int y, int nFlags);
        /// global entrypoint for opencv mouse callbacks
        static void onMouseEvent(int nEvent, int x, int y, int nFlags, void* pData);
        cv::Size m_oLastDisplaySize,m_oLastTileSize;
        bool m_bContinuousUpdates,m_bFirstDisplay;
        cv::Mat m_oLastDisplay;
        std::function<void(int,int,int,int)> m_lInternalCallback;
        std::function<void(const CallbackData&)> m_lExternalCallback;
    private:
        DisplayHelper(const DisplayHelper&) = delete;
        DisplayHelper& operator=(const DisplayHelper&) = delete;
    };

    /// list of archive types supported by lv::write and lv::read
    enum MatArchiveList {
        MatArchive_FILESTORAGE,
        MatArchive_PLAINTEXT,
        MatArchive_BINARY
    };

    /// writes matrix data locally using a binary/yml/text file format
    void write(const std::string& sFilePath, const cv::Mat& _oData, MatArchiveList eArchiveType=MatArchive_BINARY);
    /// reads matrix data locally using a binary/yml/text file format
    void read(const std::string& sFilePath, cv::Mat& oData, MatArchiveList eArchiveType=MatArchive_BINARY);
    /// reads matrix data locally using a binary/yml/text file format (inline version)
    inline cv::Mat read(const std::string& sFilePath, MatArchiveList eArchiveType=MatArchive_BINARY) {
        cv::Mat oData;
        lv::read(sFilePath,oData,eArchiveType);
        return oData;
    }

    /// packs the data of several matrices into a bigger one (memalloc defrag helper)
    cv::Mat packData(const std::vector<cv::Mat>& vMats, std::vector<MatInfo>* pvOutputPackInfo=nullptr);
    /// unpacks the data of a matrix into several matrices (note: no allocation is done! lifetime of mat vec is tied to lifetime of input mat)
    std::vector<cv::Mat> unpackData(const cv::Mat& oPacket, const std::vector<MatInfo>& vPackInfo);

    /// shifts the values in a matrix by an (x,y) offset (see definition for full info)
    void shift(const cv::Mat& oInput, cv::Mat& oOutput, const cv::Point2f& vDelta, int nFillType=cv::BORDER_CONSTANT, const cv::Scalar& vConstantFillValue=cv::Scalar(0,0,0,0));

    /// returns an always-empty-mat by reference
    inline const cv::Mat& emptyMat() {
        static const cv::Mat s_oEmptyMat = cv::Mat();
        return s_oEmptyMat;
    }
    /// returns an always-empty-size by reference
    inline const cv::Size& emptySize() {
        static const cv::Size s_oEmptySize = cv::Size();
        return s_oEmptySize;
    }
    /// returns an always-empty-mat-size by reference
    inline const cv::MatSize& emptyMatSize() {
        static const std::array<int,3> anDims = {0,0,0};
        static const cv::MatSize s_oEmptyMatSize = cv::MatSize(const_cast<int*>(&anDims[1]));
        return s_oEmptyMatSize;
    }
    /// returns an always-empty-mat by reference
    inline const std::vector<cv::Mat>& emptyMatArray() {
        static const std::vector<cv::Mat> s_vEmptyMatArray = std::vector<cv::Mat>();
        return s_vEmptyMatArray;
    }
    /// returns an always-empty-size by reference
    inline const std::vector<cv::Size>& emptySizeArray() {
        static const std::vector<cv::Size> s_vEmptySizeArray = std::vector<cv::Size>();
        return s_vEmptySizeArray;
    }

    /// defines an aligned memory allocator to be used in matrices
    template<size_t nByteAlign, bool bAlignSingleElem=false>
    class AlignedMatAllocator : public cv::MatAllocator {
        static_assert(nByteAlign>0,"byte alignment must be a non-null value");
    public:
        typedef AlignedMatAllocator<nByteAlign,bAlignSingleElem> this_type;
        AlignedMatAllocator() noexcept {}
        AlignedMatAllocator(const AlignedMatAllocator<nByteAlign,bAlignSingleElem>&) noexcept {}
        template<typename T2>
        this_type& operator=(const AlignedMatAllocator<nByteAlign,bAlignSingleElem>&) noexcept {return *this;}
        virtual ~AlignedMatAllocator() noexcept {}
        virtual cv::UMatData* allocate(int dims, const int* sizes, int type, void* data, size_t* step, int /*flags*/, cv::UMatUsageFlags /*usageFlags*/) const override {
            step[dims-1] = bAlignSingleElem?cv::alignSize(CV_ELEM_SIZE(type),nByteAlign):CV_ELEM_SIZE(type);
            for(int d=dims-2; d>=0; --d)
                step[d] = cv::alignSize(step[d+1]*sizes[d+1],nByteAlign);
            const size_t nTotBytes = (size_t)cv::alignSize(step[0]*size_t(sizes[0]),(int)nByteAlign);
            cv::UMatData* u = new cv::UMatData(this);
            u->size = nTotBytes;
            if(data) {
                u->data = u->origdata = static_cast<uchar*>(data);
                u->flags |= cv::UMatData::USER_ALLOCATED;
            }
            else
                u->data = u->origdata = lv::AlignedMemAllocator<uchar,nByteAlign>::allocate(nTotBytes);
            return u;
        }
        virtual bool allocate(cv::UMatData* data, int /*accessFlags*/, cv::UMatUsageFlags /*usageFlags*/) const override {
            return (data!=nullptr);
        }
        virtual void deallocate(cv::UMatData* data) const override {
            if(data==nullptr)
                return;
            lvDbgAssert(data->urefcount>=0 && data->refcount>=0);
            if(data->refcount==0) {
                if(!(data->flags & cv::UMatData::USER_ALLOCATED)) {
                    lv::AlignedMemAllocator<uchar,nByteAlign>::deallocate(data->origdata,data->size);
                    data->origdata = nullptr;
                }
                delete data;
            }
        }
    };

    /// temp function; msvc seems to disable cuda output unless it is passed as argument to an external-lib function call...?
    void doNotOptimize(const cv::Mat& m);

} // namespace lv

namespace cv { // extending cv for fixes

#if USE_OPENCV_MAT_CONSTR_FIX
    template<typename _Tp>
    Mat_<_Tp>::Mat_(int _dims, const int* _sz, _Tp* _data, const size_t* _steps) :
            Mat(_dims, _sz, DataType<_Tp>::type, _data, _steps) {}
#endif //USE_OPENCV_MAT_CONSTR_FIX

} // namespace cv