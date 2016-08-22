
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

// dataset/application parameters below
#define LITIV2012_DATASET_PATH          "/shared2/datasets/litiv/litiv2012_dataset/"
#define LITIV2012_TEST_SEQUENCE_ID      9 // [1..9]
#include "LITIV2012Utils.h"             // automatically adds all other required LITIV dataset defines
#define RESULTS_PATH                    LITIV2012_DATASET_PATH "results/"
#define USE_FULL_DEBUG_DISPLAY          1
#define USE_THERMAL_TO_VISIBLE_PROJ     1
#define USE_FILESTORAGE_RES_OUTPUT      0
#define USE_VIDEOWRITER_RES_OUTPUT      0

#if (USE_VIDEOWRITER_RES_OUTPUT&&!USE_FULL_DEBUG_DISPLAY)
#error "cannot write video output without debug display on"
#endif //(USE_VIDEOWRITER_RES_OUTPUT&&!USE_FULL_DEBUG_DISPLAY)
#if (USE_VIDEOWRITER_RES_OUTPUT||USE_FILESTORAGE_RES_OUTPUT)
#define RES_OUTPUT_FILE_PREFIX "test_"
#endif //(USE_VIDEOWRITER_RES_OUTPUT||USE_FILESTORAGE_RES_OUTPUT)
#if (USE_VIDEOWRITER_RES_OUTPUT||USE_FILESTORAGE_RES_OUTPUT)
#define RES_OUTPUT_FILE_PREFIX_FULL RESULTS_PATH RES_OUTPUT_FILE_PREFIX "seq" __LITIV_STR(LITIV2012_TEST_SEQUENCE_ID)
#if USE_VIDEOWRITER_RES_OUTPUT
#define VIDEOWRITER_OUTPUT_FILE_PATH RES_OUTPUT_FILE_PREFIX_FULL ".avi"
#endif //USE_VIDEOWRITER_RES_OUTPUT
#if USE_FILESTORAGE_RES_OUTPUT
#define FILESTORAGE_OUTPUT_FILE_PATH RES_OUTPUT_FILE_PREFIX_FULL ".yml"
#endif //USE_FILESTORAGE_RES_OUTPUT
#endif //(USE_VIDEOWRITER_RES_OUTPUT||USE_FILESTORAGE_RES_OUTPUT)

#include "MultimodalVideoRegistrAlg.h"

int main(int /*argc*/, char **/*argv*/) {
    try {

        MultimodalVideoRegistrAlg oAlg;

        cv::Mat oGTTransMat_THERMAL,oGTTransMat_VISIBLE;
        cv::Mat oPolyListMat_THERMAL,oPolyListMat_VISIBLE;
        lv::LITIV2012::ReadTestSeqGroundtruth(oGTTransMat_THERMAL,oGTTransMat_VISIBLE,oPolyListMat_THERMAL,oPolyListMat_VISIBLE);
        cv::Mat oPolyPts_THERMAL,oPolyPts_VISIBLE;
        lv::LITIV2012::ConvertPolyPtsMatsToPtsLists(oPolyListMat_THERMAL,oPolyListMat_VISIBLE,oPolyPts_THERMAL,oPolyPts_VISIBLE);
        cv::VideoCapture oCapOrig_THERMAL,oCapBGS_THERMAL,oCapOrig_VISIBLE,oCapBGS_VISIBLE;
        const int nFrameCount = lv::LITIV2012::OpenTestSeqVideos(oCapOrig_THERMAL,oCapBGS_THERMAL,oCapOrig_VISIBLE,oCapBGS_VISIBLE);
        std::cout << "(" << nFrameCount << " frames total)" << std::endl;
        cv::Mat oTempImg_THERMAL, oTempImg_VISIBLE;
        oCapOrig_THERMAL >> oTempImg_THERMAL; oCapOrig_THERMAL.set(cv::CAP_PROP_FRAME_COUNT,0);
        oCapOrig_VISIBLE >> oTempImg_VISIBLE; oCapOrig_VISIBLE.set(cv::CAP_PROP_FRAME_COUNT,0);
        const cv::Size oInputSize_THERMAL = oTempImg_THERMAL.size();
        const cv::Size oInputSize_VISIBLE = oTempImg_VISIBLE.size();
        cv::Mat oPolyMat_THERMAL(oInputSize_THERMAL,CV_8UC1,cv::Scalar_<uchar>::all(0)),oPolyMat_VISIBLE(oInputSize_VISIBLE,CV_8UC1,cv::Scalar_<uchar>::all(0));
        lv::LITIV2012::DrawPolyPtsMatsToMat(oPolyListMat_THERMAL,oPolyListMat_VISIBLE,oPolyMat_THERMAL,oPolyMat_VISIBLE);

        cv::Mat oSource_THERMAL,oSource_VISIBLE;
        cv::Mat oForeground_THERMAL,oForeground_VISIBLE;
        cv::Mat oContours_THERMAL,oContours_VISIBLE;
#if USE_THERMAL_TO_VISIBLE_PROJ
#if USE_FULL_DEBUG_DISPLAY
        const cv::Mat& oSource_ToTransform = oSource_THERMAL;
        //const cv::Mat& oSource = oSource_VISIBLE;
        cv::Mat& oContours_ToTransform = oContours_THERMAL;
        cv::Mat& oContours = oContours_VISIBLE;
#endif //USE_FULL_DEBUG_DISPLAY
        const cv::Size& oTransformedImageSize = oInputSize_VISIBLE;
        const cv::Mat& oPolyMat_ToTransform = oPolyMat_THERMAL;
        const cv::Mat& oPolyMat = oPolyMat_VISIBLE;
        const cv::Mat& oPolyPts_ToTransform = oPolyPts_THERMAL;
        const cv::Mat& oPolyPts = oPolyPts_VISIBLE;
        const cv::Mat& oGTTransMat = oGTTransMat_THERMAL;
        const cv::Mat& oGTTransMat_inv = oGTTransMat_VISIBLE;
#else //(!USE_THERMAL_TO_VISIBLE_PROJ)
#if USE_FULL_DEBUG_DISPLAY
        const cv::Mat& oSource_ToTransform = oSource_VISIBLE;
        //const cv::Mat& oSource = oSource_THERMAL;
        cv::Mat& oContours_ToTransform = oContours_VISIBLE;
        cv::Mat& oContours = oContours_THERMAL;
#endif //USE_FULL_DEBUG_DISPLAY
        const cv::Size& oTransformedImageSize = oInputSize_THERMAL;
        const cv::Mat& oPolyMat_ToTransform = oPolyMat_VISIBLE;
        const cv::Mat& oPolyMat = oPolyMat_THERMAL;
        const cv::Mat& oPolyPts_ToTransform = oPolyPts_VISIBLE;
        const cv::Mat& oPolyPts = oPolyPts_THERMAL;
        const cv::Mat& oGTTransMat = oGTTransMat_VISIBLE;
        const cv::Mat& oGTTransMat_inv = oGTTransMat_THERMAL;
#endif //(!USE_THERMAL_TO_VISIBLE_PROJ)

        cv::Mat oTransformedSource(oTransformedImageSize,CV_8UC3,cv::Scalar_<uchar>::all(0)); // debug
        cv::Mat oGTTransformedSource(oTransformedImageSize,CV_8UC3,cv::Scalar_<uchar>::all(0)); // debug
        cv::Mat oTransformedContours(oTransformedImageSize,CV_8UC3,cv::Scalar_<uchar>::all(0)); // debug
        cv::Mat oGTTransformedContours(oTransformedImageSize,CV_8UC3,cv::Scalar_<uchar>::all(0)); // debug
        cv::Mat oTransformedPolyMat(oTransformedImageSize,CV_8UC1,cv::Scalar_<uchar>::all(0)); // eval
        cv::Mat oGTTransformedPolyMat(oTransformedImageSize,CV_8UC1,cv::Scalar_<uchar>::all(0)); // eval

        cv::warpPerspective(oPolyMat_ToTransform,oGTTransformedPolyMat,oGTTransMat,oTransformedImageSize,cv::INTER_NEAREST|cv::WARP_INVERSE_MAP,cv::BORDER_CONSTANT);
        const float fGTPolyOverlapError = lv::CalcForegroundOverlapError(oPolyMat,oGTTransformedPolyMat);
        const cv::Mat oGTTransformedPolyPts = oGTTransMat_inv*oPolyPts_ToTransform;
        const cv::Point2d oGTPolyRegError = lv::CalcPolyRegError(oPolyPts,oGTTransformedPolyPts);

#if USE_VIDEOWRITER_RES_OUTPUT
        cv::VideoWriter oResultWriter(VIDEOWRITER_OUTPUT_FILE_PATH,cv::VideoWriter::fourcc('M','J','P','G'),15,cv::Size(oTransformedImageSize.width*3,oTransformedImageSize.height*2));
#endif //USE_VIDEOWRITER_RES_OUTPUT
#if USE_FILESTORAGE_RES_OUTPUT
        cv::FileStorage oResultFS(FILESTORAGE_OUTPUT_FILE_PATH,cv::FileStorage::WRITE);
        oResultFS << "config" << "{";
        oResultFS << "USE_THERMAL_TO_VISIBLE_PROJ" << USE_THERMAL_TO_VISIBLE_PROJ;
        oResultFS << "}";
        oResultFS << "results" << "[";
#endif //USE_FILESTORAGE_RES_OUTPUT

#if USE_FULL_DEBUG_DISPLAY
        bool bContinuousUpdates = false;
#endif //USE_FULL_DEBUG_DISPLAY
        int nFirstIndex = nFrameCount;
        cv::Point2d oCumulativePolyRegErrors(0,0);
        float fCumulativePolyOverlapErrors = 0.0f;
        float fCumulativeForegroundBlobOverlapErrors = 0.0f;
        for(int nCurrFrameIndex=0; nCurrFrameIndex<nFrameCount; ++nCurrFrameIndex) {
            if((nCurrFrameIndex%50)==0)
                std::cout << "# " << nCurrFrameIndex << std::endl;
            oCapOrig_THERMAL >> oSource_THERMAL;
            oCapOrig_VISIBLE >> oSource_VISIBLE;
            if(oSource_THERMAL.empty() || oSource_VISIBLE.empty())
                break;
            cv::Mat oForegroundBGR_THERMAL,oForegroundBGR_VISIBLE;
            oCapBGS_THERMAL >> oForegroundBGR_THERMAL;
            oCapBGS_VISIBLE >> oForegroundBGR_VISIBLE;
            if(oForegroundBGR_THERMAL.empty() || oForegroundBGR_VISIBLE.empty())
                break;
            cv::cvtColor(oForegroundBGR_THERMAL,oForeground_THERMAL,cv::COLOR_BGR2GRAY);
            cv::cvtColor(oForegroundBGR_VISIBLE,oForeground_VISIBLE,cv::COLOR_BGR2GRAY);
            oAlg.ProcessForeground(oForeground_THERMAL,oForeground_VISIBLE);
            const cv::Mat& oTransMat = oAlg.GetTransformationMatrix(false);
            if(!oTransMat.empty()) {
                if(nFirstIndex==nFrameCount)
                    nFirstIndex = nCurrFrameIndex;
                const cv::Mat& oTransMat_inv = oAlg.GetTransformationMatrix(true);
#if USE_FULL_DEBUG_DISPLAY
                oContours.create(oTransformedImageSize,CV_8UC3);
                oContours_ToTransform.create(oTransformedImageSize,CV_8UC3);
                MultimodalVideoRegistrAlg::PaintFGRegions(oAlg.GetLatestContours(false),cv::Scalar(0,0,255),cv::Scalar(255,0,0),oContours);
                MultimodalVideoRegistrAlg::PaintFGRegions(oAlg.GetLatestContours(true),cv::Scalar(0,255,0),cv::Scalar(0,0,255),oContours_ToTransform);
                cv::warpPerspective(oSource_ToTransform,oTransformedSource,oTransMat,oTransformedImageSize,cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_CONSTANT);
                cv::warpPerspective(oContours_ToTransform,oTransformedContours,oTransMat,oTransformedImageSize,cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_CONSTANT);
#endif //USE_FULL_DEBUG_DISPLAY
                cv::warpPerspective(oPolyMat_ToTransform,oTransformedPolyMat,oTransMat,oTransformedImageSize,cv::INTER_NEAREST|cv::WARP_INVERSE_MAP,cv::BORDER_CONSTANT);
                const float fPolyOverlapError = lv::CalcForegroundOverlapError(oPolyMat,oTransformedPolyMat);
                fCumulativePolyOverlapErrors += fPolyOverlapError;
                const cv::Mat oTransformedPolyPts = oTransMat_inv*oPolyPts_ToTransform;
                const cv::Point2d oPolyRegError = lv::CalcPolyRegError(oPolyPts,oTransformedPolyPts);
                oCumulativePolyRegErrors += oPolyRegError;
#if USE_FILESTORAGE_RES_OUTPUT
                oResultFS << "{";
                oResultFS << "nCurrFrameIndex" << nCurrFrameIndex;
                oResultFS << "oCurrPolyRegError" << oPolyRegError;
                oResultFS << "fCurrPolyOverlapError" << fPolyOverlapError;
                oResultFS << "}";
#endif //USE_FILESTORAGE_RES_OUTPUT
            }
#if USE_FULL_DEBUG_DISPLAY
            cv::warpPerspective(oSource_ToTransform,oGTTransformedSource,oGTTransMat,oTransformedImageSize,cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_CONSTANT);
            cv::warpPerspective(oContours_ToTransform,oGTTransformedContours,oGTTransMat,oTransformedImageSize,cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_CONSTANT);
            cv::Mat oTransformedSourceOverlay = (USE_THERMAL_TO_VISIBLE_PROJ?oSource_VISIBLE:oSource_THERMAL)+oTransformedSource;
            cv::putText(oTransformedSourceOverlay,"Estimated Transformation",cv::Point(20,20),cv::FONT_HERSHEY_PLAIN,0.8,cv::Scalar_<uchar>::all(255),1);
            cv::Mat oGTTransformedSourceOverlay = (USE_THERMAL_TO_VISIBLE_PROJ?oSource_VISIBLE:oSource_THERMAL)+oGTTransformedSource;
            cv::putText(oGTTransformedSourceOverlay,"GT Transformation",cv::Point(20,20),cv::FONT_HERSHEY_PLAIN,0.8,cv::Scalar_<uchar>::all(255),1);
            cv::Mat oTransformedContoursOverlay = (USE_THERMAL_TO_VISIBLE_PROJ?oContours_VISIBLE:oContours_THERMAL)+oTransformedContours;
            cv::putText(oTransformedContoursOverlay,"Estimated Transformation",cv::Point(20,20),cv::FONT_HERSHEY_PLAIN,0.8,cv::Scalar_<uchar>::all(255),1);
            cv::Mat oGTTransformedContoursOverlay = (USE_THERMAL_TO_VISIBLE_PROJ?oContours_VISIBLE:oContours_THERMAL)+oGTTransformedContours;
            cv::putText(oGTTransformedContoursOverlay,"GT Transformation",cv::Point(20,20),cv::FONT_HERSHEY_PLAIN,0.8,cv::Scalar_<uchar>::all(255),1);
            cv::Mat oTransformedPolyMatOverlay; cv::cvtColor(oPolyMat/2+oTransformedPolyMat/2,oTransformedPolyMatOverlay,CV_GRAY2BGR);
            cv::putText(oTransformedPolyMatOverlay,"Estimated Transformation",cv::Point(20,20),cv::FONT_HERSHEY_PLAIN,0.8,cv::Scalar_<uchar>::all(255),1);
            cv::Mat oGTTransformedPolyMatOverlay; cv::cvtColor(oPolyMat/2+oGTTransformedPolyMat/2,oGTTransformedPolyMatOverlay,CV_GRAY2BGR);
            cv::putText(oGTTransformedPolyMatOverlay,"GT Transformation",cv::Point(20,20),cv::FONT_HERSHEY_PLAIN,0.8,cv::Scalar_<uchar>::all(255),1);
            cv::Mat oTransformedOverlayRow;
            cv::hconcat(oTransformedPolyMatOverlay,oTransformedContoursOverlay,oTransformedOverlayRow);
            cv::hconcat(oTransformedSourceOverlay,oTransformedOverlayRow,oTransformedOverlayRow);
            cv::Mat oGTTransformedOverlayRow;
            cv::hconcat(oGTTransformedPolyMatOverlay,oGTTransformedContoursOverlay,oGTTransformedOverlayRow);
            cv::hconcat(oGTTransformedSourceOverlay,oGTTransformedOverlayRow,oGTTransformedOverlayRow);
            cv::Mat oFullOverlay;
            cv::vconcat(oTransformedOverlayRow,oGTTransformedOverlayRow,oFullOverlay);
            cv::imshow("oFullOverlay",oFullOverlay);
            int nKeyPressed;
            if(bContinuousUpdates)
                nKeyPressed = cv::waitKey(1);
            else
                nKeyPressed = cv::waitKey(0);
            if(nKeyPressed!=-1) {
                nKeyPressed %= (UCHAR_MAX+1); // fixes return val bug in some opencv versions
                std::cout << "nKeyPressed = " << nKeyPressed%(UCHAR_MAX+1) << std::endl;
            }
            if(nKeyPressed==' ')
                bContinuousUpdates = !bContinuousUpdates;
            else if(nKeyPressed==(int)'q')
                break;
#if USE_VIDEOWRITER_RES_OUTPUT
            oResultWriter.write(oFullOverlay);
#endif //USE_VIDEOWRITER_RES_OUTPUT
#endif //USE_FULL_DEBUG_DISPLAY
        }
        const cv::Point2d oAveragePolyRegError = oCumulativePolyRegErrors/(nFrameCount-nFirstIndex);
        const float fAveragePolyOverlapError = fCumulativePolyOverlapErrors/(nFrameCount-nFirstIndex);
        const float fAverageForegroundBlobOverlapError = fCumulativeForegroundBlobOverlapErrors/(nFrameCount-nFirstIndex);
        std::cout << std::endl << "-----------------" << std::endl;
        std::cout << "TEST SEQ #" << LITIV2012_TEST_SEQUENCE_ID << " RESULTS:" << std::endl;
        std::cout << "  oAveragePolyRegError=" << oAveragePolyRegError << "    (GT=" << oGTPolyRegError << ")" << std::endl;
        std::cout << "  fAveragePolyOverlapError=" << fAveragePolyOverlapError << "    (GT=" << fGTPolyOverlapError << ")" << std::endl;
        std::cout << "  fAverageForegroundBlobOverlapError=" << fAverageForegroundBlobOverlapError << std::endl;
#if USE_FILESTORAGE_RES_OUTPUT
        oResultFS << "]";
        oResultFS << "nFirstIndex" << nFirstIndex;
        oResultFS << "nFrameCount" << nFrameCount;
        oResultFS << "oAveragePolyRegError" << oAveragePolyRegError;
        oResultFS << "oGTPolyRegError" << oGTPolyRegError;
        oResultFS << "fAveragePolyOverlapError" << fAveragePolyOverlapError;
        oResultFS << "fGTPolyOverlapError" << fGTPolyOverlapError;
        oResultFS << "fAverageForegroundBlobOverlapError" << fAverageForegroundBlobOverlapError;
#endif //USE_FILESTORAGE_RES_OUTPUT
    }
    catch(const cv::Exception& err) {
        printf("cv::Exception: %s\n",err.what());
        return -1;
    }
    catch(const std::exception& err) {
        printf("std::exception: %s\n",err.what());
        return -1;
    }
    catch(...) {
        printf("unhandled exception.\n");
        return -1;
    }
    return 0;
}
