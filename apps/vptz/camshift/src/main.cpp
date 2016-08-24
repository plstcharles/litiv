
//======================================================================================
//
// This program presents the baseline tracking algorithm along with the vptz evaluator.
// The tracker is based on OpenCV's CamShift.
//
// Before running, set the input scenario/gt paths and delays using the defines below.
//
//======================================================================================

#include "litiv/vptz/virtualptz.hpp"

////////////////////////////////////////
#define USE_VPTZ_TRACKING          1       // allows testing without vptz framework (i.e. on regular sequences)
#if USE_VPTZ_TRACKING //////////////////
#define VPTZ_USE_ALL_TEST_SETS     1       // used for batch-testing on vptz framework (will fetch all test sets automatically)
#define VPTZ_USE_SINGLE_TEST_SET   0       // used to run a single test set on vptz framework (must be specified via define below)
#define VPTZ_USE_WAITSLEEP         0       // = 'simulate' full delays by 'sleeping' between frames
#define VPTZ_EXEC_DELAY_RATIO      1.0     // = normal processing delay penalty (100% time lost considered)
#define VPTZ_COMMUNICATION_DELAY   0.125   // = 125ms network ping delay between server and client
#endif //USE_VPTZ_TRACKING /////////////
#define CAMSHIFT_USE_PURE_HSV      1
#define CAMSHIFT_USE_MIXED_HSV     0
#define CAMSHIFT_USE_POSE_PREDICT  0
#define CAMSHIFT_USE_70p100_INIT   1
#define CAMSHIFT_LIMIT_SCALE_VAR   1
#define CAMSHIFT_DISPLAY_BP_IMAGE  0
#define CAMSHIFT_DISPLAY_FG_MASK   0
#define CAMSHIFT_DISPLAY_TARGET    0
////////////////////////////////////////////////////////////////////////////////////////////////////////
#if USE_VPTZ_TRACKING //////////////////////////////////////////////////////////////////////////////////
#define VPTZ_DATASET_ROOT_DIR_PATH       std::string("/some/root/directory/litiv_vptz_icip2015/")
#if VPTZ_USE_ALL_TEST_SETS /////////////////////////////////////////////////////////////////////////////
#define INPUT_TEST_SETS_PATH_PREFIX      std::string("testsets/")
#define OUTPUT_EVAL_FILES_PATH_PREFIX    std::string("results_camshift/")
#define INTPUT_TEST_SETS_NAMES           {"all","articulated_objects","cluttered_background", \
                                          "distractors","fast_motion","illumination_variation", \
                                          "low_resolution","occlusion"}
#elif VPTZ_USE_SINGLE_TEST_SET /////////////////////////////////////////////////////////////////////////
#define INPUT_TEST_SET_PATH              std::string("testsets/articulated_objects.yml")
#define OUTPUT_EVAL_FILE_PATH            std::string("results_camshift/articulated_objects.yml")
#else //(!VPTZ_USE_SINGLE_TEST_SET) ////////////////////////////////////////////////////////////////////
#define INPUT_SCENARIO_PATH              std::string("scenario5/frames/scenario5_%06d.jpg")
#define INPUT_GT_SEQUENCE_PATH           std::string("scenario5/gt/scenario5_torso_green01.yml")
#define INPUT_TARGET_MASK_PATH           std::string("target_masks/scenario5_torso_green01.png")
#define OUTPUT_EVAL_FILE_PATH            std::string("results_camshift/scenario5_torso_green01.yml")
#endif //VPTZ_USE_SINGLE_TEST_SET //////////////////////////////////////////////////////////////////////
#else //(!USE_VPTZ_TRACKING) ///////////////////////////////////////////////////////////////////////////
#define INPUT_VIDEO_PATH                 std::string("/some/root/directory/tracking/book/book.avi")
#define INPUT_TARGET_IMG_PATH            std::string("/some/root/directory/tracking/book/target.png")
#define INPUT_METADATA_FILE_PATH         std::string("/some/root/directory/tracking/book/targetLoc.yml")
#endif //(!USE_VPTZ_TRACKING) //////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// CAMSHIFT PARAMETERS ///////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
#if CAMSHIFT_USE_PURE_HSV
#if CAMSHIFT_USE_MIXED_HSV
const int nHistDims = 1;
const int anHistChannels[] = {0};
const int anHistBins[] = {16};
const float afHistRange_H[] = {0.0f,180.0f};
const float* aafHistRanges[] = {afHistRange_H};
const int anChannelPairMix[] = {0,0};
const int vmin = 10;
const int vmax = 255;
const int smin = 30;
const int pmin = 0;
#else //(!CAMSHIFT_USE_MIXED_HSV)
const int nHistDims = 2;
const int anHistChannels[] = {0,1};
const int anHistBins[] = {16,16};
const float afHistRange_H[] = {0.0f,180.0f};
const float afHistRange_S[] = {0.0f,256.0f};
const float* aafHistRanges[] = {afHistRange_H,afHistRange_S};
const int vmin = 0;
const int vmax = 255;
const int smin = 25;
const int pmin = 35;
#endif //(!CAMSHIFT_USE_MIXED_HSV)
#else //(!CAMSHIFT_USE_PURE_HSV)
const int nHistDims = 3;
const int anHistChannels[] = {0,1,2};
const int anHistBins[] = {256,256,256};
const float afHistRange_B[] = {0.0f,256.0f};
const float afHistRange_G[] = {0.0f,256.0f};
const float afHistRange_R[] = {0.0f,256.0f};
const float* aafHistRanges[] = {afHistRange_B,afHistRange_G,afHistRange_R};
#endif //CAMSHIFT_USE_PURE_HSV
#if CAMSHIFT_USE_POSE_PREDICT && USE_VPTZ_TRACKING
const float fPosPredictScaleFactor = 0.5f;
#endif // (CAMSHIFT_USE_POSE_PREDICT && USE_VPTZ_TRACKING)
#if CAMSHIFT_LIMIT_SCALE_VAR
const float fTargetBBoxMaxScaleVarFactor = 3.0f;
#endif //CAMSHIFT_LIMIT_SCALE_VAR
const float fMeanShiftMinEpsilon = 1.0;
const int nMeanShiftMaxIterCount = 15;
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
#if !USE_VPTZ_TRACKING
class TargetBox {
public:
    cv::Point top_left, bottom_right;
    TargetBox(cv::Point tl=cv::Point(0,0), cv::Point br=cv::Point(50,100));
    cv::Point center() { return 0.5*(top_left+bottom_right); };
    void changeCenter(cv::Point);
    bool adjust;
};
cv::Mat g_oTargetPickingMat;
TargetBox g_oTargetBox;
void displayRect(cv::Mat img);
void onMouse( int event, int x, int y, int, void* );
#endif //(!USE_VPTZ_TRACKING)
#if (VPTZ_USE_ALL_TEST_SETS+VPTZ_USE_SINGLE_TEST_SET)>1
#error "config error, must specify all test sets or a single test set"
#endif //(VPTZ_USE_ALL_TEST_SETS+VPTZ_USE_SINGLE_TEST_SET)>0
int main(int /*argc*/, char** /*argv*/) {
#if VPTZ_USE_ALL_TEST_SETS
    const char* asTestSets[] = INTPUT_TEST_SETS_NAMES;
    for(size_t nTestSetIdx=0; nTestSetIdx<sizeof(asTestSets)/sizeof(char*); ++nTestSetIdx) {
        const std::string sCurrTestSetPath = VPTZ_DATASET_ROOT_DIR_PATH+INPUT_TEST_SETS_PATH_PREFIX+asTestSets[nTestSetIdx]+".yml";
        const std::string sCurrResultFilePath = VPTZ_DATASET_ROOT_DIR_PATH+OUTPUT_EVAL_FILES_PATH_PREFIX+asTestSets[nTestSetIdx]+".yml";
        std::cout << "\n\n===============================\n\n  Setting up testset#" << nTestSetIdx+1 << " [" << asTestSets[nTestSetIdx] << "]\n\n===============================\n" << std::endl;
#else //(!VPTZ_USE_ALL_TEST_SETS)
    {
#endif //(!VPTZ_USE_ALL_TEST_SETS)
        try {
            cv::Mat oCurrImg;
            cv::Mat oTargetImg,oTargetMask,oTargetImg_HIST;
            int nTotPotentialFrameCount = 0;
            int nTotProcessedFrameCount = 0;
#if USE_VPTZ_TRACKING
#if VPTZ_USE_ALL_TEST_SETS
            vptz::Evaluator oTestEval( sCurrTestSetPath,sCurrResultFilePath,
                                       VPTZ_COMMUNICATION_DELAY,VPTZ_EXEC_DELAY_RATIO);
#elif VPTZ_USE_SINGLE_TEST_SET
            vptz::Evaluator oTestEval( VPTZ_DATASET_ROOT_DIR_PATH+INPUT_TEST_SET_PATH,
                                       VPTZ_DATASET_ROOT_DIR_PATH+OUTPUT_EVAL_FILE_PATH,
                                       VPTZ_COMMUNICATION_DELAY,VPTZ_EXEC_DELAY_RATIO);
#else //(!VPTZ_USE_SINGLE_TEST_SET)
            vptz::Evaluator oTestEval( VPTZ_DATASET_ROOT_DIR_PATH+INPUT_SCENARIO_PATH,
                                       VPTZ_DATASET_ROOT_DIR_PATH+INPUT_GT_SEQUENCE_PATH,
                                       VPTZ_DATASET_ROOT_DIR_PATH+INPUT_TARGET_MASK_PATH,
                                       VPTZ_DATASET_ROOT_DIR_PATH+OUTPUT_EVAL_FILE_PATH,
                                       VPTZ_COMMUNICATION_DELAY,VPTZ_EXEC_DELAY_RATIO);
#endif //(!VPTZ_USE_SINGLE_TEST_SET)
            for(int nTestIdx=0; nTestIdx<oTestEval.GetTestSetSize(); ++nTestIdx) {
                if(nTestIdx>0)
                    cv::destroyAllWindows();
                std::cout << "\nSetting up seq#" << nTestIdx+1 << "..." << std::endl;
                oTestEval.SetupTesting(nTestIdx);
                std::cout << "Processing seq#" << nTestIdx+1 << " [" << oTestEval.GetCurrTestSequenceName() << "]..." << std::endl;
                oCurrImg = oTestEval.GetInitTargetFrame();
                oTargetImg = oTestEval.GetInitTarget();
                oTargetMask = oTestEval.GetInitTargetMask();
                nTotPotentialFrameCount += oTestEval.GetPotentialTestFrameCount();
#else //(!USE_VPTZ_TRACKING)
            {
                cv::VideoCapture oCap(INPUT_VIDEO_PATH);
                if(!oCap.isOpened()) {
                    std::cout << std::endl << "Could not open video at '" << INPUT_VIDEO_PATH << "'" << std::endl;
                    return -1;
                }
                cv::Rect oTargetBBox;
                int nTargetFrameIdx;
                oTargetImg = cv::imread(INPUT_TARGET_IMG_PATH);
                cv::FileStorage oFSTargetLoc(INPUT_METADATA_FILE_PATH,cv::FileStorage::READ);
                oFSTargetLoc["loc"] >> oTargetBBox;
                oFSTargetLoc["idx"] >> nTargetFrameIdx;
                if(oTargetBBox.height==0 || oTargetBBox.width==0 || oTargetImg.empty()) {
                    oCap >> g_oTargetPickingMat;
                    if(g_oTargetPickingMat.empty())
                        return -1;
                    g_oTargetBox.changeCenter(cv::Point(g_oTargetPickingMat.cols/2,g_oTargetPickingMat.rows/2));
                    cv::namedWindow("target selection");
                    cv::setMouseCallback("target selection",onMouse);
                    displayRect(g_oTargetPickingMat);
                    while(true) {
                        if(tolower(cv::waitKey(0))==' ')
                            break;
                        oCap >> g_oTargetPickingMat;
                        if(g_oTargetPickingMat.empty())
                            return -1;
                        displayRect(g_oTargetPickingMat);
                    }
                    cv::setMouseCallback("target selection",NULL);
                    cv::destroyWindow("target selection");
                    oTargetImg = g_oTargetPickingMat(cv::Rect(g_oTargetBox.top_left,g_oTargetBox.bottom_right)).clone();
                    if(oTargetImg.empty())
                        return -1;
                }
                oCap.set(CV_CAP_PROP_POS_FRAMES,nTargetFrameIdx);
                oCap >> oCurrImg;
                if(oCurrImg.empty())
                    return -1;
                nTotPotentialFrameCount += (int)oCap.get(CV_CAP_PROP_FRAME_COUNT);
#endif //(!USE_VPTZ_TRACKING)
                lvAssert(oTargetImg.type()==CV_8UC4);
                const int nImageWidth = oCurrImg.cols;
                const int nImageHeight = oCurrImg.rows;
#if CAMSHIFT_USE_POSE_PREDICT
                const cv::Point oCenterPos(nImageWidth/2,nImageHeight/2);
#endif //CAMSHIFT_USE_POSE_PREDICT
#if CAMSHIFT_LIMIT_SCALE_VAR
                const int nMaxTargetBBoxWidth = std::min(int(oTargetImg.cols*fTargetBBoxMaxScaleVarFactor),nImageWidth);
                const int nMinTargetBBoxWidth = std::max(int(oTargetImg.cols/fTargetBBoxMaxScaleVarFactor),1);
                const int nMaxTargetBBoxHeight = std::min(int(oTargetImg.rows*fTargetBBoxMaxScaleVarFactor),nImageHeight);
                const int nMinTargetBBoxHeight = std::max(int(oTargetImg.rows/fTargetBBoxMaxScaleVarFactor),1);
#endif //CAMSHIFT_LIMIT_SCALE_VAR
#if CAMSHIFT_USE_70p100_INIT
                const int nInitTargetWidth_30p100 = std::max(int(oTargetImg.cols*0.3),1);
                const int nInitTargetHeight_30p100 = std::max(int(oTargetImg.rows*0.3),1);
                const cv::Rect oInitTargetBBox_70p100( nInitTargetWidth_30p100/2,nInitTargetHeight_30p100/2,
                                                       oTargetImg.cols-nInitTargetWidth_30p100,
                                                       oTargetImg.rows-nInitTargetHeight_30p100);
                oTargetImg = oTargetImg(oInitTargetBBox_70p100);
                if(!oTargetMask.empty())
                    oTargetMask = oTargetMask(oInitTargetBBox_70p100);
#endif //CAMSHIFT_USE_70p100_INIT
#if CAMSHIFT_USE_PURE_HSV
                cv::Mat oTargetImg_HSV,oTargetImg_HSV_mask;
                cv::cvtColor(oTargetImg,oTargetImg_HSV,cv::COLOR_BGR2HSV);
                cv::inRange(oTargetImg_HSV,cv::Scalar(0,smin,std::min(vmin,vmax)),cv::Scalar(180,256,std::max(vmin,vmax)),oTargetImg_HSV_mask);
                if(oTargetMask.empty())
                    oTargetMask = oTargetImg_HSV_mask;
                else
                    oTargetMask &= oTargetImg_HSV_mask;
#if CAMSHIFT_DISPLAY_FG_MASK
                cv::imshow("oTargetMask",oTargetMask);
#endif //CAMSHIFT_DISPLAY_FG_MASK
#if CAMSHIFT_USE_MIXED_HSV
                cv::Mat oTargetImg_HUE;
                oTargetImg_HUE.create(oTargetImg_HSV.size(),oTargetImg_HSV.depth());
                cv::mixChannels(&oTargetImg_HSV,1,&oTargetImg_HUE,1,anChannelPairMix,1);
                cv::calcHist(&oTargetImg_HUE,1,anHistChannels,oTargetMask,oTargetImg_HIST,nHistDims,anHistBins,aafHistRanges);
#else //(!CAMSHIFT_USE_MIXED_HSV)
                calcHist(&oTargetImg_HSV,1,anHistChannels,oTargetMask,oTargetImg_HIST,nHistDims,anHistBins,aafHistRanges);
#endif //(!CAMSHIFT_USE_MIXED_HSV)
#else //(!CAMSHIFT_USE_PURE_HSV)
                calcHist(&oTargetImg,1,anHistChannels,cv::Mat(),oTargetImg_HIST,nHistDims,anHistBins,aafHistRanges);
#endif //(!CAMSHIFT_USE_PURE_HSV)
                cv::normalize(oTargetImg_HIST,oTargetImg_HIST,0,UCHAR_MAX,cv::NORM_MINMAX);
#if CAMSHIFT_DISPLAY_TARGET||CAMSHIFT_DISPLAY_FG_MASK
                cv::imshow("oTargetImg",oTargetImg);
                cv::waitKey(0);
#endif //CAMSHIFT_DISPLAY_TARGET
#if USE_VPTZ_TRACKING
                cv::Rect oTargetBBox((oCurrImg.cols-oTargetImg.cols)/2,(oCurrImg.rows-oTargetImg.rows)/2,oTargetImg.cols,oTargetImg.rows); // assuming target starts centered in first frame
#endif //USE_VPTZ_TRACKING
                cv::Mat oCurrImg_display = oCurrImg.clone();
                cv::rectangle(oCurrImg_display,oTargetBBox,cv::Scalar(0,255,0),3);
                while(oCurrImg_display.cols>1024 || oCurrImg_display.rows>768)
                    cv::resize(oCurrImg_display,oCurrImg_display,cv::Size(0,0),0.5,0.5);
                cv::imshow("oCurrImg_display",oCurrImg_display);
                cv::waitKey(0);
#if USE_VPTZ_TRACKING
                oTestEval.BeginTesting();
#endif //USE_VPTZ_TRACKING
                while(!oCurrImg.empty()) {
#if USE_VPTZ_TRACKING
#if CAMSHIFT_USE_POSE_PREDICT
                    cv::Point oTargetCenterPos = (oTargetBBox.tl()+oTargetBBox.br())*0.5;
                    cv::Point oExpectedDisplacement = oTargetBBox.area()?((oTargetCenterPos-oCenterPos)*fPosPredictScaleFactor):cv::Point(0,0);
                    cv::Point oNewCenterPos = oTargetCenterPos+oExpectedDisplacement;
                    oNewCenterPos.x = std::min(std::max(oNewCenterPos.x,0),nImageWidth-1);
                    oNewCenterPos.y = std::min(std::max(oNewCenterPos.y,0),nImageHeight-1);
#else //(!CAMSHIFT_USE_POSE_PREDICT)
                    cv::Point oNewCenterPos((oTargetBBox.tl()+oTargetBBox.br())*0.5);
#endif //(!CAMSHIFT_USE_POSE_PREDICT)
                oCurrImg = oTestEval.GetNextFrame(oNewCenterPos,VPTZ_USE_WAITSLEEP);
#else //(!USE_VPTZ_TRACKING)
                    oCap >> oCurrImg;
#endif //(!USE_VPTZ_TRACKING)
                    if(oCurrImg.empty())
                        break;
                    lvAssert(oCurrImg.type()==CV_8UC4);
                    cv::Mat oCurrImg_BP;
#if CAMSHIFT_USE_PURE_HSV
                    cv::Mat oCurrImg_HSV, oCurrImg_MASK;
                    cv::cvtColor(oCurrImg,oCurrImg_HSV,cv::COLOR_BGR2HSV);
                    cv::inRange(oCurrImg_HSV,cv::Scalar(0,smin,std::min(vmin,vmax)),cv::Scalar(180,256,std::max(vmin,vmax)),oCurrImg_MASK);
#if CAMSHIFT_DISPLAY_FG_MASK
                    cv::imshow("oCurrImg_MASK",oCurrImg_MASK);
#endif //CAMSHIFT_DISPLAY_FG_MASK
#if CAMSHIFT_USE_MIXED_HSV
                    cv::Mat oCurrImg_HUE;
                    oCurrImg_HUE.create(oCurrImg_HSV.size(), oCurrImg_HSV.depth());
                    cv::mixChannels(&oCurrImg_HSV,1,&oCurrImg_HUE,1,anChannelPairMix,1);
                    cv::calcBackProject(&oCurrImg_HUE,1,0,oTargetImg_HIST,oCurrImg_BP,aafHistRanges);
#else //(!CAMSHIFT_USE_MIXED_HSV)
                    cv::calcBackProject(&oCurrImg_HSV,1,anHistChannels,oTargetImg_HIST,oCurrImg_BP,aafHistRanges);
#endif //(!CAMSHIFT_USE_MIXED_HSV)
                    oCurrImg_BP &= oCurrImg_MASK;
                    cv::threshold(oCurrImg_BP,oCurrImg_BP,pmin,255,cv::THRESH_TOZERO);
#else //(!CAMSHIFT_USE_PURE_HSV)
                    cv::calcBackProject(&oCurrImg,1,anHistChannels,oTargetImg_HIST,oCurrImg_BP,aafHistRanges);
#endif //(!CAMSHIFT_USE_PURE_HSV)
                    oTargetBBox.width = std::max(oTargetBBox.width,1);
                    oTargetBBox.height = std::max(oTargetBBox.height,1);
                    cv::CamShift(oCurrImg_BP,oTargetBBox,cv::TermCriteria(cv::TermCriteria::EPS|cv::TermCriteria::MAX_ITER,nMeanShiftMaxIterCount,fMeanShiftMinEpsilon));
#if CAMSHIFT_LIMIT_SCALE_VAR
                    if(oTargetBBox.width>nMaxTargetBBoxWidth || oTargetBBox.width<nMinTargetBBoxWidth) {
                        int nWidthDiff = ((oTargetBBox.width>nMaxTargetBBoxWidth)?nMaxTargetBBoxWidth:nMinTargetBBoxWidth)-oTargetBBox.width;
                        oTargetBBox.x = std::max(std::min(oTargetBBox.x-nWidthDiff/2,nImageWidth-1),0);
                        oTargetBBox.width += nWidthDiff;
                        if(oTargetBBox.x+oTargetBBox.width>=nImageWidth)
                            oTargetBBox.width = nImageWidth-oTargetBBox.x;
                    }
                    if(oTargetBBox.height>nMaxTargetBBoxHeight || oTargetBBox.height<nMinTargetBBoxHeight) {
                        int nHeightDiff = ((oTargetBBox.height>nMaxTargetBBoxHeight)?nMaxTargetBBoxHeight:nMinTargetBBoxHeight)-oTargetBBox.height;
                        oTargetBBox.y = std::max(std::min(oTargetBBox.y-nHeightDiff/2,nImageHeight-1),0);
                        oTargetBBox.height += nHeightDiff;
                        if(oTargetBBox.y+oTargetBBox.height>=nImageHeight)
                            oTargetBBox.height = nImageHeight-oTargetBBox.y;
                    }
#endif //CAMSHIFT_LIMIT_SCALE_VAR
#if USE_VPTZ_TRACKING
                    oTestEval.UpdateCurrentResult(oTargetBBox,!(VPTZ_USE_SINGLE_TEST_SET||VPTZ_USE_ALL_TEST_SETS));
#endif //USE_VPTZ_TRACKING
                    if(oTargetBBox.width>0 && oTargetBBox.height>0)
                        cv::rectangle(oCurrImg,oTargetBBox,cv::Scalar(0,255,0),3);
#if USE_VPTZ_TRACKING
                    cv::rectangle(oCurrImg,oTestEval.GetLastGTBoundingBox(),cv::Scalar(0,255,255));
#endif //USE_VPTZ_TRACKING
                    cv::Mat oCurrImg_BP_display = oCurrImg_BP.clone();
                    oCurrImg_display = oCurrImg.clone();
                    while(oCurrImg_display.cols>1024 || oCurrImg_display.rows>768) {
                        cv::resize(oCurrImg_BP_display,oCurrImg_BP_display,cv::Size(0,0),0.5,0.5);
                        cv::resize(oCurrImg_display,oCurrImg_display,cv::Size(0,0),0.5,0.5);
                    }
#if CAMSHIFT_DISPLAY_BP_IMAGE
                    cv::imshow("oCurrImg_BP",oCurrImg_BP_display);
#endif //CAMSHIFT_DISPLAY_BP_IMAGE
                    cv::imshow("oCurrImg_display",oCurrImg_display);
                    cv::waitKey(1);
                    ++nTotProcessedFrameCount;
                }
#if USE_VPTZ_TRACKING
                oTestEval.EndTesting();
#endif //USE_VPTZ_TRACKING
            }
            std::cout << std::endl;
            std::cout << "nTotPotentialFrameCount = " << nTotPotentialFrameCount << std::endl;
            std::cout << "nTotProcessedFrameCount = " << nTotProcessedFrameCount << std::endl;
            std::cout << std::endl;
        }
        catch(const cv::Exception& e) {
            std::cerr << "top level caught cv::Exception:\n" << e.what() << std::endl;
#if VPTZ_USE_ALL_TEST_SETS
            break;
#endif //VPTZ_USE_ALL_TEST_SETS
        }
        catch(const std::exception& e) {
            std::cerr << "top level caught std::exception:\n" << e.what() << std::endl;
#if VPTZ_USE_ALL_TEST_SETS
            break;
#endif //VPTZ_USE_ALL_TEST_SETS
        }
        catch(...) {
            std::cerr << "top level caught unknown exception." << std::endl;
#if VPTZ_USE_ALL_TEST_SETS
            break;
#endif //VPTZ_USE_ALL_TEST_SETS
        }
    }
}

#if !USE_VPTZ_TRACKING
TargetBox::TargetBox(cv::Point tl, cv::Point br) {
    top_left = tl;
    bottom_right = br;
    adjust = false;
}

void TargetBox::changeCenter(cv::Point cen) {
    cv::Point offset = cen-center();
    top_left += offset;
    bottom_right += offset;
}

void displayRect(cv::Mat img) {
    cv::Mat img_display = img.clone();
    cv::rectangle( img_display, cv::Rect(g_oTargetBox.top_left,g_oTargetBox.bottom_right), cv::Scalar(0,255,0) );
    cv::circle( img_display, g_oTargetBox.center(), 2, cv::Scalar(0,0,255), 3);
    cv::circle( img_display, g_oTargetBox.bottom_right, 2, cv::Scalar(0,0,255), 3);
    imshow("target selection", img_display); cv::waitKey(1);
}

void onMouse( int event, int x, int y, int, void* ) {
    int error = 10;
    if( event == cv::EVENT_LBUTTONDOWN ) g_oTargetBox.adjust = true;
    else if( event == cv::EVENT_LBUTTONUP ) g_oTargetBox.adjust = false;
    else if( event == cv::EVENT_MOUSEMOVE && g_oTargetBox.adjust) {
        if( cv::norm( cv::Point(x,y)-g_oTargetBox.center() ) < error ) g_oTargetBox.changeCenter(cv::Point(x,y));
        else if( cv::norm( cv::Point(x,y)-g_oTargetBox.bottom_right ) < error ) g_oTargetBox.bottom_right = cv::Point(x,y);
        else if( fabs(x-g_oTargetBox.top_left.x) < error ) g_oTargetBox.top_left.x = x;
        else if( fabs(x-g_oTargetBox.bottom_right.x) < error ) g_oTargetBox.bottom_right.x = x;
        else if( fabs(y-g_oTargetBox.top_left.y) < error ) g_oTargetBox.top_left.y = y;
        else if( fabs(y-g_oTargetBox.bottom_right.y) < error ) g_oTargetBox.bottom_right.y = y;
    }
    else if( event == cv::EVENT_LBUTTONDBLCLK ) g_oTargetBox.changeCenter(cv::Point(x,y));
    displayRect(g_oTargetPickingMat);
}
#endif //(!USE_VPTZ_TRACKING)
