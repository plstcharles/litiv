
//======================================================================================
//
// This program is used to view ground truth annotations while controlling the camera.
//
// Before running, set the input scenario/gt paths using the defines below.
//
//======================================================================================

#include "litiv/vptz/virtualptz.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define INPUT_SCENARIO_PATH        "/some/root/directory/litiv_vptz_icip2015//scenario3/scenario3.avi"
#define INPUT_GT_SEQUENCE_PATH     "/some/root/directory/litiv_vptz_icip2015/scenario3/gt/scenario3_torso02.yml"
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

cv::Point g_oLastMouseClickPos;
void onMouse(int nEventCode, int x, int y, int, void*) {
    if(nEventCode==cv::EVENT_LBUTTONDOWN) {
        std::cout << "Clicked at [" << x << "," << y << "]" << std::endl;
        g_oLastMouseClickPos = cv::Point(x,y);
    }
}

int main(int /*argc*/, char** /*argv*/) {
    try {
        cv::FileStorage oInputGT(INPUT_GT_SEQUENCE_PATH, cv::FileStorage::READ);
        int nGTFrameWidth = oInputGT["frameImageWidth"];
        int nGTFrameHeight = oInputGT["frameImageHeight"];
        double dGTVerticalFOV = oInputGT["verticalFOV"];
        vptz::Camera oCamera(INPUT_SCENARIO_PATH);
        vptz::GTTranslator oGTTranslator(&oCamera, nGTFrameWidth, nGTFrameHeight, dGTVerticalFOV);
        cv::Point oTargetPos_XY;
        cv::Point2d oTargetPos_HX;
        cv::Mat oCurrView;
        int nBBoxWidth, nBBoxHeight;
        int nFrameIdx;

        oCamera.Set(vptz::PTZ_CAM_FRAME_POS, 0);
        oCurrView = oCamera.GetFrame();
        cv::namedWindow("Current View");
        cv::imshow("Current View", oCurrView);
        cv::waitKey(2);
        cv::setMouseCallback("Current View", onMouse, 0);
        g_oLastMouseClickPos = cv::Point(oCurrView.cols/2, oCurrView.rows/2);

        cv::FileNode oGTNode = oInputGT["basicGroundTruth"];
        bool bPaused = true;
        for(auto oGTFrame=oGTNode.begin(); oGTFrame!=oGTNode.end(); ++oGTFrame) {
            nFrameIdx = (*oGTFrame)["framePos"];
            nBBoxWidth = (*oGTFrame)["width"];
            nBBoxHeight = (*oGTFrame)["height"];
            oTargetPos_HX.x = (*oGTFrame)["horizontalAngle"];
            oTargetPos_HX.y = (*oGTFrame)["verticalAngle"];
            std::cout << "\t#" << nFrameIdx << std::endl;
            oCamera.Set(vptz::PTZ_CAM_FRAME_POS, nFrameIdx);
            while(true) {
                oCamera.GoToPosition(g_oLastMouseClickPos);
                oCurrView = oCamera.GetFrame();
                if(oCurrView.empty())
                    break;
                g_oLastMouseClickPos = cv::Point(oCurrView.cols/2, oCurrView.rows/2);
                cv::circle(oCurrView, cv::Point(oCurrView.cols/2,oCurrView.rows/2), 3, cv::Scalar(0,255,0), 5);
                oGTTranslator.GetGTTargetPoint(oTargetPos_HX.x, oTargetPos_HX.y, oTargetPos_XY);
                cv::circle(oCurrView, oTargetPos_XY, 3, cv::Scalar(0,255,255), 5);
                cv::Rect bb;
                oGTTranslator.GetGTBoundingBox(oTargetPos_HX.x, oTargetPos_HX.y, nBBoxWidth, nBBoxHeight, bb);
                cv::rectangle(oCurrView, bb, cv::Scalar(0,255,255));
                cv::imshow("Current View", oCurrView);
                char cKey = (char)cv::waitKey(1);
                if(cKey==' ')
                    bPaused = !bPaused;
                else if(cKey!=-1)
                    break;
                if(!bPaused)
                    break;
            }
        }
        return 0;
    }
    catch(const std::exception& e) {
        std::cerr << "top level caught std::exception:\n" << e.what() << std::endl;
    }
    catch(...) {
        std::cerr << "top level caught unknown exception." << std::endl;
    }
}
