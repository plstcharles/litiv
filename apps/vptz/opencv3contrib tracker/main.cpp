

#include "litiv/vptz/virtualptzExtend.hpp"
#include <opencv2/tracking.hpp>

#define VPTZ_USE_ALL_TEST_SETS     1    // used for batch-testing on vptz framework (will fetch all test sets automatically)
#define VPTZ_USE_SINGLE_TEST_SET   0    // used to run a single test set on vptz framework (must be specified via define below)
#define VPTZ_USE_WAITSLEEP         0    // = 'simulate' full delays by 'sleeping' between frames
#define VPTZ_EXEC_DELAY_RATIO      0	// = normal processing delay penalty (100% time lost considered)
#define VPTZ_COMMUNICATION_DELAY   0	//0.125   // = 125ms network ping delay between server and client
#define DISPLAY_CAMERA_IMAGE       1	//1 to display camera view
#define DISPLAY_PANORAMICA_IMAGE   1	//1 to display panoramic view
#define INPUT_TEST_TRACKER_NAMES	{"MIL"}//{"MIL","TLD"}//{"KCF","MEDIANFLOW","MIL","TLD","BOOSTING"}
#define INPUT_TEST_POSE_PREDICT_SCALE	{0.0f,1.0f}

#define VPTZ_DATASET_ROOT_DIR_PATH       std::string("")
#define INPUT_TEST_SETS_PATH_PREFIX      std::string("testsets/")
#define OUTPUT_EVAL_FILES_PATH_PREFIX    std::string("results-yml/")
#define INTPUT_TEST_SETS_NAMES          {"cluttered_background"} /*{"all","articulated_objects","cluttered_background", \
                                          "distractors","fast_motion","illumination_variation", \
                                          "low_resolution","occlusion"}*/
cv::Point g_oLastMouseClickPos;		//output coordinates in panoramic image
void onMouse(int nEventCode, int x, int y, int, void*) {
	if (nEventCode == cv::EVENT_LBUTTONDOWN) {
		std::cout << "Clicked at [" << x << "," << y << "]" << std::endl;
		g_oLastMouseClickPos = cv::Point(x, y);
	}
}
int main(int /*argc*/, char** /*argv*/) {
	std::string _TrackerName;
	const char* TrackerNames[] = INPUT_TEST_TRACKER_NAMES;
	const float POSE_PREDICT_SCALE[] = INPUT_TEST_POSE_PREDICT_SCALE;
	for (int _TrackerNums = 0; _TrackerNums < sizeof(TrackerNames) / sizeof(char*); _TrackerNums++)//KCF,MEDIANFLOW,MIL,BOOSTING,TLD
	{
		for (int _POSE_PREDICT_SCALE_iter = 0; _POSE_PREDICT_SCALE_iter <= 0 /*/ sizeof(POSE_PREDICT_SCALE[0])*/; _POSE_PREDICT_SCALE_iter++)
		{
			_TrackerName = TrackerNames[_TrackerNums];//KCF,MEDIANFLOW,MIL,BOOSTING,TLD,OAB
			std::cout << "using " << _TrackerName << " as tracker. predict scale:" << POSE_PREDICT_SCALE[_POSE_PREDICT_SCALE_iter] << std::endl;
			const char* asTestSets[] = INTPUT_TEST_SETS_NAMES;
			for (size_t nTestSetIdx = 0; nTestSetIdx < sizeof(asTestSets) / sizeof(char*); ++nTestSetIdx) {//all, articulated_objects, cluttered_background, distractors...
				const std::string sCurrTestSetPath = VPTZ_DATASET_ROOT_DIR_PATH + INPUT_TEST_SETS_PATH_PREFIX + asTestSets[nTestSetIdx] + ".yml";/*testset yml path*/
				char TpredictScaleStr[3];
				char TexecDelayRatioStr[3];
				_itoa(POSE_PREDICT_SCALE[_POSE_PREDICT_SCALE_iter], TpredictScaleStr, 10);
				_itoa(VPTZ_EXEC_DELAY_RATIO, TexecDelayRatioStr, 10);
				const std::string sCurrResultFilePath = VPTZ_DATASET_ROOT_DIR_PATH + OUTPUT_EVAL_FILES_PATH_PREFIX + _TrackerName + "/" + asTestSets[nTestSetIdx] + "-" + _TrackerName + "-predictScale" + TpredictScaleStr + 
					"-delayRatio" + TexecDelayRatioStr + ".yml";/*output yml path*/
				std::cout << "\n\n===============================\n\n  Setting up testset#" << nTestSetIdx + 1 << " [" << asTestSets[nTestSetIdx] << "]\n\n===============================\n" << std::endl;
				try {
					cv::Mat oCurrImg;
					cv::Mat oTargetImg, oTargetMask;
					int nTotPotentialFrameCount = 0;
					int nTotProcessedFrameCount = 0;
					vptz::EvaluatorEx oTestEval(sCurrTestSetPath, sCurrResultFilePath,
						VPTZ_COMMUNICATION_DELAY, VPTZ_EXEC_DELAY_RATIO, POSE_PREDICT_SCALE[_POSE_PREDICT_SCALE_iter]);


					for (int nTestIdx = 0; nTestIdx <  oTestEval.GetTestSetSize(); ++nTestIdx) {
						if (nTestIdx > 0)
							cv::destroyAllWindows();
						std::cout << "\nSetting up seq#" << nTestIdx + 1 << "..." << std::endl;
						oTestEval.SetupTesting(nTestIdx);
						std::cout << "Processing seq#" << nTestIdx + 1 << " [" << oTestEval.GetCurrTestSequenceName() << "]..." << std::endl;
						oCurrImg = oTestEval.GetInitTargetFrame();
						oTargetImg = oTestEval.GetInitTarget();
						nTotPotentialFrameCount += oTestEval.GetPotentialTestFrameCount();
						CV_Assert(oTargetImg.type() == CV_8UC4);//if the oTargetImg.type != CV_8UC4, then the program will terminate.
						const int nImageWidth = oCurrImg.cols;
						const int nImageHeight = oCurrImg.rows;
						cv::Rect2d oTargetBBox((oCurrImg.cols - oTargetImg.cols) / 2, (oCurrImg.rows - oTargetImg.rows) / 2, oTargetImg.cols, oTargetImg.rows); // assuming target starts centered in first frame
		/*init Opencv3 Tracker*/
						cv::Ptr<cv::Tracker> tracker;
						tracker = cv::Tracker::create(_TrackerName);
						tracker->init(oCurrImg, oTargetBBox);
						oTestEval.UpdateCurrentPanoramicStatus();
#if DISPLAY_CAMERA_IMAGE

						cv::Mat oCurrImg_display = oCurrImg.clone();
						cv::rectangle(oCurrImg_display, oTargetBBox, cv::Scalar(0, 255, 0), 1);//draw Ground Truth Rectangle on the image
						while (oCurrImg_display.cols > 1024 || oCurrImg_display.rows > 768)
							cv::resize(oCurrImg_display, oCurrImg_display, cv::Size(0, 0), 0.5, 0.5);
						cv::namedWindow("oCurrImg_display", 1);
						cv::moveWindow("oCurrImg_display", 0, 0);
						cv::imshow("oCurrImg_display", oCurrImg_display);
#endif//DISPLAY_CAMERA_IMAGE
#if DISPLAY_PANORAMICA_IMAGE


						cv::namedWindow("panoramic frame", 0);
						cv::resizeWindow("panoramic frame", 900 * 1.3, 450 * 1.3);
						std::cout << "[debug]m_OriFrame cols(Circumference of Sephemore):" << oTestEval.m_pCamera->panoImage.cols;
						std::cout << " Sephere radius: " << oTestEval.GetSphereRadius() << std::endl;
						cv::Mat m_OriFrame(oTestEval.GetPanoramicFrame());
						cv::Point tgtPointTop(oTestEval.GetPanoramicTargetPoint().x, 0);
						cv::Point tgtPointBottom(oTestEval.GetPanoramicTargetPoint().x, 1400);
						cv::circle(m_OriFrame, oTestEval.GetPanoramicTargetPoint(), 5, cv::Scalar(255, 255, 0), 4);
						cv::line(m_OriFrame, tgtPointTop, tgtPointBottom, cv::Scalar(0, 255, 0), 4);
						cv::imshow("panoramic frame", m_OriFrame);
						cv::setMouseCallback("panoramic frame", onMouse, 0);
						g_oLastMouseClickPos = cv::Point(m_OriFrame.cols / 2, m_OriFrame.rows / 2);
#endif // DISPLAY_PANORAMICA_IMAGE
						std::cout <<"target coordinates [x,y] in panoramic image:"<< oTestEval.GetPanoramicTargetPoint().x << ","<<oTestEval.GetPanoramicTargetPoint().y << std::endl;
						oTestEval.BeginTesting();
						while (!oCurrImg.empty()) {
							//std::cout << "--------------------------------" << std::endl;
							double tempExecTime = oTestEval.GetCurrCameraProperty(vptz::PTZ_CAM_MOTION_DELAY) + oTestEval.GetCurrCameraProperty(vptz::PTZ_CAM_EXECUTION_DELAY);
#if accelerationVelocity// s=v*t+(a*t^2)/2
							double p_offset_x = oTestEval.GetAccelerationX()*tempExecTime*tempExecTime / 2 + int(oTestEval.GetTgtSpeedX()* (oTestEval.GetCurrCameraProperty(vptz::PTZ_CAM_MOTION_DELAY) + oTestEval.GetCurrCameraProperty(vptz::PTZ_CAM_EXECUTION_DELAY)));//prediction_offset
							double p_offset_y = oTestEval.GetAccelerationY()*tempExecTime*tempExecTime / 2 + int(oTestEval.GetTgtSpeedY()* (oTestEval.GetCurrCameraProperty(vptz::PTZ_CAM_MOTION_DELAY) + oTestEval.GetCurrCameraProperty(vptz::PTZ_CAM_EXECUTION_DELAY)));//prediction_offset

#else //original and menVelocity s=vt
							double p_offset_x = int(oTestEval.GetTgtSpeedX()* (oTestEval.GetCurrCameraProperty(vptz::PTZ_CAM_MOTION_DELAY) + oTestEval.GetCurrCameraProperty(vptz::PTZ_CAM_EXECUTION_DELAY)));//prediction_offset
							double p_offset_y = int(oTestEval.GetTgtSpeedY()* (oTestEval.GetCurrCameraProperty(vptz::PTZ_CAM_MOTION_DELAY) + oTestEval.GetCurrCameraProperty(vptz::PTZ_CAM_EXECUTION_DELAY)));//prediction_offset
#endif accelerationVelocity
							//double p_offset_x = std::min(int(oTestEval.GetTgtSpeedX()* (oTestEval.GetCurrCameraProperty(vptz::PTZ_CAM_MOTION_DELAY) + oTestEval.GetCurrCameraProperty(vptz::PTZ_CAM_EXECUTION_DELAY))), 300);//prediction_offset
							//double p_offset_y = std::min(int(oTestEval.GetTgtSpeedY()* (oTestEval.GetCurrCameraProperty(vptz::PTZ_CAM_MOTION_DELAY) + oTestEval.GetCurrCameraProperty(vptz::PTZ_CAM_EXECUTION_DELAY))), 300);//prediction_offset
							//std::cout << "[debug]" << std::setw(25) << "predicted offset is: " << p_offset_x << "," << p_offset_y << std::endl;
							cv::Point oNewCenterPos((oTargetBBox.tl() + oTargetBBox.br())*0.5);
							if (oTestEval.UpdateSpeedAccordingToWhetherObjTurnAround(oNewCenterPos))//
							{
								p_offset_x = 0;
								p_offset_y = 0;
							}
							oNewCenterPos.x += p_offset_x*POSE_PREDICT_SCALE[_POSE_PREDICT_SCALE_iter];
							oNewCenterPos.y += p_offset_y*POSE_PREDICT_SCALE[_POSE_PREDICT_SCALE_iter];
							oNewCenterPos.x = std::min(std::max(oNewCenterPos.x, 0), nImageWidth - 1);
							oNewCenterPos.y = std::min(std::max(oNewCenterPos.y, 0), nImageHeight - 1);
							oCurrImg = oTestEval.GetNextFrame(oNewCenterPos, VPTZ_USE_WAITSLEEP);
							cv::Mat oCurrImgRGB;
							if (oCurrImg.channels() == 4)
								cv::cvtColor(oCurrImg, oCurrImgRGB, cv::COLOR_BGRA2BGR);//hint:oCurrImg is BGRA with 4 channel, thus convert to 3 channel to fill in the Tracker::update() parameter.
							else
								oCurrImg.copyTo(oCurrImgRGB);
							oTargetBBox.width = std::max(int(oTargetBBox.width), 1);
							oTargetBBox.height = std::max(int(oTargetBBox.height), 1);
							oTestEval.SetTrackingResult(tracker->update(oCurrImgRGB, oTargetBBox));//tracker update!oTargetBBox will be used in next loop.
							if (oCurrImg.empty())
								break;
							CV_Assert(oCurrImg.type() == CV_8UC4);
							oTestEval.UpdateCurrentResult(cv::Rect(oTargetBBox), !(VPTZ_USE_SINGLE_TEST_SET || VPTZ_USE_ALL_TEST_SETS));
#if DISPLAY_CAMERA_IMAGE

							if (oTargetBBox.width > 0 && oTargetBBox.height > 0)
								cv::rectangle(oCurrImg, oTargetBBox, cv::Scalar(0, 255, 0), 1);
							cv::rectangle(oCurrImg, oTestEval.GetLastGTBoundingBox(), cv::Scalar(0, 255, 255));
							oCurrImg_display = oCurrImg.clone();
							while (oCurrImg_display.cols > 1024 || oCurrImg_display.rows > 768) {
								cv::resize(oCurrImg_display, oCurrImg_display, cv::Size(0, 0), 0.5, 0.5);
							}
							cv::imshow("oCurrImg_display", oCurrImg_display);//althrough tracking box is in camera frame, but in debug mode we no need it.
#endif//DISPLAY_CAMERA_IMAGE
							oTestEval.UpdateCurrentPanoramicStatus();
#if DISPLAY_PANORAMICA_IMAGE
							oTestEval.m_pCamera->panoImage.copyTo(m_OriFrame);
							tgtPointTop.x = oTestEval.GetPanoramicTargetPoint().x;
							tgtPointBottom.x = oTestEval.GetPanoramicTargetPoint().x;
							cv::circle(m_OriFrame, oTestEval.GetPanoramicTargetPoint(), 5, cv::Scalar(255, 255, 0), 4);
							cv::line(m_OriFrame, tgtPointTop, tgtPointBottom, cv::Scalar(0, 255, 0), 4);
							cv::imshow("panoramic frame", m_OriFrame);
#endif //DISPLAY_PANORAMICA_IMAGE
							++nTotProcessedFrameCount;

							int key = cv::waitKey(1);
							if (key == 27)
							{
								std::cout << "----jump detected.---" << std::endl;
								break;
							}
						}
						/************************************************************************/
						/*@Output:                                                              */
						/*TPE:average offset between GT center and Tracker center
						/*TPO:average offset between GT center and image center
						/*BBOR:average overlapping ration between GT bounding box and bounding box
						/*TF:out of view images/valid images
						/************************************************************************/
						oTestEval.EndTesting();
					}
					std::cout << std::endl;
					std::cout << "nTotPotentialFrameCount = " << nTotPotentialFrameCount << std::endl;
					std::cout << "nTotProcessedFrameCount = " << nTotProcessedFrameCount << std::endl;
					std::cout << std::endl;
				}
				catch (const cv::Exception& e) {
					std::cerr << "top level caught cv::Exception:\n" << e.what() << std::endl;
					break;
				}
				catch (const std::exception& e) {
					std::cerr << "top level caught std::exception:\n" << e.what() << std::endl;
					break;
				}
				catch (...) {
					std::cerr << "top level caught unknown exception." << std::endl;
					break;
				}
			}
		}//end_POSE_SACLE
	}//end _TrackerNums
}

