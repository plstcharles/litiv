
//
//                              Virtual PTZ Camera API
//
//  - This work was done by Gengjie Chen while at Polytechnique Montreal, Summer 2014 -
//   - It is now maintained by Pierre-Luc St-Charles as part of the LITIV framework -
//
//        For the VPTZ API licensing info, see LICENSE.txt in the current folder.
//      For the LITIV framework licensing info, see LICENSE.txt in the root folder.
//
//
// The Camera class uses a panoramic video or image as input and constructs a
// spherical model. In the center of the sphere, the visual camera can pan, tilt and
// zoom to determine the view of the PTZ camera. The Evaluator class cooperates with
// Camera and evaluate the result of tracker
//
// The API is based on OpenGL (via freeglut & GLEW) and OpenCV >=2.4.9, the inputs
// and outputs are all in OpenCV format, (e.g. cv::Mat, cv::Point and cv::VideoCapture).
//
// @@@@ TODO: (05/2015)
//   - update manual with latest project architecture
//   - impl filenode config read/write functions for Camera
//   - completely wrap Camera inside evaluator class (link interfaces)
//

#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif //ndefined(_USE_MATH_DEFINES)
#include "litiv/vptz/utils.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#if !USE_VPTZ_STANDALONE
#include "litiv/utils/defines.hpp"
#if !HAVE_GLSL
#error "vptz requires full OpenGL support"
#endif //(!HAVE_GLSL)
#include "litiv/utils/parallel.hpp"
#include "litiv/utils/distances.hpp"
#include "litiv/utils/cxx.hpp"
#define VPTZ_API
#endif //(!USE_VPTZ_STANDALONE)

#define VPTZ_MINIMUM_BBOX_RADIUS 3

namespace vptz {

    class VPTZ_API Exception : public std::runtime_error {
    public:
        template<typename... Targs>
        Exception(const char* sErrMsg, const char* sFunc, const char* sFile, int nLine, Targs&&... args) :
                std::runtime_error(cv::format((std::string("vptz::Exception in function '%s' from %s(%d) : \n")+sErrMsg).c_str(),sFunc,sFile,nLine,std::forward<Targs>(args)...)),
                m_eErrn(GL_NO_ERROR),
                m_acErrMsg(sErrMsg),
                m_acFuncName(sFunc),
                m_acFileName(sFile),
                m_nLineNumber(nLine) {}
        const GLenum m_eErrn;
        const char* const m_acErrMsg;
        const char* const m_acFuncName;
        const char* const m_acFileName;
        const int m_nLineNumber;
    };

    enum CameraPropertyFlag {
        // get (read) only, 6 properties
        PTZ_CAM_IS_VIDEO,
        PTZ_CAM_FRAME_NUM,
        PTZ_CAM_HORI_ANGLE_CHANGE,
        PTZ_CAM_VERTI_ANGLE_CHANGE,
        PTZ_CAM_GET_FRAME_OVERHEAD,
        PTZ_CAM_EXECUTION_DELAY,
        PTZ_CAM_MOTION_DELAY,

        // both get (read) and set (write), 11 properties
        PTZ_CAM_FRAME_RATE,
        PTZ_CAM_FRAME_POS,
        PTZ_CAM_VERTI_FOV,
        PTZ_CAM_OUTPUT_WIDTH,
        PTZ_CAM_OUTPUT_HEIGHT,
        PTZ_CAM_HORI_ANGLE,
        PTZ_CAM_VERTI_ANGLE,
        PTZ_CAM_HORI_SPEED,
        PTZ_CAM_VERTI_SPEED,
        PTZ_CAM_EXECUTION_DELAY_RATIO,
        PTZ_CAM_COMMUNICATION_DELAY
    };

    class VPTZ_API Camera {
    public:
        /// Default Constructor; the default values are taken from the datasheet of the SONY network camera SNC-RZ50N.
        Camera( const std::string& input_file_path, // input video or image file path
                double verti_FOV = 90.0,
                double output_width = 640.0,
                double output_height = 480.0,
                double hori_angle = 0.0,
                double verti_angle = 90.0,
                double hori_speed = 300.0,
                double verti_speed = 300.0,
                double communication_delay = 0.2);
        ~Camera();

        /// moves the view to center the target point; throws if the point is out of image bounds. The User can also set the horizontal and vertical (pan and tilt) angles via the Set(...) function.
        void GoToPosition(int x, int y);
        /// moves the view to center the target point; throws if the point is out of image bounds. The User can also set the horizontal and vertical (pan and tilt) angles via the Set(...) function.
        void GoToPosition(cv::Point target);
        /// begins playing the video sequence; throws if the frame index is invalid.
        void BeginPlaying(int nFrameIdx=0);
        /// simulate the response delay of a PTZ camera; returns false if the video ends before the end of the waiting period.
        bool WaitDelay(bool bSleep=false);
        /// returns the current frame from the video sequence given the current FoV; specific frames can be obtained by using Set(...) prior to calling
        const cv::Mat& GetFrame();
        /// manual update function to replace the panoramic image (for image mode only)
        void UpdatePanoImage(cv::Mat& image);
        /// global property Getter, see 'CameraPropertyFlag' enum for options; throws if flag is invalid.
        double Get(CameraPropertyFlag flag);
        /// global property Setter, see 'CameraPropertyFlag' enum for options; throws if flag or value is invalid.
        void Set(CameraPropertyFlag flag, double value);

        const std::string m_sInputPath;

    private:

        // video information (basic parameters)
        bool isVideo;                            // true for video, false for image, get
        cv::VideoCapture panoCapture;            // panoramic video capture
        cv::Mat panoImage;                       // panoramic image
        double m_dFrameRate;                     // frame rate of input video (frame per second), 0 for image get, set
        int m_nScenarioFrameCount;               // total number of frames of input video, 1 for image, get
        int m_nCurrFrameIdx;                     // current frame position (number, [0, frameNum-1]), get, set
        GLuint m_nTexID;
        GLUquadricObj* m_pSphereObj;
        cv::Mat m_oViewportFrame;

        // constant parameters for OpenGL rendering (basic parameters)
        double vertiFOV;                         // vertical FOV angle of virtual camera (degree, (0, 180)), get, set
        double outputWidth, outputHeight;        // width and height of output image (pixel, >=1), get, set
        double horiAngle;                        // horizontal (phi) direction angle of virtual camera (degree, (-180, 180]), get, set
        double vertiAngle;                       // vertical (theta) direction angle of virtual camera (degree, [0, 180]), get, set

        // constant parameters for frame update
        double horiSpeed, vertiSpeed;            // horizontal (pan) and vertical (tilt) speeds of the camera (degree per second, >0), get, set
        double executionDelayRatio;              // ratio multiplied to execution delay, default value is 1.0 (>0), get, set

        // variables for frame update
        double firstTick;                        // for playing the video
        double horiAngleChange,vertiAngleChange; // required changes of horizontal and vertical direction angles (degree), get
        double getFrameOverhead;                 // the time of running GetFrame() (second, >=0), get
        double executionBeginTime;               // beginning time of other execution, including tracking and rendering (second)
        double executionEndTime;                 // ending time of other execution, including tracking and rendering (second)
        double currentTime;                      // current playing time since beginning, ignoring the getFrameOverhead (second)

        // three kinds of delay
        double executionDelay;                   // real execution delay of tracker (second, >=0), get
        double motionDelay;                      // motion delay (second, >=0), get
        double communicationDelay;               // communication delay (second, >=0), get, set

        // OpenGL internals
        int sphereGridSize;                      // grid size of the sphere
        std::unique_ptr<lv::gl::Context> m_pContext;

        Camera(const Camera&) = delete;
        Camera& operator=(const Camera&) = delete;
    };

    class VPTZ_API GTTranslator {
        friend class Evaluator;
    public:
        /// default Constructor based on existing vptz camera; throws if any input parameter is invalid
        GTTranslator( Camera* pCam,
                      int bgt_output_width = 640,
                      int bgt_output_height = 480,
                      double bgt_verti_FOV = 90.0);
        /// default Constructor; throws if any input parameter is invalid
        GTTranslator( int cur_output_width = 640,
                      int cur_output_height = 480,
                      double cur_verti_FOV = 90.0,
                      double cur_hori_angle = 0.0,
                      double cur_verti_angle = 0.0,
                      int bgt_output_width = 640,
                      int bgt_output_height = 480,
                      double bgt_verti_FOV = 90.0);

        /// updates viewing direction, only needed when there is no bound camera; throws if any input parameter is invalid
        void UpdateViewAngle( double cur_hori_angle,      // current horizontal (phi) direction angle of virtual (degree, (-180, 180])
                              double cur_verti_angle);    // current vertical (theta) direction angle of virtual (degree, [0, 180])
        /// tranlates gt target 2d point for particular outputWidth, outputHeight, vertiFOV, horiAngle, vertiAngle; throws if any input parameter is invalid
        /// note: returns false if the direct difference between the target direction and camera direction is larger than 90 degree
        bool GetGTTargetPoint( double bgtHoriAngle,        // horizontal (phi) direction angle of virtual camera in basic ground truth (degree, (-180, 180])
                               double bgtVertiAngle,       // vertical (theta) direction angle of virtual camera in basic ground truth (degree, [0, 180])
                               cv::Point& tgtTargetPoint); // translated ground-truth target point, reference (0,0) is bottom-left point
        /// tranlates ground-truth target bounding box for perticular outputWidth, outputHeight, vertiFOV, horiAngle, vertiAngle; throws if any input parameter is invalid
        /// note: returns false if the direct difference between the target direction and camera direction is larger than 90 degree
        bool GetGTBoundingBox( double bgtHoriAngle,        // horizontal (phi) direction angle of virtual camera in basic ground truth (degree, (-180, 180])
                               double bgtVertiAngle,       // vertical (theta) direction angle of virtual camera in basic ground truth (degree, [0, 180])
                               int bgtBBoxWidth,           // width of the bounding box in basic ground truth (pixel, [0,bgtOutputWidth])
                               int bgtBBoxHeight,          // height of the bounding box in basic ground truth (pixel, [0,bgtOutputHeight])
                               cv::Rect& tgtBoundingBox);  // translated ground-truth bounding box of, reference (0,0) is bottom-left point

    private:
        // used for current PTZ simulator
        Camera* m_pCamera;            // ptr to bound virtual camera, NULL when initialize without virtual camera
        int curOutputWidth, curOutputHeight;    // current width and height of camera output image (pixel, >=1)
        double curVertiFOV;                     // current vertical FOV angle of virtual camera (degree, (0, 180))
        double curHoriAngle;                    // horizontal (phi) direction angle of virtual camera (degree, (-180, 180])
        double curVertiAngle;                   // vertical (theta) direction angle of virtual camera (degree, [0, 180])

        // used to get basic ground truth
        int bgtOutputWidth, bgtOutputHeight;    // width and height of camera output image in ground truth(pixel, >=1)
        double bgtVertiFOV;                     // horizontal FOV angle of the virtual camera in ground truth (degree, (0, 180))
    };

    class VPTZ_API Evaluator {
    public:
        /// custom test constructor; throws if it cannot open the input file or its content is invalid
        Evaluator( const std::string& sInputScenarioPath,      // input video or image file path
                   const std::string& sInputGTSequencePath,    // input basic ground truth file path (.yml)
                   const std::string& sInputTargetMaskPath,    // input target mask image file path (can be empty)
                   const std::string& sOutputEvalFilePath,     // output evaluation result path (.yml)
                   double dCommDelay=0.5,
                   double dExecDelayRatio=0.0,
                   int nFirstTestFrameIdx=-1,
                   int nLastTestFrameIdx=INT_MAX);
        /// testset-based constructor; throws if it cannot open the input file or its content is invalid
        Evaluator( const std::string& sInputTestSetPath,
                   const std::string& sOutputEvalFilePath,
                   double dCommDelay=0.5,
                   double dExecDelayRatio=0.0);
        ~Evaluator();

        /// returns the frame count in the current test set
        int GetTestSetSize();
        /// returns the frame index in the current test set
        int GetCurrTestIdx();
        /// sets up test set for a given frame index (camera setup, output file setup)
        void SetupTesting(int nTestIdx=0);
        /// sets up last internal test components and triggers 'BeginPlaying(...)' on the vptz camera
        void BeginTesting();
        /// cleans internal state, computes overall metrics and prints them to console
        void EndTesting();
        /// redirects Get(...) calls to the camera
        double GetCurrCameraProperty(CameraPropertyFlag flag);
        /// returns the current test sequence name (for display purposes)
        std::string GetCurrTestSequenceName();
        /// returns the maximum number of frames that could be used in the current test sequence
        int GetPotentialTestFrameCount();
        /// returns the target image used to initialize tracking models
        cv::Mat GetInitTarget();
        /// returns the target image mask used to remove the background from the target image (should not be required by regular trackers)
        cv::Mat GetInitTargetMask();
        /// returns the full frame where the initialization target can be located
        cv::Mat GetInitTargetFrame();
        /// returns the bounding box where the initialization target is for the full frame
        cv::Rect GetInitTargetBBox();
        /// moves the camera to the expected target position (if needed), and returns the next image from the tracking sequence (or an empty frame when it is over)
        cv::Mat GetNextFrame(const cv::Point& oExpectedTargetPosition, bool bUseWaitDelaySleep=false);
        /// moves the camera to the expected target position (if needed), and returns the next image from the tracking sequence (or an empty frame when it is over)
        cv::Mat GetNextFrame(double dExpectedTargetAngle_H, double dExpectedTargetAngle_V, bool bUseWaitDelaySleep=false);
        /// returns the last ground truth bounding box, for display purposes only
        cv::Rect GetLastGTBoundingBox();
        /// feeds current tracking results to the evaluator which compares them to the groundtruth and saves them for later
        void UpdateCurrentResult(cv::Rect& rCurrTargetBBox, bool bPrintResult=false);

    private:
        void Setup( const std::string& sInputScenarioPath,
                    const std::string& sInputGTSequencePath,
                    const std::string& sInputTargetMaskPath,
                    int nFirstTestFrameIdx,
                    int nLastTestFrameIdx);

        std::unique_ptr<GTTranslator> m_pTranslator;
        std::unique_ptr<Camera> m_pCamera;
        cv::FileStorage m_oOutputEvalFS;
        struct TestMetadata {
            std::string sTestName;
            std::string sInputScenarioPath;
            std::string sInputGTSequencePath;
            std::string sInputTargetMaskPath;
            int nFirstTestFrameIdx;
            int nLastTestFrameIdx;
        };
        std::vector<TestMetadata> m_voTestSet;
        std::string m_sDatasetRootPath;
        const bool m_bUsingMultiTestSet;
        cv::FileStorage m_oCurrGTSequence_FS;
        cv::FileNode m_voCurrGTSequence_FN;

        double m_dCommDelay;
        double m_dExecDelayRatio;
        int m_nScenarioFrameCount;
        int m_nTestFrameCount;
        int m_nFirstGTSeqFrameIdx;
        int m_nFirstTestFrameIdx;
        int m_nLastTestFrameIdx;
        int m_bgtInitBBoxwidth, m_bgtInitBBoxheight;
        double m_bgtInitHoriAngle, m_bgtInitVertiAngle;
        cv::Rect m_tgtInitBBox;
        cv::Mat m_oInitTargetFrame;
        cv::Mat m_oInitTargetMask;
        cv::Mat m_oInitTarget;
        cv::Mat m_oCurrFrame;
        bool m_bRunning, m_bReady, m_bQueried;

        int m_nCurrTestIdx;
        int m_nCurrFrameIdx;                        // frame position, updated from bound Camera through GTTranslator
        int m_bgtBBoxwidth, m_bgtBBoxheight;        // width and height of basic ground-truth bounding box (pixel)
        double m_bgtHoriAngle, m_bgtVertiAngle;     // horizontal (phi) and vertical (theta) direction angle of virtual camera in basic ground truth (degree)

        // Translated ground truth info for current frame, reference (0,0) is bottom-left point
        // If difference between target and camera direction is larger than 90 degrees, tgtTargetPoint = (INT_MAX, INT_MAX)
        cv::Point tgtTargetPoint;                   // translated ground-truth target point (pixel)
        cv::Rect tgtBoundingBox;                    // translated ground-truth bounding box (pixel)

        int m_nCurrOutOfViewFrameCount;
        int m_nCurrProcessedFrameCount;
        double m_dCurrTargetPointErrorSum;
        double m_dCurrTargetPointOffsetSum;
        double m_dCurrBBoxOverlapRatioSum;
        double m_dOutOfViewFrameRatio_FullAvg;
        double m_dProcessedFrameRatio_FullAvg;
        double m_dTargetPointError_FullAvg;
        double m_dTargetPointOffset_FullAvg;
        double m_dBBoxOverlapRatio_FullAvg;
        double m_dTrackFragmentation_FullAvg;
        int m_nTotSeqsTested;
        int m_nTotTestFrameCount;
    };

    /// utility function: maps a 2D point on the virtual camera image to its horizontal and vertical angles in the sphere (throws if invalid args)
    void VPTZ_API PTZPointXYtoHV( int target2dX,                 // in: x coordinate of target point on the output image (pixel, [0, camOutputWidth-1])
                                  int target2dY,                 // in: y coordinate of target point on the output image (pixel, [0, camOutputHeight-1])
                                  double& tarHoriAngle,          // out: horizontal (phi) direction angle of target (degree)
                                  double& tarVertiAngle,         // out: vertical (theta) direction angle of target (degree)
                                  int camOutputWidth = 640.0,    // in: width of camera output image (pixel, >=1)
                                  int camOutputHeight = 480.0,   // in: height of camera output image (pixel, >=1)
                                  double camVertiFOV = 90.0,     // in: vertical FOV angle of the virtual camera (degree, (0, 180))
                                  double camHoriAngle = 0.0,     // in: horizontal (phi) direction angle of camera (degree, (-180, 180])
                                  double camVertiAngle = 90.0);  // in: vertical (theta) direction angle of camera (degree, [0, 180])
    /// utility function: maps a 2D point on the virtual camera image to its horizontal and vertical angles in the sphere (throws if invalid args)
    void VPTZ_API PTZPointXYtoHV( cv::Point2i targetXY,
                                  cv::Point2d& targetHV,
                                  int camOutputWidth = 640.0,
                                  int camOutputHeight = 480.0,
                                  double camVertiFOV = 90.0,
                                  double camHoriAngle = 0.0,
                                  double camVertiAngle = 90.0);

    /// utility function: maps a point from its horizontal and vertical angles to its 2D coordinate on the virtual camera image (throws if invalid args)
    /// note: returns false if the direct difference between the target direction and camera direction is larger than 90 degree
    bool VPTZ_API PTZPointHVtoXY( double tarHoriAngle,           // in: horizontal (phi) direction angle of target (degree, (-180, 180])
                                  double tarVertiAngle,          // in: vertical (theta) direction angle of target (degree, [0, 180])
                                  int& target2dX,                // out: x coordinate of target point on the output image (pixel, may exceed [0, camOutputWidth-1])
                                  int& target2dY,                // out: y coordinate of target point on the output image (pixel, may exceed [0, camOutputHeight-1])
                                  int camOutputWidth = 640.0,    // in: width of camera output image (pixel, >=1)
                                  int camOutputHeight = 480.0,   // in: height of camera output image (pixel, >=1)
                                  double camVertiFOV = 90.0,     // in: vertical FOV angle of the virtual camera (degree, (0, 180))
                                  double camHoriAngle = 0.0,     // in: horizontal (phi) direction angle of camera (degree, (-180, 180])
                                  double camVertiAngle = 90.0);  // in: vertical (theta) direction angle of camera (degree, [0, 180])
    /// utility function: maps a point from its horizontal and vertical angles to its 2D coordinate on the virtual camera image (throws if invalid args)
    /// note: returns false if the direct difference between the target direction and camera direction is larger than 90 degree
    bool VPTZ_API PTZPointHVtoXY( cv::Point2d targetHV,
                                  cv::Point2i& targetXY,
                                  int camOutputWidth = 640.0,
                                  int camOutputHeight = 480.0,
                                  double camVertiFOV = 90.0,
                                  double camHoriAngle = 0.0,
                                  double camVertiAngle = 90.0);

} // namespace vptz
