// switch for adjusting params
#define ADJUST_PARAMETER

// switch for whether running on darwin
// #define RUN_ON_DARWIN

#include "RobocupVision_HighKick.h"

cv::VideoCapture cp(0);
cv::Mat frame;
RobocupVision_HK robocup_vision;
RobocupResult_HK gabage;
AllParameters_HK ap;

 
int main(int argc, char const *argv[]) {
    if (!cp.isOpened()) {
        cerr<<"open camera fail"<<endl;
        return -1;
    }
    // give ap init values
    ap = robocup_vision;

    cv::namedWindow("set_pretreat_param", CV_WINDOW_NORMAL);
    cv::createTrackbar("gaus_kernal_size",  "set_pretreat_param",    &ap.gaus_kernal_size,           66);

    cv::namedWindow("set_sideline_param", CV_WINDOW_NORMAL);
    cv::createTrackbar("thre_min",          "set_sideline_param",   &ap.sideline_min,               255);
    cv::createTrackbar("thre_max",          "set_sideline_param",   &ap.sideline_max,               255);
    cv::createTrackbar("hori_kernal_size",  "set_sideline_param",   &ap.sideline_hori_kernal_size,  66);

    cv::namedWindow("set_hough_params", CV_WINDOW_NORMAL);
    cv::createTrackbar("mor_kernal_size",   "set_hough_params",     &ap.mor_kernal_size,            23);
    cv::createTrackbar("line_vote",         "set_hough_params",     &ap.line_vote_thre,             333);

    while (1) {
        cp >> frame;
        if (frame.empty()) {
            cerr<<"frame empty, waiting for camare init..."<<endl;
            continue;
        }
        #ifdef RUN_ON_DARWIN
            cv::flip(frame, frame, -1);
            cv::resize(frame, frame, cv::Size(320, 240));
        #else
            cv::resize(frame, frame, cv::Size(320, 240));
        #endif
        robocup_vision.set_all_parameters(ap);
        robocup_vision.imageProcess(frame, &gabage);
        


        cv::imshow("living_frame", frame);
        char key = cv::waitKey(50);        
        if (key == 'q') {
            return 0;
        }
        else if (key == 's') {
            robocup_vision.StoreParameters();
            return 0;
        }
    }
    return 0;
}
