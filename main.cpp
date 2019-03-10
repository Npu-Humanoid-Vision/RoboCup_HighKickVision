// switch for adjusting params
#define ADJUST_PARAMETER

// switch for whether running on darwin
// #define RUN_ON_DARWIN

#include "RobocupVision.h"

cv::VideoCapture cp(0);
cv::Mat frame;
RobocupVision robocup_vision;
RobocupResult gabage;
AllParameters ap;


int main(int argc, char const *argv[]) {
    if (!cp.isOpened()) {
        cerr<<"open camera fail"<<endl;
        return -1;
    }
    // give ap init values
    ap = robocup_vision;

    cv::namedWindow("set_glass_params", CV_WINDOW_NORMAL);
    cv::createTrackbar("h_min", "set_glass_params", &ap.gls_h_min, 255);
    cv::createTrackbar("h_max", "set_glass_params", &ap.gls_h_max, 255);
    cv::createTrackbar("h_direc", "set_glass_params", &ap.gls_h_direc, 1);
    cv::createTrackbar("l_min", "set_glass_params", &ap.gls_l_min, 255);
    cv::createTrackbar("l_max", "set_glass_params", &ap.gls_l_max, 255);
    cv::createTrackbar("s_min", "set_glass_params", &ap.gls_s_min, 255);
    cv::createTrackbar("s_max", "set_glass_params", &ap.gls_s_max, 255);
    cv::createTrackbar("ero_times", "set_glass_params", &ap.gls_ero_times, 9);
    cv::createTrackbar("dil_times", "set_glass_params", &ap.gls_dil_times, 9);

    cv::namedWindow("set_ball_params", CV_WINDOW_NORMAL);
    cv::createTrackbar("l_min", "set_ball_params", &ap.ball_l_min, 255);
    cv::createTrackbar("l_max", "set_ball_params", &ap.ball_l_max, 255);
    cv::createTrackbar("ero_times", "set_ball_params", &ap.ball_ero_times, 9);
    cv::createTrackbar("dil_times", "set_ball_params", &ap.ball_dil_times, 9);

    cv::namedWindow("set_sld_win_params", CV_WINDOW_NORMAL);
    cv::createTrackbar("win_num", "set_sld_win_params", &ap.sld_win_num, 40);
    cv::createTrackbar("win_rows", "set_sld_win_params", &ap.sld_win_rows, 120);
    cv::createTrackbar("win_stride", "set_sld_win_params", &ap.sld_stride, 30);
    cv::createTrackbar("win_thre_rate", "set_sld_win_params", &ap.sld_thre_rate, 100);
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
        


        cv::imshow("glass_binary_image", robocup_vision.glass_binary_image_);
        if (gabage.ball_valid_) {
            cv::rectangle(frame, robocup_vision.ball_result_rect_, cv::Scalar(0, 255, 255));// ball marked in yellow
        }
        cv::imshow("ball_binary_image", robocup_vision.ball_binary_image_);
        for (int i  = 0; i < robocup_vision.slide_window_num_; i++) {
            cv::rectangle(frame, robocup_vision.slide_wins_[i], cv::Scalar(255, 255, 0));
        }
        if (gabage.sideline_valid_) {
            cv::line(frame, gabage.sideline_begin_, gabage.sideline_end_, cv::Scalar(0, 0, 255), 4);
        }
        cv::imshow("vision_result", frame);
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
