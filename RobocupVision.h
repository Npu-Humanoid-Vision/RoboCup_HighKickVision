#ifndef ROBOCUPVISION_H
#define ROBOCUPVISION_H

// Lable for SVM Glassifier
#define Ball_POS 1
#define Ball_NEG 0

#define ADJUST_PARAMETER

#include <opencv2/opencv.hpp>
#include <fstream> 
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;

#ifdef ADJUST_PARAMETER // need asj

// showing image in debugging 
#define SHOW_IMAGE(imgName) \
    namedWindow("imgName", WINDOW_AUTOSIZE); \
    moveWindow("imgName", 300, 300); \
    imshow("imgName", imgName); \
    waitKey(5000); \
    destroyWindow("imgName"); \


class ImgProcResult{

public:
    ImgProcResult(){};
    ~ImgProcResult(){};
    virtual void operator=(ImgProcResult &res) = 0;
private:
protected:
    ImgProcResult* res;
};

class ImgProc{

public:
    ImgProc(){};
    ~ImgProc(){};
    virtual void imageProcess(cv::Mat img, ImgProcResult *Result) =0;
private:
protected:
    ImgProcResult *res;

};
#else

#include "imgproc.h"
#define SHOW_IMAGE(imgName) ;

#endif 

class RobocupResult : public ImgProcResult {
public: // data menber
    // sideline detection relate
    bool sideline_valid_;
    double sideline_slope_;
    cv::Point2i sideline_center_;
    cv::Point2i sideline_begin_;
    cv::Point2i sideline_end_;

    // ball detection relate
    bool ball_valid_;
    cv::Point2i ball_center_;

    // goal detection relate
    bool goal_valid_;
    cv::Point2i goal_center_;

    // robo detection relate


    // location relate
public:
    RobocupResult() {
        sideline_valid_ = false;
        ball_valid_     = false;
        goal_valid_     = false; 
    }

    virtual void operator=(ImgProcResult& res) {
        RobocupResult* tmp  = dynamic_cast<RobocupResult*>(&res);
        
        sideline_valid_     = tmp->sideline_valid_;
        sideline_slope_     = tmp->sideline_slope_;
        sideline_center_    = tmp->sideline_center_; 
        sideline_begin_     = tmp->sideline_begin_;
        sideline_end_       = tmp->sideline_end_;

        ball_valid_         = tmp->ball_valid_;
        ball_center_        = tmp->ball_center_;

        goal_valid_         = tmp->goal_valid_;
        goal_center_        = tmp->goal_center_;
    }

    void operator=(RobocupResult& res) {
        sideline_valid_     = res.sideline_valid_;
        sideline_slope_     = res.sideline_slope_;
        sideline_center_    = res.sideline_center_;
        sideline_begin_     = res.sideline_begin_;
        sideline_end_       = res.sideline_end_;

        ball_valid_         = res.ball_valid_;
        ball_center_        = res.ball_center_;

        goal_valid_         = res.goal_valid_;
        goal_center_        = res.goal_center_;
    }
};

struct AllParameters {
    // glass relate
    int gls_h_min;
    int gls_h_max;
    int gls_h_direc;
    int gls_l_min;
    int gls_l_max;
    int gls_s_min;
    int gls_s_max;
    int gls_ero_times;
    int gls_dil_times;

    // ball relate
    int ball_l_min;
    int ball_l_max;
    int ball_ero_times;
    int ball_dil_times;

    int ball_rect_area_thre;

    // slide window relate
    int sld_win_num;
    int sld_win_rows;
    int sld_stride;
    int sld_thre_rate;

    // feel comfortable from Alex Beng !
    template<typename XXX>
    void operator=(XXX& robocup_vision) {
        gls_h_min            =robocup_vision.glass_h_min_thre_;
        gls_h_max            =robocup_vision.glass_h_max_thre_;
        gls_h_direc          =robocup_vision.glass_h_direction_forward_;
        gls_l_min            =robocup_vision.glass_l_min_thre_;
        gls_l_max            =robocup_vision.glass_l_max_thre_;
        gls_s_min            =robocup_vision.glass_s_min_thre_;
        gls_s_max            =robocup_vision.glass_s_max_thre_;
        gls_ero_times        =robocup_vision.glass_erode_times_;
        gls_dil_times        =robocup_vision.glass_dilate_times_;

        ball_l_min           =robocup_vision.ball_l_min_thre_;
        ball_l_max           =robocup_vision.ball_l_max_thre_;
        ball_ero_times       =robocup_vision.ball_erode_times_;
        ball_dil_times       =robocup_vision.ball_dilate_times_;

        ball_rect_area_thre  =robocup_vision.ball_rect_area_thre_;

        sld_win_num          =robocup_vision.slide_window_num_;
        sld_win_rows         =robocup_vision.slide_window_rows_;
        sld_stride           =robocup_vision.slide_stride_;
        sld_thre_rate        =robocup_vision.slide_win_thre_rate_*100;
    }
};

class RobocupVision : public ImgProc {
public:
    RobocupVision();

public:
    void imageProcess(cv::Mat input_image, ImgProcResult* output_result);   // external interface
    
    void Pretreat(cv::Mat raw_image);                                       // all pretreatment, image enhancement for the src_image and etc

    cv::Mat ProcessGlassColor();                                            // get the Glass binary image

    cv::Mat ProcessBallColor();                                             // get the ball binary image

    std::vector<cv::Point2i> GetSideLineBySldWin(cv::Mat binary_image);     // get the rough sideline by using slide windows and least squares fit

    std::vector<cv::Rect> GetPossibleBallRect(cv::Mat binary_image);         // get the possible ball's rects in the ball binary image with the help of sideline

    cv::Mat GetHogVec(cv::Rect roi);                                        // get the hog feature vector of roi in src_img 

public:
    void LoadEverything();                                                  // load parameters from the file AS WELL AS the SVM MODEL !!!!

    void StoreParameters();                                                 // Store parameters to file

    void set_all_parameters(AllParameters ap);                              // when setting parameters in main.cpp

    void WriteImg(cv::Mat src, string folder_name, int num);                // while running on darwin, save images

public: // data menbers
    // father of everything
    cv::Mat src_image_;

    // for Pretreat
    cv::Mat src_hls_channels_[3];
    cv::Mat src_hsv_channels_[3];
    cv::Mat pretreated_image_;      // the after-enhancement src image 
    
    // for ProcessXColor
    cv::Mat glass_binary_image_;
    int glass_h_min_thre_;
    int glass_h_max_thre_;
    int glass_h_direction_forward_;
    int glass_l_min_thre_;
    int glass_l_max_thre_;
    int glass_s_min_thre_;
    int glass_s_max_thre_;
    int glass_erode_times_;
    int glass_dilate_times_;    // the fear of dis-robust
    cv::Mat ball_binary_image_;
    int ball_l_min_thre_;
    int ball_l_max_thre_;
    int ball_erode_times_;
    int ball_dilate_times_;

    // for GetSideLineBySldWin
    int slide_window_num_;          // get from file
    int slide_window_cols_;         // compute from frame's cols and nums
    int slide_window_rows_;         // get from file
    std::vector<cv::Rect> slide_wins_;
    int slide_stride_;              // get from file
    double slide_win_thre_rate_;    // get from file
    std::vector<cv::Point2i> sideline_border_discrete_points_;

    // for GetPossibleBallRect
    std::vector<cv::Rect> ball_possible_rects_;
    cv::Rect ball_result_rect_;
    int ball_rect_area_thre_;

    // for GetHogVec
    // null
    // all can be motified in the function body (*^_^*)
    // because it is seldom motified, which need re-train the whole svm classifier

    // for WriteImg
    int start_file_num_;
    int max_file_num_;

    // for SVM classifier
    CvSVM ball_classifier_;
    string ball_model_name_;     

    // result & etc
    RobocupResult final_result_;
};  

#endif