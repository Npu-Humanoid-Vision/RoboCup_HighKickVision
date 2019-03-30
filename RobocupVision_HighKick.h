#ifndef ROBOCUPVISION_HIGHKICK_H
#define ROBOCUPVISION_HIGHKICK_H

#define ADJUST_PARAMETER

#include <opencv2/opencv.hpp>
#include <fstream> 
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;

#ifdef ADJUST_PARAMETER // need asj

// showing image in debugging 
#define SHOW_IMAGE(window_name, imgName) \
    namedWindow(window_name, WINDOW_AUTOSIZE); \
    moveWindow(window_name, 300, 300); \
    imshow(window_name, imgName); \
    waitKey(2000); \
    destroyWindow(window_name); \


class ImgProcResult{

public:
    ImgProcResult(){};
    ~ImgProcResult(){};
    virtual void operator=(ImgProcResult &res) = 0;
private:
protected:
    ImgProcResult* res;
};

class ImgProc {

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

class RobocupResult_HK : public ImgProcResult {
public: // data menber
    // sideline detection relate
    bool sideline_valid_;
    double sideline_slope_;
    cv::Point2i sideline_center_;
    cv::Point2i sideline_begin_;
    cv::Point2i sideline_end_;

public:
    RobocupResult_HK() {
        sideline_valid_ = false;
    }

    virtual void operator=(ImgProcResult& res) {
        RobocupResult_HK* tmp  = dynamic_cast<RobocupResult_HK*>(&res);
        
        sideline_valid_     = tmp->sideline_valid_;
        sideline_slope_     = tmp->sideline_slope_;
        sideline_center_    = tmp->sideline_center_; 
    }

    void operator=(RobocupResult_HK& res) {
        sideline_valid_     = res.sideline_valid_;
        sideline_slope_     = res.sideline_slope_;
        sideline_center_    = res.sideline_center_;
    }
};

struct AllParameters {
    // preproc relate
    int gaus_kernal_size;

    // line relate
    int sideline_min;
    int sideline_max;
    int sideline_hori_kernal_size;

    // hough lines relate
    int mor_kernal_size;
    int line_vote_thre;

    // feel comfortable from Alex Beng !
    template<typename XXX>
    void operator=(XXX& robocup_vision) {
        int gaus_kernal_size;

        int sideline_min;
        int sideline_max;
        int sideline_hori_kernal_size;

        int mor_kernal_size;
        int line_vote_thre;
    }
};

enum {H, S, V, L, A, B};

class RobocupVision_HK : public ImgProc {
public:
    RobocupVision_HK();

public:
    void imageProcess(cv::Mat input_image, ImgProcResult* output_result);   // external interface
    
    cv::Mat Pretreat(cv::Mat raw_image);                                    // all pretreatment, image enhancement for the src_image and etc

    cv::Mat GetUsedChannel(cv::Mat& src_img, int flag);

    cv::Mat MorTreate(cv::Mat binary_image);

    std::vector<cv::Vec2f> StandardHough(cv::Mat binary_image);             // standard hough line

public:
    void LoadEverything();                                                  // load parameters from the file AS WELL AS the SVM MODEL !!!!

    void StoreParameters();                                                 // Store parameters to file

    void set_all_parameters(AllParameters ap);                              // when setting parameters in main.cpp

    void WriteImg(cv::Mat src, string folder_name, int num);                // while running on darwin, save images

public: // data menbers
    // father of everything
    cv::Mat src_image_;

    // for Pretreat
    int gaus_kernal_size_;
    cv::Mat pretreated_image_;      // the after-enhancement src image 
    
    // for ProcessXColor
    cv::Mat sideline_used_channel_;
    cv::Mat sideline_binary_image_;
    cv::Mat sideline_mor_treated_binary_image_;
    int sideline_min_;
    int sideline_max_;
    int sideline_hori_kernal_size_;

    // for standard hough line
    int mor_kernal_size_;
    int line_vote_thre_;
    std::vector<cv::Vec2f> lines_;

    // for WriteImg
    int start_file_num_;
    int max_file_num_;

    // result & etc
    RobocupResult_HK final_result_;
};  

#endif