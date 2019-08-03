#ifndef ROBOCUPVISION_HIGHKICK_H
#define ROBOCUPVISION_HIGHKICK_H

#define ADJUST_PARAMETER

#include <opencv2/opencv.hpp>
#include <fstream> 
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
using namespace std;
using namespace cv;

enum {H, S, V, L, A, B};

#ifdef ADJUST_PARAMETER // need asj

// showing image in debugging 
#define SHOW_IMAGE(window_name, img) \
    namedWindow(window_name, WINDOW_NORMAL); \
    imshow(window_name, img); \
    waitKey(1); \

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
#define SHOW_IMAGE(window_name, img) ;

#endif 

class RobocupResult_HK : public ImgProcResult {
public: // data menber
    // sideline detection relate
    bool sideline_valid_;
    double sideline_angle_;
    cv::Point2i sideline_center_;

public:
    RobocupResult_HK() {
        sideline_valid_ = false;
    }

    virtual void operator=(ImgProcResult& res) {
        RobocupResult_HK* tmp  = dynamic_cast<RobocupResult_HK*>(&res);
        
        sideline_valid_     = tmp->sideline_valid_;
        sideline_angle_     = tmp->sideline_angle_;
        sideline_center_    = tmp->sideline_center_; 
    }

    void operator=(RobocupResult_HK& res) {
        sideline_valid_     = res.sideline_valid_;
        sideline_angle_     = res.sideline_angle_;
        sideline_center_    = res.sideline_center_;
    }
};

struct AllParameters_HK {
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
        gaus_kernal_size = robocup_vision.gaus_kernal_size_;

        sideline_min = robocup_vision.sideline_min_;
        sideline_max = robocup_vision.sideline_max_;
        sideline_hori_kernal_size = robocup_vision.sideline_hori_kernal_size_;

        mor_kernal_size = robocup_vision.mor_kernal_size_;
        line_vote_thre = robocup_vision.line_vote_thre_;
    }
};


class RobocupVision_HK : public ImgProc {
public:
    RobocupVision_HK();

public:
    void imageProcess(cv::Mat input_image, ImgProcResult* output_result);   // external interface
    
    cv::Mat GetUsedChannel(cv::Mat& src_img, int flag);

    cv::Mat MorTreate(cv::Mat binary_image);

    void StandardHough(cv::Mat mor_gradiant, std::vector<cv::Vec2f>&);             // standard hough line
    // return vec2f[0] mean lines' rho 
    // vec2f[1] mean lines' theta

    void PbbHough(cv::Mat binary_image, std::vector<cv::Vec4i>&);                   // probabilistic Hough transform
    // 两端
    // Point(lines[i][0], lines[i][1]),
    // Point(lines[i][2], lines[i][3]) 

public:
    void LoadEverything();                                                  // load parameters from the file AS WELL AS the SVM MODEL !!!!

    void StoreParameters();                                                 // Store parameters to file

    void set_all_parameters(AllParameters_HK ap);                              // when setting parameters in main.cpp

    void WriteImg(cv::Mat src, string folder_name, int num);                // while running on darwin, save images

    void SegmentNms(std::vector<cv::Vec4i>& segments_in, std::vector<bool>& out_flags, double nms_thre); 
    // nms for segment scoring in length & iou in position 

    // For NMS
    bool SortSegments(cv::Vec4i s_1, cv::Vec4i s_2);    // for stl's sort

    // For iou
    double SegmentsIou(cv::Vec4i s_1, cv::Vec4i s_2);

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
    std::vector<cv::Vec4i> segments_;

    // for WriteImg
    int start_file_num_;
    int max_file_num_;

    // for nms
    // int nms_thre_;

    // result & etc
    RobocupResult_HK final_result_;
};  


#endif