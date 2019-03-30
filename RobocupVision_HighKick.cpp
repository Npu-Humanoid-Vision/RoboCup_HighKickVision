#include "RobocupVision_HighKick.h"

RobocupVision_HK::RobocupVision_HK() {
    start_file_num_ = 0;
    max_file_num_   = 500;
    LoadEverything();
}

void RobocupVision_HK::imageProcess(cv::Mat input_image, ImgProcResult* output_result) {
    // pretreate image
    pretreated_image_ = Pretreat(input_image);

    // sideline thre
    sideline_used_channel_  = GetUsedChannel(pretreated_image_, L);
    sideline_binary_image_ = sideline_used_channel_>sideline_min_ & sideline_used_channel_<sideline_max_;

    // mor treat
    sideline_mor_treated_binary_image_ = MorTreate(sideline_binary_image_);

    lines_ = StandardHough(sideline_mor_treated_binary_image_);

    // judge line result

    // dynamic cast 
    (*dynamic_cast<RobocupResult_HK*>(output_result)) = final_result_;
#undef ADJUST_PARAMETER
#ifndef ADJUST_PARAMETER
    WriteImg(src_image_, "src_img", start_file_num_);
    // paint result on the src_image_ before being written
    WriteImg(src_image_, "center_img", start_file_num_++);
#endif
}

void RobocupVision::imageProcess(cv::Mat input_image, ImgProcResult* output_result) {
    Pretreat(input_image);

    // pix thre
    glass_binary_image_   = ProcessGlassColor();
    ball_binary_image_    = ProcessBallColor();

    // fit the sideline discrete points by least quares method
    cv::Mat mat_a(slide_window_num_, 2, CV_64FC1);
    cv::Mat mat_x(2, 1, CV_64FC1);
    cv::Mat mat_b(slide_window_num_, 1, CV_64FC1);
    sideline_border_discrete_points_ = GetSideLineBySldWin(glass_binary_image_);
    for (int i = 0; i < slide_window_num_; i++) {
        mat_a.at<double>(i, 0) = sideline_border_discrete_points_[i].x;
        mat_a.at<double>(i, 1) = 1.;

        mat_b.at<double>(i, 0) = sideline_border_discrete_points_[i].y;
    }
    cv::Mat mat_a_t = mat_a.t();
    mat_x = (mat_a_t*mat_a).inv(DECOMP_LU)*mat_a_t*mat_b;

    // Judge the sideline result
    if (fabs(mat_x.at<double>(0, 0)) > -1) {    // it should be stable (￣▽￣)""
        final_result_.sideline_valid_   = true;
        final_result_.sideline_slope_   = mat_x.at<double>(0, 0);
        final_result_.sideline_center_  = cv::Point2i(src_image_.cols/2, 
                                                    mat_x.at<double>(0, 0)*src_image_.cols/2.+mat_x.at<double>(1, 0));
        final_result_.sideline_begin_   = cv::Point2i(0, 
                                                    mat_x.at<double>(1, 0));
        final_result_.sideline_end_     = cv::Point2i(src_image_.cols, 
                                                    mat_x.at<double>(0, 0)*src_image_.cols+mat_x.at<double>(1, 0));
    }

    // Get Ball Relate within the line area
    ball_possible_rects_ = GetPossibleBallRect(ball_binary_image_);
    // feed ball possible rect to ball classifier 
    std::vector<cv::Rect> ball_pos_lable_rects;
    for (std::vector<cv::Rect>::iterator iter = ball_possible_rects_.begin();
         iter != ball_possible_rects_.end(); iter++) {
        cv::Mat roi_hog_vec = GetHogVec(*iter);
        int lable = (int)ball_classifier_.predict(roi_hog_vec);
        if (lable == Ball_POS) {
            ball_pos_lable_rects.push_back(*iter);
        }
    }

    // Judge the ball result
    if (ball_pos_lable_rects.size() == 0) {
        final_result_.ball_valid_ = false;
    }
    else if (ball_pos_lable_rects.size() == 1) {
        final_result_.ball_valid_ = true;
        ball_result_rect_ = ball_pos_lable_rects[0];
        final_result_.ball_center_ = cv::Point2d(ball_result_rect_.x + cvRound(ball_result_rect_.width/2),
                                                 ball_result_rect_.y + cvRound(ball_result_rect_.height/2));
    }
    else {
        final_result_.ball_valid_ = false;
    }

    (*dynamic_cast<RobocupResult*>(output_result)) = final_result_;
#ifndef ADJUST_PARAMETER
    WriteImg(src_image_, "src_img", start_file_num_);
    if (final_result_.sideline_valid_) {
        cv::line(src_image_, final_result_.sideline_begin_, final_result_.sideline_end_, cv::Scalar(0, 0, 255), 3);
    }
    if (final_result_.ball_valid_) {
        cv::rectangle(src_image_, ball_result_rect_, cv::Scalar(0, 255, 0));
    }
    WriteImg(src_image_, "center_img", start_file_num_++);
#endif 
}

void RobocupVision::Pretreat(cv::Mat raw_image) {
    // init src_image_
    src_image_ = raw_image.clone();

    // image enhancement
    cv::GaussianBlur(src_image_, pretreated_image_, cv::Size(5, 5), 0, 0);

    // get hls channels
    cv::Mat t_hls;
    cv::cvtColor(pretreated_image_, t_hls, CV_BGR2HLS_FULL);
    cv::split(t_hls, src_hls_channels_);

    return ;
}

cv::Mat RobocupVision::ProcessGlassColor() {
    cv::Mat mask = src_hls_channels_[1] >= glass_l_min_thre_ & src_hls_channels_[1] <= glass_l_max_thre_
                    & src_hls_channels_[2] >= glass_s_min_thre_ & src_hls_channels_[2] <= glass_s_max_thre_;

    cv::Mat thre_result;
    if (glass_h_direction_forward_) {
        thre_result = src_hls_channels_[0] <= glass_h_min_thre_ & src_hls_channels_[0] >= glass_h_max_thre_;
    }
    else {
        thre_result = src_hls_channels_[0] >= glass_h_max_thre_ | src_hls_channels_[0] <= glass_h_min_thre_;
    }

    thre_result = thre_result & mask;

    cv::erode(thre_result, thre_result, cv::Mat(5, 5, CV_8UC1), cv::Point(-1, -1), glass_erode_times_);
    cv::dilate(thre_result, thre_result, cv::Mat(5, 5, CV_8UC1), cv::Point(-1, -1), glass_dilate_times_);

    return thre_result;
}

cv::Mat RobocupVision::ProcessBallColor() {
    cv::Mat thre_result = src_hls_channels_[1] >= ball_l_min_thre_ & src_hls_channels_[1] <= ball_l_max_thre_;
    cv::erode(thre_result, thre_result, cv::Mat(5, 5, CV_8UC1), cv::Point(-1, -1), ball_erode_times_);
    cv::dilate(thre_result, thre_result, cv::Mat(5, 5, CV_8UC1), cv::Point(-1, -1), ball_dilate_times_);

    return thre_result;
}

std::vector<cv::Point2i> RobocupVision::GetSideLineBySldWin(cv::Mat binary_image) {
    std::vector<cv::Point2i> slide_wins_points;

    // init the slide windows
    // then slide wins to get edge
    // then get discrete points of the edge
    // then return the points
    slide_window_cols_ = src_image_.cols/slide_window_num_;
    for (int i = 0; i < slide_window_num_; i++) {
        // in case that stack overflow
        if (slide_wins_.size() < slide_window_num_) {
            slide_wins_.push_back(cv::Rect(i*slide_window_cols_, 0, slide_window_cols_, slide_window_rows_));
        }
        else {
            slide_wins_[i] = cv::Rect(i*slide_window_cols_, 0, slide_window_cols_, slide_window_rows_);
        }
        
        bool win_valid = true;
        int pix_counter = 0;
        cv::Mat win_roi;
        do {
            pix_counter = 0;
            win_roi = binary_image(slide_wins_[i]);

            // SHOW_IMAGE(win_roi);
            for (cv::Mat_<uchar>::iterator iter = win_roi.begin<uchar>(); iter != win_roi.end<uchar>(); iter++) {
                if (*iter == 255) {
                    pix_counter++;
                }
            }
            if (pix_counter*1.0/slide_wins_[i].area() > slide_win_thre_rate_) {
                win_valid = false;
                break;
            }

            if (slide_wins_[i].y+2*slide_stride_ < src_image_.rows) {
                slide_wins_[i].y += slide_stride_;
            }
            else {
                win_valid = false;
            }
        } while (win_valid);

        slide_wins_points.push_back(cv::Point2i(slide_wins_[i].x + slide_window_cols_/2, slide_wins_[i].y + slide_window_rows_*slide_win_thre_rate_));
    }
    return slide_wins_points;
}

std::vector<cv::Rect> RobocupVision::GetPossibleBallRect(cv::Mat binary_image) {
    std::vector<std::vector<cv::Point> > contours;
    std::vector<std::vector<cv::Point> > contours_poly;
    std::vector<cv::Rect> pos_rect;

    cv::Mat image_for_contours = binary_image.clone();
    cv::findContours(image_for_contours, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    contours_poly.resize(contours.size());
    cv::Rect t_rect;
    for (unsigned int i = 0; i < contours.size(); i++) {
        cv::approxPolyDP(contours[i], contours_poly[i], 3, false);
        t_rect =  cv::boundingRect(contours_poly[i]);
        if ((t_rect.x + t_rect.width/2)*final_result_.sideline_slope_ + final_result_.sideline_begin_.y > t_rect.y 
             && t_rect.area() > ball_rect_area_thre_) {
            pos_rect.push_back(t_rect);
        }
    }
    return pos_rect;
}

cv::Mat RobocupVision::GetHogVec(cv::Rect roi) {
    cv::Mat roi_mat = pretreated_image_(roi).clone();
    cv::resize(roi_mat, roi_mat, cv::Size(32, 32));
    
    cv::HOGDescriptor hog_des(cv::Size(32, 32), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
    std::vector<float> hog_vec;
    hog_des.compute(roi_mat, hog_vec);

    cv::Mat t(hog_vec);
    cv::Mat hog_vec_mat = t.t();
    return hog_vec_mat;
}

void RobocupVision::LoadEverything() {
#ifdef ADJUST_PARAMETER
    std::ifstream in_file("./7.txt");
#else  
    std::ifstream in_file("../source/data/set_sprint_param/7.txt");
#endif
    
    if (!in_file) {
        cerr<<"Error:"<<__FILE__
                <<":line"<<__LINE__<<endl
                <<"     Complied on"<<__DATE__
                <<"at"<<__TIME__<<endl;
    }
    int i = 0;
    string line_words;
    cout<<"Loading Parameters"<<endl;
    while (in_file >> line_words) {
        cout<<line_words<<endl;
        std::istringstream ins(line_words);
        switch(i++) {
        case 0:
            ins >> glass_h_min_thre_;
            break;
        case 1:
            ins >> glass_h_max_thre_;
            break;
        case 2:
            ins >> glass_h_direction_forward_;
            break;
        case 3:
            ins >> glass_l_min_thre_;
            break;
        case 4:
            ins >> glass_l_max_thre_;
            break;
        case 5:
            ins >> glass_s_min_thre_;
            break;
        case 6:
            ins >> glass_s_max_thre_;
            break;
        case 7:
            ins >> glass_erode_times_;
            break;
        case 8:
            ins >> glass_dilate_times_;
            break;
        case 9:
            ins >> ball_l_min_thre_;
            break;
        case 10:
            ins >> ball_l_max_thre_;
            break;
        case 11:
            ins >> ball_erode_times_;
            break;
        case 12:
            ins >> ball_dilate_times_;
            break;
        case 13:
            ins >> ball_rect_area_thre_;
            break;
        case 14:
            ins >> slide_window_num_;
            break;
        case 15:
            ins >> slide_window_rows_;
            break;
        case 16:
            ins >> slide_stride_;
            break;
        case 17:
            ins >> slide_win_thre_rate_;
            break;
        case 18:
            ins >> ball_model_name_;
            break;
        }
    }
#ifdef ADJUST_PARAMETER
    ball_classifier_.load(ball_model_name_.c_str());
#else
    ball_classifier_.load(("../source/data/SetRobocupParams/"+svm_model_name_).c_str().c_str());
#endif
}

void RobocupVision::StoreParameters() {
    std::ofstream out_file("./7.txt");
    if (!out_file) {
        cerr<<"Error:"<<__FILE__
                <<":line"<<__LINE__<<endl
                <<"     Complied on"<<__DATE__
                <<"at"<<__TIME__<<endl;
    }
    out_file << setw(3) << setfill('0') << glass_h_min_thre_            <<"___glass_h_min_thre_"<<endl;
    out_file << setw(3) << setfill('0') << glass_h_max_thre_            <<"___glass_h_max_thre_"<<endl;
    out_file << setw(3) << setfill('0') << glass_h_direction_forward_   <<"___glass_h_direction_forward_"<<endl;
    out_file << setw(3) << setfill('0') << glass_l_min_thre_            <<"___glass_l_min_thre_"<<endl;
    out_file << setw(3) << setfill('0') << glass_l_max_thre_            <<"___glass_l_max_thre_"<<endl;
    out_file << setw(3) << setfill('0') << glass_s_min_thre_            <<"___glass_s_min_thre_"<<endl;
    out_file << setw(3) << setfill('0') << glass_s_max_thre_            <<"___glass_s_max_thre_"<<endl;
    out_file << setw(3) << setfill('0') << glass_erode_times_           <<"___glass_erode_times_"<<endl;
    out_file << setw(3) << setfill('0') << glass_dilate_times_          <<"___glass_dilate_times_"<<endl;

    out_file << setw(3) << setfill('0') << ball_l_min_thre_             <<"___ball_l_min_thre_"<<endl;
    out_file << setw(3) << setfill('0') << ball_l_max_thre_             <<"___ball_l_max_thre_"<<endl;
    out_file << setw(3) << setfill('0') << ball_erode_times_            <<"___ball_erode_times_"<<endl;
    out_file << setw(3) << setfill('0') << ball_dilate_times_           <<"___ball_dilate_times_"<<endl;
    out_file << setw(3) << setfill('0') << ball_rect_area_thre_         <<"___ball_rect_area_thre_"<<endl;

    out_file << setw(3) << setfill('0') << slide_window_num_            <<"___slide_window_num_"<<endl;
    out_file << setw(3) << setfill('0') << slide_window_rows_           <<"___slide_window_rows_"<<endl;
    out_file << setw(3) << setfill('0') << slide_stride_                <<"___slide_stride_"<<endl;
    out_file << setw(3) << setfill('0') << slide_win_thre_rate_         <<"___slide_win_thre_rate_"<<endl;
    out_file << setw(3) << setfill('0') << ball_model_name_;
}

void RobocupVision::set_all_parameters(AllParameters ap) {
    glass_h_min_thre_           = ap.gls_h_min;
    glass_h_max_thre_           = ap.gls_h_max;
    glass_h_direction_forward_  = ap.gls_h_direc;
    glass_l_min_thre_           = ap.gls_l_min;
    glass_l_max_thre_           = ap.gls_l_max;
    glass_s_min_thre_           = ap.gls_s_min;
    glass_s_max_thre_           = ap.gls_s_max;
    glass_erode_times_          = ap.gls_ero_times;
    glass_dilate_times_         = ap.gls_dil_times;

    ball_l_min_thre_            = ap.ball_l_min;
    ball_l_max_thre_            = ap.ball_l_max;
    ball_erode_times_           = ap.ball_ero_times;
    ball_dilate_times_          = ap.ball_dil_times;

    ball_rect_area_thre_        = ap.ball_rect_area_thre;

    slide_window_num_           = ap.sld_win_num;
    slide_window_rows_          = ap.sld_win_rows;
    slide_stride_               = ap.sld_stride;
    slide_win_thre_rate_        = 0.01*ap.sld_thre_rate;
}

void RobocupVision::WriteImg(cv::Mat src, string folder_name, int num) {
    stringstream t_ss;
    string path = "../source/data/con_img/";
    if (start_file_num_ <= max_file_num_) {
        path += folder_name;
        path += "/";

        t_ss << num;
        path += t_ss.str();
        t_ss.str("");
        t_ss.clear();
        // path += std::to_string(num); 

        path += ".jpg";

        cv::imwrite(path,src);
    }
}