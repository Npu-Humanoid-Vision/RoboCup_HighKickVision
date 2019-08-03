#include "RobocupVision_HighKick.h"

RobocupVision_HK::RobocupVision_HK() {
    start_file_num_ = 0;
    max_file_num_   = 500;
    LoadEverything();
}

void RobocupVision_HK::imageProcess(cv::Mat input_image, ImgProcResult* output_result) {
    // pretreate image
    src_image_ = input_image.clone();
    cv::GaussianBlur(src_image_, pretreated_image_, cv::Size(2*gaus_kernal_size_+1, 2*gaus_kernal_size_+1), 0, 0);

    SHOW_IMAGE("pretreated_frame", pretreated_image_);

    // sideline thre
    sideline_used_channel_  = GetUsedChannel(pretreated_image_, L);
    sideline_binary_image_ = sideline_used_channel_>sideline_min_ & sideline_used_channel_<sideline_max_;

    SHOW_IMAGE("sideline_thre", sideline_binary_image_);

    // mor treat
    sideline_mor_treated_binary_image_ = MorTreate(sideline_binary_image_);

    PbbHough(sideline_mor_treated_binary_image_, segments_);

    // std::vector<bool> nms_flags;
    // std::vector<cv::Vec4i> final_segs;
    // std::vector<double> final_segs_angle;
    cv::Vec4i max_seg;
    if (segments_.size() == 0) {
        final_result_.sideline_valid_ = false;
        max_seg = cv::Vec4i(0,0,0,0);
    }
    else {
        final_result_.sideline_valid_ = true;
        
        double angle;
        double max_len = 0;
        for (int i=0; i<segments_.size(); i++) {
            if (segments_[i][0] == segments_[i][2]) {
                angle = 90;
            }
            else {
                double dx = segments_[i][0]-segments_[i][2]; 
                double dy = segments_[i][1]-segments_[i][3];
                angle = atan(dy/dx);
                if (angle < 0) {
                    angle += 180;
                }
                // cout<<angle<<endl;
                if (angle > 135 || angle < 45) {
                    double t_len = sqrt(dx*dx + dy*dy);
                    // cout<<t_len<<t_len<<endl;
                    if (t_len > max_len) {
                        max_len = t_len;
                        // cout<<"yaya"<<endl;
                        max_seg = segments_[i];
                        // cout<<"ya"<<max_seg<<endl;
                        final_result_.sideline_angle_ = angle;
                        final_result_.sideline_center_ = cv::Point2i(
                            segments_[i][0] + 0.5*dx,
                            segments_[i][1] + 0.5*dy
                        );
                    }
                    // final_segs.push_back(segments_[i]);
                    // final_segs_angle.push_back(angle);
                }

            }
        }      
        // ABORT NMS !
        // SegmentNms(final_segs, nms_flags, nms_thre_);
    }


#ifdef ROBOCUP
    StandardHough(sideline_mor_treated_binary_image_, lines_);


    std::vector<cv::Vec3f> final_lines;
    cv::Vec3f  t_vec;
    // judge line result
    if (lines_.size() == 0) {
        final_result_.sideline_valid_ = false;   
    }
    else {
        // do sth magic to delete the same line
        final_result_.sideline_valid_ = true;

        double angle;
        for (int i=0; i<lines_.size(); i++) {
            angle = lines_[i][1]/CV_PI*180.;
            if (angle < 135 && angle > 45) {
                // cout<<"yayaya"<<angle<<endl;
                final_lines.push_back(cv::Vec3f(lines_[i][0], lines_[i][1], angle));
            }
        }
#if 0
        for (int i=0; i<lines_.size(); i++) {
            // cout<<lines_[i][0]<<' '<<lines_[i][1]<<endl;
            cout<<lines_[i][1]/CV_PI*180<<endl;
        }
#endif
    }

    // judge the final line result
    if (final_lines.size() == 0) {
        final_result_.sideline_valid_ = false;
    }
    else {
        // compute, draw and show
        double final_rho =  0.0;
        double final_theta = 0.0;
        double final_angle = 0.0;
        cv::Mat for_line_result = src_image_.clone();
        for (int i=0; i<final_lines.size(); i++) {
            final_rho   += final_lines[i][0];
            final_theta += final_lines[i][1];
            final_angle += final_lines[i][2];

            cv::Point2i a, b;
            double sin_theta = sin(final_lines[i][1]);
            double cos_theta = cos(final_lines[i][1]);
            a.x = 0;
            a.y = final_lines[i][0]/sin_theta;

            b.x = final_lines[i][0]/cos_theta;
            b.y = 0;
            cv::line(for_line_result, a, b, cv::Scalar(255, 0, 0), 1);
        }
        final_rho /= final_lines.size();
        final_theta /= final_lines.size();
        final_angle /= final_lines.size();
        final_result_.sideline_angle_ = final_angle;

        double sin_theta = sin(final_theta);
        double cos_theta = cos(final_theta);
        final_result_.sideline_center_.x = src_image_.cols/2;
        final_result_.sideline_center_.y = (final_rho-final_result_.sideline_center_.x*cos_theta)/sin_theta;
    
        cv::circle(for_line_result, final_result_.sideline_center_, 3, cv::Scalar(0, 255, 0), 3);
        cv::Point2i a, b;
        a.x = 0;
        a.y = final_rho/sin_theta;

        b.x = final_rho/cos_theta;
        b.y = 0;
        cv::line(for_line_result, a, b, cv::Scalar(0, 0, 255), 1);

        cout<<"get final line!"<<endl
            <<"rho: "<<final_rho<<endl
            <<"theta: "<<final_theta<<endl
            <<"angle: "<<final_angle<<endl<<endl;

        SHOW_IMAGE("final_line_result", for_line_result);
    }
#endif    

    // dynamic cast 
    (*dynamic_cast<RobocupResult_HK*>(output_result)) = final_result_;

#ifdef ADJUST_PARAMETER
    WriteImg(src_image_, "src_img", start_file_num_);
    // paint result on the src_image_ before being written
    for (int i=0; i<segments_.size(); i++) {
        cv::line(src_image_, cv::Point(segments_[i][0], segments_[i][1]), cv::Point(segments_[1][2], segments_[i][3]), cv::Scalar(0, 0, 255));        
    }
    // cout<<max_seg<<endl;
    if (final_result_.sideline_valid_) {
        cv::line(src_image_, cv::Point(max_seg[0], max_seg[1]), cv::Point(max_seg[2], max_seg[3]), cv::Scalar(0, 255, 0), 3);
    }
    SHOW_IMAGE("result", src_image_);
    WriteImg(src_image_, "center_img", start_file_num_++);
#endif
}

cv::Mat RobocupVision_HK::GetUsedChannel(cv::Mat& src_img, int flag) {
    cv::Mat t;
    cv::Mat t_cs[3];
    switch (flag) {
        case 0:
        case 1:
        case 2:
            cv::cvtColor(src_img, t, CV_BGR2HSV_FULL);
            cv::split(t, t_cs);
            return t_cs[flag];
        case 3:
        case 4:
        case 5:
            cv::cvtColor(src_img, t, CV_BGR2Lab);
            cv::split(t, t_cs);
            return t_cs[flag-3];
    }
}

cv::Mat RobocupVision_HK::MorTreate(cv::Mat binary_image) {
    cv::Mat hori_erode_kernal = cv::getStructuringElement(MORPH_RECT, cv::Size(sideline_hori_kernal_size_, 1));
    cv::Mat hori_erode_result;
    
    cv::erode(binary_image, hori_erode_result, hori_erode_kernal);
    SHOW_IMAGE("hori_erode", hori_erode_result);

    cv::Mat mor_gradiant;
    cv::Mat grad_kernal = cv::getStructuringElement(MORPH_RECT, cv::Size(mor_kernal_size_*2+1, mor_kernal_size_*2+1));
    cv::morphologyEx(hori_erode_result, mor_gradiant, MORPH_GRADIENT, grad_kernal);
    SHOW_IMAGE("mor_gradiant", mor_gradiant);

    return mor_gradiant;
}

void RobocupVision_HK::StandardHough(cv::Mat mor_gradiant, std::vector<cv::Vec2f>& lines) {
    HoughLines(mor_gradiant, lines, 1, CV_PI/180, line_vote_thre_, 0, 0);
    return ;
}

void RobocupVision_HK::PbbHough(cv::Mat binary_image, std::vector<cv::Vec4i>& lines) {
    HoughLinesP(binary_image, lines, 1, CV_PI/180, line_vote_thre_, 0, 0);
}

void RobocupVision_HK::LoadEverything() {
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
    std::string line_words;
    std::cout<<"Loading Parameters..."<<std::endl;
    while (in_file >> line_words) {
        std::cout<<line_words<<std::endl;
        std::istringstream ins(line_words);
        switch(i++) {
            case 0:
                ins >> gaus_kernal_size_;
                break;
            case 1:
                ins >> sideline_min_;
                break;
            case 2:
                ins >> sideline_max_;
                break;
            case 3:
                ins >> sideline_hori_kernal_size_;
                break;
            case 4:
                ins >> mor_kernal_size_;
                break;
            case 5:
                ins >> line_vote_thre_;
                break;
        }
    }
}

void RobocupVision_HK::StoreParameters() {
    std::ofstream out_file("./7.txt");
    if (!out_file) {
        cerr<<"Error:"<<__FILE__
            <<":line"<<__LINE__<<endl
            <<"     Complied on"<<__DATE__
            <<"at"<<__TIME__<<endl;
    }

    std::cout<<"Storing Parameters.."<<std::endl;
    out_file << setw(3) << setfill('0') << gaus_kernal_size_ 
                                    << "___gaus_kernal_size_"<<std::endl;
    out_file << setw(3) << setfill('0') << sideline_min_ 
                                    << "___sideline_min_"<<std::endl;
    out_file << setw(3) << setfill('0') << sideline_max_ 
                                    << "___sideline_max_"<<std::endl;
    out_file << setw(3) << setfill('0') << sideline_hori_kernal_size_ 
                                    << "___sideline_hori_kernal_size_"<<std::endl;
    out_file << setw(3) << setfill('0') << mor_kernal_size_ 
                                    << "___mor_kernal_size_"<<std::endl;
    out_file << setw(3) << setfill('0') << line_vote_thre_ 
                                    << "___line_vote_thre_";
}

void RobocupVision_HK::set_all_parameters(AllParameters_HK ap) {
    gaus_kernal_size_          = ap.gaus_kernal_size          ;                    
    sideline_min_              = ap.sideline_min              ;                  
    sideline_max_              = ap.sideline_max              ;       
    sideline_hori_kernal_size_ = ap.sideline_hori_kernal_size ;               
    mor_kernal_size_           = ap.mor_kernal_size           ;       
    line_vote_thre_            = ap.line_vote_thre            ;       
}

void RobocupVision_HK::WriteImg(cv::Mat src, string folder_name, int num) {
    std::stringstream t_ss;
    std::string path = "../source/data/con_img/";
    if (start_file_num_ <= max_file_num_) {
        path += folder_name;
        path += "/";

        t_ss << num;
        path += t_ss.str();
        
        path += ".jpg";
        cv::imwrite(path, src);
    }
}

// void RobocupVision_HK::SegmentNms(std::vector<cv::Vec4i>& segments_in, std::vector<bool>& out_flags, double nms_thre) {
//     sort(segments_in.begin(), segments_in.end(), SortSegments);

//     for (size_t i=0; i<segments_in.size(); i++) {
//         for (size_t j=0; j<segments_in.size(); j++) {
//             if (SegmentsIou(segments_in[i], segments_in[j]) < nms_thre) {
//                 out_flags[j] = false;
//             }
//         }
//     }
// }

// bool RobocupVision_HK::SortSegments(cv::Vec4i s_1, cv::Vec4i s_2) {
//     double len_1 = sqrt(
//         (s_1[0]-s_1[2])*(s_1[0]-s_1[2]) 
//     +   (s_1[1]-s_1[3])*(s_1[1]-s_1[3])  
//     );
//     double len_2 = sqrt(
//         (s_2[0]-s_2[2])*(s_2[0]-s_2[2]) 
//     +   (s_2[1]-s_2[3])*(s_2[1]-s_2[3])  
//     );

//     return len_1 >= len_2;
// }

// double RobocupVision_HK::SegmentsIou(cv::Vec4i s_1, cv::Vec4i s_2) {
//     cv::Point2f mid_1(
//         (s_1[0]-s_1[2])/2.,
//         (s_1[1]-s_1[3])/2.
//     );
//     cv::Point2f mid_2(
//         (s_2[0]-s_2[2])/2.,
//         (s_2[1]-s_2[3])/2.
//     );

//     return sqrt((mid_1.x-mid_2.x)*(mid_1.x-mid_2.x) + (mid_1.y-mid_2.y)*(mid_1.y-mid_2.y));
// }
