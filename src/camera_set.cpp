#include "camera_set.h"

double camera_set::get_depth(double xl, double xr)
{
    double d = xl - xr; //视差
    double depth = baseline * fx / d;

    return depth;
}

Eigen::Vector3d camera_set::pixel2camera(const Eigen::Vector2d &p_p, double depth)
{
    // double x ,y,z;
    // cout<<" p_pfhdfghdfghdfh"<<p_p<<endl;
    // x= (p_p ( 0,0 )-cx_ ) /fx_;
    //  y=(p_p ( 1,0 )-cy_ ) /fy_;
    //  z=depth*(1/sqrt(pow(x,2)+pow(y,2)+1));
    return Eigen::Vector3d(
        (p_p(0, 0) - cx) * depth / fx,
        (p_p(1, 0) - cy) * depth / fy,
        depth
        // x,y,z
    );
}

Eigen::Vector3d camera_set::camera2drone(const Eigen::Vector3d &p_p,
                                         const Eigen::Vector3d &p_drone)
{
    Eigen::Vector3d p_tra;
    p_tra(0, 0) = p_p(2, 0) - 0.52 + p_drone(0, 0);
    //  cout<<" p_pfhdfghdfghdfh"<<p_p<<endl;
    p_tra(1, 0) = p_p(0, 0) + 0.095 + p_drone(1, 0);
    p_tra(2, 0) = p_p(1, 0) + p_drone(2, 0);
    // Eigen::Matrix3d E;
    //  E<<0.26,0,0,0,-0.0475,0,0,0,0;
    return p_tra;
}

Eigen::Vector3d camera_set::pixel2drone(const Eigen::Vector2d &p_p,
                                        double depth,
                                        const Eigen::Vector3d &p_drone)
{
    return camera2drone(pixel2camera(p_p, depth), p_drone);
}

Eigen::Vector3d camera_set::pixel2world(const Eigen::Vector2d &p_p,
                                        double depth,
                                        const Eigen::Vector3d &p_drone_world,
                                        Eigen::Matrix3d matrix_rotation)
{
    Eigen::Vector3d p_tra, p_camera;

    p_camera = (pixel2camera(p_p, depth));
    p_tra(0, 0) = p_camera(2, 0) - 0.52;
    p_tra(1, 0) = p_camera(0, 0) + 0.0475;
    p_tra(2, 0) = p_camera(1, 0);
    return matrix_rotation * p_tra + p_drone_world;
}

void camera_set::GrabStereo(const sensor_msgs::ImageConstPtr &msgLeft,
                            const sensor_msgs::ImageConstPtr &msgRight)
{
    // ROS_INFO("Image grab.");
    circles_left.resize(100);
    circles_right.resize(100);
    cv_bridge::CvImageConstPtr cv_ptrLeft;
    try
    {
        cv_ptrLeft = cv_bridge::toCvShare(msgLeft);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrRight;
    try
    {
        cv_ptrRight = cv_bridge::toCvShare(msgRight);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    imLeft_raw = cv_ptrLeft->image;
    imRight_raw = cv_ptrRight->image;
    if (!imLeft_raw.data || !imRight_raw.data)
    {
        return;
    }

    cv::Mat channel_left[3];
    cv::Mat channel_right[3];
    split(imLeft_raw, channel_left);
    channel_left[0] = channel_left[0].mul(0.1 * g_nHm);                    // B    (mul: per-element matrix multiplication)
    channel_left[1] = channel_left[1].mul(0.1 * g_nHm);                    // G
    channel_left[2] = channel_left[2] - channel_left[0] - channel_left[1]; // R
    channel_left[2] = 3 * channel_left[2];
    channel_left[2].copyTo(imLeft_tmp);
    split(imRight_raw, channel_right);
    channel_right[0] = channel_right[0].mul(0.1 * g_nHm);                      // B    (mul: per-element matrix multiplication)
    channel_right[1] = channel_right[1].mul(0.1 * g_nHm);                      // G
    channel_right[2] = channel_right[2] - channel_right[0] - channel_right[1]; // R
    channel_right[2] = 3 * channel_right[2];
    channel_right[2].copyTo(imRight_tmp);
    
    GaussianBlur(imLeft_tmp, imLeft_tmp, Size(9, 9), 2, 2);
    GaussianBlur(imRight_tmp, imRight_tmp, Size(9, 9), 2, 2);

    cv::threshold(imLeft_tmp, imLeft_tmp, 125, 255, cv::THRESH_BINARY_INV);
    cv::threshold(imRight_tmp, imRight_tmp, 125, 255, cv::THRESH_BINARY_INV);

    std::vector<std::vector<Point>> contours_left;
    std::vector<std::vector<Point>> contours_right;
    findContours(imLeft_tmp, contours_left, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    findContours(imRight_tmp, contours_right, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    int idex_left, idex_right;
    idex_left = contours_left.size() - 2;
    idex_right = contours_right.size() - 2;
    if (idex_left < 0 || idex_right < 0) return;
    size_t count_left = contours_left[idex_left].size();
    size_t count_right = contours_right[idex_right].size();
    if (count_left < 40 || count_left > 1000 || count_right < 40 || count_right > 1000) return;
    Mat pointsf_left, pointsf_right;
    Mat(contours_left[idex_left]).convertTo(pointsf_left, CV_32F);
    Mat(contours_right[idex_right]).convertTo(pointsf_right, CV_32F);
    RotatedRect box_left = fitEllipse(pointsf_left);
    RotatedRect box_right = fitEllipse(pointsf_right);

    vector<Mat> channels;
    for (int i = 0; i < 3; i++) {
        channels.push_back(imLeft_tmp);
    }
    merge(channels, imLeft_out);
    ellipse(imLeft_out, box_left, Scalar(0, 0, 255), 3, 8);
    imshow("HoughResult_left", imLeft_out);
    waitKey(2);

    double depth = get_depth(box_left.center.x, box_right.center.x);
    result << box_left.center.x, box_left.center.y, depth;
    
    // HoughCircles(imLeft_tmp, circles_left, HOUGH_GRADIENT, 1, imLeft_tmp.rows / 5, 200, 30, 0, 0);
    // HoughCircles(imRight_tmp, circles_right, HOUGH_GRADIENT, 1, imRight_tmp.rows / 5, 200, 30, 0, 0);

    // if (circles_left.size() > 1 || circles_right.size() > 1) {
    //     imLeft_out = imLeft_tmp.clone();
    //     imRight_out = imRight_tmp.clone();
    //     imshow("HoughResult_right", imRight_out);
    //     imshow("HoughResult_left", imLeft_out);
    //     waitKey(2);
    //     return;
    // }

    // for (size_t i = 0; i < 1; i++)
    // {
    //     Point center_right(cvRound(circles_right[i][0]), cvRound(circles_right[i][1]));
    //     int radius_right = cvRound(circles_right[i][2]);

    //     Point center_left(cvRound(circles_left[i][0]), cvRound(circles_left[i][1]));
    //     int radius_rleft = cvRound(circles_left[i][2]);

    //     circle(imLeft_tmp, center_left, 3, Scalar(255, 255, 255), -1, 8, 0);
    //     // 绘制圆轮廓
    //     circle(imLeft_tmp, center_left, radius_rleft, Scalar(255, 255, 255), 3, 8, 0);

    //     // 绘制圆中心
    //     circle(imRight_tmp, center_right, 3, Scalar(255, 255, 255), -1, 8, 0);
    //     // 绘制圆轮廓
    //     circle(imRight_tmp, center_right, radius_right, Scalar(255, 255, 255), 3, 8, 0);
    // }

    // for (size_t i = 0; i < 1; i++)
    // {
    //     double depth = get_depth(circles_left[i][0], circles_right[i][0]);
    //     if (depth < 0 || depth == INFINITY)
    //     {
    //         return;
    //     }
    //     result << circles_left[i][0], circles_left[i][1], depth;
    // }
}

void camera_set::draw()
{
    if (!imLeft_tmp.data || !imRight_tmp.data)
    {
        return;
    }
    // cv::imshow("HoughResult_right", imRight_tmp);
    cv::imshow("HoughResult_left", imLeft_tmp);
}

void camera_set::print()
{
    cout << "cx " << cx << endl;
}

// int main(int argc, char **argv)
// {
//   camera_set camera_set1;
//   camera_set1.print();

//   return 0;
// }
