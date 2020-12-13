#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <memory>
#include <fstream>

int main() {
    std::vector<cv::KeyPoint> keypoints;
    cv::Ptr<cv::Feature2D> surf = cv::xfeatures2d::SURF::create();
    cv::Mat image;
    
    image = cv::imread("./lena.png");
    
    cv::Mat mask = cv::Mat(image.size().height, 
                           image.size().width,
                           CV_8UC1, cv::Scalar(255));
    
    cv::Mat descriptors;
    
    surf->detectAndCompute(image, mask, keypoints, descriptors);
    
    cv::drawKeypoints(image, keypoints, image);
    cv::imshow("image", image);
    
    // save to a csv all the descriptors
    std::ofstream out("surf_descriptors.csv");
    for(int i=0; i<descriptors.rows; i++){
        for(int j=0; j<descriptors.cols; j++){
                out << (float)descriptors.at<float>(i, j) <<',';
        }
        out << '\n';
    }
    
    std::cout<<"There are "<<descriptors.rows<<" keypoints"<<std::endl;
    std::cout<<"Each surf descriptor has "<<descriptors.cols<<" values"<<std::endl;
  
    cv::waitKey(0);
    return 0;
}
