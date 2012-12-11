//
//  testViewController.m
//  NewProject
//
//  Created by Jessica Todd on 11/16/12.
//  Copyright (c) 2012 Jessica Todd. All rights reserved.
//

#import "testViewController.h"

#include <vector>
#include <Eigen/Dense>
using namespace Eigen;

@interface testViewController ()
@end

@implementation testViewController

@synthesize mat, image, mat1, image1;
@synthesize imageView = _imageView;

std::vector <Eigen::Vector4d> points4D;

- (void)viewDidLoad
{
    [super viewDidLoad];
	// Do any additional setup after loading the view, typically from a nib.
    image = [UIImage imageNamed:@"/IMG_0055.JPG"];
    image1 = [UIImage imageNamed:@"/IMG_0056.JPG"];
    mat = [self cvMatFromUIImage:image];
    mat1 = [self cvMatFromUIImage:image1];
    cv::resize(mat, mat, cvSize(352, 288));
    cv::resize(mat1, mat1, cvSize(352, 288));
    cv::cvtColor(mat, mat, CV_BGR2GRAY);
    cv::cvtColor(mat1, mat1, CV_BGR2GRAY);
    
    Matrix3d K(Matrix3d::Identity());
    
    K(0,0) = 432.02360;
    K(0,1) = 0.0;
    K(0,2) = 177.67501;
    K(1,1) = 473.77855;
    K(1,2) = 134.13891;
    
    // All the variables! ALL OF THEM!
    std::vector<uchar> inliers;
    cv::Mat descriptors, img_matches, fun, tri, inliers1, inliers2;
    std::vector<cv::DMatch> matches, matches2, better, final;
    std::vector<cv::KeyPoint> keypoints, keypoints2;
    std::vector<cv::Point2f> points, points2;
    Eigen::Matrix3d fund;
    std::vector <Eigen::Vector4d> points3D;
    
    cv::ORB ex(80, 1.2, 8, 31, 0, 2, 0, 10);
    ex(mat, cv::Mat(), keypoints, descriptors);
    ex(mat1, cv::Mat(), keypoints2, descriptors2);
    
    cv::BFMatcher m(cv::NORM_HAMMING, true);
    
    // Certain numbers made the application crash for no particular reason, opencv is weird
    if ((descriptors2.rows == 0 && descriptors.rows > 0) || (descriptors.rows == 0 && descriptors2.rows > 0)) {
        NSLog(@":K crash!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        return;
    } else {
        // Do the matchy thing!
        m.match(descriptors, descriptors2, matches);
        m.match(descriptors2, descriptors, matches2);
    }
    
    // Need to filter out good matches here (k1->k2 && k2->k1)
    for (int i = 0; i < matches.size(); i++) {
        for (int j = 0; j < matches2.size(); j++) {
            if (matches[i].queryIdx == matches2[j].trainIdx) {
                if (matches[i].trainIdx == matches2[j].queryIdx ) {
                    better.push_back(matches[i]);
                }
            }
        }
    }
    
    // Take the good matches and push them over here
    for (int i = 0; i < better.size(); i++) {
        points2.push_back(keypoints2[better[i].trainIdx].pt);
        points.push_back(keypoints[better[i].queryIdx].pt);
    }
    
    fun = cv::findFundamentalMat(points, points2, inliers, cv::FM_RANSAC, 1.0, 0.9999);
    
    for (int i = 0; i < points.size(); i++) {
        if (inliers[i] > 0) {
            final.push_back(better[i]);
        }
    }
    
    points.clear();
    points2.clear();
    
    for (int i = 0 ; i < final.size(); i++) {
        points2.push_back(keypoints2[final[i].trainIdx].pt);
        points.push_back(keypoints[final[i].queryIdx].pt);
    }
    
    for (int i = 0; i < fun.cols; i++) {
        for (int j = 0; j < fun.rows; j++) {
            fund(i, j) = fun.at<double>(i, j);
        }
    }
    
    std::vector<Eigen::Matrix<double, 3, 4>> P(2);
    std::vector<Eigen::Vector3d> p(2);
    
    P[0].block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    P[0].col(3) = Eigen::Vector3d::Zero();
    P[1] = [self calcProj:fund];
    
    //    for (int i = 0; i < fun.cols; i++) {
    //        for (int j = 0; j < fun.rows; j++) {
    //            NSLog(@"Fundamental matrix: %f", fun.at<double>(i, j));
    //        }
    //    }
    //
    //    for (int i = 0; i < P[0].cols(); i++) {
    //        for (int j = 0; j < P[0].rows(); j++) {
    //            NSLog(@"Projection matrix: %f   %f", P[0](j, i), P[1](j, i));
    //        }
    //    }
    Eigen::Matrix<double, 3, 3> f;
    
    for (int i = 0; i < fun.cols; i++) {
        for  (int j = 0; j < fun.rows; j++) {
            f(i, j) = fun.at<double>(i, j);
        }
    }
    
    Eigen::Vector3d a, b;
    
    for (int i = 0; i < points2.size(); i++) {
        a << points.at(i).x, points.at(i).y, 1;
        b << points2.at(i).x, points2.at(i).y, 1;
        debugStuff(a, b, f);
    }
    
    checkStuff(P[0], P[1], f);
    
    
    //    for (int i = 0; i < P[1].rows(); i++) {
    //        NSLog(@"%f", P[1](i, 3));
    //    }
    
    //    NSLog(@"Points : %li", points.size());
    Eigen::Vector3d p2d_h;
    
    // Need to pull out the points that are inliers into Vector3D's
    // Then write the triangulate method
    for (int i = 0; i < final.size(); i++) {
        //      NSLog(@"1: (%f, %f) 2: (%f, %f)", points.at(i).x, points.at(i).y, points2.at(i).x, points2.at(i).y);
        p[0] << points.at(i).x, points.at(i).y, 1;
        p[1] << points2.at(i).x, points2.at(i).y, 1;
        
        points4D.push_back(triangulate(P, p));
    }
    
    //    for (int i = 0; i < points4D.size(); i++) {
    //        NSLog(@"(%f %f %f %f)", points4D[i].x(), points4D[i].y(), points4D[i].z(), points4D[i].w());
    //    }
    
    for (int i = 0; i < points4D.size(); i++) {
        points3D.push_back(convertCartesian(points4D[i]));
    }
    
    std::vector<cv::Point3d> c;
    cv::Point3d s;
    Eigen::Vector3d result;
    
    for (int i = 0; i < points3D.size(); i++) {
        result = P[0] * points3D[i];
        cv::circle(mat, cvPoint(result.x()/result.z(), result.y()/result.z()), 4, cv::Scalar(255, 83, 96));
        NSLog(@"Original: %f %f", points.at(i).x, points.at(i).y);
        NSLog(@"Calculat: %f %f", result.x()/result.z(), result.y()/result.z());
        //NSLog(@"Triangulised point: %f %f %f %f", points3D[i].x(), points3D[i].y(), points3D[i].z(), points3D[i].w());
    }
    
    
    //drawMatches(mat, keypoints, mat1, keypoints2, final, img_matches);
    image = [self UIImageFromCVMat:mat];
    _imageView.image = image;
}

Eigen::Vector4d triangulate(const std::vector<Eigen::Matrix<double, 3, 4>>&
                            P, const std::vector<Eigen::Vector3d>& p) {
    
    if (P.size() < 2) {
        NSLog(@"Cannot triangulate a point from %li views", P.size());
        return Eigen::Vector4d::Zero();
    }
    
    Eigen::MatrixXd A;
    A.resize(2*P.size(), 4);
    
    Eigen::Vector4d p1, p2, p3;
    double x, y;
    
    for (size_t i = 0; i < P.size(); ++i) {
        p1 = P[i].row(0);
        p2 = P[i].row(1);
        p3 = P[i].row(2);
        x = p[i](0)/p[i](2);
        y = p[i](1)/p[i](2);
        A.row(2*i)   = x*p3 - p1;
        A.row(2*i+1) = y*p3 - p2;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    return svd.matrixV().col(3);
}

Eigen::Matrix<double, 3, 4> joinMatrixVector(Matrix3d m, Vector3d v) {
    Eigen::Matrix<double, 3, 4> join;
    
    // Glue the rotation and translation matrices together
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            join(i, j) = m(i, j);
        }
    }
    
    for(int i=0; i<3; i++) {
        join(i, 3) = v(i, 0);
    }
    
    return join;
}

Eigen::Vector4d convertCartesian(Eigen::Vector4d points) {
    // Gets rid of the w for a set of points (from homogeneous to cartesian)
    Eigen::Vector4d p;
    Eigen::Vector4d ps;
    p.x() = points.x() / points.w();
    p.y() = points.y() / points.w();
    p.z() = points.z() / points.w();
    p.w() = 1.0;
    return p;
}

void debugStuff(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Matrix<double, 3, 3> f) {
    Eigen::Matrix<double, 1, 3> c;
    for (int i = 0; i < b.rows(); i++) {
        for (int j = 0; j < b.cols(); j++) {
            c(j, i) = b(i, j);
        }
    }
    
    Eigen::Matrix<double, 1, 3> d = c * f;
    double i = d * a;
    NSLog(@"%f", i);
}

void checkStuff (Eigen::Matrix<double, 3, 4> p1, Eigen::Matrix<double, 3, 4> p2, Eigen::Matrix<double, 3, 3> f) {
    Eigen::Matrix<double, 4, 3> c, d;
    Eigen::Matrix<double, 4, 4> e;
    
    for (int i = 0; i < p2.rows(); i++) {
        for (int j = 0; j < p2.cols(); j++) {
            c(j, i) = p2(i, j);
        }
    }
    
    d = c * f;
    e = d * p1;
    
    for (int i = 0; i < e.cols(); i++) {
        for (int j = 0; j < e.rows(); j++) {
            NSLog(@"%f", e(i, j));
        }
        NSLog(@"\n");
    }
}

-(Eigen::Matrix<double, 3, 4>) calcProj:(Eigen::Matrix3d) fundamental {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(fundamental, Eigen::ComputeFullU |
                                          Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d skew, mult;
    Eigen::Matrix<double, 3, 4> join, done;
    
    Eigen::Vector3d v = U.col(2);
    
    skew = [self skewSymmetric:v];
    mult = skew * fundamental;
    join = joinMatrixVector(mult, v);
    
    //    NSLog(@"Join: %i  %i", join.rows(), join.cols());
    //
    //    for (int i = 0; i < join.rows(); i++) {
    //        for (int j = 0; j < join.cols(); j++) {
    //            NSLog(@"%f", join(i, j));
    //        }
    //        NSLog(@"\n");
    //    }
    
    return join;
}

-(Eigen::Matrix3d) skewSymmetric:(Eigen::Vector3d) v {
    Eigen::Matrix3d skew;
    
    //    NSLog(@"%i  %i   %d", skew.cols(), skew.rows(), skew.size());
    skew(0, 0) = 0.0;
    skew(0, 1) = -v(2);
    skew(0, 2) = v(1);
    skew(1, 0) = v(2);
    skew(1, 2) = -v(0);
    skew(2, 0) = -v(1);
    skew(2, 1) = v(0);
    
    return skew;
}

- (cv::Mat) cvMatFromUIImage:(UIImage *)i
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(i.CGImage);
    CGFloat cols = i.size.width;
    CGFloat rows = i.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), i.CGImage);
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
    return cvMat;
}

-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                              //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}


- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end