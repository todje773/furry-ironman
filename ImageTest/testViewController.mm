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

std::vector <Eigen::Vector4d> points3D;

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
    
    // All the variables! ALL OF THEM!
    cv::Mat descriptors, h1, h2, img_keypoints, img_matches, fun, camProOne, camProTwo, points, points2, n;
    std::vector<cv::DMatch> matches, matches2, better;
    std::vector<cv::KeyPoint> keypoints, keypoints2;
    std::vector<cv::Point2f> k, k1;
    std::vector<uchar> inliers;

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
            k1.push_back(keypoints2[better[i].trainIdx].pt);
            k.push_back(keypoints[better[i].queryIdx].pt);
        }
        
        // Distortion and calibration matrices, pre-computed with Matlab
        double tmp[5][1] = {0.01519, -0.05698, 0.00683, 0.01324, 0.0};
        cv::Mat distortion = cv::Mat(5, 1, CV_64F, tmp);
        double temp[3][3] = {{432.02360, 0.0, 177.67501}, {0.0, 473.77855, 134.13891}, {0.0, 0.0, 1.0}};
        cv::Mat camMatrix= cv::Mat(3, 3, CV_64F, temp);
        
        if (k.size() > 0 && k1.size() > 0) {
            // From here on everything is magic, even though by now we have determined that magic is not real
            Matrix3d K(Matrix3d::Identity());
            
            K(0,0) = 432.02360;
            K(0,1) = 0.0;
            K(0,2) = 177.67501;
            K(1,1) = 473.77855;
            K(1,2) = 134.13891;
            
            fun = cv::findFundamentalMat(k, k1, inliers, cv::FM_RANSAC, 1.0, 0.9999);
            size_t inlierCount = 0;
            for (size_t i = 0; i < inliers.size(); ++i) {
                if (inliers[i] > 0) {
                    ++inlierCount;
                }
            }
            
            Eigen::Matrix3d Kinv = K.inverse();
            
            cv::Mat inliers1(inlierCount, 2, CV_64F);
            cv::Mat inliers2(inlierCount, 2, CV_64F);
            
            size_t ix = 0;
            Eigen::Vector3d p2d_h;
            
            for (int i = 0; i < inliers.size(); ++i) {
                if (inliers[i] > 0) {
                    p2d_h << k.at(i).x, k.at(i).y, 1;
                    p2d_h = Kinv*p2d_h;
                    inliers1.at<double>(ix, 0) = p2d_h(0)/p2d_h(2);
                    inliers1.at<double>(ix, 1) = p2d_h(1)/p2d_h(2);
                    p2d_h << k1.at(i).x, k1.at(i).y, 1;
                    p2d_h = Kinv*p2d_h;
                    inliers2.at<double>(ix, 0) = p2d_h(0)/p2d_h(2);
                    inliers2.at<double>(ix, 1) = p2d_h(1)/p2d_h(2);
                    ++ix;
                }
            }
            
            // Prints number of inliers
            // NSLog(@"(%i, %i)  (%i, %i)", inliers1.rows, inliers1.cols, inliers2.rows, inliers2.cols);
            
            if (inlierCount >= 8) {
                cv::Mat E = cv::findFundamentalMat(inliers1, inliers2, cv::FM_8POINT);
                for (int i = 0; i < E.rows; i++) {
                    for (int j = 0; j < E.cols; j++) {
                        NSLog(@"Ess: %f", E.at<double>(i, j));
                    }
                    NSLog(@"");
                }
                Eigen::Matrix3d R;
                Eigen::Vector3d t;
                points = (cv::Mat) k;
                points2 = (cv::Mat) k1;
                decomposeEssentialMatrix(E, inliers1, inliers2, &R, &t);
                
                if (R.determinant() < 0) {
                    NSLog(@"Found negative determinant, recomputing E");
                    decomposeEssentialMatrix(-E, points, points2, &R, &t);
                }
                
                NSLog(@"translation: %f %f %f", t.x(), t.y(), t.z());
                
                NSLog(@"rotation:");
                for (int i = 0; i < R.RowsAtCompileTime; i++) {
                    for (int j = 0; j < R.ColsAtCompileTime; j++) {
                        NSLog(@"%f", R(i, j));
                    }
                }
                
                camProOne = getProjectionMatrix(camMatrix, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
                camProTwo = getProjectionMatrix(camMatrix, R, t);
                
                std::vector<Eigen::Matrix<double, 3, 4>> P(2);
                std::vector<Eigen::Vector3d> p(2);
                
                P[0].block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
                P[0].col(3) = Eigen::Vector3d::Zero();
                P[1].block(0,0,3,3) = R;
                P[1].col(3) = t;
                
                for (int i = 0; i < inliers1.rows; i++) {
                    p[0] << inliers1.at<double>(i,0), inliers1.at<double>(i,1), 1;
                    p[1] << inliers2.at<double>(i,0), inliers2.at<double>(i,1), 1;
                    
                    points3D.push_back(triangulate(P, p));
                }
                
                // Prints the inliers 3D points
                //                    for (int i = 0; i < points3D.size(); i++) {
                //                        NSLog(@"(%f, %f, %f, %f)", points3D[i].x(), points3D[i].y(), points3D[i].z(), points3D[i].w());
                //                    }
            }
        }
        drawMatches(mat, keypoints, mat1, keypoints2, better, img_matches);
        image = [self UIImageFromCVMat:img_matches];
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

cv::Mat getProjectionMatrix(cv::Mat intrinsicMatrix, Matrix3d rotationMatrix, Vector3d translationVector) {
    cv::Mat cameraMatrix = cvCreateMat(3,4,CV_64FC1);
    cv::Mat rotationTranslationMatrix = cvCreateMat(3,4,CV_64FC1);
    
    // Glue the rotation and translation matrices together
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            rotationTranslationMatrix.at<double>(i, j) = rotationMatrix(i, j);
        }
    }
    
    for(int i=0; i<3; i++) {
        rotationTranslationMatrix.at<double>(i, 3) = translationVector(i, 0);
    }
    
        for (int i = 0; i < rotationTranslationMatrix.rows; i++) {
            for (int j = 0; j < rotationTranslationMatrix.cols; j++) {
                NSLog(@"%f", rotationTranslationMatrix.at<double>(i, j));
            }
            NSLog(@"\n");
        }

    
    cv::Mat mul = intrinsicMatrix * rotationTranslationMatrix;
    return mul;
}

//
//-(Eigen::Matrix<double, 3, 4>) calcProj:(cv::Mat) fundamental {
//     
//    return NULL;
//}


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

bool testPointInFrontOfCameras(const Eigen::Matrix3d R, const
                               Eigen::Vector3d& t, const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) {
    std::vector<Eigen::Matrix<double, 3, 4>> P(2);
    std::vector<Eigen::Vector3d> p(2);
    
    P[0].block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    P[0].col(3) = Eigen::Vector3d::Zero();
    P[1].block(0,0,3,3) = R;
    P[1].col(3) = t;
    
    p[0] = p1;
    p[1] = p2;
    
    Eigen::Vector4d p3d_h = triangulate(P, p);
    
    if (p3d_h(2) / p3d_h(3) < 0) return false;
    
    Eigen::Vector3d p2d_h = P[1]*p3d_h;
    
    if (p2d_h(2) < 0) return false;
    return true;
}

int countPointsInFrontOfCameras(const Eigen::Matrix3d R, const
                                Eigen::Vector3d& t, const cv::Mat& points1, const cv::Mat& points2) {
    int count = 0;
    Eigen::Vector3d test1, test2;
    for (size_t i = 0; i < points1.rows; ++i) {
        test1 << points1.at<double>(i,0), points1.at<double>(i,1), 1;
        test2 << points2.at<double>(i,0), points2.at<double>(i,1), 1;
        
        if (testPointInFrontOfCameras(R, t, test1, test2)) ++count;
    }
    return count;
}

void decomposeEssentialMatrix(const cv::Mat& F, const cv::Mat&
                              points1, const cv::Mat& points2, Eigen::Matrix3d* R, Eigen::Vector3d* t) {
    Eigen::Matrix3d E;
    
    for (size_t r = 0; r < 3; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            E(r,c) = F.at<double>(r,c);
        }
    }
    
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU |
                                          Eigen::ComputeFullV);
    
    Eigen::Vector3d svals = svd.singularValues();
    
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    //Phil - Fix a negative determinant
    Eigen::Matrix3d tempR = U*V.transpose();
    
    if(tempR.determinant() < 0.0) {
        NSLog(@"decomposeEssentialMatrix: Rotation matrix has negative determinant! Fixing it");
        V.col(2) *= -1; //Negate the last column
        //tempR = svd.matrixU()*V.transpose();
    }
    
    Eigen::Matrix3d W(Eigen::Matrix3d::Zero());
    W(0,1) = -1;
    W(1,0) = 1;
    W(2,2) = 1;
    
    Eigen::Vector3d u3 = U.col(2);
    
    Eigen::Matrix3d R1 = U*W*V.transpose();
    Eigen::Matrix3d R2 = U*W.transpose()*V.transpose();
    
    int maxCount = 0;
    int thisCount = countPointsInFrontOfCameras(R1, u3, points1, points2);
    
    if (thisCount > maxCount) {
        NSLog(@"Essential Matrix Decomposition: R1, +u3");
        *R = R1;
        *t = u3;
        maxCount = thisCount;
    }
    thisCount = countPointsInFrontOfCameras(R1, -u3, points1, points2);
    if (thisCount > maxCount) {
        NSLog(@"Essential Matrix Decomposition: R1, -u3");
        *R = R1;
        *t = -u3;
        maxCount = thisCount;
    }
    thisCount = countPointsInFrontOfCameras(R2, u3, points1, points2);
    if (thisCount > maxCount) {
        NSLog(@"Essential Matrix Decomposition: R2, +u3");
        *R = R2;
        *t = u3;
        maxCount = thisCount;
    }
    thisCount = countPointsInFrontOfCameras(R2, -u3, points1, points2);
    if (thisCount > maxCount) {
        NSLog(@"Essential Matrix Decomposition: R2, -u3");
        *R = R2;
        *t = -u3;
        maxCount = thisCount;
    }
    
    int numInFront = maxCount;
    int numBehind = points1.rows - numInFront;
    
    NSLog(@"Found %i points in front of cameras, but %i behind", numInFront, numBehind);
}

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end