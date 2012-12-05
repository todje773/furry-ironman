//
//  testViewController.h
//  ImageTest
//
//  Created by Jessica Todd on 12/4/12.
//  Copyright (c) 2012 Jessica Todd. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface testViewController : UIViewController {
    UIImage *image, *image1;
    cv::Mat mat, mat2, descriptors2;
}

@property (strong, nonatomic) IBOutlet UIImageView *imageView;

@property cv::Mat mat;
@property cv::Mat mat1;
@property (nonatomic, retain) UIImage *image;
@property (nonatomic, retain) UIImage *image1;

@end
