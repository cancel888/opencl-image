//
//  main.cpp
//  lab_4_2
//
//  Created by Nikita on 29.04.15.
//  Copyright (c) 2015 Nikita. All rights reserved.
//

#include <iostream>
#include <OpenCL/OpenCL.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <fstream>
#include <string>
#include <stdlib.h>

#include "kernel.cl.h"

using namespace std;

int main() {
    
    dispatch_queue_t dq = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
    
    dispatch_semaphore_t dsema = dispatch_semaphore_create(0);
    
    ifstream in;
    in.open("conf.properties", ios::in);
    
    int type;
    string fileName;
    
    in >> type >> fileName;
    
    cv::Mat image = cv::imread(fileName, 1);
    
    size_t width = image.cols;
    size_t height = image.rows;
    
    if (!image.data) {
        cout << "Image problem.." << endl;
        
        return 0;
    }
    
    char* pixels = (char *) malloc(sizeof(char) * width * height * 4);
    
    cl_image_format format;
    format.image_channel_order = CL_RGB;
    format.image_channel_data_type = CL_UNORM_INT8;
    
    cl_mem input_image = gcl_create_image(&format, width, height, 1, NULL);
    cl_mem output_image = gcl_create_image(&format, width, height, 1, NULL);
    
    if (type == 1) {
        
        dispatch_async(dq, ^{
            
            cl_ndrange range = {
                2,                  // Using a two-dimensional execution.
                {0},                // Start at the beginning of the range.
                {width, height},    // Execute width * height work items.c
                {0}                 // And let OpenCL decide how to divide
                // the work items into work-groups.
            };
            
            const size_t origin[3] = { 0, 0, 0 };
            const size_t region[3] = { width, height, 1 };
            
            gcl_copy_ptr_to_image(input_image, image.data, origin, region);
            
            exec1_kernel(&range, input_image, output_image);
            
            gcl_copy_image_to_ptr(pixels, output_image, origin, region);
            
            dispatch_semaphore_signal(dsema);
            
        });
        
    }
    else if (type == 2) {
        
        dispatch_async(dq, ^{
            
            cl_ndrange range = {
                2,                  // Using a two-dimensional execution.
                {0},                // Start at the beginning of the range.
                {width, height},    // Execute width * height work items.c
                {0}                 // And let OpenCL decide how to divide
                // the work items into work-groups.
            };
            
            const size_t origin[3] = { 0, 0, 0 };
            const size_t region[3] = { width, height, 1 };
            
            gcl_copy_ptr_to_image(input_image, image.data, origin, region);
            
            exec2_kernel(&range, input_image, output_image);
            
            gcl_copy_image_to_ptr(pixels, output_image, origin, region);
            
            dispatch_semaphore_signal(dsema);
            
        });
        
    }
    else {
        cout << "..." << endl;
        
        return 0;
    }
    
    dispatch_semaphore_wait(dsema, DISPATCH_TIME_FOREVER);
    
    vector<int> param;
    param.push_back(CV_IMWRITE_JPEG_QUALITY);
    param.push_back(90);
    
    cv::Mat res((int) height, (int) width, CV_8UC3, pixels);
    
    cv::imwrite("out.jpg", res, param);
    
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    cv::imshow( "Display window", res);
    
    cv::waitKey(0);
    
    clReleaseMemObject(input_image);
    clReleaseMemObject(output_image);
    free(pixels);
    dispatch_release(dq);
    
    return 0;
}

























