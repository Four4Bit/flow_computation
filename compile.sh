#!/bin/sh

CODEFILE=${1:-'optical_flow.cpp'}
BINARYFILE=${2:-'optical_flow'}

OPENCV_INSTALL="D:/opencv4.5/opencv/build/x64/vc15/"


g++ -std=c++11 -ID:/opencv4.5/opencv/build/include -LD:/opencv4.5/opencv/build/x64/vc15/lib -g -o optical_flow.exe optical_flow.cpp -lopencv_calib3d -lopencv_core -lopencv_cudaarithm -lopencv_cudabgsegm -lopencv_cudacodec -lopencv_cudafeatures2d -lopencv_cudafilters -lopencv_cudaimgproc -lopencv_cudalegacy -lopencv_objdetect -lopencv_cudaoptflow -lopencv_cudastereo -lopencv_cudawarping -lopencv_cudev -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_videoio -lopencv_video -lopencv_videostab
