import cv2
import numpy as np
import time 
import matplotlib.pyplot as plt

def generate_window(row, col, image, blockSize):
    window = (image[row:row + blockSize, col:col + blockSize])
    return window

def disparitymap(imgL,imgR):   #standard results

    stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*9,  # Must be a multiple of 16
    blockSize=9,
    P1=9 * 3 * 2,
    P2=64 * 3 * 2,
    disp12MaxDiff=5,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63)

    dispMap = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    dispMap[dispMap == 0] = 0.1  # Small nonzero value

    # plt.imshow(dispMap)
    #plt.show()

    return dispMap

def left_right_disparity(limg,rimg):  #resuts very weak

    # Create stereo matchers
    stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*8,  # Must be a multiple of 16
    blockSize=9,
    P1=9 * 3 * 2,
    P2=64 * 3 * 2,
    disp12MaxDiff=5,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63)

    # Compute left and right disparity maps
    disp_L = stereo.compute(limg, rimg).astype(np.float32) / 16.0
    disp_R = stereo.compute(rimg, limg).astype(np.float32) / 16.0

    # Reproject right disparity to left view
    #disp_R_to_L = cv2.remap(disp_R, np.array(range(disp_R.shape[1])), np.array(range(disp_R.shape[0])), cv2.INTER_LINEAR)

    # Compute absolute difference
    confidence_mask = np.abs(disp_L - disp_R) < 50  # Keep only small differences

    # Apply confidence mask
    filtered_disparity = np.where(confidence_mask, disp_L, 0)

    filtered_disparity[filtered_disparity == 0] = 0.1  # Small nonzero value

    #plt.imshow(filtered_disparity)
    #plt.show()

    return filtered_disparity

def wls_disparity(limg, rimg):

    lmbda = 40
    sigma = 1.7

    #create stereo matcher
    stereo_left = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*10,  # Must be a multiple of 16
    blockSize=5,
    P1=9 * 3 * 2,
    P2=64 * 3 * 2,
    disp12MaxDiff=5,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)

    # Compute disparity maps
    disparity_left = stereo_left.compute(limg, rimg).astype(np.float32) / 16.0
    disparity_right = stereo_right.compute(rimg, limg).astype(np.float32) / 16.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo_left)
    wls_filter.setLambda(lmbda)  # Smoothness strength
    wls_filter.setSigmaColor(sigma)  # Edge-preserving filter strength

    wls_disparity = wls_filter.filter(disparity_left, limg, disparity_map_right=disparity_right)

    confidence = wls_filter.getConfidenceMap()
    wls_disparity[confidence<240] = float('nan')

    plt.imshow(wls_disparity)
    plt.show()

    return wls_disparity

def subpixel_disparity(limg, rimg):

    #create stereo matcher
    stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*8,  # Must be a multiple of 16
    blockSize=9,
    P1=9 * 3 * 2,
    P2=64 * 3 * 2,
    disp12MaxDiff=5,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63)

    # Compute disparity (fixed-point format, multiply by 16 for precision)
    disparity = stereo.compute(limg, rimg).astype(np.float32) / 16.0

    # Apply bilateral filter to smooth disparities
    subpixel_disparity = cv2.bilateralFilter(disparity, d=15, sigmaColor=25, sigmaSpace=25)

    plt.imshow(subpixel_disparity)
    plt.show()

    return subpixel_disparity