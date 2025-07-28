import numpy as np
import cv2
import matplotlib.pyplot as plt


baseline = 95
focal_lenght = 659.186

def depth_map(dispMap):
    print("Calculating depth....")
    
    depth = (focal_lenght * baseline) / dispMap

    depth = np.clip(depth, 0, 3000)
    depthmap = plt.imshow(depth,cmap='jet_r')
    plt.colorbar(depthmap)
    plt.show()
    
    return depth

