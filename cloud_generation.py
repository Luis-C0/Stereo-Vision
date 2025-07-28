from math import nan
import numpy as np
import open3d as o3d

fx = 659.186  # Focal length in pixels (from camera intrinsics)
fy = 659.186  # Typically the same as f_x
cx = 640  # Optical center x (image width / 2)
cy = 360  # Optical center y (image height / 2)

min_depth = 200  # 40 cm
max_depth = 900  # 100 cm


def get_points(DMap):
    
    height,width = DMap.shape[0], DMap.shape[1]
    points = []

    for row in range(height):
        for col in range(width):

            X = (col - cx) * DMap[row,col] / fx
            Y = (row - cy) * DMap[row,col] / fy
            Z = DMap[row,col]
            points.append([X,Y,Z])
    
    points = np.array(points)

    bounding_box = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=(-300, -500, min_depth),  # (Xmin, Ymin, Zmin)
    max_bound=(300, 500, max_depth))    # (Xmax, Ymax, Zmax)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    filtered_pcd = pcd.crop(bounding_box)

    cl, ind = filtered_pcd.remove_statistical_outlier(nb_neighbors=100,std_ratio=0.5)

    #o3d.visualization.draw_geometries([cl])

    return cl
