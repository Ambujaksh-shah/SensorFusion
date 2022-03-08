# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import torch
import zlib
import open3d as o3d
# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools


# visualize lidar point-cloud
def show_pcl(pcl):

    ####### ID_S1_EX2 START #######     
    vis_lidar = o3d.visualization.VisualizerWithKeyCallback()
    vis_lidar.createwindow(window_name='Lidar PCL view')
    pcd = o3d.geometry.PointCloud() # creating an instance for point cloud
    pcd.points = o3d.utility.Vector3dVector(pcl[:,:3])
    vis_lidar.add_geometry(pcd)
    
    vis_lidar.register_key_callback(262, next_frame_callback)	#262 : Right arrow key 
    vis_lidar.register_key_callback(27, exit_callback)	#27  : Escape
    vis_lidar.update_renderer()
    vis_lidar.poll_events()
    vis_lidar.run() 
    ####### ID_S1_EX2 END #######     
def next_frame_callback(vis_lidar):
    vis_lidar.close()

def exit_callback(vis_lidar):
    vis_lidar.destroy_window()


# visualize range image
def show_range_image(frame, lidar_name):

    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0]

    if len(lidar.ri_return1.range_image_compressed) > 0:
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    
    ri_range = ri[:,:,0]
    ri_intensity = ri[:,:,1]

    ri_range[ri_range<0] = 0.0
    ri_intensity[ri_intensity<0] = 0.0
    ri_range = ((ri_range *255 ) / (np.amax(ri_range) - np.amin(ri_range)))
    ri_intensity = np.percentile(ri_intensity,99)/2 *((ri_intensity*255) /(np.percentile(ri_intensity,99) -            np.percentile(ri_intensity,1)))
    
    img_range_intensity = np.vstack([ri_range, ri_intensity])    
    
    return img_range_intensity


# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs):

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  

    
    bev_discret = (configs.lim_x[1] - configs.lim_x[0])/configs.bev_height
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl_cpy[:,0] = np.int_(np.floor(lidar_pcl_cpy[:,0]/bev_discret))
    lidar_pcl_cpy[:,1] = np.int_(np.floor(lidar_pcl_cpy[:,1]/bev_discret) + (configs.bev_width + 1)/2)
    #show_pcl(lidar_pcl_cpy)
    
    

    intensity_map = np.zeros((configs.bev_height +1, configs.bev_width + 1 ))
    lidar_pcl_cpy[lidar_pcl_cpy[:,3]>1.0, 3] =1.0
    idx_intensity = np.lexsort((-lidar_pcl_cpy[:,3], lidar_pcl_cpy[:,1], lidar_pcl_cpy[:,0]))
    lidar_pcl_top = lidar_pcl_cpy[idx_intensity] 

    _, indices = np.unique(lidar_pcl_cpy[:,0:2], axis=0, return_index = True)
    lidar_pcl_top = lidar_pcl_top[indices]
    intensity_map[np.int_(lidar_pcl_top[:,0]), np.int_(lidar_pcl_top[:,1])] = lidar_pcl_top[:,3]/(np.amax(lidar_pcl_top[:,3])-np.amin(lidar_pcl_top[:,3]))

    ''' #temp Intensity visualization
    intensity = intensity_map *256
    image = intensity.astype(np.uint8)
    while(1):
        cv2.imshow('Image', image)
        if cv2.waitKey(10) & 0xff == 27:
            break
    cv2.destroyAllWindows()
    '''

    height_map = np.zeros((configs.bev_height+1, configs.bev_width+1))
    height_map[np.int_(lidar_pcl_top[:,0]), np.int_(lidar_pcl_top[:,1])] = lidar_pcl_top[:,2]/float(np.abs(configs.lim_z[1]-configs.lim_z[0]))
 
    ''' #temp height map visualization
    image_height = height_map*256
    image = image_height.astype(np.uint8)
    while(1):
        cv2.imshow("Image", image)
        if cv2.waitKey(10) & 0xff == 27:
            break
    cv2.destroyAllWindows()
    '''
   
    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) 
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts
        
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps


