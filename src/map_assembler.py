#!/usr/bin/env python
"""
Zhiang Chen
Jan 2021
map_assembler reads pcd files, assembles the point clouds, and display the assembled point cloud.
"""
import open3d as o3d
import rospkg
import os
import rospy
import numpy as np
from utils.open3d_ros_conversion import *
from sensor_msgs.msg import PointCloud2

def assemble_map():
    rp = rospkg.RosPack()
    pkg_path = rp.get_path('target_mapping')
    pcd_path = os.path.join(pkg_path, 'pcd')
    pcd_files = [os.path.join(pcd_path, f) for f in os.listdir(pcd_path) if f.endswith('.pcd')]
    if len(pcd_files) == 0:
        return 0
    points = []
    colors = []
    for pcd_file in pcd_files:
        pcd = o3d.io.read_point_cloud(pcd_file)
        points.append(np.asarray(pcd.points))
        colors.append(np.asarray(pcd.colors))

    map_points = np.concatenate(points, axis=0)
    map_colors = np.concatenate(colors, axis=0)
    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(map_points)
    map_pcd.colors = o3d.utility.Vector3dVector(map_colors)
    return map_pcd

def save_o3d_pcd(o3d_pcd, file_name):
    assert file_name.endswith('.pcd')
    o3d.io.write_point_cloud(file_name, o3d_pcd)

def display_pcd(pcd=None, file_name=''):
    if pcd == None:
        assert file_name.endswith('.pcd')
        pcd = o3d.io.read_point_cloud(file_name)
    o3d.visualization.draw_geometries([pcd])

class MapAssembler(object):
    def __init__(self):
        self.map_pub = rospy.Publisher("/pbr_map", PointCloud2, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1), self.timerCallback)
        #self.pc_sub = rospy.Subscriber('/rtabmap/cloud_map', PointCloud2, self.callback, queue_size=1)
        rospy.loginfo("map_assembler has been initialized!")

    def callback(self, ros_pc):
        print(len(ros_pc.data))
        o3d_pcd = convertCloudFromRosToOpen3d(ros_pc)
        save_o3d_pcd(o3d_pcd, '0.pcd')

    def timerCallback(self, timer):
        o3d_pc = assemble_map()
        if o3d_pc == 0:
            return 0
        ros_pc = convertCloudFromOpen3dToRos(o3d_pc, 'map')
        self.map_pub.publish(ros_pc)



if __name__ == '__main__':
    rospy.init_node('map_assember', anonymous=False)
    ma = MapAssembler()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node killed!")
