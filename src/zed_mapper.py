#!/usr/bin/env python
"""
Zhiang Chen
Sept 2021
"""

import os
import rospy
import rospkg
from zed_interfaces.srv import start_3d_mapping, start_3d_mappingRequest
from zed_interfaces.srv import stop_3d_mapping, stop_3d_mappingRequest
import target_mapping.msg
import actionlib
from sensor_msgs.msg import PointCloud2
from utils.open3d_ros_conversion import convertCloudFromRosToOpen3d, convertCloudFromOpen3dToRos
import open3d as o3d
import copy
import numpy as np

class Zed_Mapping_Client(object):
    def __init__(self):
        self.mapped = False
        rp = rospkg.RosPack()
        pkg_path = rp.get_path('target_mapping')
        self.pcd_path = os.path.join(pkg_path, 'pcd')

        self.as_start_map = actionlib.SimpleActionServer("/zed_mapping_client/start_mapping", target_mapping.msg.StartZedMappingAction, execute_cb=self.start_mapping_Callback, auto_start=False)
        self.as_start_map.start()
        self.as_stop_map = actionlib.SimpleActionServer("/zed_mapping_client/stop_mapping", target_mapping.msg.StopZedMappingAction, execute_cb=self.stop_mapping_Callback, auto_start=False)
        self.as_stop_map.start()
        self.as_save_map = actionlib.SimpleActionServer("/zed_mapping_client/save_mapping", target_mapping.msg.SaveZedMapAction, execute_cb=self.save_mapping_Callback, auto_start=False)
        self.as_save_map.start()

        self.start_zed_map_srv_client = rospy.ServiceProxy('/zed2/zed_node/start_3d_mapping', start_3d_mapping)
        rospy.wait_for_service('/zed2/zed_node/start_3d_mapping')
        
        self.stop_zed_map_srv_client = rospy.ServiceProxy('/zed2/zed_node/stop_3d_mapping', stop_3d_mapping)
        rospy.wait_for_service('/zed2/zed_node/stop_3d_mapping')
        stop_request = stop_3d_mappingRequest()
        result = self.stop_zed_map_srv_client(stop_request)

        self.pc_sub = rospy.Subscriber('/zed2/zed_node/mapping/fused_cloud', PointCloud2, self.mapCallback, queue_size=1)


        rospy.loginfo("zed_mapping_client has been initialized!")


    def mapCallback(self, pc_msg):
        print('mapping')
        self.pc_map = pc_msg
        self.mapped = True

    def start_mapping_Callback(self, goal):
        print('start ZED mapping')
        start_request = start_3d_mappingRequest()
        start_request.resolution = 0.01
        start_request.max_mapping_range = -1
        start_request.fused_pointcloud_freq = 1
        result = self.start_zed_map_srv_client(start_request)
        self.as_start_map.set_succeeded()
        self.start_mapping = True


    def stop_mapping_Callback(self, goal):
        print('stop ZED mapping')
        rospy.sleep(1.)  # give map subscriber 1 second to save map
        stop_request = stop_3d_mappingRequest()
        result = self.stop_zed_map_srv_client(stop_request)
        self.as_stop_map.set_succeeded()


    def save_mapping_Callback(self, goal):
        print('save ZED mapping')
        pc_map_msg = copy.copy(self.pc_map)
        o3d_pc = convertCloudFromRosToOpen3d(pc_map_msg)
        pts = np.asarray(o3d_pc.points)
        clrs = np.asarray(o3d_pc.colors)
        map_pcd = o3d.geometry.PointCloud()
        map_pcd.points = o3d.utility.Vector3dVector(pts)
        map_pcd.colors = o3d.utility.Vector3dVector(clrs)
        pcd_name = os.path.join(self.pcd_path, "target_zed.pcd")
        o3d.io.write_point_cloud(pcd_name, map_pcd)
        self.as_save_map.set_succeeded()


if __name__ == '__main__':
    rospy.init_node('zed_mapping_client', anonymous=False)
    zed_mapping_client = Zed_Mapping_Client()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node killed!")

