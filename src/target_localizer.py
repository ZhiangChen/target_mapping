#!/usr/bin/env python
"""
Zhiang Chen
Feb 2020
Localize targets using particle filter
"""

import sys
import os
import rospy
import rospkg
from darknet_ros_msgs.msg import BoundingBoxes
from darknet_ros_msgs.msg import ObjectCount
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from image_geometry import PinholeCameraModel
from geometry_msgs.msg import PoseStamped, TwistStamped
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import copy
from numpy.linalg import inv
from numpy.linalg import det


class TargetTracker(object):
    def __init__(self, particle_nm=1000):
        self.nm = particle_nm
        self.target_points = [] # a list of Nx3 ndarrays

        camera_info = rospy.wait_for_message("/r200/depth/camera_info", CameraInfo)

        self.pinhole_camera_model = PinholeCameraModel()
        self.pinhole_camera_model.fromCameraInfo(camera_info)

        self.sub_bbox = message_filters.Subscriber('/bbox_tracker/bounding_boxes', BoundingBoxes)
        self.sub_pose = message_filters.Subscriber('/mavros/local_position/pose', PoseStamped)
        # self.sub_vel = message_filters.Subscriber('/mavros/local_position/velocity_local', TwistStamped)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_bbox, self.sub_pose], queue_size=10, slop=0.1)
        # self.ts = message_filters.TimeSynchronizer([self.sub_bbox, self.sub_pose, self.sub_vel], 10)
        self.ts.registerCallback(self.callback)
        print("target_localizer initialized!")

    def callback(self, bbox_msg, pose_msg):
        pose = pose_msg.pose
        bboxes = bbox_msg.bounding_boxes
        for i, bbox in enumerate(bboxes):
            cone = self.reprojectBBoxesCone(bbox)
            if not self.checkPointsInCone(cone, pose):
                self.generatePoints(cone, self.nm)
            else:
                self.updatePoints(bbox)

        variances = self.computeTargetsVariance()
        self.publish_targets()

    def reprojectBBoxesCone(self, bbox):
        x1 = bbox.xmin
        y1 = bbox.ymin
        x2 = bbox.xmax
        y2 = bbox.ymax
        #  ray1  ray2
        #  ray3  ray4
        #  see the camera coordinate system: https://github.com/ZhiangChen/target_mapping
        #  and also the api description: http://docs.ros.org/diamondback/api/image_geometry/html/c++/classimage__geometry_1_1PinholeCameraModel.html#ad52a4a71c6f6d375d69865e40a117ca3
        ray1 = self.pinhole_camera_model.projectPixelTo3dRay((x1, y1))
        ray2 = self.pinhole_camera_model.projectPixelTo3dRay((x2, y1))
        ray3 = self.pinhole_camera_model.projectPixelTo3dRay((x1, y2))
        ray4 = self.pinhole_camera_model.projectPixelTo3dRay((x2, y2))
        ray1 = np.asarray(ray1)/ray1[2]
        ray2 = np.asarray(ray2)/ray2[2]
        ray3 = np.asarray(ray3)/ray3[2]
        ray4 = np.asarray(ray4)/ray4[2]
        cone = np.asarray((ray1, ray2, ray3, ray4))
        #print(cone)
        return cone

    def checkPointsInCone(self, cone, pose):
        # False: no points; True: points
        if len(self.target_points) == 0:
            return False

        ray1, ray2, ray3, ray4 = cone
        #       norm1
        # norm4        norm2
        #       norm3
        norm1 = np.cross(ray1, ray2)
        norm2 = np.cross(ray2, ray3)
        norm3 = np.cross(ray3, ray4)
        norm4 = np.cross(ray4, ray1)
        for points in self.target_points:
            # convert points to camera coordinate system
            pass

        return None

    def generatePoints(self, cone, pose, nm=1000):
        # register new target
        # 1. generate points on unit surface, in camera coordinate system
        # 2. randomize these points by varying elevations, in camera coordinate system
        # 3. convert to world coordinate system
        pass

    def updatePoints(self, bbox):
        # return information gain
        # if there are multiple targets in the bbox, then update them all individually
        pass


    def computeTargetsVariance(self):
        return None

    def publish_targets(self):
        pass

    def deregisterTarget(self, id):
        pass


if __name__ == '__main__':
    rospy.init_node('target_localizer', anonymous=False)
    target_tracker = TargetTracker()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node killed!")