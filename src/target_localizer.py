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
        bboxes = bbox_msg.bounding_boxes
        cones = self.reprojectBBoxes2Cones(bboxes)
        cone_occupancy = self.checkPointsInCones(cones)
        for i, bbox in enumerate(bboxes):
            if not cone_occupancy[i]:
                self.generatePoints(bbox, self.nm)
            else:
                self.updatePoints(bbox)

        variances = self.computeTargetsVariance()
        self.publish_targets()

    def reprojectBBoxes2Cones(self, bboxes):
        pass

    def checkPointsInCones(self, cones):
        # False: no points; True: points
        pass

    def generatePoints(self, bbox, nm):
        # register new target
        pass

    def updatePoints(self, bbox):
        # return information gain
        # if there are multiple targets in the bbox, then update them all individually
        pass


    def computeTargetsVariance(self):
        pass

    def publish_targets(self):
        pass


if __name__ == '__main__':
    rospy.init_node('target_localizer', anonymous=False)
    target_tracker = TargetTracker()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node killed!")