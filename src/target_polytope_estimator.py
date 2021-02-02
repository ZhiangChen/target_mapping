#!/usr/bin/env python
"""
Zhiang Chen
Jan 2021
estimate target shape using polytope
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
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import copy
from numpy.linalg import inv
from numpy.linalg import det
import tf
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
import scipy.stats as stats
from visualization_msgs.msg import Marker, MarkerArray
import visualization_msgs
import PyKDL
import target_mapping.msg
import actionlib
from threading import Thread
import ros_numpy
import polytope as PT

class Polyhedron(object):
    def __init__(self, bbox, camera_model, T_world2camera):
        self.boarder = 5  # bbox boarder in pixel
        self.bbox = bbox
        self.pinhole_camera_model = camera_model
        self.bbox_center = ((bbox.xmin+bbox.xmax)/2., (bbox.ymin+bbox.ymax)/2.)
        ray_c = self.pinhole_camera_model.projectPixelTo3dRay(self.bbox_center)
        self.ray_c = np.asarray(ray_c) / ray_c[2]  # ray in camera frame, (x, y, z)
        self.T_world2camera = T_world2camera
        intrinsic_matrix = np.zeros((3, 4))
        intrinsic_matrix[:3, :3] = self.pinhole_camera_model.fullIntrinsicMatrix()
        self.P = np.matmul(intrinsic_matrix, self.T_world2camera)  # projection matrix

    def get_bbox_center_ray(self):
        u, v = self.bbox_center
        a1 = (v*self.P[2, 0] - self.P[1, 0], v*self.P[2, 1] - self.P[1, 1], v*self.P[2, 2] - self.P[1, 2], v*self.P[2, 3] - self.P[1, 3])
        a2 = (u*self.P[2, 0] - self.P[0, 0], u*self.P[2, 1] - self.P[0, 1], u*self.P[2, 2] - self.P[0, 2], u*self.P[2, 3] - self.P[0, 3])
        return a1+a2  # [[a, b, c, d], [e, f, g, h]]

    def get_polyhedron(self):
        cone = self.get_bbox_cone()
        ray1, ray2, ray3, ray4 = cone
        #       norm1
        # norm4        norm2
        #       norm3
        # the order of cross product determines the normal vector direction
        norm1 = np.cross(ray1, ray2)
        norm2 = np.cross(ray2, ray3)
        norm3 = np.cross(ray3, ray4)
        norm4 = np.cross(ray4, ray1)
        # norm * (x - x') = 0, where x' is camera origin, * is dot multiplication
        # halfspace: norm * x >= norm * x'
        origin_c = self.T_world2camera[:3, 3]
        A = np.array([norm1, norm2, norm3, norm4])
        b = - np.array([np.dot(norm1, origin_c), np.dot(norm2, origin_c), np.dot(norm3, origin_c), np.dot(norm4, origin_c)])
        # adding negative sign because the format is different, Ax <= b
        p = PT.Polytope(A, b)
        return p

    def get_bbox_cone(self):
        """
        :return: cone = np.asarray((ray1, ray2, ray3, ray4))
        """
        x1 = self.bbox.xmin - self.boarder
        y1 = self.bbox.ymin - self.boarder
        x2 = self.bbox.xmax + self.boarder
        y2 = self.bbox.ymax + self.boarder
        #  ray1  ray2
        #  ray3  ray4
        #  see the camera coordinate system: https://github.com/ZhiangChen/target_mapping
        #  and also the api description: http://docs.ros.org/diamondback/api/image_geometry/html/c++/classimage__geometry_1_1PinholeCameraModel.html#ad52a4a71c6f6d375d69865e40a117ca3
        ray1 = self.pinhole_camera_model.projectPixelTo3dRay((x1, y1))
        ray2 = self.pinhole_camera_model.projectPixelTo3dRay((x2, y1))
        ray3 = self.pinhole_camera_model.projectPixelTo3dRay((x2, y2))
        ray4 = self.pinhole_camera_model.projectPixelTo3dRay((x1, y2))
        ray1 = np.asarray(ray1)/ray1[2]
        ray2 = np.asarray(ray2)/ray2[2]
        ray3 = np.asarray(ray3)/ray3[2]
        ray4 = np.asarray(ray4)/ray4[2]
        cone = np.asarray((ray1, ray2, ray3, ray4))
        return cone

class TargetPolytope(object):
    def __init__(self, camera_model):
        self.polyhedrons = []
        self.pinhole_camera_model = camera_model

    def add_polyhedron(self, bbox, T_world2camera):
        poly = Pyhedron(bbox, self.pinhole_camera_model, T_world2camera)
        self.polyhedrons.append(poly)

    def get_polytope(self):
        polys = [poly.get_polyhedron() for poly in self.polyhedrons]
        N = len(polys)
        if N > 1:
            polyt = polys[0].intersect(polys[1])
            for i in range(2, N):
                polyt = polyt.intersect(polys[i])

            return polyt
        else:
            return None


class PolytopeEstimator(TargetPolytope):
    def __init__(self, camera_model, T_camera2base, hit_updating_N=3, missed_hit_M=5):
        super(PolytopeEstimator, self).__init__(camera_model)
        self.T_camera2base = T_camera2base
        self.estimated = False
        self.false_estimating_redflag = False
        self.check_seq = 0  # check target sequence number
        # self.hit_img = 0  # hit image number
        self.missed_hit_M = missed_hit_M  # the maximum hit number missed in tracking
        self.hit_updating = 0  # hit number of points updating
        self.hit_updating_N = hit_updating_N  # the minimum hit updating number to enable localizing
        self.updated = False
        self.trans_threshold = 1.
        self.rot_threshold = 15 / 180. * np.pi
        self.last_pose = Pose()

    def update_transform(self, pose):
        self.T_base2world = self.get_transform_from_pose(pose)
        self.T_camera2world = np.matmul(self.T_base2world, self.T_camera2base)
        self.T_world2camera = inv(self.T_camera2world)

    def get_transform_from_pose(self, pose):
        trans = tf.transformations.translation_matrix((pose.position.x, pose.position.y, pose.position.z))
        rot = tf.transformations.quaternion_matrix((pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))
        T = np.matmul(trans, rot)
        return T

    def update_pose(self, pose):
        self.last_pose = pose
        self.update_transform(pose)

    def assert_keyframe(self, pose):
        # translation difference
        position = np.asarray((pose.position.x, pose.position.y, pose.position.z))
        position_ = np.asarray((self.last_pose.position.x, self.last_pose.position.y, self.last_pose.position.z))
        trans_diff = np.linalg.norm(position_ - position)
        if trans_diff > self.trans_threshold:
            return True
        euler = tf.transformations.euler_from_quaternion((pose.orientation.x, pose.orientation.y,
                                                          pose.orientation.z, pose.orientation.w))
        euler_ = tf.transformations.euler_from_quaternion((self.last_pose.orientation.x, self.last_pose.orientation.y,
                                                           self.last_pose.orientation.z, self.last_pose.orientation.w))
        rot_diff = abs(euler_[2] - euler[2])
        if rot_diff > self.rot_threshold:
            return True
        return False







