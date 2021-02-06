#!/usr/bin/env python
"""
Zhiang Chen
Jan 2021
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
import threading
import ros_numpy
from utils.open3d_ros_conversion import convertCloudFromRosToOpen3d, convertCloudFromOpen3dToRos

ROS_BAG = True

target_status_dict = {'mapped': 0,
                      'localized': 1,
                      'pre_localizing': 3,
                      'false_localizing': 4,
                      'false_localizing_examining': 5,
                      'localizing': 6}

uav_status_dict = {'searching': 0,
                   'localizing': 1,
                   'mapping': 2}

update_status_dict = {'localized': 0,
                      'true_keyframe': 1,
                      'false_keyframe': 2,
                      'mapped': 3}

class TargetPoints(object):
    def __init__(self, camera_model, T_camera2base, hit_updating_N=6, missed_hit_M=10):
        self.nm = 2000
        #self.z_min = 3  # granite dell # the range of particles along z axis in camera coord system
        #self.z_max = 18
        self.z_min = 6  # blender terrain
        self.z_max = 23
        self.noise_z = 0.1
        self.noise_xy = 0.1
        self.boarder = 10
        self.generation_boarder_scale = 3  # the bbox boarder when generating points is self.boarder * self.generation_boarder_scale
        self.epsilon = 0.8  # the larger epsilon is, the heavier uniform distribution weights are
        self.pinhole_camera_model = camera_model
        self.image_W, self.image_H = self.pinhole_camera_model.fullResolution()
        self.T_camera2base = T_camera2base
        self.mapped = False
        self.localized = False
        self.false_localization_redflag = False
        self.check_seq = 0  # check target sequence number
        # self.hit_img = 0  # hit image number
        self.missed_hit_M = missed_hit_M  # the maximum hit number missed in tracking
        self.hit_updating = 0  # hit number of points updating
        self.hit_updating_N = hit_updating_N  # the minimum hit updating number to enable localizing
        self.trans_threshold = 1.5
        self.rot_threshold = 30/180.*np.pi
        self.KL_D = 1e10  # KL divergence
        self.DE = 1e10  # differential entropy
        self.last_pose = Pose()
        self.points_3d = np.random.rand(self.nm, 3)
        self.points_2d = np.zeros((self.nm, 2))
        self.marker = Marker()
        self.eigen_w = np.ones(3)  # eigen values
        self.eigen_v = np.eye(3)  # eigen vectors
        self.quat = (0, 0, 0, 1)  # quat of ellipsoid computed from eigen values and vectors, (x, y, z, w)

    def update_transform(self, pose):
        self.T_base2world = self.get_transform_from_pose(pose)
        self.T_camera2world = np.matmul(self.T_base2world, self.T_camera2base)
        self.T_world2camera = inv(self.T_camera2world)

    def update_pose(self, pose):
        self.last_pose = pose
        self.update_transform(pose)

    def update_points(self, bbox, pose):
        # self.hit_img = self.hit_bbox
        self.hit_updating += 1
        self.check_seq = self.hit_updating
        if self.mapped:
            return update_status_dict['mapped']

        if self.localized:
            return update_status_dict['localized']

        if self.assert_keyframe(pose):
            self.importance_sampling(bbox)
            self.update_pose(pose)
            return update_status_dict['true_keyframe']
        else:
            return update_status_dict['false_keyframe']

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
        """
        err_matrix = (np.matmul(rot_.transpose(), rot) - np.matmul(rot, rot_.transpose())) / 2.
        x3 = err_matrix[1, 0]
        x2 = err_matrix[0, 2]
        x1 = err_matrix[2, 1]
        rot_diff = abs(x1) + abs(x2) + abs(x3)
        print(rot_diff/3.14*180)
        """
        rot_diff = abs(euler_[2] - euler[2])
        if rot_diff > self.rot_threshold:
            return True
        return False

    def importance_sampling(self, bbox):
        points = copy.deepcopy(self.points_3d)
        # add noise
        w = np.random.normal(size=(self.nm, 2)) * self.noise_xy
        points[:, :2] = points[:, :2] + w
        w = np.random.normal(size=self.nm) * self.noise_z
        points[:, 2] = points[:, 2] + w
        # compute weights (importance)
        bbox_size = (bbox.xmax - bbox.xmin) * (bbox.ymax - bbox.ymin)
        uniform_imp = self.x_in_bbox_bool / bbox_size  # importance from uniform distribution
        mean = ((bbox.xmin + bbox.xmax) / 2.0, (bbox.ymin + bbox.ymax) / 2.0)
        cov = ((((bbox.xmax - bbox.xmin) / 2.0) ** 2, 0), (0, ((bbox.ymax - bbox.ymin) / 2.0) ** 2))
        normal_2d = stats.multivariate_normal(mean, cov)
        normal_imp = normal_2d.pdf(self.points_2d)  # importance from normal distribution
        imp = (1 - self.epsilon) * normal_imp + self.epsilon * uniform_imp
        W = imp / np.sum(imp)  # normalize importance
        # resampling
        acc_W = np.cumsum(W)  # accumulated W
        u_samples = np.random.rand(self.points_2d.shape[0])  # uniform samples
        resample_ids = np.searchsorted(acc_W, u_samples)
        new_points = np.take(points, resample_ids, axis=0)
        self.KL_D, self.DE = self.measure_points(self.points_3d, new_points)
        self.points_3d = new_points
        self.project_points()

    def resampling_uniform_dist(self):
        marker_rot = tf.transformations.quaternion_matrix(self.quat)
        points = copy.deepcopy(self.points_3d)
        points = np.insert(points, 3, 1, axis=1).transpose()
        points_rot = np.matmul(inv(marker_rot), points)
        new_points = points_rot[:3, :].transpose()
        x_max = new_points[:, 0].max()
        x_min = new_points[:, 0].min()
        x = np.random.rand(self.nm) * (x_max - x_min) + x_min
        new_points[:, 0] = x
        new_points = np.insert(new_points, 3, 1, axis=1).transpose()
        new_points = np.matmul(marker_rot, new_points)
        new_points = new_points[:3, :].transpose()
        self.KL_D, self.DE = self.measure_points(self.points_3d, new_points)
        self.points_3d = new_points
        self.project_points()

    def expanded_resampling_uniform_dist(self, resampler_boader=1.):
        marker_rot = tf.transformations.quaternion_matrix(self.quat)
        points = copy.deepcopy(self.points_3d)
        points = np.insert(points, 3, 1, axis=1).transpose()
        points_rot = np.matmul(inv(marker_rot), points)
        new_points = points_rot[:3, :].transpose()
        x_max = new_points[:, 0].max() #+ resampler_boader
        x_min = new_points[:, 0].min() #- resampler_boader
        y_max = new_points[:, 1].max() + resampler_boader
        y_min = new_points[:, 1].min() - resampler_boader
        z_max = new_points[:, 2].max() + resampler_boader
        z_min = new_points[:, 2].min() - resampler_boader
        x = np.random.rand(self.nm) * (x_max - x_min) + x_min
        y = np.random.rand(self.nm) * (y_max - y_min) + y_min
        z = np.random.rand(self.nm) * (z_max - z_min) + z_min
        # x = np.linspace(x_min, x_max, self.nm)
        # y = np.linspace(y_min, y_max, self.nm)
        # z = np.linspace(z_min, z_max, self.nm)
        new_points = np.asarray((x, y, z)).transpose()
        new_points = np.insert(new_points, 3, 1, axis=1).transpose()
        new_points = np.matmul(marker_rot, new_points)
        new_points = new_points[:3, :].transpose()
        self.KL_D, self.DE = self.measure_points(self.points_3d, new_points)
        self.points_3d = new_points
        self.project_points()

    def register_points(self, bbox, pose):
        self.update_pose(pose)
        cone = self.reproject_bbox_cone(bbox)
        self.generate_points(cone)

    def generate_points(self, cone):
        # cone = np.asarray((ray1, ray2, ray3, ray4))
        # 1. generate points on unit surface, in camera coordinate system
        # 2. randomize these points by varying elevations, in camera coordinate system
        # 3. convert to world coordinate system
        a = np.random.rand(4, self.nm)  # 4 x nm
        scaling = np.diag(1.0 / np.sum(a, axis=0))  # nm x nm
        a_ = np.matmul(a, scaling)  # 4 x nm
        points = np.matmul(cone.transpose(), a_)  # 3 x nm
        z = np.random.rand(self.nm) * (self.z_max - self.z_min) + self.z_min
        z = np.diag(z)  # nm x nm
        points_c = np.matmul(points, z)  # 3 x nm
        points_c = np.insert(points_c, 3, 1, axis=0)  # 4 x nm
        points_w = np.matmul(self.T_camera2world, points_c)  # 4 x nm
        points_w = points_w[:3, :].transpose()  # nm x 3
        # 3d points
        self.points_3d = points_w
        # 2d points
        self.project_points()

    def project_points(self):
        points = copy.deepcopy(self.points_3d)
        points_w = np.insert(points, 3, 1, axis=1).transpose()
        points_c = np.matmul(self.T_world2camera, points_w)
        points_c = points_c[:3, :]
        points_c = points_c.transpose()
        x = map(self.pinhole_camera_model.project3dToPixel, points_c)
        self.points_2d = np.asarray(x)

    def get_eigens(self):
        # center = np.mean(self.points_3d, axis=0)
        w, v = np.linalg.eig(np.cov(self.points_3d.transpose()))
        eigx_n = PyKDL.Vector(v[0, 0], v[1, 0], v[2, 0])
        eigy_n = -PyKDL.Vector(v[0, 1], v[1, 1], v[2, 1])
        eigz_n = PyKDL.Vector(v[0, 2], v[1, 2], v[2, 2])
        eigx_n.Normalize()
        eigy_n.Normalize()
        eigz_n.Normalize()
        self.quat = PyKDL.Rotation(eigx_n, eigy_n, eigz_n).GetQuaternion()
        self.eigen_w = w
        self.eigen_v = v
        return w, v, self.quat

    def get_statistic(self):
        return self.KL_D, self.DE

    def check_points_in_image(self, pose):
        u = np.all(np.asarray((0 <= self.points_2d[:, 0], self.points_2d[:, 0] <= self.image_W)), axis=0)
        v = np.all(np.asarray((0 <= self.points_2d[:, 1], self.points_2d[:, 1] <= self.image_H)), axis=0)
        x_in_image_bool = np.all(np.asarray((u, v)), axis=0)
        if np.sum(x_in_image_bool) <= self.nm/20:
            return False
        else:
            # self.hit_img += 1
            return True

    def check_points_in_bbox(self, bbox, pose):
        self.update_transform(pose)
        self.project_points()
        # if not self.check_points_in_image(pose):
        #     return False
        xmin = bbox.xmin #- self.boarder
        xmax = bbox.xmax #+ self.boarder
        ymin = bbox.ymin #- self.boarder
        ymax = bbox.ymax #+ self.boarder
        u = np.all(np.asarray((xmin <= self.points_2d[:, 0], self.points_2d[:, 0] <= xmax)), axis=0)
        v = np.all(np.asarray((ymin <= self.points_2d[:, 1], self.points_2d[:, 1] <= ymax)), axis=0)
        self.x_in_bbox_bool = np.all(np.asarray((u, v)), axis=0)
        if np.sum(self.x_in_bbox_bool) > 0:
            # self.hit_bbox += 1
            return True
        else:
            return False

    def check_status(self, update_seq=True):
        if update_seq:
            self.check_seq += 1
        if self.mapped:
            return target_status_dict['mapped']

        if self.localized:
            return target_status_dict['localized']

        if self.hit_updating < self.hit_updating_N:
            if not self.false_localization_redflag:
                if (self.check_seq - self.hit_updating) > self.missed_hit_M:
                    self.false_localization_redflag = True
                    self.reset_status()
                    return target_status_dict['false_localizing_examining']
                else:
                    return target_status_dict['pre_localizing']
            else:
                if self.check_seq > self.missed_hit_M:
                    return target_status_dict['false_localizing']
                else:
                    return target_status_dict['false_localizing_examining']

        if (self.check_seq - self.hit_updating) > self.missed_hit_M:
            self.false_localization_redflag = True
            self.reset_status()
            return target_status_dict['false_localizing_examining']
        else:
            self.false_localization_redflag = False
            return target_status_dict['localizing']

    def reset_status(self):
        self.check_seq = 0
        self.hit_updating = 0
        # self.hit_img = 0

    def set_localized(self):
        self.localized = True

    def reset_localized(self):
        self.localized = False

    def set_mapped(self):
        self.mapped = True

    def update_points_after_mapping(self, points):
        if points.shape[0] > self.nm:
            idx = np.array([False] * points.shape[0])
            idx[:self.nm] = True
            np.random.shuffle(idx)
            self.points_3d = points[idx]
        else:
            self.nm = points.shape[0]
            self.points_3d = points

    def measure_points(self, p0, p1):
        """
        :param p0: old points
        :param p1: new points
        :return:
        """
        u0 = p0.mean(axis=0)
        u1 = p1.mean(axis=0)
        cov0 = np.cov(p0.transpose())
        cov1 = np.cov(p1.transpose())
        # KL( P_old | P_new)
        term1 = np.matmul(inv(cov1), cov0).trace()
        term2 = np.matmul(np.matmul((u1-u0).transpose(), inv(cov1)), (u1-u0))
        term4 = np.log(det(cov1)/det(cov0))
        kld = (term1 + term2 - 3 + term4)*0.5
        de = 4.2568155996140185 + np.log(det(cov1)) / 2.0
        return kld, de

    def reproject_bbox_cone(self, bbox):
        """
        :param bbox:
        :return: cone = np.asarray((ray1, ray2, ray3, ray4))
        """
        x1 = bbox.xmin - self.boarder * self.generation_boarder_scale
        y1 = bbox.ymin - self.boarder * self.generation_boarder_scale
        x2 = bbox.xmax + self.boarder * self.generation_boarder_scale
        y2 = bbox.ymax + self.boarder * self.generation_boarder_scale
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

    def get_transform_from_pose(self, pose):
        trans = tf.transformations.translation_matrix((pose.position.x, pose.position.y, pose.position.z))
        rot = tf.transformations.quaternion_matrix((pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))
        T = np.matmul(trans, rot)
        return T

    def get_points_3d(self):
        return self.points_3d

class TargetTracker(object):
    def __init__(self):
        self.targets = []
        self.markers = []
        self.pose = Pose()
        self.processing_target_id = 0
        self.localizing_target_id = 0
        self.uav_status = uav_status_dict['searching']
        self.seq = 0
        self.process_status = {target_status_dict['mapped']: self.process_mapped,
                               target_status_dict['localized']: self.process_localized,
                               target_status_dict['pre_localizing']: self.process_pre_localizing,
                               target_status_dict['false_localizing']: self.process_false_localizing,
                               target_status_dict['localizing']: self.process_localizing,
                               target_status_dict['false_localizing_examining']: self.process_false_localizing_examining}
        #self.trigger_localizing_DE = 3.  # granite dell
        self.trigger_localizing_DE = 2.65  # blender terrain
        self.trigger_resampling_KLD = 0.1
        self.trigger_mapping_KLD = 0.01
        self.trigger_mapping_N = 3  # the minimum number of keyframes with satisfying KLD to trigger mapping
        self.seq_examine_mapping = 0
        self.check_edge_box_boarder = 5
        self.target_spacing_threshold = 1.  # mapping will be skipped if the target center distance are smaller than this value

        self.pub_markers = rospy.Publisher("/target_localizer/ellipsoids", MarkerArray, queue_size=1)
        self.client = actionlib.SimpleActionClient("/path_planner/target_plan", target_mapping.msg.TargetPlanAction)
        #self.client.wait_for_server()

        # changed
        self.publish_image = True
        if self.publish_image:
            self.update_img = False
            self.bridge = CvBridge()
            self.pub = rospy.Publisher("/target_localizer/detection_image", Image, queue_size=1)
            #raw_image_sub = rospy.Subscriber('/bbox_tracker/detection_image', Image, self.image_callback, queue_size=1)
            raw_image_sub = rospy.Subscriber('/darknet_ros/detection_image', Image, self.image_callback, queue_size=2)
            self.img = np.zeros(1)

        if not ROS_BAG:
            rospy.loginfo("checking tf from camera to base_link ...")
            listener = tf.TransformListener()
            while not rospy.is_shutdown():
                try:
                    now = rospy.Time.now()
                    listener.waitForTransform("base_link", "r200", now, rospy.Duration(5.0))
                    (trans, rot) = listener.lookupTransform("base_link", "r200", now)
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
        else:
            trans_vec = (0.1, 0,  - 0.01)
            trans = tf.transformations.translation_matrix(trans_vec)
            #quaternion = (-0.6743797, 0.6743797, - 0.2126311, 0.2126311)
            quaternion = (-0.6830127,  0.6830127, -0.1830127,  0.1830127)
            rot = tf.transformations.quaternion_matrix(quaternion)

        self.T_camera2base = np.matmul(trans, rot)


        self.pcl_pub = rospy.Publisher("/target_localizer/points", PointCloud2, queue_size=10)

        camera_info = rospy.wait_for_message("/r200/rgb/camera_info", CameraInfo)
        self.pinhole_camera_model = PinholeCameraModel()
        self.pinhole_camera_model.fromCameraInfo(camera_info)
        self.image_W, self.image_H = self.pinhole_camera_model.fullResolution()

        self.sub_bbox = message_filters.Subscriber('/bbox_tracker/bounding_boxes_drop', BoundingBoxes, queue_size=10, buff_size=2**24)
        #self.sub_bbox = message_filters.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, queue_size=10,buff_size=2**24)
        self.sub_pose = message_filters.Subscriber('/mavros/local_position/pose', PoseStamped, queue_size=10, buff_size=2**24)
        # self.sub_vel = message_filters.Subscriber('/mavros/local_position/velocity_local', TwistStamped)
        # self.timer = rospy.Timer(rospy.Duration(.5), self.timerCallback)


        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_bbox, self.sub_pose], queue_size=10, slop=0.05)
        #self.ts = message_filters.TimeSynchronizer([self.sub_bbox, self.sub_pose], 10)
        self.ts.registerCallback(self.callback)

        # self.sub1 = message_filters.Subscriber('/mavros/odometry/in', Odometry, queue_size=10, buff_size=2 ** 24)
        # self.sub_bbox = message_filters.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes)
        # self.sub2 = message_filters.Subscriber('/mavros/local_position/pose', PoseStamped, queue_size=10, buff_size=2 ** 24)
        # self.sub_vel = message_filters.Subscriber('/mavros/local_position/velocity_local', TwistStamped)
        self.timer = rospy.Timer(rospy.Duration(0.5), self.timerCallback)
        # self.ts1 = message_filters.ApproximateTimeSynchronizer([self.sub1, self.sub2], queue_size=10, slop=0.1)
        # self.ts = message_filters.TimeSynchronizer([self.sub_bbox, self.sub_pose], 10)
        # self.ts1.registerCallback(self.callback1)

        print("target_localizer initialized!")

    def timerCallback(self, timer):
        self.publishTargetPoints()
        self.publishTargetMarkers()  # this is needed because markers will be used for localization and mapping
        N = len(self.targets)
        for id in reversed(range(N)):
            target = self.targets[id]
            target_status = target.check_status()
            if target_status == target_status_dict['false_localizing']:
                self.processing_target_id = id
                target = self.targets[self.processing_target_id]
                print("target id: " + str(self.processing_target_id))
                self.process_status[target_status](target)

    def callback(self, bbox_msg, pose_msg):
        self.timer.shutdown()
        # processing observation,
        # True, so that timer callback process is disabled, and other target points will be saved
        self.seq += 1
        self.pose = pose_msg.pose
        bboxes = bbox_msg.bounding_boxes
        enable_localization = False
        if len(self.targets) > 0:
            for bbox in bboxes:
                if self.checkBBoxOnEdge(bbox):
                    continue
                bbox_in_target_bools = []
                for id, target in enumerate(self.targets):
                    if target.check_points_in_bbox(bbox, self.pose):
                        bbox_in_target_bools.append(True)
                        update_status = target.update_points(bbox, self.pose)
                        if update_status == update_status_dict['localized']:
                            enable_localization = True  # enable localizeTarget
                        if update_status == update_status_dict['true_keyframe']:
                            enable_localization = True
                        if update_status == update_status_dict['false_keyframe']:
                            continue
                if len(bbox_in_target_bools) == 0:
                    if self.uav_status == uav_status_dict['searching']:
                        self.generateTarget(bbox)
        else:
            map(self.generateTarget, bboxes)
            enable_localization = True

        self.publishTargetPoints()
        self.publishTargetMarkers()  # this is needed because markers will be used for localization and mapping

        if enable_localization:
            self.localizeTarget()
        self.timer = rospy.Timer(rospy.Duration(0.5), self.timerCallback)

    def generateTarget(self, bbox):
        if self.checkBBoxOnEdge(bbox):
            return None
        target = TargetPoints(self.pinhole_camera_model, self.T_camera2base)
        target.register_points(bbox, self.pose)
        self.targets.append(target)

    def checkBBoxOnEdge(self, bbox):
        x1 = bbox.xmin
        y1 = bbox.ymin
        x2 = bbox.xmax
        y2 = bbox.ymax
        if x1 < self.check_edge_box_boarder:
            return True  # it is on the edge
        if y1 < self.check_edge_box_boarder:
            return True
        if (self.image_W - x2) < self.check_edge_box_boarder:
            return True
        if (self.image_H - y2) < self.check_edge_box_boarder:
            return True
        return False

    def localizeTarget(self):
        if self.processing_target_id - len(self.targets) == 0:
            return
        elif self.processing_target_id - len(self.targets) > 0:
            self.processing_target_id -= 1
            return

        N = len(self.targets)
        for id in reversed(range(N)):
            target = self.targets[id]
            target_status = target.check_status()
            if target_status == target_status_dict['false_localizing']:
                self.processing_target_id = id
                target = self.targets[self.processing_target_id]
                print("target id: " + str(self.processing_target_id))
                self.process_status[target_status](target)

        target_statuses = [target.check_status(update_seq=False) for target in self.targets]
        for id, target_status in enumerate(target_statuses):
            if target_status == target_status_dict['localized']:
                self.processing_target_id = id
                target = self.targets[self.processing_target_id]
                # check if the target has already been mapped
                if self.assert_mapped_by_others(id, target_statuses):
                    self.process_status[target_status_dict['false_localizing']](target)
                else:
                    # else, then map
                    print("target id: " + str(self.processing_target_id))
                    self.process_status[target_status](target)

        for id, target_status in enumerate(target_statuses):
            if target_status == target_status_dict['localizing']:
                self.processing_target_id = id
                target = self.targets[self.processing_target_id]
                print("target id: " + str(self.processing_target_id))
                self.process_status[target_status](target)
                self.update_img = True


    def process_pre_localizing(self, target):
        pass

    def process_localizing(self, target):
        KLD, DE = target.get_statistic()
        print('KLD: ' + str(KLD))
        print('DE: ' + str(DE))

        if DE < self.trigger_localizing_DE:
            if self.uav_status == uav_status_dict['searching']:
                #target.resampling_uniform_dist()
                #target.expanded_resampling_uniform_dist()
                self.requestLocalizing()
                self.localizing_target_id = self.processing_target_id
                rospy.sleep(2.)
            if (self.uav_status == uav_status_dict['localizing']) & \
                    (KLD <= self.trigger_mapping_KLD):
                self.seq_examine_mapping += 1
                if self.seq_examine_mapping >= self.trigger_mapping_N:
                    target.set_localized()
                    self.seq_examine_mapping = 0
            else:
                self.seq_examine_mapping = 0


    def process_false_localizing_examining(self, target):
        return

    def process_false_localizing(self, target):
        rospy.loginfo('false localizing')
        del self.targets[self.processing_target_id]
        if self.uav_status != uav_status_dict['searching']:
            print("processing target id: " + str(self.processing_target_id))
            print("localizing target id: " + str(self.localizing_target_id))
            if self.processing_target_id == self.localizing_target_id:
                self.continueSearch()
            if self.processing_target_id < self.localizing_target_id:
                self.localizing_target_id -= 1


    def process_localized(self, target):
        result = self.requestMapping()
        if result.success:
            rospy.loginfo('mapped')
            target.set_mapped()
            ros_pts = result.pointcloud_map
            o3d_pts = convertCloudFromRosToOpen3d(ros_pts)
            pts = np.asarray(o3d_pts.points)
            target.update_points_after_mapping(pts)
            self.continueSearch()
        else:
            rospy.loginfo('false mapping')
            del self.targets[self.processing_target_id]
            self.continueSearch()


    def process_mapped(self, target):
        print('mapped')

    def requestLocalizing(self):
        rospy.loginfo('requesting localizing')
        goal = target_mapping.msg.TargetPlanGoal()
        goal.header.stamp = rospy.Time.now()
        goal.id.data = self.processing_target_id
        goal.mode.data = uav_status_dict['localizing']
        goal.markers = self.markers
        self.client.send_goal(goal)
        self.uav_status = uav_status_dict['localizing']

    def requestMapping(self):
        rospy.loginfo('requesting mapping')
        goal = target_mapping.msg.TargetPlanGoal()
        goal.header.stamp = rospy.Time.now()
        goal.id.data = self.processing_target_id
        goal.mode.data = uav_status_dict['mapping']
        goal.markers = self.markers
        self.client.send_goal(goal)
        self.uav_status = uav_status_dict['mapping']
        rospy.sleep(5.)
        self.client.wait_for_result()
        result = self.client.get_result()
        return result

    def continueSearch(self):
        rospy.loginfo('resuming searching')
        goal = target_mapping.msg.TargetPlanGoal()
        goal.header.stamp = rospy.Time.now()
        goal.id.data = self.processing_target_id
        goal.mode.data = uav_status_dict['searching']
        goal.markers = self.markers
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()
        self.uav_status = uav_status_dict['searching']

    def image_callback(self, data):
        self.image_header = data.header
        raw_image = self.bridge.imgmsg_to_cv2(data).astype(np.uint8)
        if len(raw_image.shape) > 2:
            self.img = raw_image
        if self.update_img:
            if len(self.targets) <= self.processing_target_id:
                return
            target = self.targets[self.processing_target_id]
            points_2d = target.points_2d
            img = self.draw_points(self.img, points_2d)
            image_msg = self.bridge.cv2_to_imgmsg(img, 'bgr8')
            self.pub.publish(image_msg)
            self.update_img = False


    def draw_points(self, image, pts):
        for pt in pts:
            image = cv2.circle(image, (int(pt[0]), int(pt[1])), radius=2, color=(255, 0, 0), thickness=2)
        return image

    def assert_mapped_by_others(self, id, statuses):
        mapped_ids = [i for i, status in enumerate(statuses) if status==target_status_dict['mapped']]
        target_marker = self.markers.markers[id]
        target_x = target_marker.pose.position.x
        target_y = target_marker.pose.position.y
        target_z = target_marker.pose.position.z
        target_pos = np.array((target_x, target_y, target_z))
        for mapped_id in mapped_ids:
            other_marker = self.markers.markers[mapped_id]
            other_x = other_marker.pose.position.x
            other_y = other_marker.pose.position.y
            other_z = other_marker.pose.position.z
            other_pos = np.array((other_x, other_y, other_z))
            dis = np.linalg.norm(target_pos - other_pos)
            if dis < self.target_spacing_threshold:
                return True

        return False


    def publishTargetMarkers(self):
        markerArray = MarkerArray()
        for id, target in enumerate(self.targets):
            pts = target.get_points_3d()
            center = np.mean(pts, axis=0)
            w, v, quat = target.get_eigens()
            marker = self.markerVector(id * 3, w * 10, v, center, quat, pts)
            markerArray.markers.append(marker)

        if len(self.targets) > 0:
            self.pub_markers.publish(markerArray)
            self.markers = markerArray

    def markerVector(self, id, scale, rotation, position, quaternion, pts):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "target_mapping"
        marker.id = id
        marker.type = visualization_msgs.msg.Marker.SPHERE
        marker.action = visualization_msgs.msg.Marker.ADD
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]
        marker.color.a = 1.0
        marker.color.r = 1
        marker.color.g = 0
        marker.color.b = 0

        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        R = np.eye(4)
        R[:3, :3] = rotation
        q = tf.transformations.quaternion_from_matrix(R)

        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]
        points = [Point(p[0], p[1], p[2]) for p in pts]
        marker.points = points
        return marker

    def publishTargetPoints(self):
        # convert target_points to pointcloud messages
        # if len(self.targets) > 0:
        all_points = []
        for target in self.targets:
            all_points = all_points + target.get_points_3d().tolist()
        if len(all_points) == 0:
            all_points = [[0, 0, 0]]

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        # create pcl from points
        pc_msg = pcl2.create_cloud_xyz32(header, all_points)
        self.pcl_pub.publish(pc_msg)


if __name__ == '__main__':
    rospy.init_node('target_localizer', anonymous=False)
    target_tracker = TargetTracker()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node killed!")