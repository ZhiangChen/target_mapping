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
import tf
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
import scipy.stats as stats
from visualization_msgs.msg import Marker, MarkerArray
import visualization_msgs
from geometry_msgs.msg import Point
import PyKDL
import target_mapping.msg
import actionlib

ROS_BAG = True

class TargetTracker(object):
    def __init__(self, particle_nm=1000, z_range=(0.2, 4), a=0.5):
        self.nm = particle_nm
        self.a = a  # https://github.com/ZhiangChen/target_mapping/wiki/Target-Localization-and-Body-Estimation-by-3D-Points
        self.w = []
        self.target_points = []  # a list of Nx3 ndarrays, in world coord system
        self.z_min = z_range[0]  # the range of particles along z axis in camera coord system
        self.z_max = z_range[1]
        self.searching_id = 0
        self.localized = []
        self.mapped = []
        self.markers = MarkerArray()
        self.KL_D = []  # KL divergence
        self.DE = []  # differential entropy
        self.status = 0  # 0: searching; 1: localizing; 2: mapping

        self.pub_markers = rospy.Publisher("/target_localizer/ellipsoids", MarkerArray, queue_size=1)
        self.client = actionlib.SimpleActionClient("/path_planner/target_plan", target_mapping.msg.TargetPlanAction)
        self.client.wait_for_server()

        self.publish_image = True
        if self.publish_image:
            self.bridge = CvBridge()
            self.pub = rospy.Publisher("/target_localizer/detection_image", Image, queue_size=1)
            raw_image_sub = rospy.Subscriber('/bbox_tracker/detection_image', Image, self.image_callback, queue_size=1)
            self.img = np.zeros(1)

        if not ROS_BAG:
            rospy.loginfo("checking tf from camera to base_link ...")
            listener = tf.TransformListener()
            while not rospy.is_shutdown():
                try:
                    now = rospy.Time.now()
                    listener.waitForTransform("base_link", "r200", now, rospy.Duration(4.0))
                    (trans, rot) = listener.lookupTransform("base_link", "r200", now)
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
        else:
            trans_vec = (0.1, 0,  - 0.01)
            trans = tf.transformations.translation_matrix(trans_vec)
            quaternion = (-0.6743797, 0.6743797, - 0.2126311, 0.2126311)
            rot = tf.transformations.quaternion_matrix(quaternion)

        self.T_camera2base = np.matmul(trans, rot)

        self.pcl_pub = rospy.Publisher("/target_localizer/points", PointCloud2, queue_size=10)

        camera_info = rospy.wait_for_message("/r200/depth/camera_info", CameraInfo)

        self.pinhole_camera_model = PinholeCameraModel()
        self.pinhole_camera_model.fromCameraInfo(camera_info)

        self.image_W, self.image_H = self.pinhole_camera_model.fullResolution()

        self.sub_bbox = message_filters.Subscriber('/bbox_tracker/bounding_boxes', BoundingBoxes)
        self.sub_pose = message_filters.Subscriber('/mavros/local_position/pose', PoseStamped)
        # self.sub_vel = message_filters.Subscriber('/mavros/local_position/velocity_local', TwistStamped)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_bbox, self.sub_pose], queue_size=100, slop=0.1)
        # self.ts = message_filters.TimeSynchronizer([self.sub_bbox, self.sub_pose, self.sub_vel], 10)
        self.ts.registerCallback(self.callback)
        print("target_localizer initialized!")

    def callback(self, bbox_msg, pose_msg):
        pose = pose_msg.pose
        bboxes = bbox_msg.bounding_boxes

        for i, bbox in enumerate(bboxes):
            if not self.checkBBoxOnEdge(bbox):
                cone = self.reprojectBBoxesCone(bbox)
                ids = self.checkPointsInCone(cone, pose)
                if not ids:
                    self.generatePoints(cone, pose, self.nm)
                else:
                    #self.updatePoints(ids, cone, pose)
                    self.updatePointsDF(ids, bbox, pose)

        print("target #: %d" % len(self.target_points))
        variances = self.computeTargetsVariance()
        self.publishTargetPoints()
        self.localizeTarget()

    def checkBBoxOnEdge(self, bbox, p=20):
        x1 = bbox.xmin
        y1 = bbox.ymin
        x2 = bbox.xmax
        y2 = bbox.ymax
        if x1 < p:
            return True # it is on the edge
        if y1 < p:
            return True
        if (self.image_W - x2) < p:
            return True
        if (self.image_H - y2) < p:
            return True

        return False

    def reprojectBBoxesCone(self, bbox):
        """
        :param bbox:
        :return: cone = np.asarray((ray1, ray2, ray3, ray4))
        """
        x1 = bbox.xmin - 10
        y1 = bbox.ymin - 10
        x2 = bbox.xmax + 10
        y2 = bbox.ymax + 10
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
        # the order of cross product determines the normal vector direction
        norm1 = np.cross(ray1, ray2)
        norm2 = np.cross(ray2, ray3)
        norm3 = np.cross(ray3, ray4)
        norm4 = np.cross(ray4, ray1)
        H = np.asarray((norm1, norm2, norm3, norm4))

        ids = []
        for i,points in enumerate(self.target_points):
            # convert points to camera coordinate system
            T_base2world = self.getTransformFromPose(pose)
            T_camera2world = np.matmul(T_base2world, self.T_camera2base)
            T_world2camera = inv(T_camera2world)
            points_w = np.insert(points, 3, 1, axis=1).transpose()
            points_c = np.matmul(T_world2camera, points_w)
            points_c = points_c[:3, :]
            points_occupancy = np.all(np.matmul(H, points_c) >= 0, axis=0)
            if np.any(points_occupancy):
                ids.append(i)
        if len(ids) >= 1:
            return ids
        else:
            return False

    def generatePoints(self, cone, pose, nm=1000):
        # register new target
        # cone = np.asarray((ray1, ray2, ray3, ray4))
        # 1. generate points on unit surface, in camera coordinate system
        # 2. randomize these points by varying elevations, in camera coordinate system
        # 3. convert to world coordinate system
        a = np.random.rand(4, nm)  # 4 x nm
        scaling = np.diag(1.0 / np.sum(a, axis=0)) # nm x nm
        a_ = np.matmul(a, scaling)  # 4 x nm
        points = np.matmul(cone.transpose(), a_)  # 3 x nm
        z = np.random.rand(nm) * (self.z_max - self.z_min) + self.z_min
        z = np.diag(z)  # nm x nm
        points_c = np.matmul(points, z)  # 3 x nm
        points_c = np.insert(points_c, 3, 1, axis=0) # 4 x nm

        T_base2world = self.getTransformFromPose(pose)
        T_camera2world = np.matmul(T_base2world, self.T_camera2base)
        points_w = np.matmul(T_camera2world, points_c)  # 4 x nm
        points_w = points_w[:3, :].transpose()  # nm x 3
        self.target_points.append(points_w)
        self.w.append(1)
        self.KL_D.append(-1)
        self.DE.append(-1)
        self.localized.append(False)
        self.mapped.append(False)

    def updatePoints(self, ids, cone, pose):
        # return information gain
        # if there are multiple targets in the bbox, then update them all individually
        ray1, ray2, ray3, ray4 = cone
        norm1 = np.cross(ray1, ray2)
        norm2 = np.cross(ray2, ray3)
        norm3 = np.cross(ray3, ray4)
        norm4 = np.cross(ray4, ray1)
        H = np.asarray((norm1, norm2, norm3, norm4))
        T_base2world = self.getTransformFromPose(pose)
        T_camera2world = np.matmul(T_base2world, self.T_camera2base)
        T_world2camera = inv(T_camera2world)

        for id in ids:
            # add Gaussian noise to all points
            points = self.target_points[id]
            w = np.random.normal(size=(1000, 3)) * self.w[id]
            self.w[id] = self.w[id]*0.9 + 0.001
            points = points + w
            # convert to camera coord sys
            points_w = np.insert(points, 3, 1, axis=1).transpose()
            points_c = np.matmul(T_world2camera, points_w)
            points_c = points_c[:3, :]
            # update points
            HX = np.matmul(H, points_c)
            points_occupancy = np.all(HX >= 0, axis=0)
            Z = points_c[:, points_occupancy] # points in bbox
            Y = points_c[:, np.invert(points_occupancy)] # points not in bbox
            D = np.matmul(H, Z).min(axis=0)
            sigma = self.a * float(self.nm - Z.shape[1])/self.nm
            sigma = np.max((sigma, 0.2))
            W = stats.norm(0, sigma).pdf(D)
            print(sigma) # debug: make sure sigma is converging
            W = W/np.sum(W) # normalize W
            acc_W = np.cumsum(W) # accumulated W
            u_samples = np.random.rand(Y.shape[1]) # uniform samples
            resample_ids = np.searchsorted(acc_W, u_samples)
            Y_ = np.take(Z, resample_ids, axis=1)
            new_points_c = np.concatenate((Z, Y_), axis=1)
            # convert to world coord sys
            new_points_c = np.insert(new_points_c, 3, 1, axis=0)
            new_points_w = np.matmul(T_camera2world, new_points_c)[:3, :].transpose()

            self.target_points[id] = new_points_w

    def updatePointsDF(self, ids, bbox, pose, epsilon=0.):
        # update points using depth filter
        T_base2world = self.getTransformFromPose(pose)
        T_camera2world = np.matmul(T_base2world, self.T_camera2base)
        T_world2camera = inv(T_camera2world)

        for id in ids:
            if self.localized[id]:
                continue
            points = self.target_points[id]
            old_points = copy.deepcopy(points)
            #w = np.random.normal(size=(self.nm, 3)) * self.w[id]
            #self.w[id] = self.w[id] * 0.9 + 0.001
            w = np.random.normal(size=(self.nm, 2)) * 0.05
            points[:, :2] = points[:, :2] + w
            w = np.random.normal(size=self.nm) * 0.02
            points[:, 2] = points[:, 2] + w
            points_w = np.insert(points, 3, 1, axis=1).transpose()
            points_c = np.matmul(T_world2camera, points_w)
            points_c = points_c[:3, :]
            points_c = points_c.transpose()
            x = map(self.pinhole_camera_model.project3dToPixel, points_c)
            x = np.asarray(x)
            xmin = bbox.xmin - 10
            xmax = bbox.xmax + 10
            ymin = bbox.ymin - 10
            ymax = bbox.ymax + 10
            u = np.all(np.asarray((xmin <= x[:, 0], x[:, 0] <= xmax)), axis=0)
            v = np.all(np.asarray((ymin <= x[:, 1], x[:, 1] <= ymax)), axis=0)
            x_inbbox_bool = np.all(np.asarray((u, v)), axis=0)
            bbox_size = (bbox.xmax - bbox.xmin) * (bbox.ymax - bbox.ymin)
            uniform_imp = x_inbbox_bool / bbox_size  # importance from uniform distribution
            if self.publish_image:
                if self.img.shape[0] == self.image_H:
                    x_inbbox = x[x_inbbox_bool]
                    image = self.draw_points(copy.deepcopy(self.img), x_inbbox)
                    image_msg = self.bridge.cv2_to_imgmsg(image, 'rgb8')
                    self.pub.publish(image_msg)

            mean = ((bbox.xmin + bbox.xmax)/2.0, (bbox.ymin + bbox.ymax)/2.0)
            cov = ((((bbox.xmax - bbox.xmin)/2.0)**2, 0), (0, ((bbox.ymax - bbox.ymin)/2.0)**2))
            normal_2d = stats.multivariate_normal(mean, cov)
            normal_imp = normal_2d.pdf(x)  # importance from normal distribution
            imp = (1 - epsilon) * normal_imp + epsilon * uniform_imp
            W = imp / np.sum(imp)  # normalize importance
            # resampling
            acc_W = np.cumsum(W)  # accumulated W
            u_samples = np.random.rand(x.shape[0])  # uniform samples
            resample_ids = np.searchsorted(acc_W, u_samples)
            new_points = np.take(points, resample_ids, axis=0)

            self.target_points[id] = new_points
            kld, de = self.measurePoints(old_points, new_points)
            self.KL_D[id] = kld
            self.DE[id] = de

    def localizeTarget(self):
        if len(self.target_points) <= self.searching_id:
            return
        points = self.target_points[self.searching_id]
        kld = self.KL_D[self.searching_id]
        de = self.DE[self.searching_id]
        if kld == -1:
            return

        # start localization
        if self.status == 0:
            self.requestLocalizing()

        # check if new localization planning is needed
        if (not self.checkPointsInImage) & (self.status == 1):
            self.requestLocalizing()

        # confirm localization when KL divergence is too small
        if (kld < 5) & (self.status == 1):
            self.localized[self.searching_id] = True
            self.requestMapping()
            return

        # deregister when differential entropy is too small or too large

        # start mapping
        if self.status == 2:
            result = self.requestMapping()  # this will wait until mapping is done
            self.mapped[self.searching_id] = result
            self.searching_id += 1
            return

    def checkPointsInImage(self):
        return True


    def continueSearch(self):
        self.status = 0

    def requestLocalizing(self):
        self.status = 1

    def requestMapping(self):
        self.status = 2

    def image_callback(self, data):
        self.image_header = data.header
        raw_image = self.bridge.imgmsg_to_cv2(data).astype(np.uint8)
        if len(raw_image.shape) > 2:
            self.img = raw_image

    def draw_points(self, image, pts):
        for pt in pts:
            image = cv2.circle(image, (int(pt[0]), int(pt[1])), radius=2, color=(0, 0, 255), thickness=2)
        return image

    def computeTargetsVariance(self):
        markerArray = MarkerArray()
        for id, pts in enumerate(self.target_points):
            center = np.mean(pts, axis=0)
            w, v = np.linalg.eig(np.cov(pts.transpose()))
            eigx_n = PyKDL.Vector(v[0, 0], v[1, 0], v[2, 0])
            eigy_n = -PyKDL.Vector(v[0, 1], v[1, 1], v[2, 1])
            eigz_n = PyKDL.Vector(v[0, 2], v[1, 2], v[2, 2])
            eigx_n.Normalize()
            eigy_n.Normalize()
            eigz_n.Normalize()
            rot = PyKDL.Rotation(eigx_n, eigy_n, eigz_n)
            quat = rot.GetQuaternion()
            marker = self.markerVector(id*3, w*10, v, center, quat)
            markerArray.markers.append(marker)

        if len(self.target_points) > 0:
            self.pub_markers.publish(markerArray)
            self.markers = copy.deepcopy(markerArray)

    def markerVector(self, id, scale, rotation, position, quaternion):
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
        return marker

    def deregisterTarget(self, id):
        pass

    def getTransformFromPose(self, pose):
        trans = tf.transformations.translation_matrix((pose.position.x, pose.position.y, pose.position.z))
        rot = tf.transformations.quaternion_matrix((pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))
        T = np.matmul(trans, rot)
        return T

    def publishTargetPoints(self):
        # convert target_points to pointcloud messages
        all_points = []
        for points in self.target_points:
            all_points = all_points + points.tolist()
            
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        # create pcl from points
        pc_msg = pcl2.create_cloud_xyz32(header, all_points)
        self.pcl_pub.publish(pc_msg)

    def measurePoints(self, old_points, points):
        p0 = copy.deepcopy(old_points)
        p1 = copy.deepcopy(points)
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

if __name__ == '__main__':
    rospy.init_node('target_localizer', anonymous=False)
    target_tracker = TargetTracker()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node killed!")