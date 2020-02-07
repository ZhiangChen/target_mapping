#!/usr/bin/env python
"""
Zhiang Chen
Jan 2020
The tracking loop is in a callback function
"""

import sys
import os
import rospy
import rospkg
from darknet_ros_msgs.msg import BoundingBoxes
from darknet_ros_msgs.msg import ObjectCount
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import copy
from numpy.linalg import inv
from numpy.linalg import det


class BBoxTracker(object):
    def __init__(self):
        bbox_nn_sub = rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.bbox_nn_callback, queue_size=1)
        raw_image_sub = rospy.Subscriber('/r200/depth/image_raw', Image, self.raw_image_callback, queue_size=1)

        self.bridge = CvBridge()
        self.pub = rospy.Publisher("/bbox_tracker/detection_image", Image, queue_size=1)
        self.refined_bboxes_msg = BoundingBoxes()
        self.pub1 = rospy.Publisher("/bbox_tracker/bounding_boxes", BoundingBoxes, queue_size=1)
        self.bboxes_new = []

        self.X = []  # a list of darknet_ros_msgs/BoundingBox
        self.Cov = []  # this is in Euclidean coordinate system
        self.image_new = np.zeros(1)
        self.image_old = np.zeros(1)
        self.startXs = np.empty((20, 0), int)
        self.startYs = np.empty((20, 0), int)
        self.bboxes_klt = np.empty((0, 4, 2), float)

        self.observation_done = False

        self.Q = np.diag((.2, .2, .2, .2))  # observation model noise covariance in Euclidean coordinate system
        self.R = np.diag((10., 10., 0., 10., 10., 0.))  # motion model noise covariance in Homogeneous coordinate system
        rospy.loginfo("bbox_tracker initialized!")

    def bbox_nn_callback(self, data):
        """
        rosmsg info darknet_ros_msgs/BoundingBoxes
        float64 probability
        int64 xmin
        int64 ymin
        int64 xmax
        int64 ymax
        int16 id
        string Class
        :param data:
        :return:
        """
        self.bboxes_new = data.bounding_boxes  # bboxes is a list of darknet_ros_msgs.msg.BoundingBox

    def raw_image_callback(self, data):
        raw_image = self.bridge.imgmsg_to_cv2(data).astype(np.uint8)
        if len(raw_image.shape) > 2:
            self.image_new = raw_image

        # update observation model
        if not self.observation_done:
            if len(self.bboxes_new) > 0:
                if len(self.image_new.shape) == 3:
                    self.observation_done = True
                    seconds = rospy.get_time()
                    # clear cache
                    bboxes_new = copy.deepcopy(self.bboxes_new)
                    X = copy.deepcopy(self.X)
                    Cov = copy.deepcopy(self.Cov)
                    self.bboxes_new = []
                    # update image
                    self.image_old = copy.deepcopy(self.image_new)
                    self.image_new = np.zeros(1)
                    for i, bbox_new in enumerate(bboxes_new):
                        # check if new bbox is existing
                        idx = self.__checkRegistration(bbox_new, X)
                        if idx == -1:
                            X.append(bbox_new)
                            Cov.append(self.Q)
                            bbox_klt = self.__bbox_msg2np([bbox_new])
                            startXs, startYs = getFeatures(cv2.cvtColor(self.image_old, cv2.COLOR_RGB2GRAY), bbox_klt,
                                                           use_shi=False)
                            self.startXs = np.append(self.startXs, startXs, axis=1)
                            self.startYs = np.append(self.startYs, startYs, axis=1)
                            self.bboxes_klt = np.append(self.bboxes_klt, bbox_klt, axis=0)

                        else:
                            self.__updateObservation(bbox_new, idx, X, Cov)

                    img = copy.deepcopy(self.image_old)
                    image_msg = self.__draw_BBox(img, X)
                    self.pub.publish(image_msg)
                    seconds = rospy.get_time() - seconds
                    self.X = copy.deepcopy(X)
                    self.Cov = copy.deepcopy(Cov)
                    print("obs " + str(len(self.X)) + " " + str(seconds))
                    self.__publish_bbox()

        # update motion model
        if len(self.image_new.shape) == 3:
            if len(self.X) > 0:
                image_new = copy.deepcopy(self.image_new)
                self.image_new = np.zeros(1)

                seconds = rospy.get_time()
                # self.__updateMotion(image_new, prediction=self.observation_done)
                self.__updateMotion(image_new, prediction=True)
                seconds = rospy.get_time() - seconds
                print("mot " + str(len(self.X)) + " " + str(seconds) + " " + str(self.observation_done))
                self.observation_done = False

                self.image_old = copy.deepcopy(image_new)
                image_msg = self.__draw_BBox(copy.deepcopy(self.image_old), self.X)
                self.pub.publish(image_msg)
                self.__publish_bbox()

    def __checkRegistration(self, bbox_new, X, threshold=0.3):
        """
        :param bbox_new:
        :return: -1: new bbox; idx: existing bbox
        """
        bbox = (bbox_new.xmin, bbox_new.ymin, bbox_new.xmax, bbox_new.ymax)
        for i, x in enumerate(X):
            bbox_ = (x.xmin, x.ymin, x.xmax, x.ymax)
            iou = self.__bbox_IoU(bbox, bbox_)
            if iou >= threshold:
                return i
        return -1

    def __updateObservation(self, bbox_new, idx, X, Cov):
        """
        Update observation model in Euclidean coordinate system
        :param bbox_new:
        :param idx:
        :param X:
        :param Cov:
        :return:
        """
        bbox_old = X[idx]
        cov = Cov[idx]
        K = cov.dot(inv(cov + self.Q))
        z = np.array((bbox_new.xmin, bbox_new.ymin, bbox_new.xmax, bbox_new.ymax)).astype(float)
        x = np.array((bbox_old.xmin, bbox_old.ymin, bbox_old.xmax, bbox_old.ymax)).astype(float)
        x = x + K.dot((z - x))
        cov = (np.identity(4) - K).dot(cov)
        X[idx].xmin = x[0]
        X[idx].ymin = x[1]
        X[idx].xmax = x[2]
        X[idx].ymax = x[3]
        Cov[idx] = cov
        bbox_klt = self.__bbox_msg2np([X[idx]])
        startXs, startYs = getFeatures(cv2.cvtColor(self.image_old, cv2.COLOR_RGB2GRAY), bbox_klt, use_shi=False)
        self.startXs[:, idx] = startXs[:, 0]
        self.startYs[:, idx] = startYs[:, 0]
        self.bboxes_klt[idx] = bbox_klt[0]

    def __updateMotion(self, new_image, prediction):
        newXs, newYs = estimateAllTranslation(self.startXs, self.startYs, self.image_old, new_image)

        if prediction:
            # print(1, self.Cov[0])
            Xs, Ys, self.bboxes_klt, self.Cov = applyGeometricTransformation(self.startXs, self.startYs, newXs, newYs, self.bboxes_klt, self.Cov, self.R)
            # print(1, self.Cov[0])
        else:
            Xs, Ys, self.bboxes_klt = applyGeometricTransformation(self.startXs, self.startYs, newXs, newYs, self.bboxes_klt)

        # update coordinates
        self.startXs = Xs
        self.startYs = Ys

        # update registration
        bboxes_klt = copy.deepcopy(self.bboxes_klt)
        height, width, _ = self.image_old.shape
        for i, bbox in reversed(list(enumerate(bboxes_klt))):
            # deregistration 1: measure differential entropy to remove false positive detection
            cov = self.Cov[i]
            DE = 5.675754132818691 + np.log(det(cov))/2.0 # differential entropy for Multivariate normal distribution
            if DE >= 17.:
                del self.X[i]
                del self.Cov[i]
                self.bboxes_klt = np.delete(self.bboxes_klt, i, 0)
                self.startXs = np.delete(self.startXs, i, 1)
                self.startYs = np.delete(self.startYs, i, 1)
                continue
            # deregistration 2: remove bbox out of frame
            if (bbox.min() < 0) or (bbox[2, 0] > width) or (bbox[2, 1] > height):
                del self.X[i]
                del self.Cov[i]
                self.bboxes_klt = np.delete(self.bboxes_klt, i, 0)
                self.startXs = np.delete(self.startXs, i, 1)
                self.startYs = np.delete(self.startYs, i, 1)

            else:
                # update bbox
                self.X[i].xmin = bbox[0][0]
                self.X[i].ymin = bbox[0][1]
                self.X[i].xmax = bbox[2][0]
                self.X[i].ymax = bbox[2][1]
                # generate new features if needed
                n_features_left = np.sum(self.startXs[:, i] != -1)
                if n_features_left < 15:
                    print('Generate New KLT Features')
                    bbox = np.expand_dims(self.bboxes_klt[i], 0)
                    startXs, startYs = getFeatures(cv2.cvtColor(self.image_old, cv2.COLOR_RGB2GRAY), bbox)
                    self.startXs[:, i] = startXs[:, 0]
                    self.startYs[:, i] = startYs[:, 0]

    def __bbox_msg2np(self, X):
        bboxes = np.zeros((len(X), 4, 2))
        for i, x in enumerate(X):
            bboxes[i, :, :] = np.array(((x.xmin, x.ymin), (x.xmax, x.ymin), (x.xmax, x.ymax), (x.xmin, x.ymax))).astype(float)
        return bboxes

    def __bbox_IoU(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def __draw_BBox(self, image, X, Image_msg=True):
        for x in X:
            image = cv2.rectangle(image, (int(x.xmin), int(x.ymin)), (int(x.xmax), int(x.ymax)), (0, 255, 255), 2)
        if Image_msg:
            image = self.bridge.cv2_to_imgmsg(image, 'rgb8')
        return image

    def __publish_bbox(self):
        if len(self.X) != 0:
            self.refined_bboxes_msg.header.stamp = rospy.Time.now()
            self.refined_bboxes_msg.bounding_boxes = copy.deepcopy(self.X)
            self.pub1.publish(self.refined_bboxes_msg)


if __name__ == '__main__':
    rospack = rospkg.RosPack()
    pack_path = rospack.get_path('target_tracking')
    sys.path.insert(0, pack_path)
    from KLT_Feature_Tracking.getFeatures import getFeatures
    from KLT_Feature_Tracking.estimateAllTranslation import estimateAllTranslation
    from KLT_Feature_Tracking.applyGeometricTransformation import applyGeometricTransformation
    rospy.init_node('bbox_tracker', anonymous=False)
    bbox_tracker = BBoxTracker()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node killed!")
