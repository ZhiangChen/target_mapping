#!/usr/bin/env python
"""
Zhiang Chen
Jan 2020
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



class BBoxTracker(object):
    def __init__(self):
        bbox_nn_sub = rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.bbox_nn_callback)
        raw_image_sub = rospy.Subscriber('/r200/depth/image_raw', Image, self.raw_image_callback)

        self.bridge = CvBridge()
        #self.pub = rospy.Publisher("/camera_reader", Image, queue_size=1)  # debug

        self.bboxes_new = []

        self.X = []  # a list of darknet_ros_msgs/BoundingBox
        self.Cov = []
        self.image_old = np.zeros(1) 

        self.Q = np.diag((1., 1., 1., 1.,))  # observation model noise covariance
        self.R = np.diag((2., 2., 2., 2.,))  # motion model noise covariance
        rospy.loginfo("bbox_tracker initialized!")

        while not rospy.is_shutdown():
            # update observation model
            if len(self.bboxes_new) > 0:
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
                    else:
                        self.__updateObservation(bbox_new, idx, X, Cov)
                        # todo: check if X is updated

                self.X = copy.deepcopy(X)
                self.Cov = copy.deepcopy(Cov)

            if len(self.image_new.shape) == 3:
                image_new = copy.deepcopy(self.image_new)
                self.image_new = np.zeros(1)
                if len(self.X) > 0:
                    if len(self.image_old.shape) != 3:
                        # todo: add new image; initialize KLT tracker
                        None
                    else:
                        self.__updateMotion(image_new, self.image_old, self.X)
                        self.image_old = copy.deepcopy(image_new)

    def bbox_nn_callback(self, data):
        """
        rosmsg info darknet_ros_msgs/BoundingBox
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
        if len(raw_image.shape)>2:
            self.image_new = raw_image
            #_image = self.bridge.cv2_to_imgmsg(self.image_new, 'rgb8')
            #self.pub.publish(_image)


    def __checkRegistration(self, bbox_new, X, threshold=0.5):
        """
        :param bbox_new:
        :return: -1: new bbox; idx: existing bbox
        """
        bbox = (bbox_new.xmin, bbox_new.ymin, bbox_new.xmax, bbox_new.ymax)
        for i, x in X:
            bbox_ = (x.xmin, x.ymin, x.xmax, x.ymax)
            iou = self.__bbox_IoU(bbox, bbox_)
            if iou >= threshold:
                return i

        return -1

    def __updateObservation(self, bbox_new, idx, X, Cov):
        bbox_old = X[idx]
        cov = Cov[idx]
        K = cov * inv(cov + self.Q)
        z = np.array((bbox_new.xmin, bbox_new.ymin, bbox_new.xmax, bbox_new.ymax))
        x = np.array((bbox_old.xmin, bbox_old.ymin, bbox_old.xmax, bbox_old.ymax))
        x = x + K * (z - x)
        cov = (np.identity(4) - K) * cov
        X[idx] = x
        Cov[idx] = cov

    def __updateMotion(self, new_image, X):
        return -1

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
