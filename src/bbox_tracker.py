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


class BBoxTracker(object):
    def __init__(self):
        bbox_nn_sub = rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.bbox_nn_callback)
        raw_image_sub = rospy.Subscriber('/r200/depth/image_raw', Image, self.raw_image_callback)

        self.bridge = CvBridge()
        #self.pub = rospy.Publisher("/camera_reader", Image, queue_size=1)  # debug

        self.bboxes_new = []

        self.X = []
        self.Var = []
        self.image_old = np.zeros(1)
        rospy.loginfo("bbox_tracker initialized!")

        while not rospy.is_shutdown():
            # update observation model
            if len(self.bboxes_new) > 0:
                # clear cache
                bboxes_new = copy.deepcopy(self.bboxes_new)
                X = copy.deepcopy(self.X)
                self.bboxes_new = []
                # update image
                self.image_old = copy.deepcopy(self.image_new)
                self.image_new = np.zeros(1)
                for i, bbox_new in enumerate(bboxes_new):
                    # check if new bbox is existing
                    idx = self.__checkRegistration(bbox_new, X)
                    if idx == -1:
                        self.__registerBBox(bbox_new, X)
                        # todo: check if X is updated
                    else:
                        self.__updateObservation(bbox_new, idx, X)
                        # todo: check if X is updated

                self.X = copy.deepcopy(X)

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
        self.bboxes_new = data.bounding_boxes  # bboxes is a list of darknet_ros_msgs.msg.BoundingBox

    def raw_image_callback(self, data):
        raw_image = self.bridge.imgmsg_to_cv2(data).astype(np.uint8)
        if len(raw_image.shape)>2:
            self.image_new = raw_image
            #_image = self.bridge.cv2_to_imgmsg(self.image_new, 'rgb8')
            #self.pub.publish(_image)

    def __registerBBox(self, bbox, X):
        return -1

    def __checkRegistration(self, bbox_new, X):
        """

        :param bbox_new:
        :return: -1: new bbox; idx: existing bbox
        """
        return -1

    def __updateObservation(self, bbox_new, idx, X):
        return -1, -1

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
