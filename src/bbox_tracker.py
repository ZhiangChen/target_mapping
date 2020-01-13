#!/usr/bin/env python
"""
Zhiang Chen
Jan 2020
"""

import sys
import os
import rospy
from darknet_ros_msgs.msg import BoundingBoxes


class BBoxTracker(object):
    def __init__(self):
        bbox_nn_sub = rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.bbox_nn_callback)

        rospy.loginfo("bbox_tracker initialized!")

    def bbox_nn_callback(self, data):
        bboxes = data.bounding_boxes  # bboxes is a list of darknet_ros_msgs.msg.BoundingBox
        for bbox in bboxes:
            name = bbox.Class
            prob = bbox.probability
            xmin = bbox.xmin
            







if __name__ == '__main__':
    klt_tracker_path = os.path.abspath('..')
    sys.path.insert(0, klt_tracker_path)
    from KLT_Feature_Tracking.getFeatures import getFeatures
    from KLT_Feature_Tracking.estimateAllTranslation import estimateAllTranslation
    from KLT_Feature_Tracking.applyGeometricTransformation import applyGeometricTransformation

    rospy.init_node('bbox_tracker', anonymous=False)
    bbox_tracker = BBoxTracker()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Node killed!")
