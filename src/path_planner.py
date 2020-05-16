#!/usr/bin/env python
"""
Zhiang Chen
May 2020
path planner
"""
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import UInt8
from visualization_msgs.msg import Marker, MarkerArray
import uav_motion.msg
import actionlib
import numpy as np

class PathPlanner(object):
    def __init__(self):
        current_pose_sub_ = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.poseCallback, queue_size=1)
        status_sub_ = rospy.Subscriber('/path_planner/status', UInt8, self.statusCallback, queue_size=1)
        markers_sub = rospy.Subscriber("/target_localizer/ellipsoids", self.markersCallback, MarkerArray, queue_size=1)
        self.client_ = actionlib.SimpleActionClient('waypoints', uav_motion.msg.waypointsAction)
        self.client_.wait_for_server()
        self.status_ = 0
        self.current_pose_ = PoseStamped()
        self.saved_pose_ = PoseStamped()
        self.marker_ = Marker()

        self.initSearch()
        rospy.loginfo("Path planner has been initialized!")


    def initSearch(self):
        pass

    def poseCallback(self, pose):
        pass

    def markersCallback(self, markers):
        pass

    def statusCallback(self, status):
        if status.data != self.status_:
            pass

    def getLocalizerPath(self):
        pass

    def getMapperPath(self):
        pass

    def recoverSearch(self):
        pass


if __name__ == '__main__':
    rospy.init_node('path_planner', anonymous=False)
    path_planner = PathPlanner()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node killed!")