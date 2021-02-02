#!/usr/bin/env python
"""
Zhiang Chen
Jan 2021
convert nav_msgs/Odometry to geometry_msgs/TransformStamped for pointcloud registration using voxblox
"""

import rospy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

class Odom2tfmsg(object):
    def __init__(self):
        self.tfmsg_pub = rospy.Publisher("/camera_transform", TransformStamped, queue_size=1)
        self.current_odom_sub_ = rospy.Subscriber('/mavros/global_position/local', Odometry, self.callback,
                                             queue_size=1)
        rospy.loginfo("ros node odom2tfmsg has been initialized")

    def callback(self, data):
        tfmsg = TransformStamped()
        tfmsg.header = data.header
        tfmsg.child_frame_id = "pointcloud_camera"
        tfmsg.transform.translation.x = data.pose.pose.position.x
        tfmsg.transform.translation.y = data.pose.pose.position.y
        tfmsg.transform.translation.z = data.pose.pose.position.z
        tfmsg.transform.rotation = data.pose.pose.orientation
        self.tfmsg_pub.publish(tfmsg)

if __name__ == '__main__':
    rospy.init_node('odom2tfmsg', anonymous=False)
    converter = Odom2tfmsg()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node killed!")