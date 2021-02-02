#!/usr/bin/env python
"""
Zhiang Chen
Jan 2021
tf broadcast between odom and base_link
"""

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import tf_conversions
import tf2_ros
import geometry_msgs.msg

class Odom_TF_Broadcaster(object):
    def __init__(self):
        #self.sub = rospy.Subscriber('/gazebo/odom', Odometry, self.callback, queue_size=1)
        #self.sub = rospy.Subscriber('/mavros/odometry/in', Odometry, self.callback, queue_size=1)
        self.sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.ps_callback, queue_size=1)
        self.br = tf2_ros.TransformBroadcaster()
        rospy.loginfo("odom_tf_broadcaster has been initialized!")

    def callback(self, odom_msg):
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = odom_msg.header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        p = odom_msg.pose.pose.position
        t.transform.translation.x = p.x
        t.transform.translation.y = p.y
        t.transform.translation.z = p.z
        q = odom_msg.pose.pose.orientation
        t.transform.rotation.x = q.x
        t.transform.rotation.y = q.y
        t.transform.rotation.z = q.z
        t.transform.rotation.w = q.w
        self.br.sendTransform(t)

    def ps_callback(self, ps_msg):
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = ps_msg.header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        p = ps_msg.pose.position
        t.transform.translation.x = p.x
        t.transform.translation.y = p.y
        t.transform.translation.z = p.z
        q = ps_msg.pose.orientation
        t.transform.rotation.x = q.x
        t.transform.rotation.y = q.y
        t.transform.rotation.z = q.z
        t.transform.rotation.w = q.w
        self.br.sendTransform(t)


if __name__ == '__main__':
    rospy.init_node('odom_tf_broadcaster', anonymous=False)
    br = Odom_TF_Broadcaster()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node killed!")