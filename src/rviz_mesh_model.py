#!/usr/bin/env python
"""
Zhiang Chen
Jan 2021
visualize mesh model in rviz
"""

import rospy
from visualization_msgs.msg import Marker, MarkerArray
import visualization_msgs

def create_mesh_marker():
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "target_mapping"
    marker.id = 0
    marker.type = visualization_msgs.msg.Marker.MESH_RESOURCE
    # marker.type = visualization_msgs.msg.Marker.CYLINDER
    marker.mesh_resource = "package://gazebo_sim_models/models/granite_dell/granite_dell.dae"
    marker.action = visualization_msgs.msg.Marker.ADD
    marker.scale.x = 1.
    marker.scale.y = 1.
    marker.scale.z = 1.
    marker.color.a = .5
    marker.color.r = .5
    marker.color.g = .5
    marker.color.b = 0.
    marker.pose.position.x = 0.005115
    marker.pose.position.y = -0.008963
    marker.pose.position.z = -0.133592
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1.
    return marker

def create_cylinder_marker(pos=[0,0,0], qua=[0,0,0,1], scale=[1,1,1]):
    """
    :param pos: [x, y, z]
    :param qua: [x, y, z, w]
    :param scale: [diameter_x, diameter_y, height]; the first two params are diameters for an ellipse
    :return:
    """
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "target_mapping"
    marker.id = 0
    #marker.type = visualization_msgs.msg.Marker.MESH_RESOURCE
    marker.type = visualization_msgs.msg.Marker.CYLINDER
    #marker.mesh_resource = "package://gazebo_sim_models/models/granite_dell/granite_dell.dae"
    marker.action = visualization_msgs.msg.Marker.ADD
    marker.scale.x = scale[0]
    marker.scale.y = scale[1]
    marker.scale.z = scale[2]
    marker.color.a = .5
    marker.color.r = .5
    marker.color.g = .5
    marker.color.b = 0.
    marker.pose.position.x = pos[0]
    marker.pose.position.y = pos[1]
    marker.pose.position.z = pos[2]
    marker.pose.orientation.x = qua[0]
    marker.pose.orientation.y = qua[1]
    marker.pose.orientation.z = qua[2]
    marker.pose.orientation.w = qua[3]
    return marker


if __name__ == '__main__':
    rospy.init_node('rviz_mesh_model', anonymous=False)
    pub = rospy.Publisher('mesh_model', Marker, queue_size=10)
    r = rospy.Rate(.1)  # 10hz
    while not rospy.is_shutdown():
        #marker = create_mesh_marker()
        marker = create_cylinder_marker(scale=[1, 1, 1])
        pub.publish(marker)
        r.sleep()