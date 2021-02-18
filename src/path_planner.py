#!/usr/bin/env python
"""
Zhiang Chen
May 2020
path planner
"""
import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from std_msgs.msg import Int8
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
import uav_motion.msg
import actionlib
import numpy as np
from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion
from std_srvs.srv import Empty
import copy
from threading import Thread
import target_mapping.msg
import tf
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from rtabmap_ros.srv import ResetPose, ResetPoseRequest
from utils.open3d_ros_conversion import convertCloudFromRosToOpen3d, convertCloudFromOpen3dToRos
import open3d as o3d
import rospkg
import yaml
import os
import time
import matplotlib.pyplot as plt
import visualization_msgs

rp = rospkg.RosPack()
pkg_path = rp.get_path('target_mapping')
config_path = os.path.join(pkg_path, 'config', 'target_mapping.yaml')
yaml_file = open(config_path)
params = yaml.load(yaml_file, Loader=yaml.FullLoader)

class PathPlanner(object):
    def __init__(self):
        self.id_ = -1
        self.current_pose_ = PoseStamped()
        self.current_pose_.pose.orientation.w = 1
        self.saved_pose_ = PoseStamped()
        self.marker_ = Marker()
        self.cylinder_marker_ = Marker()
        self.got_cylinder_marker_ = False
        self.goal_position_ = Point()
        self.goal_yaw_ = 0
        self.plan_mode_ = 0
        self.alpha = params['alpha']
        self.bcem_alpha = params['bcem_alpha']
        self.half_vfov = params['half_vfov']  # half vertical fov for mapping
        # self.alpha is the camera angle, which is supposed to be 60 degrees according to the camera mount angle.
        # However, if we set it as 60 degrees, the lower-bound scanning ray will be too long
        # For example, alpha = 60 degrees, half FoV = 20 degrees, distance to keep is 1.5 meters.
        # Then the vertical distance from the lower-bound scanning ray is 1.5*tan(60+20), which is 8.5 meters.
        # The vertical distance from the upper-bound scanning ray is 1.5*tan(60-20), which is 1.3 meters.
        self.mapping = False
        rp = rospkg.RosPack()
        pkg_path = rp.get_path('target_mapping')
        self.pcd_path = os.path.join(pkg_path, 'pcd')

        self.pc_map_ = PointCloud2()
        self.path = Path()
        self.path.header.frame_id = 'map'
        self.local_path_pub = rospy.Publisher("/local_path", Path, queue_size=1)
        self.poses = []

        self.cylinder_marker_pub_ = rospy.Publisher('/path_planner/cylinder_marker', Marker, queue_size=2)

        rospy.wait_for_service('stop_sampling')
        self.stop_srv_client_ = rospy.ServiceProxy('stop_sampling', Empty)
        self.as_ = actionlib.SimpleActionServer("/path_planner/target_plan", target_mapping.msg.TargetPlanAction,
                                                execute_cb=self.targetPlanCallback, auto_start=False)
        self.as_.start()
        current_pose_sub_ = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.poseCallback,
                                             queue_size=1)
        self.client_ = actionlib.SimpleActionClient('waypoints', uav_motion.msg.waypointsAction)
        self.client_.wait_for_server()

        #self.plan_thread_ = Thread(target=self.targetPlan, args=())
        #self.plan_thread_.daemon = True
        #self.plan_thread_.start()


        self.resumeMap_srv_client_ = rospy.ServiceProxy('/rtabmap/resume', Empty)
        self.pauseMap_srv_client_ = rospy.ServiceProxy('/rtabmap/pause', Empty)
        self.newMap_srv_client_ = rospy.ServiceProxy('/rtabmap/trigger_new_map', Empty)
        self.deleteMap_srv_client_ = rospy.ServiceProxy('/rtabmap/reset', Empty)
        self.setPose_srv_client_ = rospy.ServiceProxy('/rtabmap/reset_odom_to_pose', ResetPose)
        rospy.wait_for_service('/rtabmap/resume')
        rospy.wait_for_service('/rtabmap/pause')
        rospy.wait_for_service('/rtabmap/trigger_new_map')
        #self.newMap_srv_client_()
        self.deleteMap_srv_client_()
        self.pauseMap_srv_client_()

        map_sub_ = rospy.Subscriber('/rtabmap/cloud_map', PointCloud2, self.pointcloudCallback, queue_size=1)

        rospy.loginfo("Path planner has been initialized!")


    def startSearch(self):
        positions = np.asarray(((0, 0, 24), (-15, 10, 24), (1, 12, 24), (0, 0, 20)))  # granite dell search path
        #positions = self.lawnmower(pt1=(50, 35), pt2=(-50, -35), origin=(30, 38), spacing=10, vertical=True) # pt1=(-50, -35)
        #positions = self.lawnmower(pt1=(0, 35), pt2=(-50, -35), origin=(30, 38), spacing=10, vertical=True)  # pt1=(-50, -35)
        #positions = self.add_height(positions, 17.)  # for blender_terrain, [10, 17]
        yaws = self.getHeads(positions)

        assert positions.shape[0] == len(yaws)

        for i in range(len(yaws)):
            goal = uav_motion.msg.waypointsGoal()
            goal_p = positions[i]

            self.goal_position_.x = float(goal_p[0])
            self.goal_position_.y = float(goal_p[1])
            self.goal_position_.z = float(goal_p[2])
            q = self.current_pose_.pose.orientation
            yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
            self.goal_yaw_ = yaw

            goal.positions.append(self.goal_position_)
            goal.yaws.append(yaw)
            self.client_.send_goal(goal)
            while True & (not rospy.is_shutdown()):
                rospy.sleep(1.)
                current_p = np.asarray((self.current_pose_.pose.position.x,
                                        self.current_pose_.pose.position.y,
                                        self.current_pose_.pose.position.z))
                dist = np.linalg.norm(goal_p - current_p)
                if self.got_cylinder_marker_:
                    self.cylinder_marker_pub_.publish(self.cylinder_marker_)

                if dist < 0.2:
                    break

            rospy.sleep(1.)
            goal = uav_motion.msg.waypointsGoal()
            goal.positions.append(self.goal_position_)
            goal.yaws.append(yaws[i])
            self.client_.send_goal(goal)
            rospy.sleep(5.)

    def getHeads(self, waypoints):
        yaws = []
        nm = waypoints.shape[0]
        for i in range(nm-1):
            currnt_p = waypoints[i][:2]
            nxt_p = waypoints[i+1][:2]
            dir = nxt_p - currnt_p
            yaws.append(np.arctan2(dir[1], dir[0]))

        yaws.append(0)
        return yaws

    def poseCallback(self, pose):
        self.current_pose_ = pose
        self.poses.append(pose)
        self.path.poses = self.poses
        self.local_path_pub.publish(self.path)


    def pointcloudCallback(self, pc_msg):
        if self.mapping:
            self.pc_map_ = pc_msg
            #xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc_msg)

    def targetPlanCallback(self, target_plan):
        if (self.plan_mode_ == 0) & (target_plan.mode.data != 0):
            self.saved_pose_ = copy.deepcopy(self.current_pose_)

        self.id_ = target_plan.id.data
        self.plan_mode_ = target_plan.mode.data
        if self.plan_mode_ != 0:
            self.marker_ = target_plan.markers.markers[self.id_]

        self.stop_srv_client_()
        rospy.sleep(3.)

        result = target_mapping.msg.TargetPlanResult()
        if self.plan_mode_ == 1:
            result.success = self.get_bcylinder_estimating_motion()
            self.as_.set_succeeded(result)
        elif self.plan_mode_ == 2:
            result = self.getMapping()
            self.as_.set_succeeded(result)
        elif self.plan_mode_ == 0:
            print("resuming")
            save_position = Point()
            save_position.x = self.saved_pose_.pose.position.x
            save_position.y = self.saved_pose_.pose.position.y
            save_position.z = self.saved_pose_.pose.position.z

            q = self.saved_pose_.pose.orientation
            yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]

            goal = uav_motion.msg.waypointsGoal()
            goal.positions.append(save_position)
            goal.yaws.append(yaw)
            self.client_.send_goal(goal)
            while True & (not rospy.is_shutdown()):
                rospy.sleep(1.)
                current_p = np.asarray((self.current_pose_.pose.position.x,
                                        self.current_pose_.pose.position.y,
                                        self.current_pose_.pose.position.z))
                goal_p = np.asarray((save_position.x,
                                     save_position.y,
                                     save_position.z))
                dist = np.linalg.norm(goal_p - current_p)
                if dist < 0.2:
                    break
            rospy.sleep(1.)
            goal = uav_motion.msg.waypointsGoal()
            goal.positions.append(self.goal_position_)
            goal.yaws.append(self.goal_yaw_)
            self.client_.send_goal(goal)
            result.success = True
            self.as_.set_succeeded(result)

    def get_bcylinder_estimating_motion(self):
        print('b-cylinder estimation motion')
        # 1. generate a circle
        # use the center of the marker, (x, y),
        # and the current drone height, (h), as the circle center, (x, y, h).
        # then we only need to decide the radius of the circle.
        # assume the target is always in the center of the image,
        # we can compute the angle between camera's z axis and horizontal plane, alpha.
        # the circle will be determined by object center (x, y, z), h, and alpha
        marker_position = np.asarray((self.marker_.pose.position.x, self.marker_.pose.position.y, self.marker_.pose.position.z))
        drone_position = np.asarray((self.current_pose_.pose.position.x, self.current_pose_.pose.position.y, self.current_pose_.pose.position.z))
        h = self.current_pose_.pose.position.z
        circle_center = np.asarray((self.marker_.pose.position.x, self.marker_.pose.position.y, h))
        radius = (h - marker_position[2])/np.tan(self.bcem_alpha)

        # 2. sample keypoints
        # from drone's closest point to the farthest point
        # get the closest point
        dir_cp = drone_position - circle_center
        dir_cp = dir_cp/np.linalg.norm(dir_cp)
        # cp = circle_center + dir_cp * radius
        # get the farthest point
        """
        # this is ok to find the farthest point that is farthest to the longest axis
        marker_q = (self.marker_.pose.orientation.x, self.marker_.pose.orientation.y, self.marker_.pose.orientation.z, self.marker_.pose.orientation.w)
        marker_rot = tf.transformations.quaternion_matrix(marker_q)
        marker_scale = (self.marker_.scale.x, self.marker_.scale.y, self.marker_.scale.z)
        idx = np.argmax(marker_scale)
        long_axis = marker_rot[:, idx]
        """
        # or the farthest point is the opposite of the closest point
        positions = []
        yaws = []
        N = 25  # the number of key points on the trajectory
        step = 4*np.pi/(N-1)
        yaw_cp = np.arctan2(-dir_cp[1], -dir_cp[0])
        for i in range(N):
            dir_i = self.rotateDirection(dir_cp, step*i)
            pos = circle_center + dir_i * radius
            #yaw = np.arctan2(-dir_i[1], -dir_i[0])  # this will cause some issues because atan2 is not continuous
            yaw = yaw_cp + step*i
            positions.append(pos)
            yaws.append(yaw)
        self.sendWaypoints(positions, yaws)
        return True


    def getMapping(self):
        print('mapping motion')
        # get target position
        marker_position = np.asarray((self.marker_.pose.position.x, self.marker_.pose.position.y, self.marker_.pose.position.z))
        # get target points
        points = np.asarray([(p.x, p.y, p.z) for p in self.marker_.points])
        # extract points in 3 sigma
        three_sigma_stds = points.std(axis=0) * 3
        pillar_radius_0 = three_sigma_stds[:2].max()
        pillar_top_0 = marker_position[2] + three_sigma_stds[2]
        pillar_bottom_0 = marker_position[2] - three_sigma_stds[2]
        # approximate points with a pillar
        pillar_radius_1 = np.linalg.norm(points[:, :2] - marker_position[:2], axis=1).max()  # the radius can also be defined by Gaussian sigma distance
        pillar_top_1 = points[:, 2].max()
        pillar_bottom_1 = points[:, 2].min() #+ pillar_radius * np.tan(self.alpha)
        pillar_radius = min(pillar_radius_0, pillar_radius_1)
        pillar_top = min(pillar_top_0, pillar_top_1)
        pillar_bottom = min(pillar_bottom_0, pillar_bottom_1)

        cylinder_pos = marker_position
        cylinder_scale = [pillar_radius*2, pillar_radius*2, pillar_top - points[:, 2].min()]
        self.cylinder_marker_ = self.create_cylinder_marker(pos=cylinder_pos, scale=cylinder_scale)
        self.got_cylinder_marker_ = True
        """
        # get target height (not real height, it's eigenvalue of the vertical vector)
        marker_q = (self.marker_.pose.orientation.x, self.marker_.pose.orientation.y, self.marker_.pose.orientation.z,
                    self.marker_.pose.orientation.w)
        marker_rot = tf.transformations.quaternion_matrix(marker_q)
        height = (marker_rot[:, 0] * self.marker_.scale.x)[2]
        """
        # map plan: sweep from bottom to top
        ## get circular planes
        dist = 1.5  # distance to keep between drone and the closest pillar surface
        half_vfov = self.half_vfov
        h1 = dist * np.tan(self.alpha + half_vfov)
        h2 = dist * np.tan(self.alpha - half_vfov)
        d = h1 - h2
        N = int(round(np.ceil((pillar_top - pillar_bottom) / d)))  # number of sweeping planes
        heights = [pillar_bottom + d * i + h1 for i in range(N)]
        n = 15  # number of waypoints on a circular path
        radius = pillar_radius + dist

        ## get start position
        drone_position = np.asarray((self.current_pose_.pose.position.x, self.current_pose_.pose.position.y,
                                     self.marker_.pose.position.z))
        dir_cp = drone_position - marker_position
        dir_cp = dir_cp/np.linalg.norm(dir_cp)
        ## get path points
        positions = []
        yaws = []
        last_yaw = 0
        for i in range(N):
            center = np.asarray((marker_position[0], marker_position[1], heights[i]))
            p, y = self.circularPoints(dir_cp, center, radius, n)
            positions.append(p)
            yaws.append(y)

        positions = np.asarray(positions).reshape(-1, 3)
        yaws = np.asarray(yaws).reshape(-1, 1)

        start_p = positions[0]
        start_y = yaws[0]
        point = Point(start_p[0], start_p[1], start_p[2])
        goal = uav_motion.msg.waypointsGoal()
        goal.positions.append(point)
        goal.yaws.append(start_y)
        self.client_.send_goal(goal)
        while True & (not rospy.is_shutdown()):
            rospy.sleep(1.)
            current_p = np.asarray((self.current_pose_.pose.position.x,
                                    self.current_pose_.pose.position.y,
                                    self.current_pose_.pose.position.z))

            dist = np.linalg.norm(start_p - current_p)
            if dist < 0.2:
                break
        rospy.sleep(2.)
        """
        pose = ResetPoseRequest()
        pose.x = self.current_pose_.pose.position.x
        pose.y = self.current_pose_.pose.position.y
        pose.z = self.current_pose_.pose.position.z
        q = self.current_pose_.pose.orientation

        euler = euler_from_quaternion((q.x, q.y, q.z, q.w))
        pose.roll = euler[0]
        pose.pitch = euler[1]
        pose.yaw = euler[2]
        #self.setPose_srv_client_(pose)
        """
        self.mapping = True
        self.resumeMap_srv_client_()

        self.sendWaypoints(positions[1:], yaws[1:])
        last_p = positions[-1]
        while True & (not rospy.is_shutdown()):
            rospy.sleep(1.)
            current_p = np.asarray((self.current_pose_.pose.position.x,
                                    self.current_pose_.pose.position.y,
                                    self.current_pose_.pose.position.z))

            dist = np.linalg.norm(last_p - current_p)
            if dist < 0.2:
                break
        self.mapping = False
        # save pointcloud map
        print('saving map')
        pc_map_msg = copy.copy(self.pc_map_)
        o3d_pc = convertCloudFromRosToOpen3d(pc_map_msg)
        # downsampling
        o3d_pc = o3d_pc.voxel_down_sample(0.05)
        # extract points in a sphere
        sphere_center = cylinder_pos
        sphere_radius = np.linalg.norm(np.asarray(cylinder_scale)/2.)
        pts = np.asarray(o3d_pc.points)
        clrs = np.asarray(o3d_pc.colors)
        in_sphere_bools = [np.linalg.norm(pt - sphere_center) <= sphere_radius for pt in pts]
        in_pts = pts[in_sphere_bools]
        in_clrs = clrs[in_sphere_bools]
        map_pcd = o3d.geometry.PointCloud()
        map_pcd.points = o3d.utility.Vector3dVector(in_pts)
        map_pcd.colors = o3d.utility.Vector3dVector(in_clrs)
        pcd_name = os.path.join(self.pcd_path, str(self.id_) + ".pcd")
        o3d.io.write_point_cloud(pcd_name, map_pcd)

        self.newMap_srv_client_()
        self.deleteMap_srv_client_()
        self.pauseMap_srv_client_()
        self.got_cylinder_marker_ = False

        result = target_mapping.msg.TargetPlanResult()
        if len(in_sphere_bools) > 0:
            result.success = True
            result.pointcloud_map = convertCloudFromOpen3dToRos(map_pcd, 'map')
        else:
            result.success = False
        return result

    def circularPoints(self, dir_cp, center, radius, n):
        positions = []
        yaws = []
        step = 2 * np.pi / n
        yaw_cp = np.arctan2(-dir_cp[1], -dir_cp[0])
        for i in range(n):
            dir_i = self.rotateDirection(dir_cp, step * i)
            pos = center + dir_i * radius
            # yaw = np.arctan2(-dir_i[1], -dir_i[0])  # this will cause some issues because atan2 is not continuous
            yaw = yaw_cp + step * i
            positions.append(pos)
            yaws.append(yaw)
        return positions, yaws



    def rotateDirection(self, d, theta):
        r = np.array(((np.cos(theta), -np.sin(theta), 0),
                      (np.sin(theta), np.cos(theta), 0),
                      (0, 0, 0,)))
        return np.matmul(r, d)

    def sendWaypoints(self, positions, yaws):
        goal = uav_motion.msg.waypointsGoal()
        for i in range(len(yaws)):
            p = positions[i]
            yaw = yaws[i]
            point = Point(p[0], p[1], p[2])
            goal.positions.append(point)
            goal.yaws.append(yaw)
        self.client_.send_goal(goal)


    def lawnmower(self, pt1, pt2, origin, spacing, vertical):
        """
        :param pt1: start point (x, y)
        :param pt2: end point (x, y)
        :param origin: uav origin (x, y)
        :param spacing:
        :param vertical:
        :return:
        """
        origin = np.array(origin)
        pt1 = np.array(pt1) - origin
        pt2 = np.array(pt2) - origin
        x1, y1 = pt1
        x2, y2 = pt2
        width = x2 - x1
        length = y2 - y1
        waypoints = [np.array((0., 0.)), pt1]
        if vertical:
            if width < 0:
                spacing = - spacing
            N = int(width / spacing / 2)
            for i in range(N):
                pt_0 = waypoints[-1]
                pt_1 = pt_0 + np.array((0, length))
                pt_2 = pt_1 + np.array((spacing, 0))
                pt_3 = pt_2 + np.array((0, -length))
                pt_4 = pt_3 + np.array((spacing, 0))
                waypoints.append(pt_1)
                waypoints.append(pt_2)
                waypoints.append(pt_3)
                waypoints.append(pt_4)
        else:
            if length < 0:
                spacing = - spacing
            N = int(length / spacing / 2)
            for i in range(N):
                pt_0 = waypoints[-1]
                pt_1 = pt_0 + np.array((width, 0))
                pt_2 = pt_1 + np.array((0, spacing))
                pt_3 = pt_2 + np.array((-width, 0))
                pt_4 = pt_3 + np.array((0, spacing))
                waypoints.append(pt_1)
                waypoints.append(pt_2)
                waypoints.append(pt_3)
                waypoints.append(pt_4)
        waypoints.append(pt2)
        return np.array(waypoints)


    def plot_path(self, waypoints):
        waypoints = np.array(waypoints)
        x = waypoints[:, 0]
        y = waypoints[:, 1]
        plt.plot(x, y)
        plt.show()

    def add_height(self, waypoints, height):
        N = waypoints.shape[0]
        new_waypoints = np.zeros((N, 3))
        new_waypoints[:, :2] = waypoints
        new_waypoints[:, 2] = height
        return new_waypoints

    def create_cylinder_marker(self, pos=[0, 0, 0], qua=[0, 0, 0, 1], scale=[1, 1, 1]):
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
        marker.type = visualization_msgs.msg.Marker.CYLINDER
        marker.action = visualization_msgs.msg.Marker.ADD
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]
        marker.color.a = .5
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.5
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        marker.pose.orientation.x = qua[0]
        marker.pose.orientation.y = qua[1]
        marker.pose.orientation.z = qua[2]
        marker.pose.orientation.w = qua[3]
        return marker



if __name__ == '__main__':
    rospy.init_node('path_planner', anonymous=False)
    path_planner = PathPlanner()
    path_planner.startSearch()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node killed!")
