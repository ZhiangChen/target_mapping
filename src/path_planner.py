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

class PathPlanner(object):
    def __init__(self):
        self.id_ = -1
        self.current_pose_ = PoseStamped()
        self.current_pose_.pose.orientation.w = 1
        self.saved_pose_ = PoseStamped()
        self.marker_ = Marker()
        self.goal_position_ = Point()
        self.goal_yaw_ = 0
        self.plan_mode_ = 0
        self.alpha = 45./180*np.pi  # camera angle
        self.mapping = False
        self.pc_map_ = PointCloud2()
        self.path = Path()
        self.path.header.frame_id = 'map'
        self.local_path_pub = rospy.Publisher("/local_path", Path, queue_size=1)
        self.poses = []

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
        #positions = np.asarray(((0, 0, 10), (-18, 0, 10), (0, 0, 6)))
        positions = np.asarray(((0, 0, 6), (-6, -6, 10), (0, 0, 5)))
        positions = np.asarray(((0, 0, 5), (-15, -15, 5), (0, 0, 5)))
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
            result.success = self.getLocalizing()
            self.as_.set_succeeded(result)
        elif self.plan_mode_ == 2:
            result.success = self.getMapping()
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

    def getLocalizing(self):
        print('localizing')
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
        radius = (h - marker_position[2])/np.tan(self.alpha)

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
        print('mapping')
        # get target position
        marker_position = np.asarray((self.marker_.pose.position.x, self.marker_.pose.position.y, self.marker_.pose.position.z))
        # get target points
        points = np.asarray([(p.x, p.y, p.z) for p in self.marker_.points])
        # approximate points with a pillar
        pillar_radius = np.linalg.norm(points[:, :2] - marker_position[:2], axis=1).max()  # the radius can also be defined by Gaussian sigma distance
        pillar_top = points[:, 2].max()
        pillar_bottom = points[:, 2].min() + pillar_radius * np.tan(self.alpha)

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
        half_vfov = 20. / 180 * np.pi
        h1 = dist * np.tan(self.alpha + half_vfov)
        h2 = dist * np.tan(self.alpha - half_vfov)
        d = h1 + h2
        N = int(np.ceil((pillar_top - pillar_bottom) / d))  # number of sweeping planes
        heights = [pillar_bottom + d * i + h1 - 1 for i in range(N)]
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
        self.newMap_srv_client_()
        self.deleteMap_srv_client_()
        self.pauseMap_srv_client_()

        return True

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



if __name__ == '__main__':
    rospy.init_node('path_planner', anonymous=False)
    path_planner = PathPlanner()
    path_planner.startSearch()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node killed!")
