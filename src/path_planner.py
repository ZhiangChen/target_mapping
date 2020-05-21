#!/usr/bin/env python
"""
Zhiang Chen
May 2020
path planner
"""
import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from std_msgs.msg import Int8
from visualization_msgs.msg import Marker, MarkerArray
import uav_motion.msg
import actionlib
import numpy as np
from tf.transformations import quaternion_from_euler
from std_srvs.srv import Empty
import copy
from threading import Thread
import target_mapping.msg
import tf

class PathPlanner(object):
    def __init__(self):
        self.id_ = -1
        self.current_pose_ = PoseStamped()
        self.current_pose_.pose.orientation.w = 1
        self.saved_pose_ = PoseStamped()
        self.marker_ = Marker()
        self.goal_pose_ = Pose()
        self.plan_mode_ = 0
        self.alpha = 45./180*np.pi  # camera angle

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

        rospy.loginfo("Path planner has been initialized!")


    def startSearch(self):
        positions = np.asarray(((0, 0, 4),  (-15, -15, 4)))
        yaws = self.getHeads(positions)
        assert positions.shape[0] == len(yaws)

        for i in range(len(yaws)):
            goal = uav_motion.msg.waypointsGoal()
            goal_p = positions[i]

            self.goal_pose_.position.x = float(goal_p[0])
            self.goal_pose_.position.y = float(goal_p[1])
            self.goal_pose_.position.z = float(goal_p[2])

            self.goal_pose_.orientation.x = self.current_pose_.pose.orientation.x
            self.goal_pose_.orientation.y = self.current_pose_.pose.orientation.y
            self.goal_pose_.orientation.z = self.current_pose_.pose.orientation.z
            self.goal_pose_.orientation.w = self.current_pose_.pose.orientation.w

            goal.poses.append(self.goal_pose_)
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
            goal_p = positions[i]
            yaw = yaws[i]
            goal_q = quaternion_from_euler(0, 0, yaw)
            self.goal_pose_.position.x = float(goal_p[0])
            self.goal_pose_.position.y = float(goal_p[1])
            self.goal_pose_.position.z = float(goal_p[2])
            self.goal_pose_.orientation.x = goal_q[0]
            self.goal_pose_.orientation.y = goal_q[1]
            self.goal_pose_.orientation.z = goal_q[2]
            self.goal_pose_.orientation.w = goal_q[3]
            goal.poses.append(self.goal_pose_)
            self.client_.send_goal(goal)
            rospy.sleep(5.)
        rospy.loginfo('Done!')

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
        elif self.plan_mode_ == 2:
            result.success = self.getMapping()
        else:
            print("resuming")
            save_pose = Pose()
            save_pose.position.x = self.saved_pose_.pose.position.x
            save_pose.position.y = self.saved_pose_.pose.position.y
            save_pose.position.z = self.saved_pose_.pose.position.z

            save_pose.orientation.x = self.saved_pose_.pose.orientation.x
            save_pose.orientation.y = self.saved_pose_.pose.orientation.y
            save_pose.orientation.z = self.saved_pose_.pose.orientation.z
            save_pose.orientation.w = self.saved_pose_.pose.orientation.w
            goal = uav_motion.msg.waypointsGoal()
            goal.poses.append(save_pose)
            self.client_.send_goal(goal)
            while True & (not rospy.is_shutdown()):
                rospy.sleep(1.)
                current_p = np.asarray((self.current_pose_.pose.position.x,
                                        self.current_pose_.pose.position.y,
                                        self.current_pose_.pose.position.z))
                goal_p = np.asarray((save_pose.position.x,
                                     save_pose.position.y,
                                     save_pose.position.z))
                dist = np.linalg.norm(goal_p - current_p)
                if dist < 0.2:
                    break
            rospy.sleep(1.)
            goal = uav_motion.msg.waypointsGoal()
            goal.poses.append(self.goal_pose_)
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
        N = 10  # the number of key points on the trajectory
        step = 0.8*np.pi/(N-1)
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
        rospy.sleep(5)
        return True

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
            q = quaternion_from_euler(0, 0, yaw)
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.position.z = float(p[2])
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            goal.poses.append(pose)
        self.client_.send_goal(goal)



if __name__ == '__main__':
    rospy.init_node('path_planner', anonymous=False)
    path_planner = PathPlanner()
    path_planner.startSearch()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node killed!")
