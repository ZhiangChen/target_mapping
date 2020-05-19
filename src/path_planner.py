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

class PathPlanner(object):
    def __init__(self):
        self.id_ = -1
        self.current_pose_ = PoseStamped()
        self.current_pose_.pose.orientation.w = 1
        self.saved_pose_ = PoseStamped()
        self.marker_ = Marker()
        self.goal_pose_ = Pose()
        self.plan_mode_ = 0

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
        positions = np.asarray(((0, 0, 4),  (-15, -15, 4), (0, 0, 4)))
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
        self.id_ = target_plan.id.data
        self.plan_mode_ = target_plan.mode.data
        if self.plan_mode_ != 0:
            self.marker_ = target_plan.markers.markers[self.id_]

        self.saved_pose_ = copy.deepcopy(self.current_pose_)
        self.stop_srv_client_()
        rospy.sleep(1.5)
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
        rospy.sleep(3.0)

        result = target_mapping.msg.TargetPlanResult()

        if self.plan_mode_ == 1:
            result.success = self.getLocalizing()
        elif self.plan_mode_ == 2:
            result.success = self.getMapping()
        else:
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
        rospy.sleep(5)
        return True

    def getMapping(self):
        print('mapping')
        rospy.sleep(5)
        return True





if __name__ == '__main__':
    rospy.init_node('path_planner', anonymous=False)
    path_planner = PathPlanner()
    path_planner.startSearch()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node killed!")