#!/usr/bin/env python
"""
Zhiang Chen
Jan 2021
iterative closest point
"""
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import open3d as o3d
from open3d_ros_conversion import convertCloudFromRosToOpen3d
import copy
import random

"""
ref: https://github.com/ClayFlannigan/icp/blob/master/icp.py
try this later: https://github.com/agnivsen/icp/blob/master/basicICP.py
"""

from sklearn.neighbors import NearestNeighbors

def random_sampling(orig_points, num_points):
    assert orig_points.shape[0] > num_points

    points_down_idx = random.sample(range(orig_points.shape[0]), num_points)
    down_points = orig_points[points_down_idx, :]

    return down_points

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

class ICP(object):
    def __init__(self, rosnode=True):
        if rosnode:
            self.pc_sub = rospy.Subscriber('/r200/depth/points', PointCloud2, self.callback, queue_size=1)
            self.id = 0

    def callback(self, pc_msg):
        o3d_pc = convertCloudFromRosToOpen3d(pc_msg)
        o3d.io.write_point_cloud(str(self.id)+".pcd", o3d_pc)
        self.id += 1

    def preprocess_point_cloud(self, pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def readfile(self, file):
        return o3d.io.read_point_cloud(file)

    def execute_global_registration(self, source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(False),
            4, [
                o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
        return result

    def refine_registration(self, source, target, source_fpfh, target_fpfh, voxel_size, trans_init):
        distance_threshold = voxel_size * 0.4
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
        result = o3d.registration.registration_icp(
            source, target, distance_threshold, trans_init,
            o3d.registration.TransformationEstimationPointToPlane())
        return result


if __name__ == '__main__':
    rospy.init_node('icp_node', anonymous=False)
    icp = ICP(rosnode=True)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node killed!")

"""
import open3d as o3d
from icp import ICP
from icp import draw_registration_result
init_trans = np.asarray(((1, 0, 0, 0),(0, 1, 0, 0),(0, 0, 1, 0),(0, 0, 0, 1)), dtype=float)
icp = ICP(False)
voxel_size = 0.05
pcd6 = icp.readfile('4.pcd')
pcd8 = icp.readfile('8.pcd')
pcd6_down, pcd6_fpfh = icp.preprocess_point_cloud(pcd6, voxel_size) 
pcd8_down, pcd8_fpfh = icp.preprocess_point_cloud(pcd8, voxel_size)
result_ransac = icp.execute_global_registration(pcd6_down, pcd8_down,
                                            pcd6_fpfh, pcd8_fpfh,
                                            voxel_size)
draw_registration_result(pcd6_down, pcd8_down, result_ransac.transformation)                                            

result_icp = icp.refine_registration(pcd6_down, pcd8_down, pcd6_fpfh, pcd8_fpfh,
                                 voxel_size, result_ransac.transformation)
draw_registration_result(pcd6_down, pcd8_down, result_icp.transformation)
threshold = voxel_size * 1.5
evaluation = o3d.registration.evaluate_registration(pcd6_down, pcd8_down, threshold, result_icp.transformation)
print(evaluation)
"""