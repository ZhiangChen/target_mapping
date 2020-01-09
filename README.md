# target_tracking

## Workflow
1. Deep learning object detection
2. KLT tracker
3. Kalman filtering to fuse bounding boxes from deep learning and KLT tracker
4. Camera model to generates a cone from the refined bounding box
5. Particle filtering to produce pointclouds of targets.

## ROS Packages
#### 1. [darknet_ros](https://github.com/leggedrobotics/darknet_ros)
It generates bounding boxes using deep neural networks.

#### 2. target_tracking
a. bbox_tracker(Kalman filter)  
subscriber_1: bbox from deep learning  
subscriber_2: bbox from KLT  
publisher_1: refined bbox  

b. target_tracker(Particle filter)  
subscriber_1: refined bbox from bbox_tracker  
subscriber_2: coarse global coordinates of camera  
publisher_1: pointcloud estimation of targets
