# Copyright (c) 2026 ETH Zurich. All rights reserved.
# SPDX-FileCopyrightText: 2026 ETH Zurich
# SPDX-License-Identifier: MIT
# Author: Jonas Kühne


"""
    ROS Bag Data Extraction Module
    
    Provides utilities to extract image and IMU data from rosbag files for processing
    by the VIO pipeline. Supports generator-based streaming of data to avoid loading
    entire datasets into memory.
"""
import rosbag
import numpy as np

class IMUData:
    """Container for IMU measurements from a rosbag message.
    
    Extracts and stores angular velocity (gyroscope) and linear acceleration
    (accelerometer) data from ROS IMU messages for use in VIO preintegration.
    """
    
    def __init__(self, msg):
        """Initialize IMU data from ROS message.
        
        Args:
            msg: ROS Imu message containing angular_velocity and linear_acceleration fields
        """
        self.rot_vel = np.array([msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z])
        self.lin_acc = np.array([msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z])

    def __repr__(self):
        return ("Angular Velocity (rad/s): [%f, %f, %f] \nLinear Velocity (m/s^2):  [%f, %f, %f]" % 
                (self.rot_vel[0],self.rot_vel[1],self.rot_vel[2],self.lin_acc[0],self.lin_acc[1],self.lin_acc[2]))
    
    def apply_transform(self, tf):
        """Apply a rigid body transformation to IMU measurements.
        
        Transforms angular velocity and linear acceleration from one coordinate frame
        to another using the rotation component of the transformation matrix.
        
        Args:
            tf (np.ndarray): 4x4 homogeneous transformation matrix
        """
        self.rot_vel = tf[:3,:3]@self.rot_vel
        self.lin_acc = tf[:3,:3]@self.lin_acc

class RosbagExtractor:
    """Utility class for extracting camera and IMU data from ROS bag files.
    
    Provides generator functions to stream camera images and IMU measurements from
    rosbag files without loading the entire dataset into memory.
    """
    
    def __init__(self, file, cam_topic, imu_topic = None):
        """Initialize rosbag extractor.
        
        Args:
            file (str): Path to rosbag file
            cam_topic (str): ROS topic name for camera images
            imu_topic (str, optional): ROS topic name for IMU measurements
        """
        self.bag = rosbag.Bag(file, "r")
        self.cam_topic = cam_topic
        self.imu_topic = imu_topic

    def image_msg_to_numpy(self,msg):
        """Convert ROS Image message to numpy array.
        
        Args:
            msg: ROS Image message (expected mono8 format)
            
        Returns:
            np.ndarray: Grayscale image as 2D numpy array
        """
        dtype = np.uint8  # mono8
        image = np.frombuffer(msg.data, dtype=dtype)
        image = image.reshape(msg.height, msg.width)
        return image

    def img_generator(self):
        """Generator for camera images from rosbag.
        
        Yields:
            tuple: (image, timestamp) where image is np.ndarray and timestamp is float (seconds)
        """
        for topic, msg, t in self.bag.read_messages(topics=[self.cam_topic]):
            cv_img = self.image_msg_to_numpy(msg)
            stamp = msg.header.stamp
            time_stamp = stamp.secs + stamp.nsecs / 1000000000
            yield cv_img, time_stamp

    def imu_generator(self):
        """Generator for IMU measurements from rosbag.
        
        Yields:
            tuple: (imu_data, timestamp) where imu_data is IMUData and timestamp is float (seconds)
        """
        for topic, msg, t in self.bag.read_messages(topics=[self.imu_topic]):
            imu_data = IMUData(msg)
            stamp = msg.header.stamp
            time_stamp = stamp.secs + stamp.nsecs / 1000000000
            yield imu_data, time_stamp

    def gt_generator(self):
        """Generator for ground truth poses from rosbag.
        
        Yields:
            tuple: (pose_message, timestamp) where timestamp is float (seconds)
        """
        for topic, msg, t in self.bag.read_messages(topics=[self.gt_topic]):
            stamp = msg.header.stamp
            time_stamp = stamp.secs + stamp.nsecs / 1000000000
            yield msg, time_stamp
