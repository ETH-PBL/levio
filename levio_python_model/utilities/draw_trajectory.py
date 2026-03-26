# Copyright (c) 2026 ETH Zurich. All rights reserved.
# SPDX-FileCopyrightText: 2026 ETH Zurich
# SPDX-License-Identifier: MIT
# Author: Jonas Kühne


"""
    Trajectory Visualization Module
    
    Provides functionality to visualize camera trajectories and export pose data
    in various formats (images, TUM trajectory format).
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

def invert_pose(pose):
    return np.linalg.inv(pose.copy())

class TrajectoryVisualizer:
    """Visualizer for camera trajectory and pose data.
    
    Renders camera trajectory as 2D plot projected onto XZ plane and exports poses
    to disk in multiple formats (TUM trajectory format and visualization images).
    """
    
    def __init__(self, width, height, offset_x, offset_y, name = ''):
        """Initialize trajectory visualizer.
        
        Args:
            width (int): Canvas width in pixels
            height (int): Canvas height in pixels
            offset_x (int): X-axis offset for trajectory drawing
            offset_y (int): Z-axis (depth) offset for trajectory drawing
            name (str): Title text to display on visualization
        """
        self.traj = np.zeros((width,height,3), dtype=np.uint8)
        self.imu = np.zeros((width,height,3), dtype=np.uint8)
        self.last_img = None
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.width = width
        self.height = height
        self.text = np.zeros((width,height,3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.text, name, (5,15), font, 0.5, (0,255,0))
        
    def add_pose(self, x, y, z, color = (0,0,255)):
        """Add a single pose point to trajectory visualization.
        
        Projects 3D position onto 2D canvas using XZ plane and draws a colored circle.
        
        Args:
            x (float): World X coordinate
            y (float): World Y coordinate (not used in 2D projection)
            z (float): World Z coordinate (depth)
            color (tuple): BGR color tuple for the point (default: red)
        """
        draw_x = int(x)+self.offset_x
        draw_y =  self.height-int(z)-self.offset_y
        cv2.circle(self.traj, (draw_x, draw_y), 1, color, 2)

    def add_imu_pose(self, x, y, z, color = (0,0,255)):
        """Add IMU-derived pose to visualization (reserved for future use).
        
        Args:
            x (float): World X coordinate
            y (float): World Y coordinate
            z (float): World Z coordinate
            color (tuple): BGR color tuple for the point
        """
        draw_x = int(x)+self.offset_x
        draw_y = self.height-int(z)-self.offset_y
        cv2.circle(self.imu, (draw_x, draw_y), 1, color, 2)

    def add_poses(self, xa, ya, za):
        """Add multiple pose points to trajectory visualization.
        
        Args:
            xa (np.ndarray): Array of X coordinates
            ya (np.ndarray): Array of Y coordinates
            za (np.ndarray): Array of Z coordinates
        """
        for x, y, z in zip(xa, ya, za):
            self.add_pose(x,y,z)

    def reset_poses(self):
        """Clear trajectory visualization canvas."""
        self.traj = np.zeros((self.width,self.height,3), dtype=np.uint8)

    def update_frame(self, img, kpts = [None]):
        """Update current frame display with keypoint annotations.
        
        Args:
            img (np.ndarray): Grayscale image to display
            kpts (list): List of keypoints (x, y) to annotate on image
        """
        self.last_img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) 
        for kpt in kpts:
            cv2.circle(self.last_img, (int(kpt[0]), int(kpt[1])), 1, (0,0,255), 2)

    def draw(self):
        """Export trajectory visualization to PNG image (updating every frame)."""
        cv2.imwrite('Vis_Camera.png', self.last_img)
        cv2.imwrite('Vis_Trajectory.png', self.traj+self.imu+self.text)

    def save_trajectory_visualization_to_file(self, filename):
        """Export final trajectory visualization to PNG image.
        
        Args:
            filename (str): Output file path for the trajectory image
        """
        cv2.imwrite(filename, self.traj)

    def save_stamped_poses_to_file(self, frames, filename):
        """Export poses in TUM trajectory format.
        
        Saves poses as timestamped 6-DOF poses (translation + quaternion rotation)
        in the format expected by TUM SLAM evaluation tools.
        
        File format: timestamp x y z qx qy qz qw
        
        Args:
            frames (list): List of Frame objects with pose and timestamp attributes
            filename (str): Output file path for the trajectory file
        """
        with open(filename, 'w+') as f_traj:
            f_traj.write("# time x y z qx qy qz qw\n")
            for frame in frames:
                pose_in_world = invert_pose(frame.pose)
                stamp = frame.t
                f_traj.write(str(stamp) + ' ')
                f_traj.write(str(pose_in_world[0,3]) + ' ')
                f_traj.write(str(pose_in_world[1,3]) + ' ')
                f_traj.write(str(pose_in_world[2,3]) + ' ')
                rot = R.from_matrix(pose_in_world[0:3,0:3])
                q_rot = rot.as_quat()
                f_traj.write(str(q_rot[0]) + ' ')
                f_traj.write(str(q_rot[1]) + ' ')
                f_traj.write(str(q_rot[2]) + ' ')
                f_traj.write(str(q_rot[3]) + '\n')
