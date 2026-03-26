# Copyright (c) 2026 ETH Zurich. All rights reserved.
# SPDX-FileCopyrightText: 2026 ETH Zurich
# SPDX-License-Identifier: MIT
# Author: Jonas Kühne


"""
    Factor Graph Data Structures Module
    
    Defines the core data structures for the SLAM system: 3D points, camera frames,
    and the sparse map that connects them. Intermediate abstraction between raw
    measurements and factor graph optimization.
"""
import numpy as np
from vio_pipeline_segments.vo_frontend import VO_frontend
from vio_pipeline_segments.gtsam_optimizer import VisualInertialOdometryGraph

def add_ones(x):
    """Convert 3D point to homogeneous coordinates.
    
    Args:
        x (np.ndarray): 3D point (1D array) or array of 3D points (2D array)
        
    Returns:
        np.ndarray: Point(s) in homogeneous coordinates with appended 1.0
    """
    if len(x.shape) == 1:
        return np.concatenate([x,np.array([1.0])], axis=0)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

class Point(object):
    """3D landmark in the sparse map.
    
    Stores world coordinates and tracks all camera frames that observe this point,
    including the keypoint index in each frame.
    """
    
    def __init__(self, graph, loc):
        """Initialize 3D point.
        
        Args:
            graph (Map): Map object for registration
            loc (np.ndarray): 3D position in world frame
        """
        self.pt = np.array(loc)
        self.frames = []
        self.idxs = []
        self.id = graph.add_point(self)
        
    def add_observation(self, frame, idx):
        """Record observation of this point in a frame.
        
        Args:
            frame (Frame): Camera frame observing this point
            idx (int): Index of the keypoint in the frame's keypoint array
        """
        assert frame.pts[idx] is None
        assert frame not in self.frames
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)
        
    def homogeneous(self):
        """Get point in homogeneous coordinates.
        
        Returns:
            np.ndarray: 4D homogeneous coordinate [x, y, z, 1]
        """
        return add_ones(self.pt)

class Frame(object):
    """Single camera frame with keypoints, descriptors, and pose.
    
    Contains raw feature data and is associated with a unique pose in the world frame.
    Can be marked as a keyframe for use in optimization.
    """
    
    def __init__(self, graph, image, K, pose=np.eye(4), t = 0):
        """Initialize camera frame.
        
        Args:
            graph (Map): Map object for registration
            image (np.ndarray): Grayscale image data
            K (np.ndarray): 3x3 camera intrinsic matrix
            pose (np.ndarray): 4x4 pose matrix (camera to world)
            t (float): Timestamp in seconds
        """
        self.image = image
        self.K = K
        self.pose = pose
        self.frontend = VO_frontend()
        self.kps, self.des = self.frontend.get_keypoints(image)
        self.pts = [None]*len(self.kps)
        self.id = graph.add_frame(self)
        self.SE3 = None
        self.is_keyframe = False
        self.t = t
        
    @property
    def normalized_kps(self):
        """Normalized coordinates in camera frame.
        
        Projects image plane coordinates into normalized camera frame by undoing
        the effect of camera intrinsics: point_normalized = K^{-1} @ [u, v, 1]^T
        
        Returns:
            np.ndarray: Nx2 array of normalized 2D coordinates
        """
        if not hasattr(self, '_normalized_kps'):
            self._normalized_kps = np.dot(np.linalg.inv(self.K), add_ones(self.kps).T).T[:, :2]
        return self._normalized_kps

class Map(object):
    """Sparse 3D map containing frames, points, and keyframes.
    
    Central data structure that maintains the spatial graph including camera poses
    and 3D landmarks. Interfaces with the optimizer for pose graph optimization.
    """
    
    def __init__(self):
        """Initialize sparse map."""
        self.frames = []
        self.points = []
        self.keyframes = []
        self.max_frame = 0
        self.max_point = 0
        self.max_keyframe = 0
        self.is_initialized = False
        self.optimizer = VisualInertialOdometryGraph()
        
    def add_point(self, point):
        """Register a 3D point in the map.
        
        Args:
            point (Point): Point object to register
            
        Returns:
            int: Unique point ID
        """
        ret = self.max_point
        self.max_point = self.max_point + 1
        self.points.append(point)
        return ret
        
    def add_frame(self, frame):
        """Register a frame in the map.
        
        Args:
            frame (Frame): Frame object to register
            
        Returns:
            int: Unique frame ID
        """
        ret = self.max_frame
        self.max_frame = self.max_frame + 1
        self.frames.append(frame)
        return ret
        
    def add_keyframe(self, keyframe):
        """Register a frame as a keyframe.
        
        Args:
            keyframe (Frame): Frame to designate as keyframe
            
        Returns:
            int: Unique keyframe ID
        """
        ret = self.max_keyframe
        self.max_keyframe = self.max_keyframe + 1
        self.keyframes.append(keyframe)
        return ret
    
    def rescale(self, scale):
        """Scale all 3D points and frame positions by a uniform scale factor.
        
        Used after visual-inertial initialization to set the metric scale of the map.
        
        Args:
            scale (float): Scale factor to apply to all positions
        """
        self.is_initialized = True
        for point in self.points:
            point.pt *= scale
        for frame in self.frames:
            frame.pose[:3,3] *= scale

    def optimize_keyframes_gtsam(self):
        """Optimize keyframe poses and landmark positions using GTSAM.
        
        Returns:
            float: Reprojection error after optimization
        """
        error = self.optimizer.isam_update(self)
        return error
