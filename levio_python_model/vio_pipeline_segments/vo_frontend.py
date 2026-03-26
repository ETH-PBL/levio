# Copyright (c) 2026 ETH Zurich. All rights reserved.
# SPDX-FileCopyrightText: 2026 ETH Zurich
# SPDX-License-Identifier: MIT
# Author: Jonas Kühne


"""
    Visual Odometry Frontend Module
    
    Handles feature detection, descriptor extraction, and feature matching between frames.
    Provides methods for pose estimation using epipolar geometry and PnP algorithms.
"""
import numpy as np
import cv2

class VO_frontend(object):
    """Visual odometry frontend for feature-based pose estimation.
    
    Implements feature detection (GFTT corners), feature description (BRIEF),
    feature matching, and pose estimation from matches.
    """
    def __init__(self):
        """Initialize frontend with feature detector and matcher.
        
        Sets up GFTT (Good Features To Track) corner detector and BRIEF descriptor
        extractor with Hamming distance matching.
        """
        self.detector = cv2.GFTTDetector_create(
                maxCorners=1000, minDistance=15.0,
                qualityLevel=0.00002, useHarrisDetector=True)
        self.descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=True)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

    def get_keypoints(self, image):
        """Detect corners and compute BRIEF descriptors.
        
        Args:
            image (np.ndarray): Input grayscale image
            
        Returns:
            tuple: (keypoints_array, descriptors) where
                - keypoints_array: Nx2 array of (x, y) coordinates
                - descriptors: NxM array of binary descriptors
        """
        kps = self.detector.detect(image)
        kps, des = self.descriptor.compute(image, kps)
        return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

    def get_matches(self, image1, image2, hamming_threshold = 255):
        """Find descriptor matches between two frames.
        
        Matches descriptors using Hamming distance (BRIEF uses Hamming).
        Returns matches sorted by distance, filtered by threshold.
        
        Args:
            image1: First frame with descriptors (des attribute)
            image2: Second frame with descriptors (des attribute)
            hamming_threshold (int): Maximum Hamming distance for valid match
            
        Returns:
            tuple: (idx1, idx2, matches_list) where
                - idx1: Indices of keypoints in frame1
                - idx2: Indices of keypoints in frame2
                - matches_list: OpenCV match objects
        """
        idx1 = []
        idx2 = []
        matches = self.matcher.match(image1.des, image2.des)
        matches = sorted(matches, key = lambda x:x.distance)
        for match in matches:
            if(match.distance > hamming_threshold):
                break
            idx1.append(match.queryIdx)
            idx2.append(match.trainIdx)
        return idx1, idx2, matches
    
    def get_matches_graph(self, frame1, graph, hamming_threshold = 255, min_observations = 2, req_graph_points = 1000, max_frames = 30):
        """Find matches between current frame and 3D points from recent frames.
        
        Matches current frame descriptors against descriptors of 3D points that have
        been observed in recent (non-keyframe) frames.
        
        Args:
            frame1: Current frame to match
            graph: Map object containing frames and points
            hamming_threshold (int): Maximum Hamming distance for valid match
            min_observations (int): Minimum observations required for a point
            req_graph_points (int): Target number of graph points to collect
            max_frames (int): Maximum number of recent frames to search
            
        Returns:
            tuple: (idx_frame, idx_graph) where
                - idx_frame: Keypoint indices in current frame
                - idx_graph: Point IDs from the 3D map
        """
        graph_des = []
        graph_ids = []
        for i in range(min(max_frames, len(graph.frames)-1)):
            prev_frame = graph.frames[-2-i]
            for pt in prev_frame.pts:
                # Check number of observations and if pt has not been added before
                if (pt and len(pt.idxs) >= min_observations and pt.frames[-1] == prev_frame):
                    graph_ids.append(pt.id)
                    graph_des.append(prev_frame.des[pt.idxs[-1]])
            if len(graph_des) > req_graph_points:
                break
        graph_des = np.array(graph_des)
        matches = self.matcher.match(frame1.des, graph_des)
        matches = sorted(matches, key = lambda x:x.distance)
        idx_frame = []
        idx_graph = []
        for match in matches:
            if(match.distance > hamming_threshold):
                break
            idx_frame.append(match.queryIdx)
            idx_graph.append(graph_ids[match.trainIdx])
        return idx_frame, idx_graph
    
    def get_matches_graph_keypoints(self, frame1, graph, hamming_threshold = 255, min_observations = 2, req_graph_points = 1000, max_frames = 30):
        """Find matches between current frame and 3D points from recent keyframes.
        
        Similar to get_matches_graph but only considers points observed in keyframes,
        useful for pose estimation with limited frame window.
        
        Args:
            frame1: Current frame to match
            graph: Map object containing keyframes and points
            hamming_threshold (int): Maximum Hamming distance for valid match
            min_observations (int): Minimum observations required for a point
            req_graph_points (int): Target number of graph points to collect
            max_frames (int): Maximum number of recent keyframes to search
            
        Returns:
            tuple: (idx_frame, idx_graph) where
                - idx_frame: Keypoint indices in current frame
                - idx_graph: Point IDs from the 3D map
        """
        graph_des = []
        graph_ids = []
        prev_keyframe_id = frame1.id
        for i in range(min(max_frames, len(graph.keyframes))):
            curr_keyframe = graph.keyframes[-1-i]
            for pt in curr_keyframe.pts:
                # Check number of observations and if pt has not been added before
                if (pt and len(pt.idxs) >= min_observations and pt.frames[-1].id < prev_keyframe_id):
                    #print(curr_keyframe.id, pt.id)
                    graph_ids.append(pt.id)
                    graph_des.append(pt.frames[-1].des[pt.idxs[-1]])
            if len(graph_des) > req_graph_points:
                break
            prev_keyframe_id = curr_keyframe.id
        graph_des = np.array(graph_des)
        matches = self.matcher.match(frame1.des, graph_des)
        matches = sorted(matches, key = lambda x:x.distance)
        idx_frame = []
        idx_graph = []
        for match in matches:
            if(match.distance > hamming_threshold):
                break
            idx_frame.append(match.queryIdx)
            idx_graph.append(graph_ids[match.trainIdx])
        return idx_frame, idx_graph

    def get_pose_essential(self, image1, idx1, image2, idx2, K):
        Rt = np.eye(4)
        E, mask = cv2.findEssentialMat(image1.kps[idx1], image2.kps[idx2], K, cv2.RANSAC)
            # Use the mask to select only inlier matches
        pts1_inliers = image1.kps[idx1][mask.ravel() == 1]
        pts2_inliers = image2.kps[idx2][mask.ravel() == 1]
        _, rot, trans, mask = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)
        Rt[:3, :3] = rot
        Rt[:3, 3] = trans.squeeze()
        return np.linalg.inv(Rt), E
    
    def get_pose_ePnP(self, frame, idx_frame, graph_points, idx_graph, K, min_matches = 28):
        Rt = np.eye(4)
        object_pts = [graph_points[idx].pt for idx in idx_graph]
        img_pts = [frame.kps[idx] for idx in idx_frame]
        if len(object_pts) < min_matches:
            print("BOOTSTRAPPING: Few Points", len(object_pts))
            return (True, Rt)
        try:
            _, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(object_pts), np.array(img_pts), K, np.zeros((4,1)), reprojectionError = np.sqrt(2.0), iterationsCount=150)
        except:
            print("BOOTSTRAPPING: Exception PnP")
            return (True, Rt)
        if inliers is None:
            print("BOOTSTRAPPING: No Inliers")
            return (True, Rt)
        rot, _ = cv2.Rodrigues(rvec)
        Rt[:3, :3] = rot
        Rt[:3, 3] = tvec.T
        return (False, Rt)

    def add_ones(self, x):
        if len(x.shape) == 1:
            return np.concatenate([x,np.array([1.0])], axis=0)
        else:
            return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

    def get_triangulation(self, image1, idx1, image2, idx2):
        kps1_normalized = np.dot(np.linalg.inv(image1.K), self.add_ones(image1.kps[idx1]).T)
        kps2_normalized = np.dot(np.linalg.inv(image2.K), self.add_ones(image2.kps[idx2]).T)
        pts_4d = cv2.triangulatePoints(image1.pose[:3, :], image2.pose[:3, :],
                                       kps1_normalized[:2], kps2_normalized[:2])
        return pts_4d.T

    def calculate_reprojection_error_squared(self, frame, idx, pt_w):
        pt_f = np.dot(frame.pose, pt_w)
        # Point is behind the camera
        if pt_f[2] < 0:
            return 10 # Return large reprojection error as proxy
        projected_point = np.dot(frame.K, pt_f[:3])
        error = (projected_point[0:2] / projected_point[2]) - frame.kps[idx]
        return np.sum(error**2)

    def check_parallax(self, frame1, idx1, frame2, idx2, K):
        relative_parallax = (frame1.kps[idx1]-frame2.kps[idx2])
        relative_parallax[:,0] = relative_parallax[:,0]/K[0,0]
        relative_parallax[:,1] = relative_parallax[:,1]/K[1,1]
        avg_rel_parallax = np.average(np.sqrt(relative_parallax[:,0]**2+relative_parallax[:,1]**2))
        return avg_rel_parallax
