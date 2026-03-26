# Copyright (c) 2026 ETH Zurich. All rights reserved.
# SPDX-FileCopyrightText: 2026 ETH Zurich
# SPDX-License-Identifier: MIT
# Author: Jonas Kühne


"""
    LEVIO: Visual-Inertial Odometry System
    
    Main entry point for the VIO pipeline. Orchestrates data loading, feature tracking,
    visual-inertial initialization, and pose graph optimization.
"""
import numpy as np
import cv2
import os 
import time

import sys
sys.path.append('../')
from vio_pipeline_segments.vo_frontend import VO_frontend
from vio_pipeline_segments.factor_graph import Frame, Map, Point
from vio_pipeline_segments.vio_initialization import VisualInertialOdometryInitializer
from utilities.rosbag_extractor import RosbagExtractor
from utilities.draw_trajectory import TrajectoryVisualizer

seqeuences = [
    'data/MH_01_easy.bag',
    'data/MH_02_easy.bag',
    'data/MH_03_medium.bag',
    'data/MH_04_difficult.bag',
    'data/MH_05_difficult.bag',
    # 'data/V1_01_easy.bag',
    # 'data/V1_02_medium.bag',
    # 'data/V1_03_difficult.bag',
    # 'data/V2_01_easy.bag',
    # 'data/V2_02_medium.bag',
    # 'data/V2_03_difficult.bag',
]

class VIOSystem():
    """Visual-Inertial Odometry system.
    
    Integrates visual odometry frontend with IMU preintegration and factor graph
    optimization. Processes camera and IMU streams to estimate 6-DOF camera trajectory.
    """
    
    def __init__(self):
        """Initialize VIO system with sensor models and enabled features.
        
        Configures:
        - PnP pose estimation for relocalization
        - Bundle adjustment optimization
        - Keyframe-based mapping
        - Camera intrinsics and distortion model
        """
        self.use_epnp = True
        self.use_optimization = True
        self.use_keyframes = True

        self.W = int(752.0)
        self.H = int(480.0)
        self.K = np.array([[458.654,0,367.215],[0,457.296,248.375],[0,0,1]])
        self.D = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])
        self.graph = Map()
        self.frontend = VO_frontend()
        self.vio_initializer = VisualInertialOdometryInitializer(self.graph.optimizer)
        self.parallax_kf = 0.1

    def run_pipeline(self, rosbag_file, out_prefix):
        """Process a complete rosbag sequence and save results.
        
        Loads camera and IMU data from rosbag, runs VIO for all frames, and exports
        trajectory estimates along with diagnostic information.
        
        Args:
            rosbag_file (str): Path to rosbag file
            out_prefix (str): Prefix for output directory name
        """
        self.visualization = TrajectoryVisualizer(800,800,400,400,name=(rosbag_file.split('/')[-1]+' with '+out_prefix))
        run_crashed = False
        data_loader = RosbagExtractor(rosbag_file, '/cam0/image_raw', '/imu0')
        imgs = data_loader.img_generator()
        imu_data_loader = data_loader.imu_generator()
        self.graph.optimizer.set_imu_data_loader(imu_data_loader)

        for img, t_img in imgs:
            try:
                x, y, z = self.process_frame(img, t_img)
            except Exception:
                run_crashed = True
                break

        out_path = rosbag_file[:-4] + '_output/00_param_search/'
        if not os.path.exists(out_path): 
            os.makedirs(out_path)
        run_path = out_path + out_prefix + f"_{time.time():.0f}/"
        os.makedirs(run_path)
        os.system("git log -n 3 > "+run_path+"git.log")
        os.system("git diff > "+run_path+"changes.diff")
        self.visualization.save_stamped_poses_to_file(self.graph.keyframes, (run_path+'keyframes_stamped_traj_estimate.txt'))
        self.visualization.save_stamped_poses_to_file(self.graph.frames, (run_path+'frames_stamped_traj_estimate.txt'))
        self.visualization.save_trajectory_visualization_to_file((run_path+'image.png'))
        if run_crashed:
            os.system("touch "+run_path+"CRASHED.log")

    def process_frame(self, image, t_image):
        """Process a single camera frame through the VIO pipeline.
        
        Performs:
        1. Image undistortion
        2. Feature detection and matching
        3. Pose estimation (EPnP or Essential matrix)
        4. Triangulation of new points
        5. Keyframe detection
        6. VIO initialization (if needed)
        7. Pose graph optimization
        
        Args:
            image (np.ndarray): Grayscale camera image
            t_image (float): Frame timestamp (seconds)
            
        Returns:
            tuple: (x, y, z) translation of the camera pose
        """
        undist_image = cv2.undistort(image, self.K, self.D)
        frame = Frame(self.graph, undist_image, self.K, t = t_image)
        print()
        print("******** Frame %d *********" % frame.id)
        if frame.id == 0:
            self.graph.add_keyframe(frame)
            self.graph.keyframes[-1].is_keyframe = True
            return 0, 0, 0
        frame1 = self.graph.frames[-1]
        frame2 = self.graph.keyframes[-1]

        if (len(self.graph.keyframes) > 1 and self.use_epnp):
            if (len(self.graph.keyframes) > 5):
                idx_frame, idx_graph = self.frontend.get_matches_graph_keypoints(frame1, self.graph, hamming_threshold=30, min_observations=3, req_graph_points=700, max_frames=15)
            else:
                idx_frame, idx_graph = self.frontend.get_matches_graph_keypoints(frame1, self.graph, hamming_threshold=30, min_observations=2, req_graph_points=700, max_frames=15)
            bootstrap, Rt = self.frontend.get_pose_ePnP(frame1, idx_frame, self.graph.points, idx_graph, self.K, min_matches=25)
            frame1.pose = Rt
        else:
            bootstrap = True

        hamming_threshold = 90
        if self.use_epnp:
            hamming_threshold = 30
        idx1, idx2, matches = self.frontend.get_matches(frame1, frame2, hamming_threshold=hamming_threshold)
        if bootstrap:
            Rt, E = self.frontend.get_pose_essential(frame1, idx1, frame2, idx2, self.K)
            keyframe_id = self.graph.keyframes[-1].id
            dt = frame1.t - frame2.t
            if keyframe_id > 5:
                v = self.graph.optimizer.previous_velocity
                Rt[:3,3] *=  dt * np.linalg.norm(v)
            else:
                # Scale translation with time, due to lack of initial scale factor
                if frame2.id == 0 and frame1.id > 10:
                    # Standing start
                    dt = 0.5
                    frame2.t = frame1.t - dt
                Rt[:3,3] *=  dt
            frame1.pose = np.dot(Rt, frame2.pose)

        # Add new observations of existing points
        for index1, index2 in zip(idx1, idx2):
            if frame2.pts[index2] != None:
                pt = frame2.pts[index2]
                err = self.frontend.calculate_reprojection_error_squared(frame1, index1, pt.homogeneous())
                if err < 2:
                    pt.add_observation(frame1, index1)
                continue

        # Check for keyframe
        keyframe = (self.frontend.check_parallax(frame1, idx1, frame2, idx2, self.K) > self.parallax_kf) or not self.use_keyframes
        if keyframe:
            self.graph.add_keyframe(frame1)
            self.graph.keyframes[-1].is_keyframe = True

            pts_4d = self.frontend.get_triangulation(frame1, idx1, frame2, idx2)
            pts_3d = pts_4d[:, :4] / pts_4d[:, 3:]
        
            new_pts_count = 0
            for i, p in enumerate(pts_3d):
                # Check if pt is already in pose graph and add new observation
                if frame2.pts[idx2[i]] != None:
                    continue
                # Check reprojection errors and add points to pose graph if sufficiently small
                err1 = self.frontend.calculate_reprojection_error_squared(frame1, idx1[i], p)
                err2 = self.frontend.calculate_reprojection_error_squared(frame2, idx2[i], p)
                if err1 > 2 or err2 > 2:
                    continue
                pt = Point(self.graph, p[0:3])
                pt.add_observation(frame2, idx2[i])
                pt.add_observation(frame1, idx1[i])
                new_pts_count += 1

            print("Adding:   %d new points" % (new_pts_count))

            if (not self.graph.is_initialized):
                # Try initialization
                self.vio_initializer.run(self.graph)
            elif(self.use_optimization):
                if(self.use_keyframes):
                    err = self.graph.optimize_keyframes_gtsam()
                    print("Optimize: %f units of error" % err)
                elif(frame.id%5 == 0):
                    err = self.graph.optimize_gtsam(iterations=50)
                    print("Optimize: %f units of error" % err)
        
        print("Map:      %d points, %d frames, %d keyframes" % (len(self.graph.points), len(self.graph.frames), len(self.graph.keyframes)))
        self.visualization.reset_poses()
        for frame in self.graph.frames:
            pose_inv = np.linalg.inv(frame.pose)
            x, y, z = 20*pose_inv[:3, 3]
            if frame.is_keyframe:
                self.visualization.add_pose(x,y,z,(255,0,0))
            else:
                self.visualization.add_pose(x,y,z)
        self.visualization.update_frame(undist_image,frame1.kps)
        self.visualization.draw()

        return frame1.pose[:3, 3]


if __name__ == "__main__":
    run_name = "000_default"
    for seqeuence in seqeuences:
        vio = VIOSystem()
        vio.run_pipeline(seqeuence, run_name)
