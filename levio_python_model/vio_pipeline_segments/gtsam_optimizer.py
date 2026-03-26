# Copyright (c) 2026 ETH Zurich. All rights reserved.
# SPDX-FileCopyrightText: 2026 ETH Zurich
# SPDX-License-Identifier: MIT
# Author: Jonas Kühne


"""
    GTSAM Factor Graph Optimizer Module
    
    Handles pose graph construction and optimization using GTSAM library.
    Integrates camera measurements and IMU preintegration factors for visual-inertial SLAM.
"""
import numpy as np
import gtsam
from gtsam.symbol_shorthand import B, V, X, L
import random
np.random.seed(42)
random.seed(42)

# inertial sensor noise model parameters (static)
# gyroscope_noise_density: 1.6968e-04     # [ rad / s / sqrt(Hz) ]   ( gyro "white noise" )  --> 5.87e-6
# gyroscope_random_walk: 1.9393e-05       # [ rad / s^2 / sqrt(Hz) ] ( gyro bias diffusion ) --> 2e-12
# accelerometer_noise_density: 2.0000e-3  # [ m / s^2 / sqrt(Hz) ]   ( accel "white noise" ) --> 8e-4
# accelerometer_random_walk: 3.0000e-3    # [ m / s^3 / sqrt(Hz) ].  ( accel bias diffusion ) --> 4.5e-8
# rate_hz: 200

def invert_pose(pose):
    """Invert a 4x4 homogeneous pose transformation.
    
    Args:
        pose (np.ndarray): 4x4 transformation matrix
        
    Returns:
        np.ndarray: Inverse of the transformation matrix
    """
    return np.linalg.inv(pose.copy())

class VisualInertialOdometryGraph(object):
    def __init__(self):
        """Initialize factor graph with noise models and calibration parameters.
        
        Configures sensor noise models, camera calibration, and IMU to camera
        transformation for the visual-inertial system.
        """
        self.g = 9.81
        self.gravity = np.array([0.0, self.g, 0.0])
        self.imu_data_loader = None
        self.prev_imu_t = 0.0
        # IMU preintegration parameters
        self.cam_to_imu_tf = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                                [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                                [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                                [0.0, 0.0, 0.0, 1.0]])
        self.imu_to_cam_tf = np.linalg.inv(self.cam_to_imu_tf)
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3]*6))
        self.odom_noise  = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.025, 0.025, 0.025, 0.1, 0.1, 0.1])) # (rot in rad, translation in m)
        self.pixel_error = 1.2
        self.results = gtsam.Values()
        self.previous_velocity = np.zeros((3,1))
        self.previous_bias = gtsam.imuBias.ConstantBias()
        self.imu_factors = []
        self.alpha_gravity = 0.5
        self.optimization_window = 12
        self.imu_bias_cov_scalar = 1.0
        self.imu_int_cov_scalar = 1.0
        self.vel_noise = 0.005
        self.update_bias = False
        self.optimizer_iterations = 100

    def set_imu_data_loader(self, imu_data_loader):
        """Set the IMU data generator and initialize noise models.
        
        Args:
            imu_data_loader: Generator function yielding (IMUData, timestamp) tuples
        """
        self.velocity_noise = gtsam.noiseModel.Isotropic.Sigma(3, self.vel_noise)
        self.BIAS_COVARIANCE = gtsam.noiseModel.Isotropic.Variance(6, 1e-8*self.imu_bias_cov_scalar)
        self.meas_noise  = gtsam.noiseModel.Isotropic.Sigma(2, self.pixel_error)  # 2D pixel noise (in pixels)
        self.imu_data_loader = imu_data_loader
        _, self.prev_imu_t = next(self.imu_data_loader)

    def update_gravity(self):
        """Update gravity estimate from recent IMU measurements.
        
        Blends gravity observed from IMU measurements with gravity predicted from
        the preintegrated measurements to refine the gravity vector estimate.
        """
        accumulated_values = self.imu_factors[-1].preintegratedMeasurements()
        v = accumulated_values.deltaVij()
        gravity_obs = -v/(np.linalg.norm(v))*self.g
        R = accumulated_values.deltaRij().matrix()
        gravity_proc = R@self.gravity
        self.gravity = ((1-self.alpha_gravity)*gravity_obs + self.alpha_gravity*gravity_proc)
        self.gravity = self.gravity/np.linalg.norm(self.gravity)*self.g

    def get_preintegrator(self, gravity):
        """Create IMU preintegration object.
        
        Args:
            gravity (np.ndarray): Gravity vector for preintegration
            
        Returns:
            gtsam.PreintegratedImuMeasurements: Preintegrator object
        """
        IMU_PARAMS = gtsam.PreintegrationParams(gravity)
        I = np.eye(3)
        IMU_PARAMS.setAccelerometerCovariance(I * 0.1**2)
        IMU_PARAMS.setGyroscopeCovariance(I * 0.01**2)
        IMU_PARAMS.setIntegrationCovariance(I * 0.001**2 * self.imu_int_cov_scalar)
        preintegrator = gtsam.PreintegratedImuMeasurements(IMU_PARAMS, self.previous_bias)
        return preintegrator

    def preintegrate_imu(self, prev_frame, cur_frame, additional_accumulator = None):
        """Preintegrate IMU measurements between two frames.
        
        Reads IMU data from the loader and integrates measurements using GTSAM.
        Creates an IMU factor connecting the two frames.
        
        Args:
            prev_frame: Previous camera frame (None for initialization)
            cur_frame: Current camera frame
            additional_accumulator: Optional secondary accumulator for simultaneous integration
            
        Returns:
            gtsam.PreintegratedImuMeasurements: Integrated measurements
        """
        incremental_accum = self.get_preintegrator(self.gravity)
        preintegrate = True
        while preintegrate:
            imu_data, t_imu = next(self.imu_data_loader)
            imu_data.apply_transform(self.imu_to_cam_tf)
            if additional_accumulator:
                additional_accumulator.integrateMeasurement(imu_data.lin_acc, imu_data.rot_vel, t_imu-self.prev_imu_t)
            incremental_accum.integrateMeasurement(imu_data.lin_acc, imu_data.rot_vel, t_imu-self.prev_imu_t)
            self.prev_imu_t = t_imu
            preintegrate = (t_imu < cur_frame.t)
        if prev_frame is not None:
            factor = gtsam.ImuFactor(X(prev_frame.id), V(prev_frame.id), X(cur_frame.id), V(cur_frame.id), B(cur_frame.id), incremental_accum)
            self.imu_factors.append(factor)
            self.update_gravity()
        return incremental_accum

    def build_gtsam_graph(self, frames, points, imu_factors, fixed_points = 1):
        """Construct GTSAM factor graph for optimization.
        
        Creates a factor graph with pose, velocity, bias, and landmark vertices,
        connected by IMU preintegration and camera projection factors.
        
        Args:
            frames (list): List of Frame objects
            points (list): List of Point objects
            imu_factors (list): List of IMU preintegration factors
            fixed_points (int): Number of initial frames to fix as anchors
            
        Returns:
            tuple: (factor_graph, initial_values) for optimization
        """
        factor_graph = gtsam.NonlinearFactorGraph()
        new_vals = gtsam.Values()
        prev_frame = frames[0]
        for frame in frames:
            if self.results.exists(X(frame.id)):
                new_vals.insert(X(frame.id), self.results.atPose3(X(frame.id)))
                new_vals.insert(V(frame.id), self.results.atVector(V(frame.id)))
                new_vals.insert(B(frame.id), self.results.atConstantBias(B(frame.id)))
            else:
                pose_in_world = invert_pose(frame.pose)
                new_vals.insert(X(frame.id), gtsam.Pose3(pose_in_world))
                # Will be zero if no previous velocity has been set (i.e., when initializing the optimization)
                new_vals.insert(V(frame.id), self.previous_velocity)
                new_vals.insert(B(frame.id), gtsam.imuBias.ConstantBias())
            # if frame.id != 0:
            #     odom = new_vals.atPose3(X(prev_frame.id)).between(new_vals.atPose3(X(frame.id)))
            #     factor_graph.add(gtsam.BetweenFactorPose3(X(prev_frame.id), X(frame.id), odom, self.odom_noise))

        for frame in frames[:fixed_points]:
            if frame.id == 0:
                factor_graph.add(gtsam.PriorFactorPose3(X(frame.id), gtsam.Pose3(np.eye(4)), self.prior_noise))
                factor_graph.add(gtsam.PriorFactorVector(V(frame.id), self.init_velocity, self.velocity_noise))
            else:
                factor_graph.add(gtsam.PriorFactorPose3(X(frame.id), self.results.atPose3(X(frame.id)), self.prior_noise))
                factor_graph.add(gtsam.PriorFactorVector(V(frame.id), self.results.atVector(V(frame.id)), self.velocity_noise))

        # odom = self.results.atPose3(X(self.last_frame_id)).between(gtsam.Pose3(pose_in_world))
        # new_factors.add(gtsam.BetweenFactorPose3(X(self.last_frame_id), X(frame.id), odom, self.odom_noise))

        prev_frame = frames[0]
        for frame, imu_factor in zip(frames[1:] ,imu_factors):
            factor_graph.add(gtsam.BetweenFactorConstantBias(B(prev_frame.id), B(frame.id), gtsam.imuBias.ConstantBias(), self.BIAS_COVARIANCE))
            factor_graph.add(imu_factor)
            prev_frame = frame
        
        K_np = frames[-1].K
        K = gtsam.Cal3_S2(K_np[0,0], K_np[1,1], 0., K_np[0,2], K_np[1,2])
        for point in points:
            # dist_cam_landmark = np.linalg.norm(pose_in_world[:3,3] - point.pt)
            # landmark_noise = gtsam.noiseModel.Isotropic.Sigma(3, max(0.35,dist_cam_landmark/5.0)) # Landmark position noise (in meters)
            if self.results.exists(L(point.id)):
                new_vals.insert(L(point.id), self.results.atPoint3(L(point.id)))
            else:
                new_vals.insert(L(point.id), point.pt)
            for cur_frame, point_idx_in_frame in zip(point.frames, point.idxs):
                if cur_frame in frames:
                    kps = cur_frame.kps[point_idx_in_frame]
                    factor_graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
                        kps, self.meas_noise, X(cur_frame.id), L(point.id), K))
        print("Dimension of optimization",len(new_vals.keys()))
        return (factor_graph, new_vals)

    def collect_subgraph(self, graph, nodes):
        """Extract a subgraph of recent keyframes and their observed points.
        
        Args:
            graph (Map): Complete map
            nodes (int): Number of recent keyframes to extract
            
        Returns:
            tuple: (keyframes_subgraph, points_subgraph)
        """
        upper_bound = graph.keyframes[-1].id
        lower_bound = graph.keyframes[-nodes].id
        points = []
        for frame in graph.keyframes[-nodes:]:
            for point in frame.pts:
                if point is None:
                    continue
                counter = 0
                for cur_frame in point.frames:
                    if cur_frame.is_keyframe and cur_frame.id >= lower_bound and cur_frame.id <= upper_bound:
                        counter += 1
                if counter > 1 and point not in points:
                    points.append(point)
        return (graph.keyframes[-nodes:], points)
    
    def optimizer_iteration(self, graph, window_size):
        """Perform one windowed optimization iteration.
        
        Extracts a subgraph of recent keyframes, builds and optimizes the factor graph,
        then updates map positions with optimized values.
        
        Args:
            graph (Map): Complete map
            window_size (int): Number of recent keyframes to include in window
            
        Returns:
            float: Final error after optimization
        """
        frames_sub, points_sub = self.collect_subgraph(graph, window_size)
        factor_graph, new_vals = self.build_gtsam_graph(frames_sub, points_sub, self.imu_factors[-(window_size-1):])

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self.optimizer_iterations)
        params.setlambdaInitial(0.01)
        params.setlambdaLowerBound(0.001)
        params.setlambdaUpperBound(10)
        params.setDiagonalDamping(True)
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(factor_graph, new_vals, params)
        self.results = self.optimizer.optimize()
        current_id = graph.keyframes[-1].id
        self.previous_velocity = self.results.atVector(V(current_id))
        if self.update_bias:
            self.previous_bias = self.results.atConstantBias(B(current_id))

        for point in points_sub:
            if self.results.exists(L(point.id)):
                point.pt = self.results.atPoint3(L(point.id))
        for frame in frames_sub:
            if self.results.exists(X(frame.id)):
                frame.pose =  invert_pose(self.results.atPose3(X(frame.id)).matrix())
        print("Coords", self.results.atPose3(X(current_id)).translation())
        return self.optimizer.error()
    
    def check_for_standstill(self, graph):
        """Detect and handle camera standstill (no movement).
        
        Updates gravity estimate from accumulated measurements when camera is stationary.
        
        Args:
            graph (Map): Current map
        """
        if graph.keyframes[-1].t - graph.keyframes[-2].t < 1.0:
            return
        prev_pose = graph.keyframes[-2].pose
        cur_pose = graph.keyframes[-1].pose
        delta = prev_pose @ np.linalg.inv(cur_pose)
        delta_position = np.linalg.norm(delta[:3,3])
        if delta_position > 0.05:
            return
        print("Standstill")
        accumulated_values = self.imu_factors[-1].preintegratedMeasurements()
        v = accumulated_values.deltaVij()
        self.gravity = -v/(np.linalg.norm(v))*self.g

    def init_gtsam(self, graph, init_velocity):
        """Initialize GTSAM optimization with estimated scale and velocity.
        
        Called after successful visual-inertial initialization.
        
        Args:
            graph (Map): Map to initialize
            init_velocity (np.ndarray): Initial velocity estimate (3D vector)
        """
        self.init_velocity = init_velocity
        error = self.optimizer_iteration(graph, len(graph.keyframes))
        print("Initial optimization error:", error)

    def isam_update(self, graph):
        self.preintegrate_imu(graph.keyframes[-2], graph.keyframes[-1])
        self.check_for_standstill(graph)
        error = self.optimizer_iteration(graph, self.optimization_window)
        return error
