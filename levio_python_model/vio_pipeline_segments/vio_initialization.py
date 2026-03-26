# Copyright (c) 2026 ETH Zurich. All rights reserved.
# SPDX-FileCopyrightText: 2026 ETH Zurich
# SPDX-License-Identifier: MIT
# Author: Jonas Kühne


"""
    Visual-Inertial Odometry Initialization Module
    
    Bootstraps VIO system by estimating scale and initial velocity from visual and
    inertial measurements. Uses gravity alignment to enable monocular metric scale recovery.
"""
import numpy as np

class VisualInertialOdometryInitializer():
    """Bootstrap scale and velocity for VIO system.
    
    Handles initialization of the VIO system from monocular vision and IMU measurements.
    Estimates gravity direction, scale factor, and initial velocity needed to transition
    from monocular to metric-scale SLAM.
    """
    
    def __init__(self, optimizer):
        """Initialize VIO bootstrapper.
        
        Args:
            optimizer (VisualInertialOdometryGraph): Reference to factor graph optimizer
        """
        self.optimizer = optimizer
        self.is_time_synced = False
        self.is_standing_start = False
        self.delta_ps = []
        self.g = 9.81
        gravity = np.array([0.0, self.g, 0.0])
        self.accum = self.optimizer.get_preintegrator(gravity)
        self.velocity = np.zeros(3)
        self.scale = 1.0

    def set_gravity(self, values):
        """Estimate gravity direction from stationary IMU measurements.
        
        Computes the negative mean acceleration (which represents gravity) and
        normalizes to standard gravity magnitude (9.81 m/s^2).
        
        Args:
            values (np.ndarray): Array of acceleration measurements
        """
        # Estimate gravity direction:
        gravity_vector_dir = -np.average(values, axis=0)
        self.gravity_vector = gravity_vector_dir/np.linalg.norm(gravity_vector_dir)*self.g

    def estimate_initial_velocity_and_scale(self, keyframes):
        """Estimate scale factor and initial velocity from visual-inertial constraints.
        
        Solves a linear least-squares problem matching visual and IMU position changes.
        Uses visual pose increments as constraints against IMU double-integrated displacement.
        
        Args:
            keyframes (list): List of initialized keyframes with poses
            
        Returns:
            bool: True if initialization is valid, False otherwise
        """
        if not self.is_standing_start:
            self.set_gravity(self.delta_ps)

        # Each interval gives 3 equations; total system is 3*(N-1) x 4.
        num_intervals = len(self.delta_ps)
        if self.is_standing_start:
            self.A = np.zeros((3 * num_intervals, 1))
        else:
            self.A = np.zeros((3 * num_intervals, 4))
        self.b = np.zeros(3 * num_intervals)
        for i, [keyframe, delta_p_imu] in enumerate(zip(keyframes[1:], self.delta_ps)):
            kf_pose = keyframe.pose.copy()
            delta_p_vo = np.linalg.inv(kf_pose)[:3,3]
            dt = keyframe.t - keyframes[0].t
            self.A[3*i:3*i+3, 0] = delta_p_vo
            if not self.is_standing_start:
                self.A[3*i:3*i+3, 1:4] = -dt*np.eye(3)
            self.b[3*i:3*i+3] = delta_p_imu + self.gravity_vector*dt**2/2

        # Perform Linear Optimization
        x, residuals, rank, s_vals = np.linalg.lstsq(self.A, self.b, rcond=None)
        # Sanity Check
        check = False
        if self.is_standing_start:
            self.velocity = np.zeros(3)
            if x[0] > 0.0:
                print('Good Init')
                print('Linear', x)
                check = True
        else:
            self.velocity = x[1:4]
        velocity_norm = np.linalg.norm(self.velocity)
        self.scale = x[0]
        print(self.scale)
        print(velocity_norm)
        if velocity_norm*0.5 < self.scale and velocity_norm*2.0 > self.scale:
            print('Good Init')
            print('Linear', x)
            print('Norm', np.linalg.norm(x[1:]))
            check = True
        return check

    def preintegrate_imu(self, graph):
        """Preintegrate IMU measurements between two consecutive keyframes.
        
        Args:
            graph (Map): Map object containing keyframes
        """
        self.optimizer.preintegrate_imu(graph.keyframes[-2],graph.keyframes[-1], self.accum)
        self.delta_ps.append(self.accum.deltaPij())

    def run(self, graph):
        """Execute VIO initialization procedure.
        
        Synchronizes timestamps, estimates gravity, accumulates IMU measurements,
        and attempts scale and velocity estimation when sufficient keyframes are available.
        
        Args:
            graph (Map): Map object to initialize
        """
        if not self.is_time_synced:
            # Integrate values prior to first value
            accumulate = self.optimizer.preintegrate_imu(None, graph.keyframes[0])
            self.is_time_synced = True
            if graph.keyframes[1].id > 10:
                self.is_standing_start = True
                self.set_gravity([accumulate.deltaVij()])
                self.optimizer.gravity = self.gravity_vector
        self.preintegrate_imu(graph)
        if len(graph.keyframes) < self.optimizer.optimization_window:
            return
        success = self.estimate_initial_velocity_and_scale(graph.keyframes)
        if(success):
            graph.rescale(self.scale)
            self.optimizer.init_gtsam(graph, self.velocity)
