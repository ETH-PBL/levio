// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#ifndef __POSE_GRAPH_H__
#define __POSE_GRAPH_H__

#include "pmsis.h"

#include "config.h"
#include "logging.h"
#include "matrix.h"
#include "vio_definitions.h"

typedef float pose_t[16];

enum DataStorageType{
    PREV_KPTS,
    PREV_DESCS,
    PREV_WP_IDS,
    WORLD_PTS,
    WORLD_DESCS,
    WORLD_PT_AGE,
    KEYFRAME_POSES,
    KEYFRAME_OBS_COUNTERS,
    KEYFRAME_VELOCITIES,
    IMU_FACTORS,
    OBSERVATIONS
};

typedef struct observation
{
    point2D_u16_t kpt;
    uint16_t      wp_id;
    uint16_t      pose_id;
} observation_t;

typedef struct imu_factor
{
    float           dt;
    point3D_float_t dp;
    point3D_float_t dv;
    float           dR[9];
} imu_factor_t;

typedef struct {
    imu_factor_t base;
    float J_r_bg[9];
    float J_p_bg[9];
    float J_p_ba[9];
    float J_v_bg[9];
    float J_v_ba[9];
} imu_factor_with_bias_t;


typedef struct
{
    point2D_u16_t*          prev_kpts;
    orb_descriptor_t*       prev_descs;
    int16_t*                prev_wp_ids;
    point3D_float_t*        world_pts;
    orb_descriptor_t*       world_descs;
    uint8_t*                world_pt_ages;
    pose_t*                 keyframe_poses;
    uint16_t*               keyframe_obs_counters;
    point3D_float_t*        keyframe_velocities;
    imu_factor_with_bias_t* imu_factors;
    observation_t*          observations;
} data_storage_t;

typedef struct pose_graph_stats
{
    pose_t* prev_pose;
    point3D_float_t prev_velocity;
    uint32_t total_features;
    uint32_t total_observations;
    uint16_t total_keyframes;
    uint16_t total_frames;
    uint16_t prev_features_total;
    uint16_t prev_features_triangulated;
    uint16_t prev_frame_number;
} pose_graph_stats_t;

/**
 * @brief Initializes the L2 data storage.
 */
void initialize_data_storage();
 
/**
 * @brief Moves data from L2 storage to L1 memory.
 * @param local Pointer to the local data (L1).
 * @param data_type Type of data to retrieve.
 */
void storage_move_in(void* local, enum DataStorageType data_type);
 
/**
 * @brief Moves data out from L1 to the presistent L2 memory.
 * @param local Pointer to the local data (L1).
 * @param data_type Type of data to store.
 */
void storage_move_out(void* local, enum DataStorageType data_type);
 
/**
 * @brief Initializes pose graph statistics structure.
 * @param cluster_device Pointer to the cluster device.
 * @return Pointer to the initialized pose_graph_stats_t structure.
 */
pose_graph_stats_t* init_pose_graph_stats(pi_device_t* cluster_device);
 
/**
 * @brief Adds a new keyframe to the pose graph.
 * @param graph_stats Pointer to pose graph statistics.
 * @param pose Pointer to the pose data.
 * @param curr_features Pointer to current ORB features.
 * @param prev_features Pointer to previous ORB features.
 * @param curr_wp_ids Pointer to current world point (landmark) IDs.
 * @param prev_wp_ids Pointer to previous world point (landmark) IDs.
 * @param work_memory Working memory structure.
 */
void pose_graph_add_keyframe(pose_graph_stats_t* graph_stats,
                             float* pose,
                             orb_features_t* curr_features,
                             orb_features_t* prev_features,
                             int16_t* curr_wp_ids,
                             int16_t* prev_wp_ids,
                             work_memory_t work_memory);
 
/**
 * @brief Adds the first keyframe to the pose graph.
 * @param graph_stats Pointer to pose graph statistics.
 * @param features Pointer to ORB features.
 * @param work_memory Working memory structure.
 */
void pose_graph_add_first_keyframe(pose_graph_stats_t* graph_stats, orb_features_t* features, work_memory_t work_memory);
 
/**
 * @brief Retrieves the previous ORB features from the pose graph.
 * @param graph_stats Pointer to pose graph statistics.
 * @param work_memory Pointer to working memory structure.
 * @return Pointer to previous ORB features.
 */
orb_features_t* pose_graph_get_prev_features(pose_graph_stats_t* graph_stats, work_memory_t* work_memory);

#endif /* __POSE_GRAPH_H__ */