// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#include "pose_graph.h"

data_storage_t* storage;

void initialize_data_storage()
{
    storage = pi_l2_malloc(sizeof(data_storage_t));
    storage->prev_kpts =             pi_l2_malloc(MAX_KEYPOINTS*sizeof(point2D_u16_t));
    storage->prev_descs =            pi_l2_malloc(MAX_KEYPOINTS*sizeof(orb_descriptor_t));
    storage->prev_wp_ids =           pi_l2_malloc(MAX_KEYPOINTS*sizeof(int16_t));
    storage->world_pts =             pi_l2_malloc(MAX_WORLD_FEATURES*sizeof(point3D_float_t));
    storage->world_descs =           pi_l2_malloc(MAX_WORLD_FEATURES*sizeof(orb_descriptor_t));
    storage->world_pt_ages =         pi_l2_malloc(MAX_WORLD_FEATURES*sizeof(uint8_t));
    storage->keyframe_poses =        pi_l2_malloc(MAX_KEYFRAMES*sizeof(pose_t));
    storage->keyframe_obs_counters = pi_l2_malloc(MAX_KEYFRAMES*sizeof(uint16_t));
    storage->keyframe_velocities =   pi_l2_malloc(MAX_KEYFRAMES*sizeof(point3D_float_t));
    storage->imu_factors =           pi_l2_malloc(MAX_KEYFRAMES*sizeof(imu_factor_with_bias_t));
    storage->observations =          pi_l2_malloc(MAX_OBSERVATIONS*sizeof(observation_t));
}

void storage_move_in(void* local, enum DataStorageType data_type)
{
    void* external;
    uint32_t data_size;
    switch (data_type)
    {
    case PREV_KPTS:
        external = (void*) storage->prev_kpts;
        data_size = MAX_KEYPOINTS*sizeof(point2D_u16_t);
        break;
    case PREV_DESCS:
        external = (void*) storage->prev_descs;
        data_size = MAX_KEYPOINTS*sizeof(orb_descriptor_t);
        break;
    case PREV_WP_IDS:
        external = (void*) storage->prev_wp_ids;
        data_size = MAX_KEYPOINTS*sizeof(int16_t);
        break;
    case WORLD_PTS:
        external = (void*) storage->world_pts;
        data_size = MAX_WORLD_FEATURES*sizeof(point3D_float_t);
        break;
    case WORLD_DESCS:
        external = (void*) storage->world_descs;
        data_size = MAX_WORLD_FEATURES*sizeof(orb_descriptor_t);
        break;
    case WORLD_PT_AGE:
        external = (void*) storage->world_pt_ages;
        data_size = MAX_WORLD_FEATURES*sizeof(uint8_t);
        break;
    case KEYFRAME_POSES:
        external = (void*) storage->keyframe_poses;
        data_size = MAX_KEYFRAMES*sizeof(pose_t);
        break;
    case KEYFRAME_OBS_COUNTERS:
        external = (void*) storage->keyframe_obs_counters;
        data_size = MAX_KEYFRAMES*sizeof(uint16_t);
        break;
    case KEYFRAME_VELOCITIES:
        external = (void*) storage->keyframe_velocities;
        data_size = MAX_KEYFRAMES*sizeof(point3D_float_t);
        break;
    case IMU_FACTORS:
        external = (void*) storage->imu_factors;
        data_size = MAX_KEYFRAMES*sizeof(imu_factor_with_bias_t);
        break;
    case OBSERVATIONS:
        external = (void*) storage->observations;
        data_size = MAX_OBSERVATIONS*sizeof(observation_t);
        break;
    default:
        return;
    }
    pi_cl_dma_cmd_t cmd;
    /* DMA Copy Data */
    pi_cl_dma_cmd((uint32_t) external, (uint32_t) local, data_size, PI_CL_DMA_DIR_EXT2LOC, &cmd);
    /* Wait for DMA transfer to finish. */
    pi_cl_dma_wait(&cmd);
}

void storage_move_out(void* local, enum DataStorageType data_type)
{
    void* external;
    uint32_t data_size;
    switch (data_type)
    {
    case PREV_KPTS:
        external = (void*) storage->prev_kpts;
        data_size = MAX_KEYPOINTS*sizeof(point2D_u16_t);
        break;
    case PREV_DESCS:
        external = (void*) storage->prev_descs;
        data_size = MAX_KEYPOINTS*sizeof(orb_descriptor_t);
        break;
    case PREV_WP_IDS:
        external = (void*) storage->prev_wp_ids;
        data_size = MAX_KEYPOINTS*sizeof(int16_t);
        break;
    case WORLD_PTS:
        external = (void*) storage->world_pts;
        data_size = MAX_WORLD_FEATURES*sizeof(point3D_float_t);
        break;
    case WORLD_DESCS:
        external = (void*) storage->world_descs;
        data_size = MAX_WORLD_FEATURES*sizeof(orb_descriptor_t);
        break;
    case WORLD_PT_AGE:
        external = (void*) storage->world_pt_ages;
        data_size = MAX_WORLD_FEATURES*sizeof(uint8_t);
        break;
    case KEYFRAME_POSES:
        external = (void*) storage->keyframe_poses;
        data_size = MAX_KEYFRAMES*sizeof(pose_t);
        break;
    case KEYFRAME_OBS_COUNTERS:
        external = (void*) storage->keyframe_obs_counters;
        data_size = MAX_KEYFRAMES*sizeof(uint16_t);
        break;
    case KEYFRAME_VELOCITIES:
        external = (void*) storage->keyframe_velocities;
        data_size = MAX_KEYFRAMES*sizeof(point3D_float_t);
        break;
    case IMU_FACTORS:
        external = (void*) storage->imu_factors;
        data_size = MAX_KEYFRAMES*sizeof(imu_factor_with_bias_t);
        break;
    case OBSERVATIONS:
        external = (void*) storage->observations;
        data_size = MAX_OBSERVATIONS*sizeof(observation_t);
        break;
    default:
        return;
    }
    pi_cl_dma_cmd_t cmd;
    /* DMA Copy Data */
    pi_cl_dma_cmd((uint32_t) external, (uint32_t) local, data_size, PI_CL_DMA_DIR_LOC2EXT, &cmd);
    /* Wait for DMA transfer to finish. */
    pi_cl_dma_wait(&cmd);
}

pose_graph_stats_t* init_pose_graph_stats(pi_device_t* cluster_device)
{
    pose_graph_stats_t* graph_stats = pi_cl_l1_malloc(cluster_device, sizeof(pose_graph_stats_t));
    graph_stats->prev_pose = pi_cl_l1_malloc(cluster_device, sizeof(pose_t));
    graph_stats->prev_velocity = (point3D_float_t){0.0f, 0.0f, 0.0f};
    graph_stats->total_features = 0;
    graph_stats->total_observations = 0;
    graph_stats->total_keyframes = 0;
    graph_stats->total_frames = 0;
    graph_stats->prev_features_total = 0;
    graph_stats->prev_features_triangulated = 0;
    graph_stats->prev_frame_number = 0;
    return graph_stats;
}

void pose_graph_add_keyframe(pose_graph_stats_t* graph_stats,
                             float* pose,
                             orb_features_t* curr_features,
                             orb_features_t* prev_features,
                             int16_t* curr_wp_ids,
                             int16_t* prev_wp_ids,
                             work_memory_t work_memory)
{
    /* Update previous keyframe data */
    memcpy(graph_stats->prev_pose, pose, 16*sizeof(float));
    storage_move_out(curr_features->kpts,PREV_KPTS);
    storage_move_out(curr_features->descs,PREV_DESCS);
    graph_stats->prev_features_total = curr_features->kpt_counter;
    graph_stats->prev_frame_number = graph_stats->total_frames - 1;

    /* Update pose */
    uint16_t curr_index = graph_stats->total_keyframes % MAX_KEYFRAMES;
    uint16_t prev_index = (graph_stats->total_keyframes-1) % MAX_KEYFRAMES;
    uint16_t curr_keyframe_number = graph_stats->total_keyframes;
    pose_t* keyframe_poses = allocate_work_memory(&work_memory, MAX_KEYFRAMES*sizeof(pose_t));
    uint16_t* keyframe_obs_counters = allocate_work_memory(&work_memory, MAX_KEYFRAMES*sizeof(uint16_t));

    storage_move_in(keyframe_poses, KEYFRAME_POSES);
    storage_move_in(keyframe_obs_counters, KEYFRAME_OBS_COUNTERS);
    memcpy(keyframe_poses[curr_index], pose, 16*sizeof(float));
    storage_move_out(keyframe_poses, KEYFRAME_POSES);
    graph_stats->total_keyframes += 1;
    if(prev_features == NULL)
    {
        /* Initializing pose graph */
        keyframe_obs_counters[curr_index] = 0;
        storage_move_out(keyframe_obs_counters, KEYFRAME_OBS_COUNTERS);
        return;
    }

    observation_t* observations = allocate_work_memory(&work_memory, MAX_OBSERVATIONS*sizeof(observation_t));
    storage_move_in(observations, OBSERVATIONS);

    graph_stats->total_observations -= keyframe_obs_counters[prev_index];
    keyframe_obs_counters[prev_index] = 0;
    keyframe_obs_counters[curr_index] = 0;
    for(uint16_t i = 0; i < MAX_KEYPOINTS; ++i)
    {
        if(prev_wp_ids[i] > -1)
        {
            uint16_t obs_offset = graph_stats->total_observations % MAX_OBSERVATIONS;
            observations[obs_offset].kpt = prev_features->kpts[i];
            observations[obs_offset].wp_id = prev_wp_ids[i];
            observations[obs_offset].pose_id = curr_keyframe_number - 1;
            keyframe_obs_counters[prev_index] += 1;
            graph_stats->total_observations += 1;
        }
    }
    for(uint16_t i = 0; i < MAX_KEYPOINTS; ++i)
    {
        if(curr_wp_ids[i] > -1)
        {
            /* Circular buffer: once full, the oldest observations are intentionally
             * overwritten by newer ones. Observations outside the current optimization
             * window (MAX_KEYFRAMES) are no longer needed and can be discarded.
             * The warning below fires only if an observation still inside the window
             * is about to be overwritten, which would corrupt the current optimization. */
            uint16_t obs_offset = graph_stats->total_observations % MAX_OBSERVATIONS;
            if ((graph_stats->total_observations >= MAX_OBSERVATIONS) &&
                (observations[obs_offset].pose_id > (curr_keyframe_number - MAX_KEYFRAMES)))
            {
                LOG_ERROR("\nWARNING: OBSERVATION BUFFER OVERFILLING\n\n");
            }
            observations[obs_offset].kpt = curr_features->kpts[i];
            observations[obs_offset].wp_id = curr_wp_ids[i];
            observations[obs_offset].pose_id = curr_keyframe_number;
            keyframe_obs_counters[curr_index] += 1;
            graph_stats->total_observations += 1;
        }
    }
    LOG_INFO("Keyframe Observation Counter Prev %d, Curr %d, Total %d\n", keyframe_obs_counters[prev_index], keyframe_obs_counters[curr_index], graph_stats->total_observations);
    storage_move_out(keyframe_obs_counters, KEYFRAME_OBS_COUNTERS);
    storage_move_out(observations, OBSERVATIONS);
}

void pose_graph_add_first_keyframe(pose_graph_stats_t* graph_stats, orb_features_t* features, work_memory_t work_memory)
{
    LOG_INFO("Initializing Pose Graph\n");
    float init_pose[16] =  {1,0,0,0,
                            0,1,0,0,
                            0,0,1,0,
                            0,0,0,1};
    pose_graph_add_keyframe(graph_stats,init_pose,features,NULL,NULL,NULL,work_memory);
    int16_t* world_pt_ids = allocate_work_memory(&work_memory, (MAX_KEYPOINTS*sizeof(int16_t)));
    uint8_t* world_point_ages = allocate_work_memory(&work_memory, MAX_WORLD_FEATURES*sizeof(uint8_t));
    for(uint16_t i = 0; i < MAX_KEYPOINTS; ++i)
    {
        world_pt_ids[i] = -1;
    }
    for(uint16_t i = 0; i < MAX_WORLD_FEATURES; ++i)
    {
        world_point_ages[i] = 100;
    }
    point3D_float_t* velocities = allocate_work_memory(&work_memory, MAX_KEYFRAMES*sizeof(point3D_float_t));
    velocities[0].x = 0.0;
    velocities[0].y = 0.0;
    velocities[0].z = 0.0;
    storage_move_out(world_pt_ids,PREV_WP_IDS);
    storage_move_out(world_point_ages, WORLD_PT_AGE);
    storage_move_out(velocities, KEYFRAME_VELOCITIES);
}

orb_features_t* pose_graph_get_prev_features(pose_graph_stats_t* graph_stats, work_memory_t* work_memory)
{
    // Pass work memory by reference for persistent allocation!
    orb_features_t* prev_features = allocate_work_memory(work_memory, sizeof(orb_features_t));
    prev_features->kpts = allocate_work_memory(work_memory, (MAX_KEYPOINTS*sizeof(point2D_u16_t)));
    prev_features->descs = allocate_work_memory(work_memory, (MAX_KEYPOINTS*sizeof(orb_descriptor_t)));
    prev_features->kpt_counter = graph_stats->prev_features_total;
    prev_features->kpt_capacity = MAX_KEYPOINTS;
    storage_move_in(prev_features->kpts,PREV_KPTS);
    storage_move_in(prev_features->descs,PREV_DESCS);
    return prev_features;
}