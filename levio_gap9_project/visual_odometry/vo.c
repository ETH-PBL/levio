// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

#include "vo.h"

#include "feature_handling/orb_gap.h"
#include "feature_handling/bf_matcher.h"
#include "triangulation.h"
#include "essential_matrix.h"
#include "epnp.h"
#include "optimizer/imu_preintegration.h"

#include "math.h"

#define Abs(a)          (((int)(a)<0)?(-(a)):(a))
#define Min(a, b)       (((a)<(b))?(a):(b))
#define Max(a, b)       (((a)>(b))?(a):(b))


#define IS_KEYFRAME 1
#define IS_REGULAR_FRAME 0

float K_data[9] = {CAMERA_FX,     0.0, CAMERA_CX,
                        0.0, CAMERA_FY, CAMERA_CY,
                        0.0,       0.0,       1.0};


/**
 * @brief DMA-transfers one image frame from L2 to L1 and runs ORB detection + descriptor computation.
 *
 * Copies the raw image from img_l2_buffer into L1 work memory, then runs either the
 * single-core or multi-core ORB pipeline depending on the SINGLE_CORE compile flag.
 * Results are written into orb_features (keypoints and descriptors).
 */
void process_frame(orb_features_t* orb_features,
                   uint8_t* img_l2_buffer,
                   work_memory_t work_memory)
{
    pi_cl_dma_cmd_t cmd;
    /* DMA Copy Frame */
    uint8_t* image_buffer = allocate_work_memory(&work_memory,(size_t) (IMG_SIZE));
    pi_cl_dma_cmd((uint32_t) img_l2_buffer, (uint32_t) image_buffer, IMG_SIZE, PI_CL_DMA_DIR_EXT2LOC, &cmd);
    /* Wait for DMA transfer to finish. */
    pi_cl_dma_wait(&cmd);
    image_data_t img = {IMG_WIDTH, IMG_HEIGHT, image_buffer};
    if(SINGLE_CORE)
    {
        orb_detect_and_compute_single_core(&img, orb_features,work_memory);
    }
    else
    {
        orb_detect_and_compute_multi_core(&img, orb_features,work_memory,NB_CORES);
    }
}

/**
 * @brief Matches current features against the world model and estimates the camera pose via EPnP RANSAC.
 *
 * Loads world descriptors from L2, runs brute-force two-way matching, then filters matches
 * by world-point age (MAX_WORLD_POINT_AGE). If enough inliers remain, calls EPnP RANSAC
 * to compute T (camera-to-world). Returns 0 on success, 1 if there are insufficient matches.
 *
 * @return 0 on success, 1 if EPnP failed (insufficient matches before or after age filter).
 */
uint8_t process_epnp_ransac(orb_features_t* curr_features,
                            pose_graph_stats_t* graph_stats,
                            matrix_2D_t* K,
                            matrix_2D_t* T,
                            feature_match_t* matches,
                            uint16_t* world_match_counter,
                            work_memory_t work_memory)
{
    orb_features_t* world_features = allocate_work_memory(&work_memory, sizeof(orb_features_t));
    world_features->kpts = NULL;
    world_features->descs = allocate_work_memory(&work_memory, (MAX_WORLD_FEATURES*sizeof(orb_descriptor_t)));
    world_features->kpt_counter = Min(graph_stats->total_features,MAX_WORLD_FEATURES);
    world_features->kpt_capacity = MAX_WORLD_FEATURES;
    storage_move_in(world_features->descs, WORLD_DESCS);

    uint16_t match_counter;
    if(SINGLE_CORE)
    {
        match_counter = bf_match_two_way_max_flow(matches,world_features,curr_features,DISABLE_MAX_FLOW,HAMMING_THRESHOLD);
    }
    else
    {
        match_counter = bf_match_two_way_max_flow_multicore(matches,world_features,curr_features,DISABLE_MAX_FLOW,HAMMING_THRESHOLD,NB_CORES);
    }
    if (match_counter < MIN_MATCHES_EPNP)
    {
        *world_match_counter = match_counter;
        mateye(T);
        pi_perf_stop(); 
        LOG_TIMING("[Feature Match World] Number of cycles %d\n",pi_perf_read(PI_PERF_CYCLES));
        pi_perf_reset();
        pi_perf_start();
        LOG_INFO("EPnP failed: insufficient matches after bf matcher (got %u, need %u)\n", match_counter, MIN_MATCHES_EPNP);
        return 1;  // EPNP_ERR_INSUFFICIENT_MATCHES
    }

    point3D_float_t* all_world_points = allocate_work_memory(&work_memory, MAX_WORLD_FEATURES*sizeof(point3D_float_t));
    point3D_float_t* matched_world_points = allocate_work_memory(&work_memory, match_counter*sizeof(point3D_float_t));
    point2D_u16_t* image_points = allocate_work_memory(&work_memory, match_counter*sizeof(point2D_u16_t));
    uint8_t* world_point_ages = allocate_work_memory(&work_memory, MAX_WORLD_FEATURES*sizeof(uint8_t));
    storage_move_in(all_world_points,WORLD_PTS);
    storage_move_in(world_point_ages, WORLD_PT_AGE);

    uint16_t filtered_match_counter = 0;
    for (uint16_t i = 0; i < match_counter; ++i){
        feature_match_t cur_match = matches[i];
        uint8_t age = world_point_ages[cur_match.feat_idx0];
        if (age < MAX_WORLD_POINT_AGE)
        {
            matched_world_points[filtered_match_counter] = all_world_points[cur_match.feat_idx0];
            image_points[filtered_match_counter] = curr_features->kpts[cur_match.feat_idx1];
            filtered_match_counter++;
        }
    }
    pi_perf_stop(); 
    LOG_TIMING("[Feature Match World] Number of cycles %d\n",pi_perf_read(PI_PERF_CYCLES));
    pi_perf_reset();
    pi_perf_start();
    if (filtered_match_counter < MIN_MATCHES_EPNP)
    {
        *world_match_counter = match_counter;
        mateye(T);
        LOG_INFO("EPnP failed: insufficient matches after age filter (got %u from %u, need %u)\n", filtered_match_counter, match_counter, MIN_MATCHES_EPNP);
        return 1;  // EPNP_ERR_AGE_FILTER
    }
    if(SINGLE_CORE)
    {
        ransac_epnp_compute_pose(matched_world_points, image_points, K, T, work_memory, filtered_match_counter, EPNP_ITERS);
    }
    else
    {
        ransac_epnp_compute_pose_multicore(matched_world_points, image_points, K, T, work_memory, filtered_match_counter, EPNP_ITERS, NB_CORES);
    }
    *world_match_counter = match_counter;
    LOG_INFO("EPnP successful\n");
    return 0;
}

/**
 * @brief Estimates relative pose via 8-point RANSAC and recovers metric scale from IMU preintegration.
 *
 * Runs RANSAC on the essential matrix from curr↔prev feature matches, recovers the relative
 * rotation and (unit-scale) translation via recoverPose, then scales the translation by
 * ||prev_velocity|| * min(dt, 1s) from IMU preintegration. The resulting T_rel is
 * composed with T_prev to produce the global pose T.
 * Falls back to T = I * T_prev if match_counter < 10.
 */
void process_eight_point_ransac(orb_features_t* curr_features,
                                orb_features_t* prev_features,
                                feature_match_t* matches,
                                matrix_2D_t* T_prev,
                                pose_graph_stats_t* graph_stats,
                                matrix_2D_t* K,
                                matrix_2D_t* T,
                                uint16_t match_counter,
                                work_memory_t work_memory)
{
    float* t_rel = allocate_work_memory(&work_memory,16*sizeof(float));
    matrix_2D_t T_rel = {t_rel,{4,4,0,0}};
    if (match_counter < MIN_MATCHES_ESSENTIAL)
    {
        mateye(&T_rel);
        matmul(&T_rel,T_prev,T);
        LOG_INFO("8-point RANSAC failed: insufficient matches (got %u, need 10, 8 for algorithm and 2 as threshold)\n", match_counter);
        matprint(T, INFO_LEVEL);
        return;
    }
    float E_data[9] = {0};
    matrix_2D_t E = {E_data,{3,3,0,0}};

    if(SINGLE_CORE)
    {
        ransacEssentialMatrix(prev_features->kpts,curr_features->kpts,matches, K, &E, work_memory, match_counter, ESSENTIAL_ITERS);
    }
    else
    {
        ransacEssentialMatrixMulticore(prev_features->kpts,curr_features->kpts,matches, K, &E, work_memory, match_counter, ESSENTIAL_ITERS, NB_CORES);
    }

    recoverPose(prev_features->kpts,curr_features->kpts,matches, K, &E, &T_rel, work_memory, match_counter, SINGLE_CORE, NB_CORES);
    
    LOG_DEBUG("Prev Velocity [%f, %f, %f]\n",graph_stats->prev_velocity.x,graph_stats->prev_velocity.y,graph_stats->prev_velocity.z);
    point3D_float_t dp;
    float dt;
    get_current_dp_and_dt(&dp, &dt);
    LOG_DEBUG("IMU Preintegration dp [%f, %f, %f], dt %f\n",dp.x, dp.y, dp.z, dt);
    float scale = vec3_norm(&graph_stats->prev_velocity)*(Min(dt, 1.0f));
    LOG_INFO("Scale: %f\n", scale);
    scaleTranslationOfTransformation(&T_rel, Max(scale, MIN_TRANSLATION_SCALE));
    matprint(&T_rel, INFO_LEVEL);

    /* Apply new transformation to previous pose */
    matmul(&T_rel,T_prev,T);
    LOG_INFO("8-point result\n");
    matprint(T, INFO_LEVEL);
}

/**
 * @brief Updates the world map by re-observing existing landmarks and triangulating new ones.
 *
 * For each world match, updates the descriptor and resets the age of landmarks that
 * re-project within a 2-pixel threshold. For frame matches where neither keypoint has a
 * known world-point ID, triangulates a new 3D point and inserts it into the first available
 * slot (determined by age threshold, adaptively lowered if the buffer is full).
 * curr_wp_ids and prev_wp_ids are updated in-place and written back to L2 storage.
 */
void update_world_points(orb_features_t* prev_features,
                         orb_features_t* curr_features,
                         feature_match_t* frame_matches,
                         feature_match_t* world_matches,
                         int16_t* prev_wp_ids,
                         int16_t* curr_wp_ids,
                         matrix_2D_t* prev_T,
                         matrix_2D_t* curr_T,
                         matrix_2D_t* K,
                         pose_graph_stats_t* graph_stats,
                         work_memory_t work_memory,
                         uint16_t frame_match_counter,
                         uint16_t world_match_counter)
{
    /* Init Data */
    point3D_float_t* world_points = allocate_work_memory(&work_memory, (MAX_WORLD_FEATURES*sizeof(point3D_float_t)));
    orb_descriptor_t* world_descs = allocate_work_memory(&work_memory, (MAX_WORLD_FEATURES*sizeof(orb_descriptor_t)));
    uint8_t* world_point_ages = allocate_work_memory(&work_memory, MAX_WORLD_FEATURES*sizeof(uint8_t));
    for(uint16_t i = 0; i < MAX_KEYPOINTS; ++i)
    {
        curr_wp_ids[i] = -1;
    }
    storage_move_in(world_points,WORLD_PTS);
    storage_move_in(world_descs, WORLD_DESCS);
    storage_move_in(world_point_ages, WORLD_PT_AGE);
    for(uint16_t i = 0; i < Min(graph_stats->total_features, MAX_WORLD_FEATURES); ++i)
    {
        world_point_ages[i] +=1;
    }

    uint16_t known_counter = 0;
    for(uint16_t i = 0; i < world_match_counter; ++i)
    {
        feature_match_t match = world_matches[i];
        uint16_t idx0 = match.feat_idx0;
        uint16_t idx1 = match.feat_idx1;
        point3D_float_t wp = world_points[idx0];
        float error = reprojection_error_squared(&wp, &curr_features->kpts[idx1], curr_T, K);
        if(error < REPROJECTION_ERROR_SQ)
        {
            /* Add new observation / Update descriptor */
            curr_wp_ids[idx1] = idx0;
            memcpy(&world_descs[idx0],&curr_features->descs[idx1],sizeof(orb_descriptor_t));
            world_point_ages[idx0] = 0;
            known_counter += 1;
        }
    }

    uint16_t new_counter = 0;
    uint16_t next_wp_index = 0;
    int16_t age_threshold = MAX_WORLD_POINT_AGE;
    for(uint16_t i = 0; i < frame_match_counter; ++i)
    {
        feature_match_t match = frame_matches[i];
        uint16_t idx0 = match.feat_idx0;
        uint16_t idx1 = match.feat_idx1;
        if ((prev_wp_ids[idx0] == -1) && (curr_wp_ids[idx1] == -1))
        {
            point3D_float_t wp = triangulatePoint(&prev_features->kpts[idx0], &curr_features->kpts[idx1], K, prev_T, curr_T, work_memory);
            if (fabsf(wp.x) < 1e-6f && fabsf(wp.y) < 1e-6f && fabsf(wp.z) < 1e-6f){
                continue;
            }
            float error0 = reprojection_error_squared(&wp, &prev_features->kpts[idx0], prev_T, K);
            float error1 = reprojection_error_squared(&wp, &curr_features->kpts[idx1], curr_T, K);
            if(error0 < REPROJECTION_ERROR_SQ && error1 < REPROJECTION_ERROR_SQ)
            {
                while (world_point_ages[next_wp_index] < age_threshold)
                {
                    next_wp_index++;
                    if(next_wp_index >= MAX_WORLD_FEATURES)
                    {
                        age_threshold -= 1;
                        next_wp_index = 0;
                        LOG_INFO("Adapting age threshold to %d\n", age_threshold);
                        if (age_threshold <= 0)
                        {
                            LOG_WARNING("\nWARNING: LANDMARK BUFFER FULL, NO FREE SLOT\n\n");
                            break;
                        }
                        if (age_threshold < MAX_KEYFRAMES)
                        {
                            LOG_WARNING("\nWARNING: LANDMARK BUFFER OVERFILLING\n\n");
                        }
                    }
                }
                if (age_threshold <= 0) break;
                int16_t offset = next_wp_index;
                memcpy(world_points+offset, &wp, sizeof(point3D_float_t));
                memcpy(world_descs+offset,curr_features->descs+idx1, sizeof(orb_descriptor_t));
                curr_wp_ids[idx1] = offset;
                /* Requires Update of pose graph of prev keyframe */
                prev_wp_ids[idx0] = offset;
                world_point_ages[offset] = 0;
                graph_stats->total_features += 1;
                new_counter += 1;
            }
        }
    }
    LOG_INFO("Newly triangulated points: %d\nRe-observed points: %d\n", new_counter, known_counter);
    storage_move_out(curr_wp_ids, PREV_WP_IDS);
    storage_move_out(world_points,WORLD_PTS);
    storage_move_out(world_descs, WORLD_DESCS);
    storage_move_out(world_point_ages, WORLD_PT_AGE);
    return;
}

uint8_t process_vo_pipeline(pose_graph_stats_t* graph_stats,
                            uint8_t* img_l2_buffer,
                            work_memory_t work_memory)
{
    LOG_WARNING("*** VO Pipeline Results ***\n");
    LOG_WARNING("Total Frames: \t%d\nKeyframes: \t%d\nFeatures: \t%d\n",graph_stats->total_frames,graph_stats->total_keyframes,graph_stats->total_features);
    LOG_WARNING("***\n\n");

    pi_perf_reset();
    pi_perf_start();
    /* Extract new features */
    orb_features_t* curr_features = allocate_work_memory(&work_memory, sizeof(orb_features_t));
    curr_features->kpts = allocate_work_memory(&work_memory, (MAX_KEYPOINTS*sizeof(point2D_u16_t)));
    curr_features->descs = allocate_work_memory(&work_memory, (MAX_KEYPOINTS*sizeof(orb_descriptor_t)));
    curr_features->kpt_counter = 0;
    curr_features->kpt_capacity = MAX_KEYPOINTS;
    process_frame(curr_features, img_l2_buffer, work_memory);
    graph_stats->total_frames += 1;
    if (graph_stats->total_keyframes == 0)
    {
        pose_graph_add_first_keyframe(graph_stats, curr_features, work_memory);
        return IS_KEYFRAME;
    }
    pi_perf_stop(); 
    LOG_TIMING("[Extract Features] Number of cycles %d\n",pi_perf_read(PI_PERF_CYCLES));


    pi_perf_reset();
    pi_perf_start();
    /* Run EPnP RANSAC */
    matrix_2D_t K = {K_data,{3,3,0,0}};
    float* t = allocate_work_memory(&work_memory,16*sizeof(float));
    matrix_2D_t T = {t,{4,4,0,0}};
    feature_match_t* world_matches = allocate_work_memory(&work_memory,MAX_WORLD_FEATURES*sizeof(feature_match_t));
    uint16_t world_match_counter = 0;
    uint8_t epnp_failed = process_epnp_ransac(curr_features,graph_stats,&K,&T,world_matches,&world_match_counter,work_memory);
    pi_perf_stop(); 
    LOG_TIMING("[EPnP] Number of cycles %d\n",pi_perf_read(PI_PERF_CYCLES));
    matprint(&T, INFO_LEVEL);

    /* Match with previous keyframe features */
    pi_perf_reset();
    pi_perf_start();
    orb_features_t* prev_features = pose_graph_get_prev_features(graph_stats, &work_memory);
    feature_match_t* frame_matches = allocate_work_memory(&work_memory, (sizeof(feature_match_t)*MAX_KEYPOINTS));
    uint16_t frame_match_counter;
    if(SINGLE_CORE)
    {
        frame_match_counter = bf_match_two_way_max_flow(frame_matches,prev_features,curr_features,DISABLE_MAX_FLOW,HAMMING_THRESHOLD);
    }
    else
    {
        frame_match_counter = bf_match_two_way_max_flow_multicore(frame_matches,prev_features,curr_features,DISABLE_MAX_FLOW,HAMMING_THRESHOLD, NB_CORES);
    }
    pi_perf_stop(); 
    LOG_TIMING("[Feature Match Frames] Number of cycles %d\n",pi_perf_read(PI_PERF_CYCLES));

    /* Solve 8-point problem using RANAC (if epnp failed) */
    pi_perf_reset();
    pi_perf_start();
    float* t_prev = (float*)graph_stats->prev_pose;
    matrix_2D_t T_prev = {t_prev,{4,4,0,0}};
    if(epnp_failed)
    {
        process_eight_point_ransac(curr_features,prev_features,frame_matches,&T_prev,graph_stats,&K,&T,frame_match_counter,work_memory);
    }
    pi_perf_stop(); 
    LOG_TIMING("[8-point] Number of cycles %d\n",pi_perf_read(PI_PERF_CYCLES));
    
    float parallax = averageParallax(prev_features->kpts,curr_features->kpts,frame_matches, &K, frame_match_counter);
    LOG_INFO("Parallax normalized %f\n", parallax);
    if(parallax < KEYFRAME_PARALLAX_THRESHOLD)
    {
        /* No keyframe, just report transformation */
        LOG_WARNING("Regular Frame with Transformation\n");
        matprint(&T, WARNING_LEVEL);
        return IS_REGULAR_FRAME;
    }

    LOG_INFO("Triangulating new world points\n");
    pi_perf_reset();
    pi_perf_start();
    int16_t* curr_wp_ids = allocate_work_memory(&work_memory, (MAX_KEYPOINTS*sizeof(int16_t)));
    int16_t* prev_wp_ids = allocate_work_memory(&work_memory, (MAX_KEYPOINTS*sizeof(int16_t)));
    storage_move_in(prev_wp_ids, PREV_WP_IDS);
    update_world_points(prev_features, curr_features, frame_matches, world_matches, prev_wp_ids, curr_wp_ids, &T_prev, &T, &K, graph_stats, work_memory, frame_match_counter, world_match_counter);
    
    pose_graph_add_keyframe(graph_stats, T.data, curr_features, prev_features, curr_wp_ids, prev_wp_ids, work_memory);

    pi_perf_stop(); 
    LOG_TIMING("[Triangulation] Number of cycles %d\n",pi_perf_read(PI_PERF_CYCLES));

    LOG_WARNING("Adding keyframe\n");
    matprint(&T, WARNING_LEVEL);

    return IS_KEYFRAME;
}