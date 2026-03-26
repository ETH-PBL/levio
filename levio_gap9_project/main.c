// Copyright (c) 2026 ETH Zurich. All rights reserved.
// SPDX-FileCopyrightText: 2026 ETH Zurich
// SPDX-License-Identifier: MIT
// Author: Jonas Kühne

/* PMSIS includes */
#include "pmsis.h"
#include "bsp/bsp.h"
#include "gaplib/ImgIO.h"

#include "definitions/type_definitions.h"
#include "feature_handling/orb_gap.h"
#include "visual_odometry/vo.h"
#include "optimizer/pose_graph_optimizer.h"
#include "optimizer/imu_preintegration.h"

#include "start_stop_indices.h"

PI_L2 uint8_t* img_buff[2];

work_memory_t total_l1_work_memory;
pose_graph_stats_t* graph_stats;

typedef struct cluster_arguments
{
    uint8_t* img_l2_buffer0;
    imu_measurement_t* current_imu_l2;
} cluster_arguments_t;

void init_data_structures(void *arg)
{
    pi_device_t* cluster_device = (pi_device_t*) arg;
    /* Memory allocation */
    graph_stats = init_pose_graph_stats(cluster_device);
    initialize_orb_storage_gap(cluster_device, IMG_WIDTH, IMG_HEIGHT);
    initialize_imu_preintegrator(cluster_device);
    total_l1_work_memory.memory_ptr = pi_cl_l1_malloc(cluster_device, L1_WORK_MEMORY_SIZE);
    total_l1_work_memory.size_left = L1_WORK_MEMORY_SIZE;
}

/* Cluster main entry, executed by core 0. */
void cluster_delegate(void *arg)
{
    work_memory_t work_memory = total_l1_work_memory;

    cluster_arguments_t* arguments = (cluster_arguments_t*) arg;

    if(graph_stats->total_keyframes > 0)
    {
        pi_perf_reset();
        pi_perf_start();
        process_imu_preintegration(arguments->current_imu_l2,work_memory);
        pi_perf_stop(); 
        LOG_TIMING("[IMU Pre-Integration] Number of cycles %d\n",pi_perf_read(PI_PERF_CYCLES));
    }
    uint8_t is_keyframe = process_vo_pipeline(graph_stats,arguments->img_l2_buffer0,work_memory);
    LOG_WARNING("Optimization required? %d\n", is_keyframe);

    if(is_keyframe)
    {
        extract_and_restart_imu_preintegration(graph_stats, work_memory);
        process_pose_graph_optimizer(graph_stats, work_memory);
    }
}

/* Fetch Image */
void fetch_image_from_host(char* ImageName, uint8_t* Input_1)
{
    ReadImageFromFile(ImageName, IMG_WIDTH, IMG_HEIGHT, 1, Input_1, IMG_SIZE*1*sizeof(char), IMGIO_OUTPUT_CHAR, 0);
}

/* Fetch Image */
void fetch_imu_from_host(char* ImageName, uint8_t* Input_1)
{
    ReadImageFromFile(ImageName, IMU_WIDTH, IMU_HEIGHT, 1, Input_1, IMU_WIDTH*IMU_HEIGHT*sizeof(char), IMGIO_OUTPUT_CHAR, 0);
}

/* Program Entry. */
int main(void)
{
    LOG_WARNING("\n\n\t *** ORB Descriptor and Detector ***\n\n");
    LOG_WARNING("Entering main controller\n");

    pi_freq_set(PI_FREQ_DOMAIN_FC, CLUSTER_FREQUENCY_HZ);
    pi_freq_set(PI_FREQ_DOMAIN_CL, CLUSTER_FREQUENCY_HZ);

    img_buff[0] = pi_l2_malloc(IMG_SIZE);
    initialize_data_storage();
    optimizer_init_l2_cache();

    uint32_t core_id = pi_core_id(), cluster_id = pi_cluster_id();
    pi_device_t* cluster_dev;
    if(pi_open(PI_CORE_CLUSTER, &cluster_dev))
    {
        LOG_ERROR("Cluster open failed !\n");
        pmsis_exit(-1);
    }

    /* Prepare cluster task and send it to cluster. */
    struct pi_cluster_task cl_task;
    pi_cluster_send_task_to_cl(cluster_dev, pi_cluster_task(&cl_task, init_data_structures, cluster_dev));

    LOG_WARNING("Base Path: %s\n",base_path);
    char img_path[64];
    char imu_path[64];
    snprintf(img_path, sizeof(img_path), base_path, start_index-1);
    fetch_image_from_host(img_path, img_buff[0]);
    cluster_arguments_t arguments = {img_buff[0], NULL};
    pi_cluster_send_task_to_cl(cluster_dev, pi_cluster_task(&cl_task, cluster_delegate, &arguments));

    imu_measurement_t current_imu_l2[IMU_HEIGHT];

    for(uint16_t current_idx = start_index; current_idx < stop_index; current_idx+= 1)
    {
        LOG_WARNING("\n***** Index = %d *****\n", current_idx);
        snprintf(img_path, sizeof(img_path), base_path, current_idx);
        fetch_image_from_host(img_path, img_buff[0]);
        snprintf(imu_path, sizeof(imu_path), base_imu_path, current_idx);
        fetch_imu_from_host(imu_path, (uint8_t*) current_imu_l2);

        cluster_arguments_t arguments = {img_buff[0], current_imu_l2};
        pi_cluster_send_task_to_cl(cluster_dev, pi_cluster_task(&cl_task, cluster_delegate, &arguments));
    }

    pi_cluster_close(cluster_dev);
    LOG_WARNING("Bye !\n");
    return 0;
}