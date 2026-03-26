# Copyright (c) 2026 ETH Zurich. All rights reserved.
# SPDX-FileCopyrightText: 2026 ETH Zurich
# SPDX-License-Identifier: MIT
# Author: Jonas Kühne

cmake_minimum_required(VERSION 3.19)

###############################################################################
# Panel Control
###############################################################################
set(TARGET_NAME "LEVIO_GAP9")
set(TARGET_SRCS 
    definitions/prng.c
    definitions/sorting.c
    definitions/work_memory.c
    definitions/matrix.c
    definitions/vio_definitions.c
    definitions/transformation.c
    definitions/pose_graph.c
    definitions/matrix3.c
    feature_handling/orb.c
    feature_handling/orb_gap.c
    feature_handling/bf_matcher.c
    visual_odometry/vo.c
    visual_odometry/triangulation.c
    visual_odometry/essential_matrix.c
    visual_odometry/epnp.c
    optimizer/imu_optimization.c
    optimizer/imu_bias_optimization.c
    optimizer/imu_calibration.c
    optimizer/pose_graph_optimizer.c
    optimizer/imu_preintegration.c
    main.c)

###############################################################################
# CMake pre initialization
###############################################################################

set(CONFIG_GAP_SDK_HOME $ENV{GAP_SDK_HOME})
include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)

include_directories(${CMAKE_SOURCE_DIR}/sensor_data/demo)
include_directories(${CMAKE_SOURCE_DIR})

project(${TARGET_NAME} C ASM)
add_executable(${TARGET_NAME} ${TARGET_SRCS})


###############################################################################
# CMake post initialization
###############################################################################
setupos(${TARGET_NAME})
