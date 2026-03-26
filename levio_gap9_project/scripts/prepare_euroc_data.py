# Copyright (c) 2026 ETH Zurich. All rights reserved.
# SPDX-FileCopyrightText: 2026 ETH Zurich
# SPDX-License-Identifier: MIT
# Author: Jonas Kühne

"""
Convert a EuRoC MAV rosbag into the PGM-based format expected by the LEVIO
GAP9 pipeline.

Camera images are undistorted and subsampled to 160x120 via array slicing.
IMU measurements are batched sequentially in groups of 10 (matching the
200 Hz / 20 Hz ratio) and serialized as 28x10 PGM files (7 floats x 10
samples).

Usage:
    python prepare_euroc_data.py <rosbag_path> [--output_dir <dir>]

Example:
    python prepare_euroc_data.py ../../levio_python_model/data/MH_01_easy.bag --output_dir ../sensor_data/demo
"""

import argparse
import os
from struct import pack

import cv2
import numpy as np

try:
    from rosbag import Bag
except ImportError:
    from rosbag_standalone import Bag

# EuRoC cam0 intrinsics
K = np.array([[458.654, 0, 367.215], [0, 457.296, 248.375], [0, 0, 1]])
D = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])

# IMU batching: 200 Hz IMU / 20 Hz camera = 10 samples per frame
IMU_SAMPLES_PER_BATCH = 10

# ROS topics
TOPICS = ['/cam0/image_raw', '/imu0']


def extract_rosbag(bag_path, output_dir):
    """Extract and convert a EuRoC rosbag to PGM format."""
    img_dir = os.path.join(output_dir, 'images')
    imu_dir = os.path.join(output_dir, 'imu')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(imu_dir, exist_ok=True)

    # Find timestamp of first camera frame
    print(f'Reading {bag_path}...')
    first_cam_stamp = None
    with Bag(bag_path, 'r') as bag:
        for topic, msg, _ in bag.read_messages(topics=TOPICS):
            if 'cam0' in topic:
                first_cam_stamp = msg.header.stamp
                break

    if first_cam_stamp is None:
        print('Error: no camera messages found in bag')
        return

    # Process messages in bag order (interleaved camera + IMU)
    image_count = 0
    imu_batch = []
    imu_batch_idx = 1

    with Bag(bag_path, 'r') as bag:
        for topic, msg, _ in bag.read_messages(topics=TOPICS):
            if 'imu' in topic and msg.header.stamp >= first_cam_stamp:
                # Accumulate IMU sample
                imu_batch.append((
                    msg.header.stamp.to_sec(),
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z,
                ))

                if len(imu_batch) >= IMU_SAMPLES_PER_BATCH:
                    # Serialize batch as PGM: each row is 7 floats (28 bytes)
                    img = []
                    for sample in imu_batch:
                        row = []
                        for val in sample:
                            row.extend(b for b in pack('f', val))
                        img.append(row)
                    img = np.array(img, dtype=np.uint8)
                    cv2.imwrite(
                        os.path.join(imu_dir, f'imu{imu_batch_idx:06d}.pgm'),
                        img)
                    imu_batch = []
                    imu_batch_idx += 1

            elif 'cam0' in topic:
                # Undistort at full resolution, then subsample via slicing
                raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width)
                undistorted = cv2.undistort(raw, K, D)
                # Subsample: every 4th pixel, crop columns 57:-57
                # Produces 160x120 image matching GAP9 intrinsics
                subsampled = undistorted[1::4, 57:-57:4]
                cv2.imwrite(
                    os.path.join(img_dir, f'frame{image_count:06d}.pgm'),
                    subsampled)
                image_count += 1

    print(f'Done. {image_count} frames, {imu_batch_idx - 1} IMU batches '
          f'written to {output_dir}')

    # Write start_stop_indices.h
    header_path = os.path.join(output_dir, 'start_stop_indices.h')
    # Path is relative to the build/ directory: ../sensor_data/<name>/
    data_prefix = '../' + os.path.relpath(
        output_dir, os.path.join(output_dir, '..', '..'))
    with open(header_path, 'w') as f:
        f.write('#ifndef __START_STOP_INDICES_H__\n')
        f.write('#define __START_STOP_INDICES_H__\n\n')
        f.write('#include "pmsis.h"\n\n')
        f.write(f'const char base_path[] = '
                f'"{data_prefix}/images/frame%06d.pgm";\n')
        f.write(f'const char base_imu_path[] = '
                f'"{data_prefix}/imu/imu%06d.pgm";\n')
        f.write(f'static uint16_t start_index = 1;\n')
        f.write(f'static uint16_t stop_index = {image_count};\n\n')
        f.write('#endif /* __START_STOP_INDICES_H__ */\n')
    print(f'Header written to {header_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Convert EuRoC rosbag to PGM format for LEVIO GAP9 pipeline')
    parser.add_argument('rosbag', help='Path to EuRoC .bag file')
    parser.add_argument('--output_dir', default='../sensor_data/demo',
                        help='Output directory (default: ../sensor_data/demo)')
    args = parser.parse_args()

    if not os.path.isfile(args.rosbag):
        print(f'Error: {args.rosbag} not found')
        return 1

    extract_rosbag(args.rosbag, args.output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
