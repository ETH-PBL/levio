# Copyright (c) 2026 ETH Zurich. All rights reserved.
# SPDX-FileCopyrightText: 2026 ETH Zurich
# SPDX-License-Identifier: MIT
# Author: Jonas Kühne

"""Visualize the trajectory from GAP9 pipeline console output.

Parses keyframe and regular frame poses from the log and renders a
top-down (XZ-plane) trajectory image using OpenCV.

Usage:
    # First, capture the pipeline output to a log file:
    cmake --build build --target run 2>&1 | tee log/run.log

    # Then visualize:
    python scripts/visualize_log.py log/run.log
"""

import argparse

import cv2
import numpy as np


def parse_poses(log_paths):
    """Extract 4x4 poses from one or more log files."""
    poses = []
    for path in log_paths:
        with open(path, "r") as f:
            fetch_counter = 0
            for line in f:
                if "Adding keyframe" in line:
                    fetch_counter = 5
                    pose = []
                    keyframe = True
                elif "Regular Frame" in line:
                    fetch_counter = 5
                    pose = []
                    keyframe = False
                elif fetch_counter == 5:
                    # Skip one line after the header
                    fetch_counter -= 1
                elif fetch_counter > 0:
                    values = [float(val) for val in line[1:-3].split(",")]
                    pose.append(values)
                    fetch_counter -= 1
                    if fetch_counter == 0:
                        poses.append((np.array(pose), keyframe))
    return poses


def draw_trajectory(poses, canvas_size=1200, scale=10):
    """Render poses as a top-down trajectory on a canvas."""
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    center = canvas_size // 2

    for pose, is_keyframe in poses:
        try:
            inv_pose = np.linalg.inv(pose)
        except np.linalg.LinAlgError:
            continue

        x, _y, z = scale * inv_pose[:3, 3]
        draw_x = int(x) + center
        draw_y = canvas_size - int(z) - center
        color = (0, 0, 255) if is_keyframe else (255, 0, 0)
        cv2.circle(canvas, (draw_x, draw_y), 1, color, 2)

        cv2.imshow("Trajectory", canvas)
        cv2.waitKey(1)

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trajectory from GAP9 pipeline log output."
    )
    parser.add_argument("log_paths", type=str, nargs="+", help="Path(s) to log file(s)")
    args = parser.parse_args()

    poses = parse_poses(args.log_paths)
    print(f"Parsed {len(poses)} poses")

    canvas = draw_trajectory(poses)

    out_path = args.log_paths[0].rsplit(".", 1)[0] + "_trajectory.png"
    cv2.imwrite(out_path, canvas)
    print(f"Saved trajectory to {out_path}")

    cv2.imshow("Trajectory", canvas)
    cv2.waitKey(3000)


if __name__ == "__main__":
    main()
