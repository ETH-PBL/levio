# LEVIO: Lightweight Embedded Visual Inertial Odometry

**Published in IEEE Sensors Journal** | [Paper (DOI: 10.1109/JSEN.2025.3644788)](https://doi.org/10.1109/JSEN.2025.3644788) | [arXiv: 2602.03294](https://arxiv.org/abs/2602.03294)

LEVIO is a complete Visual-Inertial Odometry (VIO) system that recovers metric-scale 6-DOF camera poses by fusing monocular vision with inertial measurements. It is designed from the ground up for resource-constrained processors, achieving real-time performance on an ultra-low-power SoC with under 110 KB of working memory.

This repository contains a **Python reference model** for algorithm development and evaluation, and an **optimized C implementation** targeting the [GAP9](https://github.com/GreenWaves-Technologies/) multi-core RISC-V SoC by GreenWaves Technologies.

---

## Pipeline Overview

LEVIO implements a full VIO pipeline in five stages:

```
Camera Frames ──► Feature Detection ──► Pose Estimation ──► Keyframe Management ──► Pose Graph Optimization
                       & Matching        (EPnP / 8-pt)        & Triangulation     (with IMU factors)
                                                                                        ▲
IMU Measurements ──────────────────► Preintegration ────────────────────────────────────┘
```

1. **Feature Detection & Matching** — ORB-style pipeline: FAST corner detection with Harris scoring, BRIEF binary descriptors, and brute-force Hamming distance matching. On GAP9, detection is parallelized across 8 cluster cores using a patch-based decomposition.

2. **Pose Estimation** — Two modes depending on map maturity:
   - *Bootstrap*: Essential matrix via 8-point RANSAC, with translation scale recovered from IMU velocity.
   - *Relocalization*: EPnP RANSAC against the existing 3D map for absolute pose recovery.

3. **Keyframe Management & Triangulation** — Keyframes are selected based on a parallax threshold. New 3D landmarks are triangulated from keyframe pairs and added to a sparse world map, filtered by reprojection error.

4. **IMU Preintegration** — Gyroscope and accelerometer measurements are preintegrated between keyframes using trapezoidal integration, producing compact motion factors with analytical Jacobians for bias correction.

5. **Windowed Pose Graph Optimization** — A sliding window of keyframes is jointly optimized with 3D landmarks, IMU preintegration factors, and camera reprojection factors using Levenberg-Marquardt. Gravity direction is refined online by blending IMU observations with preintegration predictions.

For full algorithmic details, see the [paper](https://arxiv.org/abs/2602.03294).

---

## Repository Structure

```
├── levio_python_model/          Python reference implementation
│   ├── main_levio.py            VIO system entry point
│   ├── vio_pipeline_segments/   Core algorithms (frontend, optimizer, initialization)
│   ├── utilities/               Data I/O and visualization
│   └── environment.yml          Conda environment specification
│
├── levio_gap9_project/          Embedded C implementation for GAP9
│   ├── main.c                   Cluster task orchestration
│   ├── definitions/             Data structures, linear algebra, configuration
│   ├── feature_handling/        ORB detection & matching (single/multi-core)
│   ├── visual_odometry/         EPnP, essential matrix, triangulation
│   ├── optimizer/               IMU preintegration & pose graph optimization
│   ├── scripts/                 Data preparation (EuRoC rosbag → PGM)
│   └── CMakeLists.txt           Build configuration (requires GAP SDK)
│
├── CITATION.cff                 Machine-readable citation metadata
└── LICENSE                      MIT License
```

See the project-specific READMEs for installation and usage:
- [Python Reference Model — Setup & Usage](levio_python_model/README.md)
- [GAP9 Embedded Implementation — Setup & Usage](levio_gap9_project/README.md)

---

## Key Design Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Descriptor | BRIEF (256-bit binary) | Hamming distance is a single XOR + popcount — ideal for cores without FPU |
| Matching | Brute-force with age filtering | Avoids KD-tree memory overhead; age filter bounds search space |
| Optimization | Windowed Gauss-Newton (8 KFs) | Bounded memory and compute; no sparse Cholesky dependency |
| Scale recovery | Linear least-squares on IMU | Single-shot initialization, no dedicated motion sequence required |
| Image resolution | 160 x 120 on GAP9 | 14.4 KB per frame fits in L1 alongside feature buffers |

---

## Python vs. GAP9 Implementation

| | Python Reference | GAP9 Embedded |
|---|---|---|
| **Language** | Python (NumPy, OpenCV, GTSAM) | C (no external dependencies) |
| **Resolution** | 752 x 480 | 160 x 120 |
| **Linear algebra** | NumPy / GTSAM (float64) | Custom matrix library (float32) |
| **Optimization** | GTSAM Levenberg-Marquardt | Custom windowed Gauss-Newton |
| **Parallelism** | Single-threaded | 8-core RISC-V cluster with shared L1 |
| **Memory model** | Heap-allocated | L2 storage + DMA to 110 KB L1 scratchpad |
| **Purpose** | Algorithm development & evaluation | Deployment on ultra-low-power hardware |

---

## Dataset

Both implementations are evaluated on the [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/euroc-mav/) (Machine Hall sequences MH01–MH05). The dataset provides synchronized stereo camera and IMU data from a micro aerial vehicle; LEVIO uses only the left camera (monocular) and the IMU.

---

## Status

This repository contains the code accompanying the LEVIO paper. It is provided as-is for reproducibility and as a reference for follow-up work. While we welcome bug reports and questions, active feature development is not planned.

---

## Citation

If you use this work, please cite:

```bibtex
@article{kuehne2026levio,
  title     = {LEVIO: Lightweight Embedded Visual Inertial Odometry for Resource-Constrained Devices},
  author    = {K{\"u}hne, Jonas and Vogt, Christian and Magno, Michele and Benini, Luca},
  journal   = {IEEE Sensors Journal},
  volume    = {26},
  number    = {3},
  pages     = {5026--5036},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/JSEN.2025.3644788}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
