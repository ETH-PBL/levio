# LEVIO — Python Reference Model

Python-based Visual-Inertial Odometry pipeline used for algorithm development and evaluation in the [LEVIO paper](https://doi.org/10.1109/JSEN.2025.3644788). Processes monocular camera images and IMU data from ROS bag files to produce metric-scale 6-DOF pose estimates.

## Installation

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
- Python 3.7+

### Setup

```bash
# Clone the repository and navigate to the Python model
cd levio_python_model

# Create and activate the conda environment
conda env create -f environment.yml
conda activate levio
```

The environment installs all required dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| NumPy | 1.18.5 | Array operations and linear algebra |
| SciPy | 1.5.0 | Rotation utilities |
| OpenCV | 4.3.0 | Feature detection, matching, image processing |
| GTSAM | 4.1.1 | Factor graph optimization and IMU preintegration |
| matplotlib | 3.3.0 | Trajectory visualization |
| rosbag-standalone | 1.17.4 | ROS bag parsing (no full ROS install required) |
| pandas | 1.3.5 | Data export |

### Dataset

Download the EuRoC MAV rosbag files from [https://projects.asl.ethz.ch/datasets/euroc-mav/](https://projects.asl.ethz.ch/datasets/euroc-mav/) and place them in a `data/` directory:

```
levio_python_model/
└── data/
    ├── MH_01_easy.bag
    ├── MH_02_easy.bag
    ├── MH_03_medium.bag
    ├── MH_04_difficult.bag
    └── MH_05_difficult.bag
```

## Usage

### Running the Pipeline

```python
from main_levio import VIOSystem

vio = VIOSystem()
vio.run_pipeline('data/MH_01_easy.bag', 'run_001')
```

Or directly:

```bash
python main_levio.py
```

By default, `main_levio.py` processes `data/MH_01_easy.bag`. Edit the `run_pipeline()` call at the bottom of the file to change the input sequence or run name.

### Output Files

Each run produces the following in the working directory:

| File | Description |
|------|-------------|
| `keyframes_stamped_traj_estimate.txt` | Keyframe poses in TUM format (timestamp tx ty tz qx qy qz qw) |
| `frames_stamped_traj_estimate.txt` | All frame poses in TUM format |
| `image.png` | Top-down trajectory visualization |

The TUM format output is compatible with standard evaluation tools such as [evo](https://github.com/MichaelGrupp/evo).

## Configuration

All parameters are set in `VIOSystem.__init__()` in [main_levio.py](main_levio.py):

### Pipeline Flags

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_epnp` | `True` | Use EPnP RANSAC for pose estimation against the 3D map |
| `use_optimization` | `True` | Enable windowed pose graph optimization |
| `use_keyframes` | `True` | Enable keyframe-based operation |

### Tuning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `parallax_kf` | 0.1 rad | Minimum parallax angle for keyframe insertion |
| `K` | EuRoC intrinsics | 3x3 camera intrinsic matrix |
| `D` | EuRoC distortion | Radial distortion coefficients |

### Camera Intrinsics (EuRoC)

```
fx = 458.654    fy = 457.296
cx = 367.215    cy = 248.375
k1 = -0.28341   k2 = 0.07396
```

To use a different camera, update `K` and `D` in the constructor and provide data as a ROS bag with topics `/cam0/image_raw` (mono8 images) and `/imu0` (IMU measurements).

## Project Structure

```
main_levio.py                          Entry point and VIO system orchestration
│
├── vio_pipeline_segments/
│   ├── vo_frontend.py                 Feature detection (GFTT + BRIEF), matching,
│   │                                  essential matrix and EPnP pose estimation
│   ├── factor_graph.py                Data structures: Frame, Point, Map
│   ├── vio_initialization.py          Metric scale and velocity recovery from
│   │                                  monocular-IMU constraints
│   └── gtsam_optimizer.py             GTSAM factor graph construction,
│                                      IMU preintegration, windowed optimization
│
└── utilities/
    ├── rosbag_extractor.py            Streaming rosbag parser for camera + IMU
    └── draw_trajectory.py             TUM format export and trajectory plotting
```

## Tested Configurations

Results reported in the paper were obtained on the following hardware/OS combinations:

- Lenovo ThinkPad T14 Gen 3 (Intel 12th Gen), Ubuntu 20.04 (native)
- Lenovo ThinkPad X13 2-in-1 Gen 5 (Intel), Ubuntu 22.04 (within Docker/WSL2)

Due to the sensitivity of the visual odometry pipeline to floating-point behavior, results may vary across different hardware configurations.

## References

- [GTSAM](https://gtsam.org/) — Factor graph optimization library
- [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/euroc-mav/) — Benchmark dataset
- [VO Pipeline by Vinohith](https://github.com/Vinohith/VO_pipeline) — Visual odometry reference
