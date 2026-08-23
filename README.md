# ZeroSLAM #

SLAM without dependencies.

```
.-----------------------------------------------.
|   _____             _____ __    _____ _____   |
|  |__   |___ ___ ___|   __|  |  |  _  |     |  |
|  |   __| -_|  _| . |__   |  |__|     | | | |  |
|  |_____|___|_| |___|_____|_____|__|__|_|_|_|  |
|                                               |
| This software is a:                           |
|  |- simple                                    |
|  |- minimal                                   |
|  |- indirect                                  |
|  |- monocular                                 |
|  |- factor-graph                              |
|  |- deterministic                             |
|  |- dependency-free                           |
|  '- visual SLAM system written in pure C++.   |
|                                               |
| No external libraries. No frills. Just SLAM.  |
|                                               |
| >   https://github.com/gpdaniels/zeroslam   < |
|                                               |
| Licensed under GPLv3                          |
| Get in touch for commercial licensing.        |
'-----------------------------------------------'
```

## Building and testing ##

Standard cmake workflow:

```
mkdir -p build
cd build
cmake ..
cmake --build . --parallel 4
ctest
```

## Scene format ##

A scene is one mcap file, `datasets/[dataset]/[scene].mcap`, holding something like:
- Raw image messages in lz4 compressed chunks (zstd chunks are rejected).
- Camera intrinsics as one camera info message per frame.
- IMU messages.
- A ground truth trajectory on `/tf` as `root -> ego`.

Sensors are topics named `/sensor/[type]_[01-99]` (`image_01`, `image_02`, ... for cameras; `imu_01`, ... for imus; `lidar_01`, `gnss_01`...) and each sensor's frame carries its full name (e.g. `sensor/image_01`).

Transforms are stored in `/tf` following `root -> ego -> sensor/[name]` with the `root -> ego` transform being the ground truth.
A per topic message `ego -> sensor/[name]` transform poses each sensor on it with the calibration extrinsics (so extrinsics could change over time).

Scene mcap files can be viewed directly in a browser with web-viewers e.g. [Lichtblick](https://lichtblick-suite.github.io/lichtblick/). 

The inspectable directory form produced by expanding a dataset mirrors the topics.
```
scene/
├── sensor/
│   ├── image_01/     # Frames named by their timestamp in nanoseconds ([ns].pgm).
│   ├── image_01.txt  # Per frame:    `[timestamp] [x] [y] [z] [qx] [qy] [qz] [qw] MODEL [fx] [fy] [cx] [cy] [[k1] [k2] [p1] [p2]]`
│   ├── image_02/     # The second camera's frames.
│   ├── image_02.txt  # The second camera's model/intrinsics/extrinsics.
│   └── imu_01.txt    # Per sample:   `[timestamp] [x] [y] [z] [qx] [qy] [qz] [qw] [wx] [wy] [wz] [ax] [ay] [az]`
└── trajectory.txt    # Ground truth: `[timestamp] [x] [y] [z] [qx] [qy] [qz] [qw]`
```

## Processing a scene ##

The tools directory contains a tool directory called `process`, target/binary is `zeroslam-process`.
This tool takes a scene mcap and runs the SLAM system on it outputting a trajectory file and pointcloud for evaluation.
```
# Build the tool.
cd build
cmake --build . --parallel 4 --target zeroslam-process

# Process a scene.
./runtime/Release/zeroslam-process ../datasets/freiburg/xyz.mcap
```

The program will output a trajectory file in TUM format and a ply pointcloud file.

## Evaluating a trajectory ##

The tools directory contains a tool directory called `evaluate`, target/binary is `zeroslam-evaluate`.
This tool aligns trajectories in the TUM format and returns the error after scaling and alignment.

Usage:
```
# Build the tool.
cd build
cmake --build . --parallel 4 --target zeroslam-evaluate

# Evaluate a trajectory with a ground truth.
./runtime/Release/zeroslam-evaluate trajectory_gt.txt trajectory_eval_1.txt

# Evaluate two trajectories against a ground truth.
./runtime/Release/zeroslam-evaluate trajectory_gt.txt trajectory_eval_1.txt trajectory_eval_2.txt

# Ensure the first pose is aligned.
./runtime/Release/zeroslam-evaluate trajectory_gt.txt trajectory_eval_1.txt --first

# Plot the trajectories from each of the x, y, or z, planes.
./runtime/Release/zeroslam-evaluate trajectory_gt.txt trajectory_eval_1.txt --plot xyz
```

## License ##

Copyright (C) 2026 Geoffrey Daniels. https://gpdaniels.com/

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License only.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
