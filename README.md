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
- Camera intrinsics as one camera info message per frame, with the principal point in the pixel-centre frame (see "Coordinate conventions" below).
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
│   ├── image_01.txt  # Per frame:    `[timestamp] [x] [y] [z] [qx] [qy] [qz] [qw] MODEL [fx] [fy] [cx] [cy] [[k1] [k2] [p1] [p2] [[k3]]]]`
│   ├── image_02/     # The second camera's frames.
│   ├── image_02.txt  # The second camera's model/intrinsics/extrinsics.
│   └── imu_01.txt    # Per sample:   `[timestamp] [x] [y] [z] [qx] [qy] [qz] [qw] [wx] [wy] [wz] [ax] [ay] [az]`
└── trajectory.txt    # Ground truth: `[timestamp] [x] [y] [z] [qx] [qy] [qz] [qw]`
```

## Coordinate conventions ##

Every frame is right-handed and every rotation is a unit quaternion written scalar-last, `[qx] [qy] [qz] [qw]`.
A transform `parent -> child` on `/tf`, and every `[x] [y] [z] [qx] [qy] [qz] [qw]` in the directory form, is the pose of the child in the parent: `p_parent = R(q) * p_child + t`.

The scene frames:
- `root` is the world: metres, z up (gravity, to within the tilt of the rig that captured it), x and y horizontal with the dataset's own arbitrary azimuth, and the origin wherever the dataset put it.
- `ego` is the platform, and it is the primary camera's (`image_01`) optical frame: x right, y down, z forward along the optical axis.
  The primary camera's `ego -> sensor/image_01` extrinsic is therefore always the identity, and `zeroslam dataset validate` rejects a scene where it is not.
- `sensor/[name]` is each other sensor's own frame, posed on the ego by its extrinsic. Cameras use the same optical convention.
  IMU samples are in the imu's own frame exactly as measured (rad/s and m/s²), not rotated onto the ego; the extrinsic says where that frame sits.
- The ground truth (`root -> ego` on `/tf`, `trajectory.txt` in the directory form) is the primary camera's camera-to-world pose in the TUM RGB-D trajectory format: the camera's position in the world and the rotation taking camera coordinates to world coordinates.

Importers are responsible for re-expressing a dataset into this convention rather than leaving it to every consumer: EuRoC's ground truth is the pose of its imu body and is carried onto cam0 with the calibration's `T_BS`; TUM RGB-D's and ETH3D's are already the rgb camera's pose in a z-up motion capture world, and TUM's accelerometer rides a half turn about the optical axis.

The SLAM output (`trajectory.txt` from `zeroslam process`, `zeroslam_pose_struct` from the C API in `include/zeroslam/zeroslam.h`) uses the same TUM camera-to-world convention: `[timestamp] [x] [y] [z] [qx] [qy] [qz] [qw]` is the camera centre in the map's world and the camera-to-world rotation.
The map's world frame is the first camera's frame: the first pose is the identity, so the world starts out with x right, y down and z along the first view. It is not gravity aligned and has no heading.
A monocular map also has no metric scale: one unit is the initialisation baseline.

An estimate and a ground truth therefore differ by a rigid pin of the first pose (`T_gt(first) * inverse(T_estimate(first))`) plus a scale.
That pin, at the fitted scale, is what `zeroslam gui` draws by default (`Scale To Truth` on, `Align To Truth` off; the fitted rotation is the optional extra and unticking the scale shows the map in its own units), and `zeroslam evaluate` fits a Sim(3) anchored on the first pose (`--first`, the default) or on the centroids (`--centroid`).

Positions alone do not always hold all three turns of that fit: a straight trajectory holds none about its own direction of travel, because every roll about that line leaves the residual unchanged.
The fit therefore takes the turns the positions do not hold from the two trajectories' own orientations at the anchor, so the aligned view is steady rather than rolling as poses arrive, and both the viewer and `zeroslam evaluate` say when an alignment is in that state (KITTI odometry 04, 394 m of motorway, always is).

### Pixel coordinates ###

The image origin is the top left corner of pixel `(0, 0)`; pixel `(i, j)` spans `[i, i + 1) x [j, j + 1)` and its centre is at `(i + 0.5, j + 0.5)`.
Three kinds of value carry image positions, and `source/core/coordinates.hpp` names them and converts between them:

| Name | Type | Range | Meaning |
|---|---|---|---|
| `pixel_index` | int | `0 .. size - 1` | the ordinal of a pixel: what a detector's probe grid emits and what image sampling, suppression and distribution consume |
| `pixel_centre` | float | `0.5 .. size - 0.5` | a position in the continuous image: what the camera model projects to and unprojects from, what the map stores, what every exported keypoint, track, line endpoint and observation carries; a centred camera has `cx = width / 2` |
| image plane | float | unit depth | `((u - cx) / fx, (v - cy) / fy)`: what the two-view, PnP, triangulation and cheirality solvers consume |

A detector works in indices, a refiner adds a fractional offset measured from the detecting index, and the sum is wrapped to a centre (`+0.5`) once, at the detector -> feature boundary, after the pyramid level is flattened onto level 0 (`index * scale + 0.5`, never `(index + 0.5) * scale`: the pyramid is decimated, so level pixel `i` is level-0 pixel `i * scale`).
Image samplers (the KLT tracker, descriptors, `image::interpolation`) take index-space positions, so a centre is converted back (`- 0.5`) before sampling, and the pixel containing a centre is its `floor`, never `round`.

Published calibrations (TUM RGB-D, EuRoC, ETH3D, any ros `camera_info` or OpenCV calibration) put pixel centres at integer coordinates, so a centred 640 wide camera is published with `cx = 319.5`.
A scene's `camera_info` (and the `[cx] [cy]` of the directory form) is in the pixel-centre frame: every importer adds `0.5` to a published `cx` and `cy` (and nothing to the focal lengths) when it writes a scene (`import::pixel_centre_principal_point` in `tools/common/import.hpp`), so scenes from different sources are consistent and nothing is shifted when one is read.
A caller of the C API likewise supplies `centre_x`/`centre_y` in the pixel-centre frame.
Shifting every feature by `+0.5` and the principal point by `+0.5` together is an exact identity under `u = fx * X / Z + cx`, so the estimators see the same rays as before.

## Processing a scene ##

The tools directory contains a tool directory called `process`, target/binary is `zeroslam-process`, run as `zeroslam process`.
This tool takes a scene mcap and runs the SLAM system on it outputting a trajectory file and pointcloud for evaluation.
```
# Build the tools.
cd build
cmake --build . --parallel 4

# Process a scene.
./runtime/Release/zeroslam process ../datasets/freiburg/xyz.mcap
```

The program will output a trajectory file in TUM format and a ply pointcloud file.

## Evaluating a trajectory ##

The tools directory contains a tool directory called `evaluate`, target/binary is `zeroslam-evaluate`, run as `zeroslam evaluate`.
This tool aligns trajectories in the TUM format and returns the error after scaling and alignment.

Usage:
```
# Build the tools.
cd build
cmake --build . --parallel 4

# Evaluate a trajectory with a ground truth.
./runtime/Release/zeroslam evaluate trajectory_gt.txt trajectory_eval_1.txt

# Evaluate two trajectories against a ground truth.
./runtime/Release/zeroslam evaluate trajectory_gt.txt trajectory_eval_1.txt trajectory_eval_2.txt

# Ensure the first pose is aligned.
./runtime/Release/zeroslam evaluate trajectory_gt.txt trajectory_eval_1.txt --first

# Plot the trajectories from each of the x, y, or z, planes.
./runtime/Release/zeroslam evaluate trajectory_gt.txt trajectory_eval_1.txt --plot xyz
```

## Fetching datasets ##

The tools directory contains a tool directory called `dataset`, target/binary is `zeroslam-dataset`, run as `zeroslam dataset`.
This tool can list, download, validate, expand, and collapse, dataset scenes hosted at [gpdaniels/slam-datasets](https://huggingface.co/datasets/gpdaniels/slam-datasets).
By default the datasets directory is assumed to be next to the tool executable (`./datasets` when that cannot be determined) override with `--datasets`.

**Note: Downloading datasets with this tool requires that the `curl` executable is installed and reachable.**

Scenes are stored as one mcap file each, `[dataset]/[scene].mcap`, and `get` accepts a whole dataset (`freiburg`) or a single scene (`freiburg/xyz`).
Downloads stream to a `.part` file renamed into place after a size check, so interrupted downloads are detectable and rerunning a download completes or repairs the files (`--force` redownloads).
Private repositories are reached with `--token` or the `HF_TOKEN` environment variable, and `--repo` selects another hub repository (huggingface only).

Usage:
```
# Build the tool.
cd build
cmake --build . --parallel 4

# List, download, and validate a scene.
./runtime/Release/zeroslam dataset list
./runtime/Release/zeroslam dataset get freiburg/xyz
./runtime/Release/zeroslam dataset validate freiburg/xyz

# Unpack a scene for inspection or editing, and pack it back.
./runtime/Release/zeroslam dataset expand ../datasets/freiburg/xyz.mcap ./xyz-expanded
./runtime/Release/zeroslam dataset collapse ./xyz-expanded ../datasets/freiburg/xyz.mcap
```

## Tracking accuracy over time ##

The tools directory contains a tool directory called `regression`, target/binary is `zeroslam-regression`, run as `zeroslam regression`.
This tool downloads (if not downloaded), validates a scene, runs the SLAM system on it, and evaluates the recorded trajectory against a ground truth.

The recorded metrics never fail the run. The exit code only reflects operational failures, as interpreting metric changes depends on the code changes.

Usage:
```
# Build the tools.
cd build
cmake --build . --parallel 4

# Download (if not downloaded) and benchmark a scene in the datasets directory.
./runtime/Release/zeroslam regression freiburg/xyz

# Benchmark only the first 150 frames of an mcap file scene.
./runtime/Release/zeroslam regression ../datasets/freiburg/xyz.mcap --frames 150

# Benchmark and evaluate against a custom ground truth.
./runtime/Release/zeroslam regression ../datasets/freiburg/xyz.mcap --ground-truth trajectory.txt
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
