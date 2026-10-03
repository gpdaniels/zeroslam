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

This builds the `zeroslam` shared library from `source/` (C++17, nothing beyond the platform's threads library), with the public headers in `include/zeroslam/`.
The default build type is `Release`, executables land in `build/runtime/[config]/` and the shared library in `build/library/[config]/` (the Windows dll sits with the executables).

Every `tests/[path].test.cpp` is one test executable for `source/[path].hpp`, compiled from the library's objects so it reaches the internal classes.
The test executables are left out of the default build: `ctest` builds each one before it runs it, or `cmake --build . --target tests` builds them all.
The scripts in `checks/` (formatting, license headers, include guards, tool layering, raw allocations, ...) run as `check_*` tests too, or together with `cmake --build . --target check`, and `cmake --build . --target format` applies clang-format (the CI pins clang-format 21.1.2).

| Option | Default | Effect |
|---|---|---|
| `ZEROSLAM_SIMD` | `ON` | Compile the AVX and AVX2 (x86-64) or NEON (arm64) kernels. The best one the CPU supports is chosen at run time, and every tier gives the same results as the portable code. |
| `ZEROSLAM_WERROR` | `OFF` | Treat compiler warnings as errors, as the CI does. |
| `ZEROSLAM_CHECKS` | `ON` | Register the `checks/` scripts as tests and add the `check` target. |
| `ZEROSLAM_SANITIZE_ADDRESS`, `ZEROSLAM_SANITIZE_UNDEFINED`, `ZEROSLAM_SANITIZE_THREAD`, `ZEROSLAM_SANITIZE_MEMORY` | `OFF` | Build with a sanitizer (gcc and clang, memory is clang only, and address and thread cannot be combined). |
| `ZEROSLAM_COVERAGE` | `OFF` | Build with coverage instrumentation (gcov with gcc, llvm source based coverage with clang). |
| `ZEROSLAM_PERF` | `OFF` | Keep frame pointers for `perf` captures (linux and `RelWithDebInfo` only). |
| `BUILD_ALL_TOOLS` | `OFF` | Build every tool. |
| `BUILD_ZEROSLAM_[TOOL]` | `OFF` | Build one tool: its directory name upper cased with `-` as `_`, e.g. `-DBUILD_ZEROSLAM_PROCESS=ON`, `-DBUILD_ZEROSLAM_IMPORT_EUROC=ON`, and `-DBUILD_ZEROSLAM_ZEROSLAM=ON` for the launcher. |

The library runs its parallel loops on one pool of worker threads sized to the machine (the calling thread included).
The `ZEROSLAM_THREADS` environment variable overrides that count (`ZEROSLAM_THREADS=1` runs everything on the calling thread).
Results are bit identical from run to run, whatever the thread count and SIMD tier, on a given platform; they are not bit identical across platforms, whose maths libraries differ in the last bits.

## Tools ##

The tools are disabled by default, enable them all with `-DBUILD_ALL_TOOLS=ON` or one at a time with their `BUILD_ZEROSLAM_[TOOL]` option.
Each `tools/[tool]` directory builds a `zeroslam-[tool]` executable, and the `zeroslam` launcher runs `zeroslam [tool] [arguments...]` as `zeroslam-[tool] [arguments...]`, looking next to itself first and then on the `PATH` (`--quiet` hides the banner, `zeroslam help` lists the tools it finds).
The examples below use the launcher, `./runtime/Release/zeroslam-process ...` works the same way without it.

| Tool | Purpose | Needs |
|---|---|---|
| `zeroslam` | The launcher. | |
| `process` | Run the SLAM system over a scene, see "Processing a scene". | |
| `evaluate` | Align trajectories to a ground truth and report their errors, see "Evaluating a trajectory". | |
| `dataset` | List, download, validate, expand and collapse scenes, see "Fetching datasets". | `curl` at run time to download |
| `regression` | Fetch, validate, process and evaluate a scene, and log the result, see "Tracking accuracy over time". | the `process` and `evaluate` tools, and `dataset` to fetch |
| `gui` | Play a scene through the SLAM system live, or view a saved map, see "Viewing a scene". | OpenGL, and X11 on linux |
| `import-euroc`, `import-tumrgbd`, `import-eth3d`, `import-kitti`, `import-lamaria` | Convert a public dataset into a scene, see "Importing datasets". | the `dataset` tool, to pack and validate the result |
| `capture-oakd` | Record a Luxonis OAK-D camera (stereo, depth and imu) into a scene. | git and network access at configure time: it fetches depthai-core and builds its vcpkg dependencies |
| `capture-kinect`, `capture-webcam` | Placeholders, not implemented yet. | |
| `api-c`, `api-cpp`, `api-python` | Demonstrations of the C, C++ and python APIs on a rendered scene, registered as `tool_api_*` tests. | python3 for `api-python` |

`tools/common` is a static library of code shared by the tools (mcap, lz4, cdr, json, trajectory metrics, files and paths) and is always built.

## Using the library ##

The interface is the C API in `include/zeroslam/zeroslam.h`, wrapped for C++ by the header only `include/zeroslam/zeroslam.hpp` (`zeroslam::system`) and for python by the ctypes module `include/zeroslam/zeroslam.py`, which loads the library from `ZEROSLAM_API_LIBRARY`, then next to itself, then the working directory, then the system search path.
The `tools/api-c`, `tools/api-cpp` and `tools/api-python` demonstrations walk through every call.

1. `zeroslam_create` a system, and optionally `zeroslam_set_configuration` it (see "Configuration").
2. `zeroslam_set_sensor_rig`, before any data, with exactly one `zeroslam_sensor_camera` entry (the system is monocular, and the other sensor types are rejected): a non-zero id and a `zeroslam_sensor_parameters_camera_struct` of the image size, the focal lengths, the principal point in the pixel-centre frame (see "Pixel coordinates"), and the distortion `[k1 k2 p1 p2 k3 k4 k5 k6]` (zeros for an undistorted pinhole).
3. `zeroslam_set_sensor_data` with each frame: the camera's id, a nanosecond timestamp that increases strictly from frame to frame, and `width * height` bytes of row major 8 bit greyscale pixels. A batch is validated whole before any of it is processed.
4. `zeroslam_get_pose` (the newest posed frame) or `zeroslam_get_pose_at_timestamp` (the frame with exactly that timestamp) at any time, and `zeroslam_finalise` at the end of a run for a last global adjustment. Poses follow "Coordinate conventions": frames before initialisation completes, and frames dropped while tracking was lost, have no pose, and the covariance is not estimated yet (it is zero).
5. `zeroslam_get_map_chunk` (every landmark, the position arguments are not used yet), `zeroslam_get_map_lines`, `zeroslam_get_map_keyframes`, `zeroslam_get_map_edges` (the covisibility and loop edges between keyframes, by timestamp) and `zeroslam_get_map_voxels` (the voxel map of the current map's landmarks as of the newest keyframe: the voxel size, and each occupied voxel's lowest corner and landmark count) export the map.
6. `zeroslam_destroy` the system.

Every call returns a `zeroslam_return_enum`.
The getters fill caller owned buffers: a null or short buffer fails with `zeroslam_return_failure_insufficient_data_length` and reports the length needed, so a first call sizes the buffer and a second fills it.

### Configuration ###

The configuration is text, one `key=value` per line.
`zeroslam_get_configuration` returns every key with its current value, and `zeroslam_set_configuration` changes just the keys it is given, rejecting the whole text (`zeroslam_return_failure_invalid_configuration`) on an unknown key or value.
The `process` and `regression` tools pass any setting through with `--config key=value`, and the `gui` takes the front end settings it offers in its controls the same way.

| Key | Default | Values |
|---|---|---|
| `verbosity` | `1` | Log level, `0` silent, `1` errors, `2` warnings, `3` notes, `4` progress, `5` debug. |
| `detector` | `fast` | `fast`, `mser`, or a structure tensor corner measure: `klt`, `forstner`, `harris`, `rohr`, `kenney`. |
| `detector_sigma` | `2.5` | The structure tensor detector's integration scale (up to `3.5`). |
| `refiner` | `subpixel` | `none`, `subpixel`, or a structure tensor measure (as `detector`) to refine corners with. |
| `refiner_sigma` | `1.5` | The structure tensor refiner's integration scale (up to `3.2`). |
| `tracker` | `klt` | `klt` pyramidal optical flow of the detected features, or `extrema` curvature extrema tracks. |
| `association` | `both` | How frames are associated: `klt` flow alone, `match` descriptor matching alone, or `both`. |
| `descriptor` | `orb` | `orb`, `teblid`, `bsift` (binarised sift), or their 512-bit forms `teblid512` (TEBLID's own 512 tests) and `bsift512` (a four level thermometer code of the sift vector). |
| `affine` | `off` | Match landmarks against their whole descriptor history, and with `descriptor=bsift` add affine (tilted) views of each descriptor. |
| `blur` | `on` | Weight measurements by the frame's blur against the recent frames. |
| `lines` | `off` | Detect, track and map line segments too: detected on the frame resampled to its pinhole camera, carried from frame to frame along the motion of the points around them, and matched to the map's lines by projection, both with the edge's polarity. |
| `line_pose` | `off` | Use the line landmarks in pose estimation (with `lines=on`), weighted less the more points hold the pose. |
| `line_angle` | `off` | Drop a line observation seen within this many degrees of end on (below `89`), or `off`. |
| `culling` | `on` | Cull redundant keyframes. |
| `local_map` | `voxels` | Where each frame's projection matching to the map takes its candidate landmarks from: the `voxels` reached by rays cast from the frame through a voxel map of the landmarks (after Muglikar, Zhang and Scaramuzza, "Voxel Map for Visual SLAM", ICRA 2020), the keyframes `covisible` with the newest, `both`, or the covisible keyframes with the voxels added when those leave a frame with few tracked landmarks or when recovering a lost or failed pose (`fallback`). The voxels find the landmarks in view that no covisible keyframe holds: on EuRoC they cut the error per run by a fifth against `covisible`. |
| `place_recognition` | `ibow` | Which place recognition proposes the loop candidates: `hbst` counts the keyframes of the descriptors in the query's leaf of a Hamming binary search tree (Schlegel and Grisetti, RA-L 2018), `ibow` scores the keyframes through an incremental vocabulary of binary words built as the map grows, with no training, and proposes the best keyframe of each island of consecutive ones (iBoW-LCD, Garcia-Fidalgo and Ortiz, RA-L 2018), leaving out the recent and covisible keyframes before it ranks them. |
| `loop_revisits` | `off` | Close a loop when the keyframe shares landmarks with an earlier, covisible keyframe and those landmarks have moved since that keyframe recorded them. The recorded positions are the ones at the earlier keyframe's insertion, so the adjustments since read as drift; ORB-SLAM3 closes no such loops. |
| `loop_confirmations` | `1` | How many keyframes must verify a loop before it closes, as ORB-SLAM3 confirms one by three: the keyframe that found it, then up to five keyframes covisible with it, each matching the loop's records by projection through its similarity, then the keyframes that follow it, a loop being dropped when two in a row fail to verify it. `1` closes a loop as soon as one keyframe verifies it. |
| `global_adjustment` | `10` | Run a global adjustment every this many inserted keyframes (with `adjustment=absolute`), or `off`. |
| `adjustment` | `absolute` | How the map is adjusted as keyframes arrive. `absolute` adjusts the world poses of a window of the newest keyframes, with a global adjustment every `global_adjustment` keyframes and a pose graph at each loop closure. `relative` is adaptive relative bundle adjustment (Sibley, Mei, Reid and Newman, RSS 2009): the keyframes are joined by relative transforms, with a similarity for each loop, each landmark is held in one keyframe's frame, and a new keyframe solves only the region whose reprojection errors it changes, so the adjustment stays local however large the map grows, at a loop closure too. It is faster, but without the periodic global adjustments a monocular map drifts further in scale, so `absolute` is the default. |
| `depth` | `inverse` | Landmark parameterisation, `inverse` anchored inverse depth or `xyz`. |
| `budget` | `free` | `fixed` caps each pyramid level at its share of the feature budget, `free` lets the distributor keep more. |
| `damping` | `off` | Halve a klt step that reverses the previous one, damping oscillation. |
| `collisions` | `2` | Drop a track closer than this many pixels (up to `64`) to a stronger one, or `off`. |
| `outliers` | `0` | Unlink a track from its landmark after this many outlier frames, `0` never. |
| `anchor` | `off` | Track patches anchored to their first frame: `translation`, `affine`, `translation_illumination`, `affine_illumination`, or `off`. |
| `anchor_refresh` | `off` | Re-anchor a patch whose alignment error exceeds this fraction (up to `1`) of its flow's rejection gate (the klt error limit for patch anchors, the wavelet phase limit for wavelet anchors), or `off`. |
| `flow` | `wavelet` | `wavelet` quaternion wavelet phase flow, or `intensity` flow (about four times faster, and loses tracking more often under fast motion and blur). |
| `wavelet_window` | `2` | The wavelet flow's half window, `1` to `6`. |
| `wavelet_levels` | `6` | The wavelet decomposition levels, `2` to `8`. |
| `wavelet_robust` | `off` | Huber weight the wavelet flow's phase residuals. |
| `wavelet_undecimated` | `off` | Use an undecimated wavelet decomposition. |
| `wavelet_seed` | `klt_fallback` | Start the wavelet flow at `rest`, from the `klt` flow, or from the klt flow and fall back to it where the wavelet flow fails (`klt_fallback`). |
| `solver` | `dense_schur` | The bundle adjustment's linear solver: `dense_schur`, `square_root` (landmarks eliminated by QR), or `automatic` (square root once there are enough poses). Graphs the square root solver cannot take fall back to the dense one. |
| `solver_precision` | `double` | The linear solver's precision, `double` or `single`. |

## Scene format ##

A scene is one mcap file, `datasets/[dataset]/[scene].mcap`, holding:
- Raw image messages (`sensor_msgs/msg/Image`: `mono8` or `rgb8` for the cameras the system runs on, `mono16` or `32FC1` for depth) in uncompressed or lz4 compressed chunks (zstd chunks are rejected).
- Camera intrinsics as one camera info message per frame on `/sensor/[name]/camera_info`, a `plumb_bob` (`k1 k2 p1 p2 [k3]`) or `rational_polynomial` (`k1 k2 p1 p2 k3 k4 k5 k6`) pinhole with no rectification, with the principal point in the pixel-centre frame (see "Coordinate conventions" below). A distortion that folds back on itself, where the radial factor stops growing or reaches a pole of the rational model, limits the model to the radius inside that point; rays and pixels beyond it are treated as not visible.
- IMU messages.
- A ground truth trajectory on `/tf` as `root -> ego`, when the dataset has one.

Sensors are topics named `/sensor/[type]_[01-99]` (`image_01`, `image_02`, ... for cameras; `imu_01`, ... for imus; `lidar_01`, `gnss_01`...) and each sensor's frame carries its full name (e.g. `sensor/image_01`).

Transforms are stored in `/tf` following `root -> ego -> sensor/[name]` with the `root -> ego` transform being the ground truth.
A per topic message `ego -> sensor/[name]` transform poses each sensor on it with the calibration extrinsics (so extrinsics could change over time).

Scene mcap files can be viewed directly in a browser with web-viewers e.g. [Lichtblick](https://lichtblick-suite.github.io/lichtblick/).

The inspectable directory form produced by expanding a scene mirrors the topics, with timestamps written as seconds with nine decimals.
```
scene/
├── sensor/
│   ├── image_01/     # Frames named by their timestamp in nanoseconds, zero padded to 20 digits ([ns].pnm, P5, P6 or Pf).
│   ├── image_01.txt  # Per frame:    `[timestamp] [x] [y] [z] [qx] [qy] [qz] [qw] [model] [fx] [fy] [cx] [cy] [distortion...]`
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
LaMAria's pseudo ground truth is the pose of its right imu (its documentation says the left camera, but the poses turn with the imu's gyroscope) and is carried onto cam0 with the calibration's `T_b_s`; its cameras are mounted on their side, so the importer turns every frame 90 degrees clockwise upright and the camera frames, intrinsics and extrinsics with it.

The SLAM output (`trajectory.txt` from `zeroslam process`, `zeroslam_pose_struct` from the C API in `include/zeroslam/zeroslam.h`) uses the same TUM camera-to-world convention: `[timestamp] [x] [y] [z] [qx] [qy] [qz] [qw]` is the camera centre in the map's world and the camera-to-world rotation.
The map's world frame is the frame of the first posed camera, the initialisation anchor: the first pose is the identity, so the world starts out with x right, y down and z along the first view. It is not gravity aligned and has no heading.
Frames before initialisation completes, and frames dropped while tracking is lost, have no pose.
A monocular map also has no metric scale: one unit starts out as the initialisation baseline (bundle adjustment then moves it), and a submap started after tracking is lost takes its first baseline from the speed the lost map last moved at, so it carries on at roughly that scale but drifts on its own until a loop closure joins it to the map.
With `adjustment=relative` the map has no fixed world frame while it is built: after each keyframe the poses and landmarks are written out from the newest keyframe through the graph of relative transforms, so the newest poses agree with the map around them, but the world origin moves from keyframe to keyframe, and parts of the map far apart in the graph, such as the two ends of a long loop, need not agree with each other.
Finalising relaxes the graph into one map that agrees with all of its transforms as well as it can, with the initialisation anchor back at the identity, before the last global adjustment.

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

A detector works in indices, a refiner adds a fractional offset measured from the detecting index, and the sum is wrapped to a centre (`+0.5`) once, at the detector -> feature boundary, after the pyramid level is flattened onto level 0 (`index * scale + 0.5`, never `(index + 0.5) * scale`: the pyramid is decimated by two per level, so level pixel `i` is level-0 pixel `i * scale` with `scale = 2^level`).
Image samplers (the KLT tracker, descriptors, `image::interpolation`) take index-space positions, so a centre is converted back (`- 0.5`) before sampling, and the pixel containing a centre is its `floor`, never `round`.

Published calibrations (TUM RGB-D, EuRoC, ETH3D, any ros `camera_info` or OpenCV calibration) put pixel centres at integer coordinates, so a centred 640 wide camera is published with `cx = 319.5`.
A scene's `camera_info` (and the `[cx] [cy]` of the directory form) is in the pixel-centre frame: every importer adds `0.5` to a published `cx` and `cy` (and nothing to the focal lengths) when it writes a scene (`import::pixel_centre_principal_point` in `tools/common/import.hpp`), so scenes from different sources are consistent and nothing is shifted when one is read.
The exception is a calibration produced by COLMAP (LaMAria's pinhole calibrations come out of its `image_undistorter`), which already puts the centre of pixel `(0, 0)` at `(0.5, 0.5)` and is used as it is.
A caller of the C API likewise supplies `centre_x`/`centre_y` in the pixel-centre frame.
Shifting every feature by `+0.5` and the principal point by `+0.5` together is an exact identity under `u = fx * X / Z + cx`, so the estimators see the same rays as before.

## Processing a scene ##

The `process` tool runs the SLAM system over a scene mcap, frame by frame through the C API, and writes the trajectory (`trajectory.txt`, TUM format) and a binary ply of the landmarks and the camera frusta (`map.ply`) into the working directory.
Colour frames are converted to greyscale, and the first frame's intrinsics are kept for the whole run.
```
# Build the tools.
cd build
cmake .. -DBUILD_ZEROSLAM_ZEROSLAM=ON -DBUILD_ZEROSLAM_PROCESS=ON
cmake --build . --parallel 4

# Process a scene.
./runtime/Release/zeroslam process ../datasets/tum-rgbd/fr1_xyz.mcap

# Process the first 300 frames, logging progress, and report the errors against a ground truth.
./runtime/Release/zeroslam process ../datasets/tum-rgbd/fr1_xyz.mcap --frames 300 --verbose 4 --truth trajectory_gt.txt

# Process with other library settings (see "Configuration").
./runtime/Release/zeroslam process ../datasets/tum-rgbd/fr1_xyz.mcap --config descriptor=teblid --config lines=on
```

The options are `--frames [count]` (the first frames only), `--skip [count]` (ignore the first frames), `--verbose [0-5]` (the log level, default `1`), `--truth [trajectory.txt]` (end with the absolute, relative and per metre errors against a TUM trajectory), `--live` (skip the final global adjustment, so the trajectory is the one tracked live) and `--config key=value` (repeatable).

## Evaluating a trajectory ##

The `evaluate` tool aligns up to two TUM format trajectories to a ground truth, pairing poses by timestamp (the nearest within 20 ms), and reports the absolute trajectory error after the Sim(3) alignment, the relative displacement errors, the error per metre of path, and the per segment scale drift.

Usage:
```
# Build the tool.
cd build
cmake .. -DBUILD_ZEROSLAM_ZEROSLAM=ON -DBUILD_ZEROSLAM_EVALUATE=ON
cmake --build . --parallel 4

# Evaluate a trajectory with a ground truth.
./runtime/Release/zeroslam evaluate trajectory_gt.txt trajectory_eval_1.txt

# Evaluate two trajectories against a ground truth.
./runtime/Release/zeroslam evaluate trajectory_gt.txt trajectory_eval_1.txt trajectory_eval_2.txt

# Anchor the alignment on the first pose pair (the default), or on the two centroids for the least squares fit.
./runtime/Release/zeroslam evaluate trajectory_gt.txt trajectory_eval_1.txt --first
./runtime/Release/zeroslam evaluate trajectory_gt.txt trajectory_eval_1.txt --centroid

# Fail (exit code 1) when an aligned rmse exceeds 5 cm.
./runtime/Release/zeroslam evaluate trajectory_gt.txt trajectory_eval_1.txt --max-rmse 0.05

# Plot the trajectories viewed along each of the x, y, and z axes (trajectory_x.ppm, trajectory_y.ppm, trajectory_z.ppm).
./runtime/Release/zeroslam evaluate trajectory_gt.txt trajectory_eval_1.txt --plot xyz
```

A scene's ground truth is written as a trajectory file with `zeroslam dataset trajectory [scene.mcap] [trajectory.txt]`.

## Fetching datasets ##

The `dataset` tool can list, download, validate, expand, and collapse dataset scenes hosted at [gpdaniels/slam-datasets](https://huggingface.co/datasets/gpdaniels/slam-datasets), and write a scene's ground truth trajectory.
By default the datasets directory is the `datasets` directory next to the tool executable (`./datasets` when that cannot be determined), override it with `--datasets`.
Configuring with the dataset tool enabled links `build/runtime/[config]/datasets` to the source tree's `datasets/` directory, so the tools share one copy.

**Note: Downloading datasets with this tool requires that the `curl` executable is installed and reachable.**

The hub holds the `tum-rgbd`, `euroc-mav`, `eth3d-slam`, `kitti-odometry` and `lamaria` datasets.
Scenes are stored as one mcap file each, `[dataset]/[scene].mcap`, and `get` accepts a whole dataset (`tum-rgbd`) or a single scene (`tum-rgbd/fr1_xyz`), validating every scene it fetches.
Downloads stream to a `.part` file renamed into place after a size check, so interrupted downloads are detectable and rerunning a download completes or repairs the files (`--force` redownloads).
Private repositories are reached with `--token` or the `HF_TOKEN` environment variable, and `--repo` selects another hub repository (huggingface only).
`validate` checks a scene by name, every scene of a dataset, or a scene mcap by path: the raw frames, a pinhole calibration per frame, and the `root -> ego -> sensor` frame tree on `/tf`.

Usage:
```
# Build the tool.
cd build
cmake .. -DBUILD_ZEROSLAM_ZEROSLAM=ON -DBUILD_ZEROSLAM_DATASET=ON
cmake --build . --parallel 4

# List, download, and validate a scene.
./runtime/Release/zeroslam dataset list
./runtime/Release/zeroslam dataset get tum-rgbd/fr1_xyz
./runtime/Release/zeroslam dataset validate tum-rgbd/fr1_xyz

# Unpack a scene for inspection or editing, and pack it back.
./runtime/Release/zeroslam dataset expand ../datasets/tum-rgbd/fr1_xyz.mcap ./fr1_xyz-expanded
./runtime/Release/zeroslam dataset collapse ./fr1_xyz-expanded ../datasets/tum-rgbd/fr1_xyz.mcap

# Write a scene's ground truth as a trajectory file.
./runtime/Release/zeroslam dataset trajectory ../datasets/tum-rgbd/fr1_xyz.mcap trajectory_gt.txt
```

## Importing datasets ##

The importers turn a dataset's own release into a scene mcap in the convention above (see "Coordinate conventions"), writing the directory form next to the mcap (the mcap's path without the extension) and packing and validating it with the `dataset` tool (found next to the importer, or with `--tools-dir`).
They read already extracted directories, not zips.

| Tool | Input | Notes |
|---|---|---|
| `import-euroc` | `[mav0-dir] [groundtruth.csv] [output.mcap]` | EuRoC MAV ASL `mav0/`, with a corrected ground truth such as open_vins' `ov_data/euroc_mav/[sequence].csv`. |
| `import-tumrgbd` | `[tumrgbd-dir] [output.mcap]` | TUM RGB-D, with the rgb, depth and accelerometer streams. `--freiburg [1-3]` sets the camera calibration, by default read from `rgb.txt`'s header or the directory name. |
| `import-eth3d` | `[eth3d-dir] [output.mcap]` | ETH3D SLAM, the monocular part plus the stereo, depth and imu parts found inside it or given with `--stereo`, `--rgbd` and `--imu`. |
| `import-kitti` | `[sequence-dir] ([poses.txt]) [output.mcap]` | KITTI odometry, `--cameras` selects which of the four cameras to import. The poses are optional, only sequences 00 to 10 publish them. |
| `import-lamaria` | `[asl-dir] [pinhole.json] [output.mcap]` | LaMAria's undistorted ASL (pinhole) release, `--ground-truth` adds the pseudo ground truth where a sequence has one. |

```
# Build the importers, and the dataset tool they pack scenes with.
cd build
cmake .. -DBUILD_ZEROSLAM_ZEROSLAM=ON -DBUILD_ZEROSLAM_DATASET=ON -DBUILD_ZEROSLAM_IMPORT_EUROC=ON
cmake --build . --parallel 4

# Import a EuRoC sequence.
./runtime/Release/zeroslam import-euroc ./MH_01_easy/mav0 ./MH_01_easy.csv ../datasets/euroc-mav/mh_01_easy.mcap
```

## Viewing a scene ##

The `gui` tool plays a scene through the SLAM system live, drawing the image with its features, the landmarks, lines, voxels, keyframes, covisibility and loop edges, and the trajectory against the scene's ground truth with live metrics (see "Coordinate conventions" for the `Scale To Truth` and `Align To Truth` controls).
It also saves a finished map (`--save-map`) and displays a saved map instead of running the system (`--load-map`), and `--play`, `--screenshot [file]`, `--screenshot-after [frames]` and `--exit-after [frames]` script it.
`--config key=value` starts it with a front end setting it offers in its controls: `tracker`, `lines`, `culling`, `association`, `detector`, `descriptor`, `flow` and `local_map`; the controls otherwise start at the library's defaults (`--help` lists them).
It needs OpenGL, with X11 on linux, Cocoa on macOS and Win32 on windows.
```
cd build
cmake .. -DBUILD_ZEROSLAM_ZEROSLAM=ON -DBUILD_ZEROSLAM_GUI=ON
cmake --build . --parallel 4
./runtime/Release/zeroslam gui ../datasets/tum-rgbd/fr1_xyz.mcap --play
```

## Tracking accuracy over time ##

The `regression` tool downloads (if not downloaded) and validates a scene, runs the `process` tool on it, evaluates the recorded trajectory against the scene's ground truth with the `evaluate` tool, and appends the metrics and the commit to a log.
The scene is a scene mcap path, or a bare `[dataset]/[scene]` name looked up in the datasets directory next to the tools.

The recorded metrics never fail the run. The exit code only reflects operational failures, as interpreting metric changes depends on the code changes.

Usage:
```
# Build the tools.
cd build
cmake .. -DBUILD_ZEROSLAM_ZEROSLAM=ON -DBUILD_ZEROSLAM_REGRESSION=ON -DBUILD_ZEROSLAM_PROCESS=ON -DBUILD_ZEROSLAM_EVALUATE=ON -DBUILD_ZEROSLAM_DATASET=ON
cmake --build . --parallel 4

# Download (if not downloaded) and benchmark a scene in the datasets directory.
./runtime/Release/zeroslam regression tum-rgbd/fr1_xyz

# Benchmark only the first 150 frames of an mcap file scene.
./runtime/Release/zeroslam regression ../datasets/tum-rgbd/fr1_xyz.mcap --frames 150

# Benchmark and evaluate against a custom ground truth.
./runtime/Release/zeroslam regression ../datasets/tum-rgbd/fr1_xyz.mcap --ground-truth trajectory.txt
```

The run's outputs go to `--work-dir` (default `./zeroslam-regression-work/[name]`) and the results are appended to `--log` (default `./regression.log`).
`--name` and `--commit` override the recorded scene name and commit (default the git `HEAD`), `--config key=value` is forwarded to the `process` tool, `--first` or `--centroid` picks the alignment anchor, and `--tools-dir` points at the other tools.

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
