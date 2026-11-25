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

The test of the inertial initialisation also runs over windows of a EuRoC MAV scene's ground truth and imu samples when `ZEROSLAM_EUROC_SCENE` names the scene's directory form (`zeroslam dataset expand`, see "Fetching datasets").

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
| `android` | Run the SLAM system live on an android phone's main camera, see "Running on Android". On a desktop it builds the test of the app's camera model and session on a rendered scene, registered as the `tool_android` test. | the android ndk to build, and the sdk's build tools and platform and a jdk to package |

`tools/common` is a static library of code shared by the tools (mcap, lz4, cdr, json, trajectory metrics, files and paths) and is always built.

## Using the library ##

The interface is the C API in `include/zeroslam/zeroslam.h`, wrapped for C++ by the header only `include/zeroslam/zeroslam.hpp` (`zeroslam::system`) and for python by the ctypes module `include/zeroslam/zeroslam.py`, which loads the library from `ZEROSLAM_API_LIBRARY`, then next to itself, then the working directory, then the system search path.
The `tools/api-c`, `tools/api-cpp` and `tools/api-python` demonstrations walk through every call.

1. `zeroslam_create` a system, and optionally `zeroslam_set_configuration` it (see "Configuration").
2. `zeroslam_set_sensor_rig`, before any data, with one or more `zeroslam_sensor_camera` entries (up to 16; the other sensor types are rejected): each a non-zero id and either a `zeroslam_sensor_parameters_camera_struct` of the image size, the focal lengths, the principal point in the pixel-centre frame (see "Pixel coordinates"), and the distortion `[k1 k2 p1 p2 k3 k4 k5 k6]` (zeros for an undistorted pinhole), or a `zeroslam_sensor_parameters_camera_model_struct` of the image size, a `zeroslam_camera_model_enum` and that model's parameters (see "Camera models"). The entry's `parameters_length` is the size of the struct it points to, which tells the two apart. The first camera is the primary camera, whose poses are reported. With several cameras, optionally `zeroslam_set_sensor_extrinsics` next (see "Camera rigs").
3. `zeroslam_set_sensor_data` with each frame: the camera's id, a nanosecond timestamp that increases strictly from frame to frame, and `width * height` bytes of row major 8 bit greyscale pixels. A batch is validated whole before any of it is processed.
   A single camera may have a depth camera registered to it, its frames given the same way (see "Depth cameras").
4. `zeroslam_get_pose` (the newest posed frame) or `zeroslam_get_pose_at_timestamp` (the frame with exactly that timestamp) at any time, and `zeroslam_finalise` at the end of a run for a last global adjustment. `zeroslam_get_camera_calibration` gives a camera's calibration as the system holds it, in a `zeroslam_sensor_parameters_camera_model_struct` of the rig's model: the rig's, refined when the `calibration` key asks for it (see "Calibrating the camera"). Poses follow "Coordinate conventions": frames before initialisation completes, and frames dropped while tracking was lost, have no pose, and the covariance is not estimated yet (it is zero).
5. `zeroslam_get_map_chunk` (every landmark, the position arguments are not used yet), `zeroslam_get_map_lines`, `zeroslam_get_map_keyframes`, `zeroslam_get_map_edges` (the covisibility and loop edges between keyframes, by timestamp) and `zeroslam_get_map_voxels` (the voxel map of the current map's landmarks as of the newest keyframe: the voxel size, and each occupied voxel's lowest corner and landmark count) export the map.
6. `zeroslam_destroy` the system.

Every call returns a `zeroslam_return_enum`.
The getters fill caller owned buffers: a null or short buffer fails with `zeroslam_return_failure_insufficient_data_length` and reports the length needed, so a first call sizes the buffer and a second fills it.

### Camera models ###

Every model's parameters start with the focal lengths and the principal point `[fx fy cx cy]`, in pixels in the pixel-centre frame, and go on with the model's own; all of them are refined together as one state vector, and each model has the jacobians of its projection with respect to the point and to every parameter.
A scene names the model in its camera info's `distortion_model`, with the model's own parameters in `d` (see "Scene format").

| Model | `zeroslam_camera_model_` | Scene `distortion_model` (`d`) | Parameters | Valid region |
|---|---|---|---|---|
| Pinhole | `pinhole` | `plumb_bob` with no `d` | `fx fy cx cy` | in front of the camera |
| OpenCV's radial tangential (plumb bob and rational) | `radial_tangential` | `plumb_bob` (`k1 k2 p1 p2 [k3]`), `rational_polynomial` (`k1 k2 p1 p2 k3 k4 k5 k6`) | `fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6` | in front, inside the radius where the radial distortion folds back or reaches a pole |
| OpenCV's whole pinhole model, with the thin prism and the tilted sensor | `opencv` | `rational_polynomial` with 12 or 14 coefficients | `fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4 tau_x tau_y` | as the radial tangential, and in front of the tilted sensor |
| Kannala-Brandt equidistant fisheye (OpenCV's fisheye, Kalibr's `pinhole-equi`) | `kannala_brandt` | `equidistant` or `kannala_brandt` (`k1 k2 k3 k4`) | `fx fy cx cy k1 k2 k3 k4` | out to where `theta_d` stops growing, up to a half turn |
| Field of view (Devernay and Faugeras), PTAM's `ATANCamera` | `field_of_view` | `fov` (`w`) | `fx fy cx cy w` | in front of the camera, `w` in `[0, pi)` |
| Unified (Geyer, Barreto, Mei without his distortion) | `unified` | `unified` (`xi`) | `fx fy cx cy xi` | `z > -min(xi, 1 / xi) * |p|`, past 90 degrees for `xi > 0` |
| Extended unified (Khomenko et al.) | `extended_unified` | `extended_unified` (`alpha beta`) | `fx fy cx cy alpha beta` | `z > -w * sqrt(beta (x^2 + y^2) + z^2)`, `w` from `alpha` |
| Double sphere (Usenko, Demmel and Cremers) | `double_sphere` | `double_sphere` (`xi alpha`) | `fx fy cx cy xi alpha` | `z > -w2 * |p|`, `w2` from `xi` and `alpha` |
| Project Aria's Fisheye624 (COLMAP's `RAD_TAN_THIN_PRISM_FISHEYE`) | `fisheye_624` | `fisheye_624` (`k0 .. k5 p0 p1 s0 .. s3`) | `fx fy cx cy k0 k1 k2 k3 k4 k5 p0 p1 s0 s1 s2 s3` | out to where `theta_d` stops growing, and where the tangential and thin prism terms keep the image from folding |

The equidistant projection is the Kannala-Brandt model with its `k` zero, and the stereographic one the unified model with `xi = 1` (and focal lengths of twice the lens's).
PTAM's calibration gives its parameters as fractions of the image size: `fx = width * p0`, `fy = height * p1`, `cx = width * p2`, `cy = height * p3` in the pixel-centre frame, and `w = p4`.
Points and pixels outside a model's valid region project and unproject to nothing, and are treated as not seen; unprojection is closed form for the pinhole, field of view, unified, extended unified and double sphere models and an iteration bounded to the valid region for the others.

Inside the library the camera is the type erased `sensor::camera::model<double>` (`source/sensor/camera/model.hpp`), which also gives the pinhole `[fx fy cx cy]` that agrees with the model to first order at its principal point, for the parts that need a pinhole: the resampling the line detector runs on, the image the epipolar re-acquisition matches on, and the scale of the pixel noise.
The two-view initialisation, the turn between frames, the perspective and essential solvers, the line measurements, the epipolar re-acquisition and the inverse depth landmarks (`depth=inverse`, the default) work on the image plane `(x / z, y / z)`, and take rays within 80 degrees of the optical axis; a wide angle camera's rays further out are tracked, triangulated, matched by projection and adjusted through its model, and with `depth=xyz` landmarks past 90 degrees from a keyframe's axis are mapped too.
A camera seeing past 180 degrees runs (the pipeline test runs a double sphere lens whose image circle reaches 123 degrees from its axis, 108 at the edge of the image), but its outer ring adds to the map only through those.

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
| `culling` | `on` | Cull redundant keyframes: one past the recent window whose landmarks are nearly all seen by three other keyframes, among them one from before it and one from after it. No landmark is removed with a keyframe. |
| `local_map` | `voxels` | Where each frame's projection matching to the map takes its candidate landmarks from: the `voxels` reached by rays cast from the frame through a voxel map of the landmarks (after Muglikar, Zhang and Scaramuzza, "Voxel Map for Visual SLAM", ICRA 2020), the keyframes `covisible` with the newest, `both`, or the covisible keyframes with the voxels added when those leave a frame with few tracked landmarks or when recovering a lost or failed pose (`fallback`). The voxels find the landmarks in view that no covisible keyframe holds: on EuRoC they cut the error per run by a fifth against `covisible`. |
| `place_recognition` | `ibow` | Which place recognition proposes the loop candidates: `hbst` counts the keyframes of the descriptors in the query's leaf of a Hamming binary search tree (Schlegel and Grisetti, RA-L 2018), `ibow` scores the keyframes through an incremental vocabulary of binary words built as the map grows, with no training, and proposes the best keyframe of each island of consecutive ones (iBoW-LCD, Garcia-Fidalgo and Ortiz, RA-L 2018), leaving out the recent and covisible keyframes before it ranks them. |
| `loop_revisits` | `off` | Close a loop when the keyframe shares landmarks with an earlier, covisible keyframe and those landmarks have moved since that keyframe recorded them. The recorded positions are the ones at the earlier keyframe's insertion, so the adjustments since read as drift; ORB-SLAM3 closes no such loops. |
| `loop_confirmations` | `3` | How many keyframes must verify a loop before it closes, as ORB-SLAM3 confirms one by three: the keyframe that found it, then up to five keyframes covisible with it, each matching the loop's records by projection through its similarity, then the keyframes that follow it, a loop being dropped when two in a row fail to verify it. `1` closes a loop as soon as one keyframe verifies it. A loop that rescales the map more than twofold (up to tenfold), as a monocular map's scale drifts over a long loop, needs three keyframes whatever this is, each agreeing with its scale within 20%. |
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
| `calibration` | `off` | Refine the camera's calibration in the adjustments of the whole map, see "Calibrating the camera": `focal` the focal lengths, `focal_centre` and the principal point, `intrinsics` and the leading distortion terms (`k1 k2 p1 p2`), `all` every parameter, each on its own. |
| `extrinsics` | `declared` | Which extrinsics of a camera rig the absolute adjustments refine (see "Camera rigs"): those `zeroslam_set_sensor_extrinsics` declared refined, none (`fixed`), or every camera's but the primary's (`refine`). |
| `imu_calibration` | `off` | Which of an imu's own calibration the adjustments of the whole map refine (see "Inertial sensors"): none, or the matrix of its `gyroscope`'s scale and misalignment. |

### Calibrating the camera ###

With `calibration` other than `off` the camera's intrinsics are a variable of the adjustments of the whole map (every `global_adjustment` keyframes, at each loop closure and in `zeroslam_finalise`): one vertex shared by every frame taken with the camera, solved with the poses and landmarks by either solver.
The frames processed after an adjustment take the refined camera, as do the frames already in the map, the tracking, the matching and the loop closure, while the caller keeps giving the calibration it started with; a different calibration given with a frame starts the calibration again from it.
The line observations (`lines=on`) take part as the points do; the window adjustments and the pose adjustments hold the camera as it was last refined, as does `adjustment=relative` until `zeroslam_finalise`.

Which parameters are refined follows what the camera model says each one is (a focal length, the principal point, the leading or the higher order distortion), so any model that says so can be refined; one that does not can only be refined `all`.
Below `all` the two focal lengths keep their ratio, the pixels' aspect ratio, which is the sensor's: a camera moving in a plane, as a vehicle does, observes the focal length along the plane and leaves the other free to stretch the scene across it.
Each refined parameter has a Gaussian prior on the given calibration, a tenth of the focal length on a focal length, a twentieth of it on the principal point, 0.1 on the leading distortion terms and 0.01 on the higher order ones, wide enough for the observations to decide what they observe and holding what they hardly do.
A monocular map does not observe the focal length until the camera has turned: from a translation alone a shallow scene stretched along the view, seen with the focal length stretched as much, gives the same images.
The refinement therefore starts at the first adjustment of the whole map with at least 10 keyframes, one of whose optical axes is at least 10 degrees from their mean.

Run from focal lengths 5% and 10% too long (`zeroslam process --focal-scale 1.05`), `calibration=focal_centre` took the absolute trajectory error of EuRoC MH_01 from 6.5 and 15.4 cm back to 1.7 cm, what the given calibration gives, the focal length coming to 463 px from 482 and 505 px as from the given 459 px; TUM fr1_desk went from 2.1 and 2.4 cm to 2.0 and 1.9 cm, and fr1_xyz from 1.4 and 1.9 cm to 1.2 cm, against 2.3 and 1.1 cm with the published calibration.
A monocular estimate of a focal length is good to a few percent, not to a checkerboard's accuracy: from every start the refinement reached the same value within a pixel or two on each sequence, 463 px on MH_01 for the 459 px of its calibration, and 536 to 541 px on fr1_desk and 526 to 529 px on fr1_xyz for the 517 px published for that camera.
Started from the published calibration, it left fr1_desk at 1.9 cm, fr1_xyz at 1.2 cm and MH_01 at 1.7 cm.
Any model's parameters are refined the same way: a synthetic adjustment recovers a Kannala-Brandt and a double sphere camera, started 6% long and with no distortion or another shape, to half a pixel over the whole image, but on the rendered circle of the wide angle test, which turns about the vertical only, `intrinsics` brought a double sphere camera nearer the truth (4.8 from 5.9 pixels rms over its field) and not to it, its focal length trading against its shape.

### Camera rigs ###

Several synchronised cameras run as one rig. `zeroslam_set_sensor_extrinsics` gives each sensor its extrinsic, the sensor-to-body pose `[x y z qx qy qz qw]` (the scene's `ego -> sensor/[name]`), and whether it is refined; a sensor given none sits at the body's origin, held. Each frame is one image of every camera with the same timestamp, in one `zeroslam_set_sensor_data` batch or over several calls; a camera given twice in a frame, or a newer image before a frame is whole, is refused.
The map's frames are posed in the primary camera's frame, whose extrinsic the gauge holds, so the reported poses stay the primary camera's and a rig of one camera is the monocular system. The other cameras' images are views of those frames: each camera tracks its own features, its tracks link to the same landmarks by projection through its extrinsic, the rig's pose comes from every camera's tracks (a generalised perspective solve, then a joint refinement), and each camera's observations enter the adjustments through its extrinsic (the `extrinsic` vertex, refined or held).
Cameras that share their view across a baseline (a stereo pair) initialise from a single frame, triangulate new landmarks across the baseline at every keyframe, and give the map a metric scale: no scale gauge is held and loops close as rigid transforms. A refined extrinsic keeps the distance from the primary camera it was declared at, the map's scale. Cameras that share no view initialise from two frames as one camera does and keep a monocular map's scale, which their baseline measures only through the rig's turns; on a planar path the turn of their extrinsic about the path's axis is not observable either.
With `calibration` on, each camera of the rig is calibrated as the primary camera is, with an intrinsics vertex of its own that its observations take after its extrinsic.
Relocalisation, loop detection and lines run on the primary camera, and `adjustment=relative` adjusts the primary camera's observations alone (the final global adjustment takes every camera's). The square root solver takes the extrinsic vertices as it does the poses.

### Depth cameras ###

A single camera's rig may also hold one `zeroslam_sensor_depth` entry, a depth camera registered to the camera (`zeroslam_sensor_parameters_depth_struct`): its images are the camera's size and pixel grid, as the aligned depth of a structured light or stereo camera comes (TUM RGB-D, the OAK-D), so pixel `(u, v)` holds the depth of what the camera's pixel `(u, v)` sees.
It names its `camera_id`, the `encoding` (16 bit unsigned values of `scale` metres each, or 32 bit float metres; zero is no measurement), the `measure` (`axial`, the z of the point in the camera frame, or `radial`, the distance along the pixel's ray), the range of depths taken, the noise model, the standard deviation `noise_constant + noise_linear d + noise_quadratic d^2` metres of a depth `d` (the Kinect's random error is about `1.425e-3 d^2`), and the `pairing_tolerance` in nanoseconds.
A depth is turned into a point by scaling the ray the camera model unprojects the pixel to, so it composes with every camera model.
Depth frames have timestamps that increase among their own, and a camera frame takes the one nearest it in time within the tolerance from those given before it or in the same call; depth that comes after its camera frame is not used.
`zeroslam::registered_depth` (C++) and `zeroslam.registered_depth` (python) fill the parameters with the Kinect's model.

With a depth camera the map has metric scale (the property a stereo rig gives it too): it starts from the first frame with depth at 100 or more of its tracks, with no second view, and its landmarks take their depths from the keyframes' depth images (and are triangulated where there is no depth, as beyond the range).
Each observation's depth is a factor in the adjustments beside its reprojection, the difference of the measured and predicted inverse depths weighted by the noise model, with a robust loss of its own, so that a depth off at an occluding edge loses its weight without the bearing; a depth beside a jump of more than 5% in the depth image is not taken, and one that disagrees with the adjusted map is dropped.
The frames' poses are refined with their depths too, no baseline holds the scale, and a loop is a rigid motion rather than a similarity.
With `adjustment=relative` the relative adjustment takes each observation's depth as a residual beside its reprojection the same way, holds no length prior for the scale, and joins a loop by a rigid link.
A scene's depth camera is a `mono16` (millimetres) or `32FC1` (metres) image topic `/sensor/depth_[01-99]` registered to `image_01`, which the importers store at the times of its frames.
A depth camera in a rig of several cameras, and depth registered to a camera through an extrinsic, are not taken yet.

### Inertial sensors ###

An imu runs with the cameras as one more sensor of the rig: a `zeroslam_sensor_imu` entry of `zeroslam_set_sensor_rig` (at most one, beside at least one camera) whose parameters are a `zeroslam_sensor_parameters_imu_struct`, the continuous time noise densities its calibration publishes (gyroscope and accelerometer white noise in rad/s/√Hz and m/s²/√Hz, their bias random walks in rad/s²/√Hz and m/s³/√Hz, as Kalibr and the EuRoC datasets give them), its rate, the magnitude of gravity (`0` for 9.81) and the offset of its clock from the cameras' in seconds.
Its entry of `zeroslam_set_sensor_extrinsics` poses its own frame on the body like a camera's (the scene's `ego -> sensor/imu_01`), refined or held, and its samples are `zeroslam_set_sensor_data` entries of one `zeroslam_sensor_measurement_imu_struct` each, the angular velocity and specific force in its own frame as measured, in time order with the frames, in the same batches or apart; a frame takes the samples up to its own timestamp, so the samples up to a frame go in before it (a frame a few sample periods past the last sample holds that sample).
The imu's samples are checked whole with a batch, and the cameras' frames between them go as they would without it.

Each keyframe preintegrates the samples since the keyframe before it (on the manifold, after Forster, Carlone, Dellaert and Scaramuzza, TRO 2017), and a culled keyframe's measurements are merged into the next one's.
Once ten keyframes over two seconds or more have been taken, the system estimates the direction of gravity, the scale where the map has none (a monocular map's), the imu's velocities and its biases from them (ORB-SLAM3's inertial only optimisation, started from a closed form solution), and initialises when their uncertainty shows the motion held them: a monocular map is then scaled into metres and keeps the world of its first camera, with gravity known in it (`metric_scale` becomes true), and a stereo rig's map, metric already, only takes gravity, the velocities and the biases.
From then on every adjustment holds consecutive keyframes with preintegrated inertial edges and bias random walk edges, a held keyframe holding its velocity and bias too, so the scale, the roll and the pitch are held by the imu; the estimate is refined over the longer window five and fifteen seconds later with the accelerometer bias freed, and a loop closes as a four degree of freedom pose graph, every keyframe's roll and pitch held.
The imu predicts a frame's pose where its tracks cannot give it, and carries up to 40 frames the tracks lose before tracking is lost (10 without it); a new submap started after a loss starts a new chain of keyframes, initialised on its own with the scale held.
The imu's extrinsic is refined with the keyframes' poses when its entry asks for it, and `zeroslam_get_sensor_extrinsics` reports it after the cameras'. `adjustment=relative` takes no imu.

With `imu_calibration=gyroscope` the adjustments also refine the gyroscope's intrinsic calibration, after Van Meter and Paul, "SLAM Dunk: A Recipe for Long-Horizon Egocentric SLAM in the Wild" (2026): one matrix `M` for the whole run, the angular velocity being `(I + M) * measured - bias`, so that its diagonal holds the scale error of each axis and the rest the misalignment of the axes, among themselves and from the accelerometer's.
Each preintegrated measurement carries its Jacobians by the nine entries of `M` beside those by the gyroscope bias, every inertial edge takes the one vertex of it, and a measurement is integrated again once `M` has moved 0.005 from where it was integrated.
The matrix starts at zero with a Gaussian prior of 0.002 on each entry, which holds what the motion does not show, and it is free only where the motion can show it: in the adjustments of the whole map (every `global_adjustment` keyframes, at each loop closure and in `zeroslam_finalise`), after the two refinements of the inertial initialisation, and once the turns between consecutive keyframes, less the mean rate's share of them, come to 0.05 rad² in sum about the axis the imu has turned about least; the keyframes those adjustments hold then free their velocities and biases, which the initialisation made with the gyroscope taken as it was.
The window and pose adjustments hold the matrix at the last estimate, the imu's predictions and the keyframes preintegrated from then on are integrated at it, and `zeroslam_get_imu_calibration` reports it with the number of adjustments that have refined it.
A simulated gyroscope scaled by about a percent and misaligned by about half a degree (entries of 0.004 to 0.012), run for thirty seconds in a rendered room, had its matrix found within 0.0017, short of the truth by what the prior holds back, and its frames ended 2.1 mrad and 7 mm from the truth for 4.9 mrad and 18 mm with the gyroscope taken as it is.
The EuRoC MAV's ADIS16448 is calibrated at the factory, and there the matrix came out with entries of 0.003 or less and the gain is small where there is one: the absolute trajectory error without a fitted scale went from 1.39 to 1.12 cm on MH_01 and from 1.21 to 1.24 cm on V1_01 with one camera, and from 1.17 to 1.16 cm and from 1.58 to 1.54 cm with the stereo pair, one run each.
The four runs agree on two entries of the matrix, the third row's first two (-0.0021 to -0.0025 and -0.0023 to -0.0030), and the two runs of each sequence agree on the others within 0.001, but the sequences differ on the rest by up to 0.0025, the first row's most (-0.003 and 0.001 on MH_01, -0.001 and 0.002 on V1_01): what a sequence makes of those is not the sensor's alone.

### Wheel odometry ###

A wheel odometer is a rig entry of type `zeroslam_sensor_wheel_odometry` beside the cameras, with a `zeroslam_sensor_parameters_wheel_odometry_struct`:
- the model: an Ackermann (car-like) vehicle measured by a wheel on its rear axle (or the mean of the rear wheels, or the drive shaft) or by a front wheel, whose calibration is the wheel radius, the wheelbase, the steering gain and the steering offset (steering angle = gain * reading + offset), or a differential drive, whose calibration is the left and right wheel radii and the track width, in metres and radians;
- `free_mask`, bit `i` refining calibration parameter `i`, with a prior of `parameter_deviations[i]` about the declared value (zero for none);
- the noise: the wheel slip, the lateral slip, the heading, the height and the tilt as standard deviations after one metre (they grow with the square root of the distance), the steering reading's per reading, and the least translation and rotation of any interval;
- `extrinsic_deviations`, a prior on the extrinsic's rotation and translation about the odometry frame's axes (zero for none).

Its extrinsic, the odometry frame's pose on the body (x forward, y left, z up, the origin at the rear axle's centre or the middle of the drive axle), is given with `zeroslam_set_sensor_extrinsics` like a camera's and refined when asked. Its samples are `zeroslam_set_sensor_data` entries of three doubles, the cumulative rotation in radians of the measured (or left) wheel and of the right wheel and the steering reading, in time order among the cameras' images; encoder ticks are turned into radians by the caller.
The samples between each two frames of an adjustment are integrated into the planar motion of the odometry frame, its covariance and its Jacobian by the calibration, and each adjustment that moves landmarks takes an odometry edge between each two consecutive frames it holds, sharing one calibration and the extrinsic, so the calibration and the extrinsic are refined with the map; a motion is integrated again when the calibration moves past its first order correction, and merged from the two on either side of a culled keyframe. The initialisation is scaled to the distance the odometry says the camera moved, which makes the map metric (no scale gauge is held and loops close as rigid transforms), and a frame the tracks do not pose is predicted by the odometry, which also carries dead reckoning on through frames nothing else poses. `zeroslam_get_odometry_calibration` reports the calibration and the extrinsic as the system holds them.
With a monocular camera a turn shows only the distance rolled over the wheelbase (or the track width), so hold one measured length; for small steering angles the gain shows only with the radius, so hold it or give it a tight prior. Planar motion cannot show the height of the odometry frame above the plane of motion: refining the extrinsic needs a tight deviation on its last component.
On a rendered drive of 30 m, a radius declared 6% small and no steering offset for 0.03 rad come out within 0.3% and 0.0005 rad, and the map within 0.4% of metric scale.

## Scene format ##

A scene is one mcap file, `datasets/[dataset]/[scene].mcap`, holding:
- Raw image messages (`sensor_msgs/msg/Image`: `mono8` or `rgb8` for the cameras the system runs on, `mono16` or `32FC1` for depth) in uncompressed or lz4 compressed chunks (zstd chunks are rejected).
- Camera intrinsics as one camera info message per frame on `/sensor/[name]/camera_info`, with no rectification (`R` the identity, `P` = `[K | 0]`) and the principal point in the pixel-centre frame (see "Coordinate conventions" below): `K` holds `fx fy cx cy`, `distortion_model` names one of the camera models and `d` holds its own parameters (see "Camera models"): `plumb_bob` (`k1 k2 p1 p2 [k3]`), `rational_polynomial` (`k1 k2 p1 p2 k3 k4 k5 k6`, or with the thin prism `s1 s2 s3 s4` and the tilt `tau_x tau_y` after them), `equidistant` (`k1 k2 k3 k4`), `fov` (`w`), `unified` (`xi`), `extended_unified` (`alpha beta`), `double_sphere` (`xi alpha`) or `fisheye_624` (`k0 .. k5 p0 p1 s0 .. s3`). A distortion that folds back on itself, where the radial factor stops growing or reaches a pole of the rational model, limits the model to the radius inside that point; rays and pixels beyond it are treated as not visible.
- IMU messages.
- Wheel odometry as `sensor_msgs/msg/JointState` messages on `/sensor/wheel_[01-99]`, the three positions the cumulative rotations of the measured (or left) wheel and the right wheel in radians and the steering reading, with one `std_msgs/msg/String` on `/sensor/wheel_[01-99]/parameters` of `key=value` words: `model` (`ackermann_rear_axle`, `ackermann_front_wheel` or `differential_drive`), `parameters`, `free_mask`, `parameter_deviations`, `noise` and `extrinsic_deviations`, numbers separated by commas, the noise the library's by default (see "Wheel odometry").
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
│   ├── imu_01.txt    # Per sample:   `[timestamp] [x] [y] [z] [qx] [qy] [qz] [qw] [wx] [wy] [wz] [ax] [ay] [az]`
│   └── wheel_01.txt  # `# [parameters]`, then per sample: `[timestamp] [x] [y] [z] [qx] [qy] [qz] [qw] [rotation_0] [rotation_1] [steering]`
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

The options are `--frames [count]` (the first frames only), `--skip [count]` (ignore the first frames), `--verbose [0-5]` (the log level, default `1`), `--truth [trajectory.txt]` (end with the absolute, relative and per metre errors against a TUM trajectory), `--live` (skip the final global adjustment, so the trajectory is the one tracked live), `--focal-scale [factor]` (scale the scene's focal lengths, to run from a wrong calibration), `--config key=value` (repeatable), `--cameras image_01,image_02` (run several cameras of the scene as a rig, the first the primary camera, each at its extrinsic on `/tf`, see "Camera rigs") and `--perturb-extrinsics degrees,metres` (start every camera's extrinsic but the primary's turned and shifted off the calibration, for `--config extrinsics=refine`).
`--depth` gives the library the scene's depth camera (`/sensor/depth_01`) with each frame, a metric RGB-D map whose errors against `--truth` are also given without a fitted scale; `--depth-noise c0,c1,c2` sets its noise model (default `0.002,0,0.005`, fitted on TUM RGB-D fr1_desk, whose depth carries the registration and timing errors of the Kinect besides its random error) and `--depth-range min,max` the depths taken (default `0.1,10` metres).
The run ends by printing the camera's calibration as the system holds it (`Calibration: [model] fx ... fy ... cx ... cy ...` and the rest of its parameters), the scene's unless `--config calibration=...` refined it.

`--imu imu_01` runs the scene's imu with the cameras (the primary camera alone when no `--cameras` are named), at its extrinsic on `/tf` and with the noise densities of the EuRoC MAV's ADIS16448 unless `--imu-noise gyroscope,accelerometer,gyroscope_walk,accelerometer_walk` gives others (a scene carries none), and `--imu-refine` refines its extrinsic (see "Inertial sensors"). `--imu-gyroscope` refines the scale and misalignment of its gyroscope (`--config imu_calibration=gyroscope`), and a run with an imu ends by printing the gyroscope's matrix as the system holds it (`Gyroscope of sensor ...`).

`--wheel wheel_01` runs the scene's wheel odometer with the camera, or with the rig of `--cameras` (see "Wheel odometry"), and writes its refined calibration and extrinsic to `odometry.txt`; the regression tool forwards it.

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

# Align without a scale, for a trajectory with metric scale (rgb-d, stereo).
./runtime/Release/zeroslam evaluate trajectory_gt.txt trajectory_eval_1.txt --centroid --rigid

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
Private repositories are reached with the `HF_TOKEN` environment variable or with `--token`, and `--repo` selects another hub repository (huggingface only).
Prefer the variable: an option is on the tool's command line, which other users of the machine can read from the process list. Either way the token reaches `curl` in a private temporary file and not on its command line.
`validate` checks a scene by name, every scene of a dataset, or a scene mcap by path: the raw frames, a calibration of one of the camera models per frame, and the `root -> ego -> sensor` frame tree on `/tf`.

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
| `import-lamaria` | `[asl-dir] [calibration.json] [output.mcap]` | LaMAria's undistorted ASL (pinhole) release with its pinhole calibration, or an ASL folder of the cameras' own fisheye images with the aria calibration (`RAD_TAN_THIN_PRISM_FISHEYE`, written as `fisheye_624` and turned upright with the images; not yet tried on the real data), `--ground-truth` adds the pseudo ground truth where a sequence has one. |

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
`--config key=value` starts it with a front end setting it offers in its controls: `tracker`, `lines`, `culling`, `association`, `detector`, `descriptor`, `flow` and `local_map`; the controls otherwise start at the library's defaults (`--help` lists them). `--cameras image_01,image_02` runs several cameras as a rig, the first the one shown, and `--wheel wheel_01` the scene's wheel odometer with them.
`--depth` gives the system the scene's depth camera with each frame of a single camera, as the `process` tool does.
`--imu imu_01` runs the scene's imu with the cameras, as the `process` tool does.
It needs OpenGL, with X11 on linux, Cocoa on macOS and Win32 on windows.
```
cd build
cmake .. -DBUILD_ZEROSLAM_ZEROSLAM=ON -DBUILD_ZEROSLAM_GUI=ON
cmake --build . --parallel 4
./runtime/Release/zeroslam gui ../datasets/tum-rgbd/fr1_xyz.mcap --play
```

## Running on Android ##

The `android` tool is an app that runs the SLAM system on a phone's main camera and draws what it makes of it: the frame it processed last with the landmarks in front of it projected through its pose, coloured from near (warm) to far (cool), a view of the map and the keyframes' path that a drag turns and a pinch zooms, and the tracking state, the time a frame takes and the size of the map.
It is a native activity with no java, running the camera through the ndk's camera2 api and drawing with OpenGL ES 2, and it is packaged with the sdk's build tools rather than gradle, so the tree holds no files but its C++ and its CMakeLists.txt (the manifest is written at configure time).
It needs android 7 (api level 24) or later and a camera the ndk's camera2 api reaches (not the `LEGACY` ones).

The camera's geometry is held still for the system: distortion correction, optical and video stabilisation are off, the crop is the whole sensor, the focus is fixed (one dioptre, about half a metre to a few metres sharp) and the frame rate range is the one that reaches 30 frames a second with the highest floor, which keeps the exposures short.
The intrinsics are the device's calibration (`LENS_INTRINSIC_CALIBRATION` and `LENS_DISTORTION`) scaled and cropped from the sensor to the stream, from the camera's characteristics or else its first capture result, or estimated from the lens's focal length and the sensor's size when it reports neither; the panel shows which, with the intrinsics.
The system takes the newest frame each time it is ready, so it falls behind the camera by a frame at most however long a frame takes, and the frames it skips are counted.
The library runs on the phone's big cores (those within 70% of the fastest core's top frequency), as its parallel loops split their work evenly.

The app starts the library from `flow=intensity` and `refiner=none` on a 320x240 stream, as a phone falls far behind the camera with the library's defaults at 640x480.
On TUM RGB-D fr1_desk, on one core of an Intel Core Ultra 9 285K, the defaults at 640x480 took 498 ms a frame (2.3 cm of absolute trajectory error), the intensity flow 126 ms (4.0 cm), the intensity flow at 320x240 62 ms (2.9 cm), and the app's settings 38 ms (3.8 cm, 31 ms on four threads).

`RESET` starts a new map, `SAVE` writes the trajectory of every posed frame (`trajectory.txt`, TUM format) and the landmarks (`map.ply`), and `RECORD` writes every frame the camera gives, with its calibration, as a scene in the directory form until it is pressed again.
Each goes into a directory named by the time, in the app's files directory, `/sdcard/Android/data/com.gpdaniels.zeroslam/files/`.
A recording collapses into a scene file to process, evaluate and view on a desktop like any other (it has no ground truth).
That directory may also hold a `configuration.txt`, read when the app starts, of `key=value` lines: the library's settings (see "Configuration") on top of the app's, and the app's own `app_camera_width` (the stream closest to it in width with the sensor's aspect ratio, `320`), `app_camera_focus` (dioptres, or `auto` for continuous autofocus) and `app_threads` (the library's threads, `0` for the big cores).
The library logs to the system log with the tag `zeroslam`.

Install the sdk's command line tools (https://developer.android.com/studio#command-tools, unpacked as `cmdline-tools/latest` in the sdk's directory) and a jdk, then the ndk, build tools, platform and adb.
Configure with the ndk's toolchain file, which finds the sdk from `ANDROID_HOME`, `ANDROID_SDK_ROOT`, `ZEROSLAM_ANDROID_SDK` or the directory the ndk is in.
The package `runtime/Release/zeroslam-android.apk` is signed with `~/.android/debug.keystore` when there is one (`ZEROSLAM_ANDROID_KEYSTORE` names another) and otherwise with a key made in the build directory, it targets api level 34 (`ZEROSLAM_ANDROID_TARGET_SDK`), and one build makes one abi (`ANDROID_ABI`, `arm64-v8a` for a phone, `x86_64` for an emulator).
```
export ANDROID_HOME=~/android-sdk
$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager "platform-tools" "platforms;android-36" "build-tools;36.1.0" "ndk;30.0.16248370"
cmake -S . -B build/android -G Ninja -DCMAKE_TOOLCHAIN_FILE=$ANDROID_HOME/ndk/30.0.16248370/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DZEROSLAM_CHECKS=OFF -DBUILD_ZEROSLAM_ANDROID=ON
cmake --build build/android --parallel 4

# Install and start the app on the phone connected over adb.
cmake --build build/android --target zeroslam-android-run

# Fetch the saved runs and recordings into a new directory, and collapse a recording into a scene with the desktop's tools.
$ANDROID_HOME/platform-tools/adb pull /sdcard/Android/data/com.gpdaniels.zeroslam/files phone
cmake -S . -B build/desktop -DBUILD_ZEROSLAM_ZEROSLAM=ON -DBUILD_ZEROSLAM_DATASET=ON -DBUILD_ZEROSLAM_PROCESS=ON
cmake --build build/desktop --parallel 4
./build/desktop/runtime/Release/zeroslam dataset collapse phone/[time]/scene phone.mcap
./build/desktop/runtime/Release/zeroslam process phone.mcap
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
`--name` and `--commit` override the recorded scene name and commit (default the git `HEAD`), `--config key=value` and `--cameras` are forwarded to the `process` tool, `--first` or `--centroid` picks the alignment anchor, and `--tools-dir` points at the other tools.
`--depth` processes with the scene's depth camera and evaluates without a scale, recorded as `[name]+depth`.
`--imu imu_01` is forwarded to the `process` tool too.

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
