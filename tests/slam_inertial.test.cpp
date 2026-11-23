/*
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
*/

#include "slam_inertial.hpp"

#include "image/image.hpp"
#include "mapping/rig.hpp"
#include "math/lie.hpp"
#include "sensor/camera.hpp"
#include "sensor/imu/measurement.hpp"
#include "simulation/imu.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

namespace {
    constexpr static const int width = 480;
    constexpr static const int height = 360;
    constexpr static const double focal = 360.0;

    // The inside of a box room, 16 m by 8 m by 16 m with gravity along -z (z up), each wall textured in hashed 25 cm cells, seen by
    // a pinhole camera posed world to camera.
    void render_room(const math::se3<double>& pose, image::image& frame) {
        const math::matrix<double, 3, 3> camera_to_world = math::transpose(pose.rotation().get_matrix());
        const math::matrix<double, 3, 1> centre = -(camera_to_world * pose.translation());
        const double half[3] = { 8.0, 8.0, 4.0 };
        unsigned char* const pixels = frame.get_data();
        for (int row = 0; row < height; ++row) {
            for (int column = 0; column < width; ++column) {
                const math::matrix<double, 3, 1> ray_camera({ (static_cast<double>(column) + 0.5 - (0.5 * width)) / focal, (static_cast<double>(row) + 0.5 - (0.5 * height)) / focal, 1.0 });
                const math::matrix<double, 3, 1> ray = camera_to_world * ray_camera;
                double nearest = 1.0e30;
                int axis = 0;
                for (int a = 0; a < 3; ++a) {
                    if (std::abs(ray[static_cast<size_t>(a)]) < 1.0e-12) {
                        continue;
                    }
                    const double bound = (ray[static_cast<size_t>(a)] > 0.0) ? half[a] : -half[a];
                    const double t = (bound - centre[static_cast<size_t>(a)]) / ray[static_cast<size_t>(a)];
                    if ((t > 0.0) && (t < nearest)) {
                        nearest = t;
                        axis = a;
                    }
                }
                const math::matrix<double, 3, 1> hit = centre + (ray * nearest);
                unsigned long long hash = static_cast<unsigned long long>(axis + 1) * 83492791ull;
                for (int a = 0; a < 3; ++a) {
                    if (a == axis) {
                        hash ^= (hit[static_cast<size_t>(a)] > 0.0) ? 0x9e3779b9ull : 0x7f4a7c15ull;
                        continue;
                    }
                    hash = (hash ^ static_cast<unsigned long long>(static_cast<long long>(std::floor(4.0 * hit[static_cast<size_t>(a)])))) * 0x100000001b3ull;
                }
                hash ^= hash >> 13;
                hash *= 0x5bd1e995ull;
                hash ^= hash >> 15;
                pixels[static_cast<size_t>((row * width) + column)] = static_cast<unsigned char>(40ull + (hash % 176ull));
            }
        }
    }

    math::matrix<double, 3, 1> centre_of(const math::matrix<double, 3, 3>& rotation, const math::matrix<double, 3, 1>& translation) {
        return math::matrix<double, 3, 1>(-(math::transpose(rotation) * translation));
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // A camera with an imu turned and offset on it, along a smooth simulated motion inside the room, the imu with the EuRoC
    // imu's noise and biases. A monocular map initialises its imu once it holds a few seconds of keyframes, which makes it
    // metric: the frames' positions come out in metres, with no scale to fit.
    simulation::imu_trajectory::options motion;
    motion.position_amplitude = 0.6;
    motion.rotation_amplitude = 0.25;
    motion.seed = 3;
    const simulation::imu_trajectory trajectory = simulation::imu_trajectory::generate(motion);
    const math::se3<double> camera_from_imu(math::so3<double>::exp({ { 0.05, -0.1, 0.02 } }), { { 0.03, -0.02, 0.05 } });
    sensor::imu::parameters noise;
    simulation::imu_sensor::options sensor_options;
    sensor_options.initial_bias.gyroscope = { { 0.003, -0.002, 0.004 } };
    sensor_options.initial_bias.accelerometer = { { 0.03, -0.02, 0.04 } };
    sensor_options.end = 12000000000ll;
    const simulation::imu_sensor::recording recording = simulation::imu_sensor::generate(trajectory, noise, sensor_options);
    // The camera looks along the imu's x axis turned to its optical frame, so that it looks across the room.
    const math::so3<double> optical = math::so3<double>(math::matrix<double, 3, 3>({ { { 0.0, -1.0, 0.0 }, { 0.0, 0.0, -1.0 }, { 1.0, 0.0, 0.0 } } }));
    const math::se3<double> imu_from_camera = math::se3<double>(optical.inverse(), math::matrix<double, 3, 1>::zero()) * camera_from_imu.inverse();
    const math::se3<double> body_from_imu = imu_from_camera.inverse();

    mapping::rig rig;
    rig.add(1, mapping::rig::sensor_kind::camera, math::se3<double>::identity(), false);
    slam system;
    system.set_rig(rig);
    system.set_imu(2, noise, body_from_imu, false);
    REQUIRE(system.reconstruction.inertial_state.declared());
    REQUIRE(!system.inertial_initialised());
    const double parameters[4] = { focal, focal, 0.5 * width, 0.5 * height };
    const std::vector<sensor::model> cameras(1, sensor::model(sensor::camera::pinhole<double>(&parameters[0], 4)));

    constexpr static const int frames = 200;
    constexpr static const long long period = 50000000ll;
    std::vector<math::se3<double>> truth;
    size_t next_sample = 0;
    for (int i = 0; i < frames; ++i) {
        const long long timestamp = 1000000000ll + (period * i);
        while ((next_sample < recording.samples.size()) && (recording.samples[next_sample].timestamp <= timestamp)) {
            REQUIRE(system.add_imu_sample(recording.samples[next_sample]));
            ++next_sample;
        }
        const double time = 1.0e-9 * static_cast<double>(timestamp);
        // T_cw = T_ci T_iw.
        truth.push_back(body_from_imu * trajectory.world_to_imu(time));
        std::vector<image::image> images;
        images.emplace_back(static_cast<size_t>(height), static_cast<size_t>(width));
        render_room(truth.back(), images[0]);
        system.set_frame_timestamp(timestamp);
        system.process_rig_frame(cameras, images);
    }
    // A sample out of order is refused.
    REQUIRE(!system.add_imu_sample(recording.samples.front()));
    system.finalise();

    REQUIRE(system.inertial_initialised());
    REQUIRE(system.reconstruction.metric_scale);
    // The gravity the system found, in the map's world (the first posed camera's frame), is the truth's turned into it.
    int first_id = -1;
    for (int i = 0; (i < frames) && (first_id < 0); ++i) {
        first_id = (system.reconstruction.frames.count(i) != 0) ? i : -1;
    }
    REQUIRE(first_id >= 0);
    const math::matrix<double, 3, 1> true_gravity = truth[static_cast<size_t>(first_id)].rotation() * trajectory.gravity();
    const math::matrix<double, 3, 1> gravity = system.reconstruction.inertial_state.gravity();
    const double cosine = ((gravity[0] * true_gravity[0]) + (gravity[1] * true_gravity[1]) + (gravity[2] * true_gravity[2])) / std::sqrt(gravity.get_length_squared() * true_gravity.get_length_squared());
    const double gravity_error = std::acos(std::min(1.0, cosine)) * 180.0 / 3.14159265358979;

    // The positions without a scale fitted: each frame's distance from the first posed one against the truth's.
    const mapping::frame& first_frame = system.reconstruction.frames.at(first_id);
    const math::matrix<double, 3, 1> first_centre = centre_of(first_frame.rotation, first_frame.translation);
    const math::matrix<double, 3, 1> first_truth = centre_of(truth[static_cast<size_t>(first_id)].rotation().get_matrix(), truth[static_cast<size_t>(first_id)].translation());
    std::vector<double> ratios;
    size_t posed = 0;
    for (const auto& [frame_id, frame] : system.reconstruction.frames) {
        ++posed;
        const math::matrix<double, 3, 1> truth_centre = centre_of(truth[static_cast<size_t>(frame_id)].rotation().get_matrix(), truth[static_cast<size_t>(frame_id)].translation());
        const double travelled = std::sqrt((truth_centre - first_truth).get_length_squared());
        if (travelled >= 0.3) {
            ratios.push_back(std::sqrt((centre_of(frame.rotation, frame.translation) - first_centre).get_length_squared()) / travelled);
        }
    }
    REQUIRE(ratios.size() > 20);
    std::sort(ratios.begin(), ratios.end());
    const double median = ratios[ratios.size() / 2];
    std::printf("Monocular inertial: %zu of %d frames posed, %zu keyframes, metric distances %.4f of the truth's (median, from %.4f to %.4f), gravity %.2f degrees off.\n", posed, frames, system.keyframe_ids().size(), median, ratios[ratios.size() / 10], ratios[(9 * ratios.size()) / 10], gravity_error);
    std::fflush(stdout);
    REQUIRE(posed > static_cast<size_t>(frames - 20));
    REQUIRE(std::abs(median - 1.0) < 0.05);
    REQUIRE(gravity_error < 2.0);
    return EXIT_SUCCESS;
}
