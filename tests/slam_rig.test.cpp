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

#include "slam_rig.hpp"

#include "image/image.hpp"
#include "mapping/calibration.hpp"
#include "mapping/rig.hpp"
#include "math/lie.hpp"
#include "sensor/camera.hpp"

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
    constexpr static const int width = 640;
    constexpr static const int height = 480;
    constexpr static const double focal = 500.0;

    // The inside of a box room, 16 m by 8 m by 16 m, each wall textured in hashed 25 cm cells, seen by a pinhole camera posed
    // world to camera.
    void render_room(const math::se3<double>& pose, image::image& frame) {
        const math::matrix<double, 3, 3> camera_to_world = math::transpose(pose.rotation().get_matrix());
        const math::matrix<double, 3, 1> centre = -(camera_to_world * pose.translation());
        const double half[3] = { 8.0, 4.0, 8.0 };
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

    // A rig of two cameras through the system, a stereo pair 30 cm apart or a pair facing apart that shares no view, the
    // second camera's images views of the reference camera's frames. The stereo pair initialises from its first frame and maps
    // in metres, and its extrinsic, declared 0.6 degrees and 5 mm across the baseline off, is refined back when asked; the pair
    // facing apart initialises from two frames as one camera does, and its second camera maps landmarks of its own.
    void test_rig(const bool stereo, const bool refine) {
        const math::se3<double> body_from_second = stereo ? math::se3<double>(math::so3<double>::identity(), { { 0.3, 0.0, 0.0 } }) : math::se3<double>(math::so3<double>::exp({ { 0.0, 3.14159265358979, 0.0 } }), { { 0.0, 0.0, -0.2 } });
        const double length = std::sqrt(body_from_second.translation().get_length_squared());
        const math::matrix<double, 3, 1> shifted = body_from_second.translation() + math::matrix<double, 3, 1>({ 0.0, 0.005, 0.0 });
        const math::se3<double> declared = refine ? math::se3<double>(math::so3<double>::exp({ { 0.006, -0.008, 0.004 } }) * body_from_second.rotation(), shifted * (length / std::sqrt(shifted.get_length_squared()))) : body_from_second;
        mapping::rig rig;
        rig.add(1, mapping::rig::sensor_kind::camera, math::se3<double>::identity(), false);
        rig.add(2, mapping::rig::sensor_kind::camera, declared, refine);
        slam system;
        system.set_rig(rig);
        REQUIRE(system.reconstruction.sensor_rig.metric());
        const double parameters[4] = { focal, focal, 0.5 * width, 0.5 * height };
        const std::vector<sensor::model> cameras(2, sensor::model(sensor::camera::pinhole<double>(&parameters[0], 4)));
        constexpr static const int frames = 60;
        std::vector<math::se3<double>> truth;
        for (int i = 0; i < frames; ++i) {
            const double t = static_cast<double>(i);
            truth.push_back(math::se3<double>(math::so3<double>::exp({ { 0.002 * std::sin(0.2 * t), 0.004 * t, 0.0 } }), { { -0.04 * t, 0.05 * std::sin(0.15 * t), 0.0 } }));
            std::vector<image::image> images;
            images.emplace_back(static_cast<size_t>(height), static_cast<size_t>(width));
            images.emplace_back(static_cast<size_t>(height), static_cast<size_t>(width));
            render_room(truth.back(), images[0]);
            render_room(body_from_second.inverse() * truth.back(), images[1]);
            system.process_rig_frame(cameras, images);
        }
        system.finalise();
        REQUIRE(system.state() == slam::tracking_state::tracking);
        // Only cameras that share their view give the map its scale.
        REQUIRE(system.reconstruction.metric_scale == stereo);
        if (stereo) {
            REQUIRE(system.reconstruction.frames.count(0) != 0);
        }
        // The second camera observes landmarks, and some it alone sees.
        size_t seen_by_second = 0;
        size_t second_only = 0;
        for (const auto& [landmark_id, landmark_observations] : system.reconstruction.observations) {
            static_cast<void>(landmark_id);
            bool first = false;
            bool second = false;
            for (const mapping::map::observation& obs : landmark_observations) {
                first = first || (obs.camera == 0);
                second = second || (obs.camera == 1);
            }
            seen_by_second += second ? 1 : 0;
            second_only += (second && !first) ? 1 : 0;
        }
        std::printf("Rig (%s%s): %zu frames posed, %zu keyframes, %zu landmarks, %zu seen by the second camera, %zu by it alone.\n", stereo ? "stereo" : "facing apart", refine ? ", refined" : "", system.reconstruction.frames.size(), system.keyframe_ids().size(), system.reconstruction.landmarks.size(), seen_by_second, second_only);
        std::fflush(stdout);
        REQUIRE(seen_by_second > (stereo ? 300u : 50u));
        REQUIRE(second_only > 50);
        REQUIRE(system.reconstruction.frames.size() > static_cast<size_t>(frames - 10));
        // The stereo map is in metres: the distance each frame has come from the first posed one is the truth's. The other is
        // in a scale of its own, the same throughout.
        int first_id = -1;
        for (int i = 0; (i < frames) && (first_id < 0); ++i) {
            first_id = (system.reconstruction.frames.count(i) != 0) ? i : -1;
        }
        REQUIRE(first_id >= 0);
        const auto centre_of = [](const math::matrix<double, 3, 3>& rotation, const math::matrix<double, 3, 1>& translation) {
            return math::matrix<double, 3, 1>(-(math::transpose(rotation) * translation));
        };
        const mapping::frame& first_frame = system.reconstruction.frames.at(first_id);
        const math::matrix<double, 3, 1> first_centre = centre_of(first_frame.rotation, first_frame.translation);
        const math::matrix<double, 3, 1> first_truth = centre_of(truth[static_cast<size_t>(first_id)].rotation().get_matrix(), truth[static_cast<size_t>(first_id)].translation());
        std::vector<double> ratios;
        for (const auto& [frame_id, frame] : system.reconstruction.frames) {
            const math::matrix<double, 3, 1> truth_centre = centre_of(truth[static_cast<size_t>(frame_id)].rotation().get_matrix(), truth[static_cast<size_t>(frame_id)].translation());
            const double travelled = std::sqrt((truth_centre - first_truth).get_length_squared());
            if (travelled >= 0.5) {
                ratios.push_back(std::sqrt((centre_of(frame.rotation, frame.translation) - first_centre).get_length_squared()) / travelled);
            }
        }
        REQUIRE(ratios.size() > 10);
        std::vector<double> sorted(ratios);
        std::sort(sorted.begin(), sorted.end());
        const double scale = stereo ? 1.0 : sorted[sorted.size() / 2];
        double worst = 0.0;
        for (const double ratio : ratios) {
            worst = std::max(worst, std::abs((ratio / scale) - 1.0));
        }
        std::printf("Rig (%s%s): scale off the truth's by %.4f at worst.\n", stereo ? "stereo" : "facing apart", refine ? ", refined" : "", worst);
        std::fflush(stdout);
        REQUIRE(worst < (stereo ? 0.02 : 0.1));
        if (refine) {
            const math::matrix<double, 6, 1> error = (system.reconstruction.sensor_rig.reference_from_sensor(system.reconstruction.sensor_rig.camera(1)) * body_from_second.inverse()).log();
            const double degrees = std::sqrt((error[0] * error[0]) + (error[1] * error[1]) + (error[2] * error[2])) * 180.0 / 3.14159265358979;
            std::printf("Rig (stereo, refined): extrinsic %.3f degrees and %.4f m off the truth, from 0.62 degrees and 5 mm.\n", degrees, std::sqrt((error[3] * error[3]) + (error[4] * error[4]) + (error[5] * error[5])));
            std::fflush(stdout);
            REQUIRE(degrees < 0.2);
        }
    }

    // Each camera of a stereo pair calibrated as the reference camera is (calibration=focal): both given focal lengths long,
    // 4% on the first and 2.5% on the second, each camera's intrinsics vertex brings its own back, through the reprojection
    // edges of its own observations. A baseline does not observe a focal length any better than one camera does, the depths
    // it triangulates scaling with the focal length it is given, so as for one camera the turns of the rig decide it, here
    // 51 degrees about one axis.
    void test_rig_calibration() {
        mapping::rig rig;
        rig.add(1, mapping::rig::sensor_kind::camera, math::se3<double>::identity(), false);
        const math::se3<double> body_from_second(math::so3<double>::identity(), { { 0.3, 0.0, 0.0 } });
        rig.add(2, mapping::rig::sensor_kind::camera, body_from_second, false);
        slam system;
        system.frontend.calibration = mapping::calibration::level::focal;
        system.set_rig(rig);
        const double first_parameters[4] = { 1.04 * focal, 1.04 * focal, 0.5 * width, 0.5 * height };
        const double second_parameters[4] = { 1.025 * focal, 1.025 * focal, 0.5 * width, 0.5 * height };
        const std::vector<sensor::model> cameras = { sensor::model(sensor::camera::pinhole<double>(&first_parameters[0], 4)), sensor::model(sensor::camera::pinhole<double>(&second_parameters[0], 4)) };
        constexpr static const int frames = 90;
        for (int i = 0; i < frames; ++i) {
            const double t = static_cast<double>(i);
            const math::se3<double> pose(math::so3<double>::exp({ { 0.003 * std::sin(0.2 * t), 0.01 * t, 0.0 } }), { { -0.04 * t, 0.05 * std::sin(0.15 * t), 0.0 } });
            std::vector<image::image> images;
            images.emplace_back(static_cast<size_t>(height), static_cast<size_t>(width));
            images.emplace_back(static_cast<size_t>(height), static_cast<size_t>(width));
            render_room(pose, images[0]);
            render_room(body_from_second.inverse() * pose, images[1]);
            system.process_rig_frame(cameras, images);
        }
        system.finalise();
        REQUIRE(system.state() == slam::tracking_state::tracking);
        // The newest keyframe holds both cameras as the adjustments left them.
        int newest = -1;
        for (const int keyframe_id : system.keyframe_ids()) {
            newest = std::max(newest, keyframe_id);
        }
        REQUIRE(newest >= 0);
        const mapping::frame& keyframe = system.reconstruction.frames.at(newest);
        REQUIRE(keyframe.views.size() == 1);
        double first[4];
        double second[4];
        REQUIRE(keyframe.camera.get_pinhole(&first[0], 4));
        REQUIRE(keyframe.views[0].camera.get_pinhole(&second[0], 4));
        std::printf("Rig calibration: focal lengths %.2f and %.2f from %.1f and %.1f, the truth %.1f.\n", first[0], second[0], first_parameters[0], second_parameters[0], focal);
        std::fflush(stdout);
        REQUIRE(std::abs((first[0] / focal) - 1.0) < 0.03);
        REQUIRE(std::abs((second[0] / focal) - 1.0) < 0.03);
        // Each camera's own vertex moved it, the second's from where the first's was not.
        REQUIRE(std::abs(first[0] - first_parameters[0]) > 0.01 * focal);
        REQUIRE(std::abs(second[0] - second_parameters[0]) > 0.01 * focal);
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    test_rig(true, false);
    test_rig(true, true);
    test_rig(false, false);
    test_rig_calibration();

    return EXIT_SUCCESS;
}
