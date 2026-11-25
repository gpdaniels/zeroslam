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

#include "slam_calibration_drift.hpp"

#include "image/image.hpp"
#include "mapping/calibration_drift.hpp"
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
    constexpr static const int frames = 120;
    constexpr static const int frames_per_chunk = 10;
    constexpr static const int chunk_count = frames / frames_per_chunk;
    // The run stands for twenty minutes of a recording, a frame of it every ten seconds, in chunks of 100 seconds.
    constexpr static const long long frame_nanoseconds = 10000000000ll;
    constexpr static const double chunk_seconds = 100.0;

    // The inside of a box room, 16 m by 8 m by 16 m, each wall textured in hashed 25 cm cells, seen by a pinhole camera of
    // the focal length given, posed world to camera.
    void render_room(const math::se3<double>& pose, const double focal_length, image::image& frame) {
        const math::matrix<double, 3, 3> camera_to_world = math::transpose(pose.rotation().get_matrix());
        const math::matrix<double, 3, 1> centre = -(camera_to_world * pose.translation());
        const double half[3] = { 8.0, 4.0, 8.0 };
        unsigned char* const pixels = frame.get_data();
        for (int row = 0; row < height; ++row) {
            for (int column = 0; column < width; ++column) {
                const math::matrix<double, 3, 1> ray_camera({ (static_cast<double>(column) + 0.5 - (0.5 * width)) / focal_length, (static_cast<double>(row) + 0.5 - (0.5 * height)) / focal_length, 1.0 });
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

    // What a run found: each chunk's focal length of each camera as a multiple of the one given, each chunk's turn of the
    // second camera from where it was declared, in degrees, and how far the frames' distances from the first are off the
    // truth's, at worst, in the map's own scale for one camera and in metres for a stereo pair.
    struct outcome final {
        std::vector<double> focal_scale[2];
        std::vector<double> turn_degrees;
        double scale_error = 0.0;
        size_t refinements = 0;
        size_t posed = 0;
    };

    // How far along its drift the run is at a frame, from nothing to all of it.
    double along(const int frame_index) {
        return static_cast<double>(frame_index) / static_cast<double>(frames - 1);
    }

    // A run through the system of one camera, or of a stereo pair 30 cm apart, about a room: the focal lengths rise by
    // focal_drift of themselves over the run, passing through the calibration given half way, and the second camera turns
    // by turn_drift radians about the vertical from where the rig declares it. The system is given the calibration and the
    // rig as declared throughout, and each frame's time unless the run is not timed.
    outcome run(const bool stereo, const bool follow, const double focal_drift, const double turn_drift, const bool timed = true) {
        const math::se3<double> declared(math::so3<double>::identity(), { { 0.3, 0.0, 0.0 } });
        slam system;
        system.frontend.calibration_drift.enabled = follow;
        system.frontend.calibration_drift.chunk_seconds = chunk_seconds;
        if (stereo) {
            mapping::rig rig;
            rig.add(1, mapping::rig::sensor_kind::camera, math::se3<double>::identity(), false);
            rig.add(2, mapping::rig::sensor_kind::camera, declared, false);
            system.set_rig(rig);
        }
        const double parameters[4] = { focal, focal, 0.5 * width, 0.5 * height };
        const std::vector<sensor::model> cameras(stereo ? 2 : 1, sensor::model(sensor::camera::pinhole<double>(&parameters[0], 4)));
        std::vector<math::se3<double>> truth;
        for (int i = 0; i < frames; ++i) {
            const double t = static_cast<double>(i);
            truth.push_back(math::se3<double>(math::so3<double>::exp({ { 0.003 * std::sin(0.2 * t), 0.008 * t, 0.0 } }), { { -0.03 * t, 0.05 * std::sin(0.15 * t), 0.0 } }));
            // The cameras as they really are at the frame: the focal lengths a little off the calibration, and the second
            // camera turned about the vertical from where the rig has it.
            const double focal_now = focal * (1.0 + (focal_drift * (along(i) - 0.5)));
            const math::se3<double> second_now(math::so3<double>::exp({ { 0.0, turn_drift * along(i), 0.0 } }), declared.translation());
            std::vector<image::image> images;
            images.emplace_back(static_cast<size_t>(height), static_cast<size_t>(width));
            render_room(truth.back(), focal_now, images[0]);
            if (stereo) {
                images.emplace_back(static_cast<size_t>(height), static_cast<size_t>(width));
                render_room(second_now.inverse() * truth.back(), focal_now, images[1]);
            }
            if (timed) {
                system.set_frame_timestamp(static_cast<long long>(i) * frame_nanoseconds);
            }
            system.process_rig_frame(cameras, images);
        }
        system.finalise();
        REQUIRE(system.state() == slam::tracking_state::tracking);

        outcome result;
        result.refinements = system.calibration_drift().get_refinements();
        result.posed = system.reconstruction.frames.size();
        for (const mapping::calibration_drift::chunk& chunk : system.calibration_drift().chunks()) {
            for (size_t camera = 0; camera < chunk.cameras.size(); ++camera) {
                double held[4] = {};
                REQUIRE(chunk.cameras[camera].camera.get_parameters(&held[0], 4));
                result.focal_scale[camera].push_back(held[0] / focal);
            }
            if (stereo) {
                result.turn_degrees.push_back((declared.rotation().inverse() * chunk.cameras[1].reference_from_camera.rotation()).log()[1] * 180.0 / 3.14159265358979);
            }
        }
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
        for (const double ratio : ratios) {
            result.scale_error = std::max(result.scale_error, std::abs((ratio / scale) - 1.0));
        }
        // Every frame the map holds has its chunk's calibration, keyframe or not; without a time a frame is in no chunk.
        if (!timed) {
            for (const auto& [frame_id, frame] : system.reconstruction.frames) {
                static_cast<void>(frame_id);
                REQUIRE(frame.calibration_chunk == -1);
            }
        }
        else if (follow) {
            size_t viewed = 0;
            for (const auto& [frame_id, frame] : system.reconstruction.frames) {
                REQUIRE(frame.calibration_chunk == (frame_id / frames_per_chunk));
                const size_t chunk = static_cast<size_t>(frame.calibration_chunk);
                double held[4] = {};
                REQUIRE(frame.camera.get_parameters(&held[0], 4));
                REQUIRE((held[0] / focal) == result.focal_scale[0][chunk]);
                // A keyframe's view has its chunk's camera and its chunk's pose on the reference camera.
                for (const mapping::frame& view : frame.views) {
                    REQUIRE(view.calibration_chunk == frame.calibration_chunk);
                    REQUIRE(view.camera.get_parameters(&held[0], 4));
                    REQUIRE((held[0] / focal) == result.focal_scale[1][chunk]);
                    REQUIRE(((declared.rotation().inverse() * view.reference_from_camera.rotation()).log()[1] * 180.0 / 3.14159265358979) == result.turn_degrees[chunk]);
                    ++viewed;
                }
            }
            if (stereo) {
                // Every chunk's cameras were both held by observations, and the rig has the newest chunk's extrinsic.
                REQUIRE(viewed > 10);
                for (const mapping::calibration_drift::chunk& chunk : system.calibration_drift().chunks()) {
                    REQUIRE((chunk.cameras.size() == 2) && (chunk.cameras[0].observations > 0) && (chunk.cameras[1].observations > 0));
                }
                const mapping::rig& rig = system.reconstruction.sensor_rig;
                REQUIRE(std::abs(((declared.rotation().inverse() * rig.reference_from_sensor(rig.camera(1)).rotation()).log()[1] * 180.0 / 3.14159265358979) - result.turn_degrees.back()) < 1.0e-9);
            }
        }
        return result;
    }

    void report(const char* const name, const outcome& result) {
        std::printf("%s: %zu frames posed, %zu adjustments followed the calibration, distances from the first frame off the truth's by %.4f at worst.\n", name, result.posed, result.refinements, result.scale_error);
        for (size_t camera = 0; camera < 2; ++camera) {
            if (result.focal_scale[camera].empty()) {
                continue;
            }
            std::printf("%s: focal length of camera %zu by chunk:", name, camera);
            for (const double scale : result.focal_scale[camera]) {
                std::printf(" %.4f", scale);
            }
            std::printf("\n");
        }
        if (!result.turn_degrees.empty()) {
            std::printf("%s: turn of the second camera by chunk, degrees:", name);
            for (const double turn : result.turn_degrees) {
                std::printf(" %.3f", turn);
            }
            std::printf("\n");
        }
        std::fflush(stdout);
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // The truth at the middle of the first chunk and of the last, as a fraction of the whole drift.
    const double first_along = 0.5 * (along(0) + along(frames_per_chunk - 1));
    const double last_along = 0.5 * (along(frames - frames_per_chunk) + along(frames - 1));

    {
        // A stereo pair whose second camera turns 0.3 degrees on the rig over the run, while both focal lengths rise by half
        // a percent. Held at what the rig declares, the baseline triangulates through an extrinsic that is more and more
        // wrong, and the map's scale goes with it; followed, each chunk has the extrinsic it was seen through.
        constexpr static const double turn = 0.3 * 3.14159265358979 / 180.0;
        const outcome held = run(true, false, 0.005, turn);
        report("Stereo pair drifting, calibration held", held);
        const outcome followed = run(true, true, 0.005, turn);
        report("Stereo pair drifting, calibration followed", followed);
        REQUIRE(held.refinements == 0);
        REQUIRE(held.focal_scale[0].empty() && held.turn_degrees.empty());
        REQUIRE(followed.refinements >= 2);
        REQUIRE(followed.focal_scale[0].size() == static_cast<size_t>(chunk_count));
        REQUIRE(followed.focal_scale[1].size() == static_cast<size_t>(chunk_count));
        REQUIRE(followed.turn_degrees.size() == static_cast<size_t>(chunk_count));
        // The turn is followed from chunk to chunk: -0.030 to 0.227 degrees for the truth's 0.011 to 0.289, each chunk
        // within 0.07 degrees of the truth at its middle.
        REQUIRE((followed.turn_degrees.back() - followed.turn_degrees.front()) > 0.15);
        REQUIRE((followed.turn_degrees.back() - followed.turn_degrees.front()) < 0.4);
        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            const double middle = 0.5 * (along(chunk * frames_per_chunk) + along(((chunk + 1) * frames_per_chunk) - 1));
            REQUIRE(std::abs(followed.turn_degrees[static_cast<size_t>(chunk)] - (0.3 * middle)) < 0.12);
        }
        // Both focal lengths rise, by 0.22% and 0.30% for the truth's 0.46% between the middles of the first chunk and the
        // last, and stay within 0.35% of the truth: a run this short finds half of a drift this small.
        for (size_t camera = 0; camera < 2; ++camera) {
            const double rise = followed.focal_scale[camera].back() - followed.focal_scale[camera].front();
            REQUIRE((rise > 0.001) && (rise < 0.008));
            REQUIRE(std::abs(followed.focal_scale[camera].front() - (1.0 + (0.005 * (first_along - 0.5)))) < 0.006);
            REQUIRE(std::abs(followed.focal_scale[camera].back() - (1.0 + (0.005 * (last_along - 0.5)))) < 0.006);
        }
        // The distances the rig has come are off the truth's by 3.3% at worst, against 7.8% with the calibration held.
        REQUIRE(followed.scale_error < (0.7 * held.scale_error));
    }

    {
        // One camera whose focal length rises by half a percent. The rise is followed, 0.25% for the truth's 0.46%; the
        // trajectory of one camera, whose scale is its own, hardly shows a focal length a quarter of a percent out, and is
        // as far off the truth's shape followed as held (1.5% and 1.4% at worst, the noise of the run).
        const outcome held = run(false, false, 0.005, 0.0);
        report("One camera drifting, calibration held", held);
        const outcome followed = run(false, true, 0.005, 0.0);
        report("One camera drifting, calibration followed", followed);
        REQUIRE(held.refinements == 0);
        REQUIRE(followed.refinements >= 1);
        REQUIRE(followed.focal_scale[0].size() == static_cast<size_t>(chunk_count));
        REQUIRE(followed.focal_scale[1].empty() && followed.turn_degrees.empty());
        const double rise = followed.focal_scale[0].back() - followed.focal_scale[0].front();
        REQUIRE((rise > 0.001) && (rise < 0.008));
        REQUIRE(std::abs(followed.focal_scale[0].front() - (1.0 + (0.005 * (first_along - 0.5)))) < 0.006);
        REQUIRE(std::abs(followed.focal_scale[0].back() - (1.0 + (0.005 * (last_along - 0.5)))) < 0.006);
        REQUIRE(followed.scale_error < (held.scale_error + 0.01));

        // Frames that come without a time are in no chunk: asked to follow the calibration, the system then holds it, and
        // the run is the one that held it.
        const outcome untimed = run(false, true, 0.005, 0.0, false);
        report("One camera drifting, calibration to be followed and no frame with a time", untimed);
        REQUIRE(untimed.refinements == 0);
        REQUIRE(untimed.focal_scale[0].empty());
        REQUIRE(untimed.posed == held.posed);
        REQUIRE(untimed.scale_error == held.scale_error);
    }

    return EXIT_SUCCESS;
}
