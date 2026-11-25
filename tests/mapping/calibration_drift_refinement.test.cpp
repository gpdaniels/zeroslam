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

#include "mapping/calibration_drift_refinement.hpp"

#include "mapping/calibration.hpp"
#include "mapping/calibration_drift.hpp"
#include "mapping/frame.hpp"
#include "mapping/map.hpp"
#include "mapping/point.hpp"
#include "mapping/rig.hpp"
#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/reprojection.hpp"
#include "optimisation/edges/rig_reprojection.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/extrinsic.hpp"
#include "optimisation/vertices/point.hpp"
#include "optimisation/vertices/pose.hpp"
#include "sensor/camera.hpp"
#include "simulation/scene.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

namespace {
    constexpr static const double focal = 500.0;
    constexpr static const double centre_x = 320.0;
    constexpr static const double centre_y = 240.0;
    // Half an hour of a run as the 160 keyframes a map might keep of it, in chunks of 45 seconds: a drift of half a
    // percent over it is then a third of what the paper's walk allows from chunk to chunk, as slow as a camera warms.
    constexpr static const int frame_count = 160;
    constexpr static const int frames_per_chunk = 4;
    constexpr static const int chunk_count = frame_count / frames_per_chunk;
    constexpr static const long long frame_nanoseconds = 11250000000ll;
    constexpr static const double chunk_seconds = 45.0;

    sensor::model pinhole_of(const double focal_x, const double focal_y) {
        const double parameters[4] = { focal_x, focal_y, centre_x, centre_y };
        return sensor::model(sensor::camera::pinhole<double>(&parameters[0], 4));
    }

    double focal_of(const sensor::model& camera, const size_t index = 0) {
        double parameters[4] = {};
        REQUIRE(camera.get_parameters(&parameters[0], 4));
        return parameters[index];
    }

    // A camera a rig carries around a cloud of points, looking at it, through a turn of about 57 degrees: its pose in the
    // world, world to camera.
    math::se3<double> pose_of(const int frame_index) {
        const double angle = 1.0 * static_cast<double>(frame_index) / static_cast<double>(frame_count - 1);
        const math::matrix<double, 3, 1> centre({ 6.0 * std::cos(angle), 1.0 + (0.3 * std::sin(3.0 * angle)), 6.0 * std::sin(angle) });
        math::matrix<double, 3, 1> forward = centre * (-1.0 / std::sqrt(centre.get_length_squared()));
        const math::matrix<double, 3, 1> down({ 0.0, 1.0, 0.0 });
        math::matrix<double, 3, 1> right({ (down[1] * forward[2]) - (down[2] * forward[1]), (down[2] * forward[0]) - (down[0] * forward[2]), (down[0] * forward[1]) - (down[1] * forward[0]) });
        right = right * (1.0 / std::sqrt(right.get_length_squared()));
        const math::matrix<double, 3, 1> below({ (forward[1] * right[2]) - (forward[2] * right[1]), (forward[2] * right[0]) - (forward[0] * right[2]), (forward[0] * right[1]) - (forward[1] * right[0]) });
        math::matrix<double, 3, 3> rotation;
        for (size_t column = 0; column < 3; ++column) {
            rotation[0][column] = right[column];
            rotation[1][column] = below[column];
            rotation[2][column] = forward[column];
        }
        return math::se3<double>(math::so3<double>(rotation), (rotation * centre) * -1.0);
    }

    // What a run's calibration really was: in each chunk the focal length of each camera, as a multiple of the one given,
    // and the second camera's pose on the first.
    struct truth final {
        std::vector<double> focal_scale[2];
        std::vector<math::se3<double>> reference_from_second;
    };

    // The truth of a run whose first camera's focal length rises by 0.5% from the first chunk to the last, as a camera's
    // does as it warms, the second's by 0.3%, each passing through the calibration given half way, and whose second camera
    // turns by 0.31 degrees and shifts by a millimetre across the 30 cm baseline from where it was declared. Without the
    // drift it is the calibration given throughout.
    truth make_truth(const bool drifting) {
        truth result;
        const math::se3<double> declared(math::so3<double>::identity(), { { 0.3, 0.0, 0.0 } });
        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            const double along = drifting ? (static_cast<double>(chunk) / static_cast<double>(chunk_count - 1)) : 0.0;
            result.focal_scale[0].push_back(1.0 + (drifting ? (0.005 * (along - 0.5)) : 0.0));
            result.focal_scale[1].push_back(1.0 + (drifting ? (0.003 * (along - 0.5)) : 0.0));
            const math::matrix<double, 3, 1> shifted = declared.translation() + math::matrix<double, 3, 1>({ 0.0, 0.001 * along, 0.0 });
            result.reference_from_second.push_back(math::se3<double>(math::so3<double>::exp({ { 0.003 * along, -0.004 * along, 0.002 * along } }), shifted * (0.3 / std::sqrt(shifted.get_length_squared()))));
        }
        return result;
    }

    // A map of the run as a system would hold it before an adjustment of the whole map: every frame in its chunk of time
    // with the calibration given, its focal lengths the truth's at the middle of the run times given_scale, posed where it
    // really was, the landmarks a few millimetres off, and each camera's observations of them through the calibration it
    // really had. The first frame of each chunk of a stereo pair has only the reference camera's, as a frame that is not a
    // keyframe lets its views go.
    mapping::map make_map(const truth& actual, const bool stereo, mapping::calibration_drift& drift, std::vector<math::matrix<double, 3, 1>>& points, const double given_scale = 1.0) {
        mapping::map m;
        const math::se3<double> declared(math::so3<double>::identity(), { { 0.3, 0.0, 0.0 } });
        if (stereo) {
            m.sensor_rig.add(1, mapping::rig::sensor_kind::camera, math::se3<double>::identity(), false);
            m.sensor_rig.add(2, mapping::rig::sensor_kind::camera, declared, false);
            m.metric_scale = true;
        }
        simulation::random_stream random(7);
        points.clear();
        for (int i = 0; i < 400; ++i) {
            points.push_back({ { random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0) } });
        }
        const sensor::model given = pinhole_of(focal * given_scale, focal * given_scale);
        for (int k = 0; k < frame_count; ++k) {
            mapping::frame f;
            f.id = k;
            f.camera = given;
            const math::se3<double> pose = pose_of(k);
            f.rotation = pose.rotation().get_matrix();
            f.translation = pose.translation();
            std::vector<sensor::model> cameras(stereo ? 2 : 1, given);
            std::vector<math::se3<double>> extrinsics = { math::se3<double>::identity() };
            if (stereo) {
                extrinsics.push_back(declared);
            }
            if (stereo && ((k % frames_per_chunk) != 0)) {
                mapping::frame view;
                view.id = k;
                view.camera = given;
                f.views.push_back(view);
            }
            f.calibration_chunk = drift.take_frame(static_cast<long long>(k) * frame_nanoseconds, cameras, extrinsics);
            REQUIRE(f.calibration_chunk == (k / frames_per_chunk));
            for (size_t index = 0; index < f.views.size(); ++index) {
                f.views[index].calibration_chunk = f.calibration_chunk;
                f.views[index].reference_from_camera = extrinsics[index + 1];
            }
            m.add_frame(f);
        }
        m.next_frame_id = frame_count;
        m.gauge_frame_id = 0;

        struct seen final {
            int frame;
            int camera;
            double x;
            double y;
        };

        std::vector<std::vector<seen>> sightings(points.size());
        for (int k = 0; k < frame_count; ++k) {
            const size_t chunk = static_cast<size_t>(k / frames_per_chunk);
            for (int camera = 0; camera < ((stereo && ((k % frames_per_chunk) != 0)) ? 2 : 1); ++camera) {
                const sensor::model actual_camera = pinhole_of(focal * actual.focal_scale[camera][chunk], focal * actual.focal_scale[camera][chunk]);
                const math::se3<double> pose = (camera == 0) ? pose_of(k) : (actual.reference_from_second[chunk].inverse() * pose_of(k));
                for (size_t i = 0; i < points.size(); ++i) {
                    const math::matrix<double, 3, 1> in_camera = pose * points[i];
                    math::matrix<double, 2, 1> pixel;
                    if (actual_camera.project(in_camera.data(), pixel.data()) && (pixel[0] > 0.5) && (pixel[0] < 639.5) && (pixel[1] > 0.5) && (pixel[1] < 479.5)) {
                        sightings[i].push_back({ k, camera, pixel[0], pixel[1] });
                    }
                }
            }
        }
        for (size_t i = 0; i < points.size(); ++i) {
            if (sightings[i].size() < 3) {
                continue;
            }
            m.add_landmark(mapping::point(static_cast<int>(i), points[i] + math::matrix<double, 3, 1>({ 0.004, -0.006, 0.005 }), math::matrix<double, 3, 1>({ 0.5, 0.5, 0.5 })));
            for (const seen& sighting : sightings[i]) {
                m.add_observation(sighting.frame, m.landmarks.at(static_cast<int>(i)), sighting.x, sighting.y, 0, 0, sighting.camera);
            }
        }
        m.next_landmark_id = static_cast<int>(points.size());
        return m;
    }

    // How far the frames' centres are from where the cameras really were, at worst.
    double worst_centre_error(const mapping::map& m) {
        double worst = 0.0;
        for (int k = 0; k < frame_count; ++k) {
            const mapping::frame& f = m.frames.at(k);
            const math::se3<double> pose = pose_of(k);
            const math::matrix<double, 3, 1> centre = -(math::transpose(f.rotation) * f.translation);
            const math::matrix<double, 3, 1> actual = -(math::transpose(pose.rotation().get_matrix()) * pose.translation());
            worst = std::max(worst, std::sqrt((centre - actual).get_length_squared()));
        }
        return worst;
    }

    // An adjustment of the whole map of one camera whose focal length drifts, with the calibration followed over time and
    // with it held: the chunks' vertices follow the focal length, every frame takes its chunk's camera, and the frames end
    // nearer to where the camera was than when one calibration has to serve the whole run. One camera observes how its
    // focal length changes far better than what it is, a longer focal length and a camera further back giving nearly the
    // same images, so the first and last chunks, which the walk ties to one neighbour only, are found the least well; and
    // the mean of the chunks stays with the prior on the calibration given, which is here the truth's mean.
    void test_focal_length_followed() {
        const truth actual = make_truth(true);
        double held_error = 0.0;
        for (int followed = 0; followed < 2; ++followed) {
            mapping::calibration_drift drift;
            mapping::calibration_drift::settings settings;
            settings.enabled = true;
            settings.chunk_seconds = chunk_seconds;
            drift.set(settings);
            std::vector<math::matrix<double, 3, 1>> points;
            mapping::map m = make_map(actual, false, drift, points);
            REQUIRE(drift.chunks().size() == static_cast<size_t>(chunk_count));
            mapping::calibration_drift::request following;
            REQUIRE(drift.make_request(mapping::calibration::level::off, false, {}, following));
            REQUIRE(following.cameras.size() == 1);
            REQUIRE(following.cameras[0].follow_focal && !following.cameras[0].follow_extrinsic);
            m.drift_request = (followed != 0) ? &following : nullptr;
            m.optimise(0, false, 100, false);
            m.drift_request = nullptr;
            const double error = worst_centre_error(m);
            if (followed == 0) {
                // Held, every frame keeps the camera given, and the poses take up what the camera cannot.
                REQUIRE(!following.refined);
                for (int k = 0; k < frame_count; ++k) {
                    REQUIRE(focal_of(m.frames.at(k).camera) == focal);
                }
                held_error = error;
                std::printf("Focal length drifting 0.5%%, calibration held: the frames end %.4f m off at worst.\n", error);
                continue;
            }
            REQUIRE(following.refined);
            REQUIRE(following.chunks.size() == static_cast<size_t>(chunk_count));
            double worst = 0.0;
            for (int chunk = 0; chunk < chunk_count; ++chunk) {
                const mapping::calibration_drift::chunk& solved = following.chunks[static_cast<size_t>(chunk)];
                REQUIRE(solved.index == chunk);
                REQUIRE(solved.cameras.size() == 1);
                REQUIRE(solved.cameras[0].observations > 1000);
                const double estimated = focal_of(solved.cameras[0].camera);
                const double expected = focal * actual.focal_scale[0][static_cast<size_t>(chunk)];
                worst = std::max(worst, std::abs((estimated / expected) - 1.0));
                if ((chunk % 13) == 0) {
                    std::printf("Focal length in chunk %d: %.3f, the truth %.3f, given %.1f.\n", chunk, estimated, expected, focal);
                }
                // The two focal lengths are each free, and tied by the prior on their ratio.
                REQUIRE(std::abs((focal_of(solved.cameras[0].camera, 1) / estimated) - 1.0) < 1.0e-4);
                // The principal point is held.
                REQUIRE(focal_of(solved.cameras[0].camera, 2) == centre_x);
                REQUIRE(focal_of(solved.cameras[0].camera, 3) == centre_y);
            }
            std::printf("Focal length drifting 0.5%%, calibration followed: within %.4f%% of the truth in every chunk, the frames end %.4f m off at worst, %.4f m with the calibration held.\n", 100.0 * worst, error, held_error);
            std::fflush(stdout);
            // The drift is 0.5%: every chunk is found to a tenth of it (0.045% at worst, in the first), the rise from the first
            // chunk to the last to more than four fifths, and the frames end a fifth as far off (4.0 mm against 22.1 mm).
            const double rise = focal_of(following.chunks.back().cameras[0].camera) - focal_of(following.chunks.front().cameras[0].camera);
            REQUIRE(worst < 6.0e-4);
            REQUIRE((rise > (0.75 * 0.005 * focal)) && (rise < (1.05 * 0.005 * focal)));
            REQUIRE(error < 0.3 * held_error);
            // Every frame of a chunk has the chunk's camera.
            for (int k = 0; k < frame_count; ++k) {
                REQUIRE(focal_of(m.frames.at(k).camera) == focal_of(following.chunks[static_cast<size_t>(k / frames_per_chunk)].cameras[0].camera));
            }
            // The state takes the result, and a chunk that starts afterwards starts from the newest.
            drift.update(following);
            REQUIRE(drift.get_refinements() == 1);
            std::vector<sensor::model> cameras(1, pinhole_of(focal, focal));
            std::vector<math::se3<double>> extrinsics(1, math::se3<double>::identity());
            REQUIRE(drift.take_frame(static_cast<long long>(frame_count) * frame_nanoseconds, cameras, extrinsics) == chunk_count);
            REQUIRE(focal_of(cameras[0]) == focal_of(following.chunks.back().cameras[0].camera));
        }
    }

    // The same camera given a focal length 2% long. How far the calibration given is trusted is what the level of the
    // constant calibration says: off, the mean of the chunks has the paper's prior on what was given, half a pixel in 240,
    // which holds it against the observations of the whole run part of the way, 0.76% long; at focal the prior is the
    // constant refinement's, a tenth of the focal length, and the chunks find the focal length as well as follow it, their
    // mean 0.002% off the truth's. Either way the rise is followed as it is from a calibration that is right.
    void test_focal_length_found() {
        const truth actual = make_truth(true);
        double mean_error[2] = { 0.0, 0.0 };
        double rise[2] = { 0.0, 0.0 };
        for (int found = 0; found < 2; ++found) {
            mapping::calibration_drift drift;
            mapping::calibration_drift::settings settings;
            settings.enabled = true;
            settings.chunk_seconds = chunk_seconds;
            drift.set(settings);
            std::vector<math::matrix<double, 3, 1>> points;
            mapping::map m = make_map(actual, false, drift, points, 1.02);
            mapping::calibration_drift::request following;
            REQUIRE(drift.make_request((found != 0) ? mapping::calibration::level::focal : mapping::calibration::level::off, false, {}, following));
            m.drift_request = &following;
            m.optimise(0, false, 100, false);
            m.drift_request = nullptr;
            REQUIRE(following.refined);
            REQUIRE(following.chunks.size() == static_cast<size_t>(chunk_count));
            for (int chunk = 0; chunk < chunk_count; ++chunk) {
                mean_error[found] += ((focal_of(following.chunks[static_cast<size_t>(chunk)].cameras[0].camera) / (focal * actual.focal_scale[0][static_cast<size_t>(chunk)])) - 1.0) / static_cast<double>(chunk_count);
            }
            rise[found] = focal_of(following.chunks.back().cameras[0].camera) - focal_of(following.chunks.front().cameras[0].camera);
            std::printf("Focal length given 2%% long and drifting 0.5%%, %s: the chunks' mean %.4f%% off the truth's, their rise %.3f pixels for the truth's %.3f.\n", (found != 0) ? "to be found" : "trusted", 100.0 * mean_error[found], rise[found], 0.005 * focal);
            std::fflush(stdout);
        }
        REQUIRE((mean_error[0] > 0.003) && (mean_error[0] < 0.015));
        REQUIRE(std::abs(mean_error[1]) < 2.0e-4);
        for (int found = 0; found < 2; ++found) {
            REQUIRE((rise[found] > (0.75 * 0.005 * focal)) && (rise[found] < (1.05 * 0.005 * focal)));
        }
    }

    // The same for a stereo pair whose second camera also turns and shifts on the rig as the run goes: each chunk's
    // extrinsic follows it at the declared baseline, both cameras' focal lengths are followed, each through its own
    // vertices, and the frames end nearer the truth, which a held extrinsic 0.31 degrees out leaves a long way off. The
    // shift of a millimetre across the baseline moves what the second camera sees by less than a tenth of a pixel, nearly as
    // a turn of it does, and is not found. Without any drift the calibration stays where it was given.
    void test_rig_followed(const bool drifting) {
        const truth actual = make_truth(drifting);
        double held_error = 0.0;
        for (int followed = (drifting ? 0 : 1); followed < 2; ++followed) {
            mapping::calibration_drift drift;
            mapping::calibration_drift::settings settings;
            settings.enabled = true;
            settings.chunk_seconds = chunk_seconds;
            drift.set(settings);
            std::vector<math::matrix<double, 3, 1>> points;
            mapping::map m = make_map(actual, true, drift, points);
            mapping::calibration_drift::request following;
            REQUIRE(drift.make_request(mapping::calibration::level::off, false, { 0.0, 0.3 }, following));
            REQUIRE(following.cameras.size() == 2);
            REQUIRE(following.cameras[1].follow_focal && following.cameras[1].follow_extrinsic && !following.cameras[0].follow_extrinsic);
            m.drift_request = (followed != 0) ? &following : nullptr;
            m.optimise(0, false, 100, false);
            m.drift_request = nullptr;
            const double error = worst_centre_error(m);
            if (followed == 0) {
                // Held, the views' observations go through the extrinsics the frames hold, the rig's being the same.
                REQUIRE(!following.refined);
                held_error = error;
                continue;
            }
            REQUIRE(following.refined);
            REQUIRE(following.chunks.size() == static_cast<size_t>(chunk_count));
            double worst_focal = 0.0;
            double worst_turn = 0.0;
            double worst_shift = 0.0;
            for (int chunk = 0; chunk < chunk_count; ++chunk) {
                const mapping::calibration_drift::chunk& solved = following.chunks[static_cast<size_t>(chunk)];
                REQUIRE(solved.cameras.size() == 2);
                for (size_t camera = 0; camera < 2; ++camera) {
                    REQUIRE(solved.cameras[camera].observations > 1000);
                    worst_focal = std::max(worst_focal, std::abs((focal_of(solved.cameras[camera].camera) / (focal * actual.focal_scale[camera][static_cast<size_t>(chunk)])) - 1.0));
                }
                const math::se3<double>& expected = actual.reference_from_second[static_cast<size_t>(chunk)];
                const math::se3<double>& estimated = solved.cameras[1].reference_from_camera;
                worst_turn = std::max(worst_turn, std::sqrt((expected.rotation().inverse() * estimated.rotation()).log().get_length_squared()));
                worst_shift = std::max(worst_shift, std::sqrt((estimated.translation() - expected.translation()).get_length_squared()));
                // The baseline keeps its declared length, which is the map's scale.
                REQUIRE(std::abs(std::sqrt(estimated.translation().get_length_squared()) - 0.3) < 1.0e-4);
                // The reference camera is the gauge.
                REQUIRE(solved.cameras[0].reference_from_camera.translation().get_length_squared() == 0.0);
            }
            if (drifting) {
                std::printf("Stereo pair drifting 0.5%% and 0.3%%, 0.31 degrees and 1 mm, calibration followed: focal lengths within %.4f%% of the truth, the extrinsic within %.4f degrees and %.5f m, the frames end %.5f m off at worst, %.5f m with the calibration held.\n", 100.0 * worst_focal, worst_turn * 180.0 / 3.14159265358979, worst_shift, error, held_error);
                // Measured: the focal lengths to 0.044%, the extrinsic to 0.0004 degrees and 0.34 mm, and the frames 3.8 mm off
                // against 236 mm.
                REQUIRE(worst_focal < 6.0e-4);
                REQUIRE(worst_turn < 5.0e-5);
                REQUIRE(worst_shift < 1.0e-3);
                REQUIRE(error < 0.1 * held_error);
            }
            else {
                std::printf("Stereo pair not drifting, calibration followed: focal lengths within %.6f%% of the truth, the extrinsic within %.6f degrees and %.7f m, the frames end %.7f m off at worst.\n", 100.0 * worst_focal, worst_turn * 180.0 / 3.14159265358979, worst_shift, error);
                REQUIRE(worst_focal < 1.0e-6);
                REQUIRE(worst_turn < 1.0e-6);
                REQUIRE(worst_shift < 1.0e-6);
                REQUIRE(error < 1.0e-5);
            }
            std::fflush(stdout);
            // Every frame's view has its chunk's camera and extrinsic.
            for (int k = 0; k < frame_count; ++k) {
                const mapping::calibration_drift::chunk& solved = following.chunks[static_cast<size_t>(k / frames_per_chunk)];
                if (m.frames.at(k).views.empty()) {
                    continue;
                }
                const mapping::frame& view = m.frames.at(k).views[0];
                REQUIRE(focal_of(view.camera) == focal_of(solved.cameras[1].camera));
                REQUIRE((view.reference_from_camera.translation() - solved.cameras[1].reference_from_camera.translation()).get_length_squared() == 0.0);
            }
            // The state takes every chunk's result, both cameras, the first frame of a chunk having only the one.
            drift.update(following);
            REQUIRE(drift.get_refinements() == 1);
            REQUIRE(drift.chunks().size() == static_cast<size_t>(chunk_count));
            for (size_t chunk = 0; chunk < drift.chunks().size(); ++chunk) {
                REQUIRE(drift.chunks()[chunk].cameras.size() == 2);
                REQUIRE(drift.chunks()[chunk].cameras[1].observations > 1000);
                REQUIRE(focal_of(drift.chunks()[chunk].cameras[1].camera) == focal_of(following.chunks[chunk].cameras[1].camera));
                REQUIRE((drift.chunks()[chunk].cameras[1].reference_from_camera.translation() - following.chunks[chunk].cameras[1].reference_from_camera.translation()).get_length_squared() == 0.0);
            }
            // An observation of a view projects through its frame's own extrinsic and not the rig's, which is still as declared:
            // after the adjustment the second camera's observations of the last chunk are within a tenth of a pixel, where the
            // rig's extrinsic would put them 2.7 pixels out.
            double worst_pixels = 0.0;
            for (const auto& [landmark_id, landmark_observations] : m.observations) {
                for (const mapping::map::observation& obs : landmark_observations) {
                    math::matrix<double, 2, 1> pixel;
                    if ((obs.camera == 1) && (obs.frame_id >= (frame_count - frames_per_chunk)) && m.project_observation(m.frames.at(obs.frame_id), obs, m.landmarks.at(landmark_id), pixel)) {
                        worst_pixels = std::max(worst_pixels, std::sqrt((pixel - obs.point).get_length_squared()));
                    }
                }
            }
            REQUIRE(worst_pixels < 0.2);
            REQUIRE((m.sensor_rig.reference_from_sensor(m.sensor_rig.camera(1)).translation() - math::matrix<double, 3, 1>({ 0.3, 0.0, 0.0 })).get_length_squared() == 0.0);
            // A pose adjustment afterwards holds the chunks' calibration, and leaves the frames where they are.
            m.optimise(frame_count, true, 20, false);
            REQUIRE(worst_centre_error(m) < (error + 1.0e-4));
        }
    }

    // The part of an adjustment on its own: which observations take which vertices, the ties between the chunks, and the
    // calibration going to the frames and coming back off them.
    void test_vertices_and_ties() {
        const sensor::model given = pinhole_of(focal, 1.02 * focal);
        mapping::rig rig;
        rig.add(1, mapping::rig::sensor_kind::camera, math::se3<double>::identity(), false);
        const math::se3<double> declared(math::so3<double>::identity(), { { 0.3, 0.0, 0.0 } });
        rig.add(2, mapping::rig::sensor_kind::camera, declared, false);
        // Frames in chunks 0, 0, 2 and 5, one frame that is in no chunk, and a second frame in chunk 2: the first frame of
        // that chunk has let its view go, as a frame that is not a keyframe does.
        const long long chunk_of_frame[6] = { 0, 0, 2, 5, -1, 2 };
        std::unordered_map<int, mapping::frame> frames;
        for (int frame_id = 0; frame_id < 6; ++frame_id) {
            mapping::frame frame;
            frame.id = frame_id;
            frame.camera = given;
            frame.calibration_chunk = chunk_of_frame[frame_id];
            mapping::frame view;
            view.id = frame_id;
            view.camera = given;
            view.calibration_chunk = chunk_of_frame[frame_id];
            view.reference_from_camera = declared;
            if (frame_id != 2) {
                frame.views.push_back(view);
            }
            frames[frame_id] = frame;
        }

        // A camera's pose on the reference as a frame holds it, and as the rig does for a frame in no chunk or without the view.
        frames.at(5).views[0].reference_from_camera = math::se3<double>(math::so3<double>::identity(), { { 0.31, 0.0, 0.0 } });
        REQUIRE(mapping::calibration_drift_refinement::reference_from_camera(rig, &frames.at(5), 1).translation()[0] == 0.31);
        REQUIRE(mapping::calibration_drift_refinement::reference_from_camera(rig, &frames.at(2), 1).translation()[0] == 0.3);
        REQUIRE(mapping::calibration_drift_refinement::reference_from_camera(rig, &frames.at(4), 1).translation()[0] == 0.3);
        REQUIRE(mapping::calibration_drift_refinement::reference_from_camera(rig, nullptr, 1).translation()[0] == 0.3);
        REQUIRE(mapping::calibration_drift_refinement::extrinsic_vertex(rig, frames.at(5), 1).is_fixed());
        REQUIRE(mapping::calibration_drift_refinement::extrinsic_vertex(rig, frames.at(5), 1).get_parameters()[0] == 0.31);
        REQUIRE(mapping::calibration_drift_refinement::extrinsic_vertex(rig, frames.at(4), 1).get_parameters()[0] == 0.3);
        frames.at(5).views[0].reference_from_camera = declared;

        mapping::calibration_drift::request following;
        following.cameras.resize(2);
        following.cameras[0].given = given;
        following.cameras[0].follow_focal = true;
        following.cameras[1].given = given;
        following.cameras[1].follow_focal = true;
        following.cameras[1].follow_extrinsic = true;
        following.cameras[1].baseline = 0.3;

        const auto add_pose = [](optimisation::factor_graph& graph) {
            optimisation::vertex pose{ optimisation::vertices::pose() };
            const double identity[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
            REQUIRE(pose.set_parameters(&identity[0], 7));
            return graph.add_vertex(static_cast<optimisation::vertex&&>(pose));
        };
        const auto add_point = [](optimisation::factor_graph& graph) {
            optimisation::vertex point{ optimisation::vertices::point() };
            const double location[3] = { 0.1, 0.2, 4.0 };
            REQUIRE(point.set_parameters(&location[0], 3));
            point.set_marginalised(true);
            return graph.add_vertex(static_cast<optimisation::vertex&&>(point));
        };

        {
            // Without a request the intrinsics are held and a view in a chunk takes a fixed extrinsic vertex of its chunk, one
            // for the frames of a chunk, where a view in no chunk takes the rig's.
            optimisation::factor_graph graph;
            mapping::calibration_drift_refinement held(nullptr, graph, frames, true);
            REQUIRE(!held.takes(0, 0));
            REQUIRE(!held.takes(0, 1));
            optimisation::vertex rig_extrinsic{ optimisation::vertices::extrinsic() };
            const std::vector<optimisation::vertex*> rig_extrinsics = { nullptr, &rig_extrinsic };
            REQUIRE(&held.extrinsics(0, 0, rig_extrinsics) == &rig_extrinsics);
            REQUIRE(&held.extrinsics(4, 1, rig_extrinsics) == &rig_extrinsics);
            const std::vector<optimisation::vertex*>& first = held.extrinsics(0, 1, rig_extrinsics);
            REQUIRE(&first != &rig_extrinsics);
            REQUIRE((first.size() == 2) && (first[0] == nullptr) && (first[1] != nullptr) && first[1]->is_fixed());
            optimisation::vertex* const of_first_chunk = first[1];
            REQUIRE(held.extrinsics(1, 1, rig_extrinsics)[1] == of_first_chunk);
            REQUIRE(held.extrinsics(5, 1, rig_extrinsics)[1] != of_first_chunk);
            REQUIRE(&held.extrinsics(2, 1, rig_extrinsics) == &rig_extrinsics);
            held.add_ties();
            held.apply();
            held.restore();
        }

        {
            // With the landmarks fixed a request is held too.
            optimisation::factor_graph graph;
            mapping::calibration_drift_refinement held(&following, graph, frames, false);
            REQUIRE(!held.takes(0, 0));
            const std::vector<optimisation::vertex*> rig_extrinsics = { nullptr, nullptr };
            REQUIRE(held.extrinsics(0, 1, rig_extrinsics)[1]->is_fixed());
            held.apply();
            REQUIRE(!following.refined);
        }

        optimisation::factor_graph graph;
        mapping::calibration_drift_refinement drift(&following, graph, frames, true);
        const std::vector<optimisation::vertex*> rig_extrinsics = { nullptr, nullptr };
        // The frames of chunks 0, 0 and 5 are observed through both cameras, the frames of chunk 2 through neither.
        std::unordered_map<int, optimisation::vertex*> intrinsics_of_frame[2];
        std::unordered_map<int, optimisation::vertex*> extrinsic_of_frame;
        for (const int frame_id : { 0, 1, 3 }) {
            REQUIRE(drift.takes(frame_id, 0));
            optimisation::edge factor{ optimisation::edges::reprojection(given, true) };
            factor.set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { 333.0, 266.0 } }.data()));
            REQUIRE(factor.add_vertex(add_pose(graph)));
            REQUIRE(factor.add_vertex(add_point(graph)));
            drift.attach(factor, frame_id, 0);
            REQUIRE(factor.num_vertices() == 3);
            optimisation::edge* const added = graph.add_edge(static_cast<optimisation::edge&&>(factor));
            REQUIRE(added != nullptr);
            intrinsics_of_frame[0][frame_id] = added->get_vertex(2);

            REQUIRE(drift.takes(frame_id, 1));
            optimisation::edge view_factor{ optimisation::edges::rig_reprojection(given, true) };
            view_factor.set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { 300.0, 266.0 } }.data()));
            REQUIRE(mapping::map::add_observation_vertices(view_factor, add_pose(graph), add_point(graph), drift.extrinsics(frame_id, 1, rig_extrinsics), 1));
            drift.attach(view_factor, frame_id, 1);
            REQUIRE(view_factor.num_vertices() == 4);
            optimisation::edge* const view_added = graph.add_edge(static_cast<optimisation::edge&&>(view_factor));
            REQUIRE(view_added != nullptr);
            extrinsic_of_frame[frame_id] = view_added->get_vertex(2);
            intrinsics_of_frame[1][frame_id] = view_added->get_vertex(3);
            REQUIRE(!extrinsic_of_frame[frame_id]->is_fixed());
        }
        // A frame in no chunk takes neither.
        REQUIRE(!drift.takes(4, 0));
        REQUIRE(!drift.takes(4, 1));
        REQUIRE(!drift.takes(9, 0));
        // The frames of a chunk share its vertices, each camera its own, and another chunk has others.
        REQUIRE(intrinsics_of_frame[0][0] == intrinsics_of_frame[0][1]);
        REQUIRE(intrinsics_of_frame[1][0] == intrinsics_of_frame[1][1]);
        REQUIRE(extrinsic_of_frame[0] == extrinsic_of_frame[1]);
        REQUIRE(intrinsics_of_frame[0][0] != intrinsics_of_frame[1][0]);
        REQUIRE(intrinsics_of_frame[0][0] != intrinsics_of_frame[0][3]);
        REQUIRE(extrinsic_of_frame[0] != extrinsic_of_frame[3]);
        // Only the focal lengths are free.
        REQUIRE(intrinsics_of_frame[0][0]->get_parameter_dimensions() == 4);
        REQUIRE(intrinsics_of_frame[0][0]->get_local_dimensions() == 2);

        drift.add_ties();
        // The first chunk's intrinsics: two observations, a prior on each focal length, one on their ratio, and a walk of
        // each to the next chunk that holds any; the last's has one observation and the same ties.
        REQUIRE(graph.get_connected_edges(intrinsics_of_frame[0][0]).size() == 7);
        REQUIRE(graph.get_connected_edges(intrinsics_of_frame[0][3]).size() == 6);
        // An extrinsic: its observations, its baseline's length, and the walk.
        REQUIRE(graph.get_connected_edges(extrinsic_of_frame[0]).size() == 4);
        REQUIRE(graph.get_connected_edges(extrinsic_of_frame[3]).size() == 3);
        // The walk between chunks 0 and 5 is over five chunks of five seconds, and each prior is one of two chunks'.
        size_t walks = 0;
        size_t priors = 0;
        for (const optimisation::edge* const tie : graph.get_connected_edges(intrinsics_of_frame[0][3])) {
            if (tie->required_vertices() == 2) {
                const double sigma = mapping::calibration_drift::settings().focal_walk * ((walks == 0) ? focal : (1.02 * focal)) * std::sqrt(5.0);
                REQUIRE(std::abs((tie->get_information()[0][0] * sigma * sigma) - 1.0) < 1.0e-9);
                ++walks;
            }
            else if ((tie->required_vertices() == 1) && (priors < 2)) {
                const double sigma = mapping::calibration_drift::focal_prior_fraction * ((priors == 0) ? focal : (1.02 * focal)) * std::sqrt(2.0);
                REQUIRE(std::abs((tie->get_information()[0][0] * sigma * sigma) - 1.0) < 1.0e-9);
                ++priors;
            }
        }
        REQUIRE((walks == 2) && (priors == 2));

        // After a solve, here the vertices set by hand: the frames take their chunks', the chunk between two that were
        // solved the way along between them, both its cameras although its first frame has one, and the frame in no chunk
        // nothing.
        const double first_solved[4] = { 505.0, 515.0, centre_x, centre_y };
        const double last_solved[4] = { 510.0, 520.0, centre_x, centre_y };
        REQUIRE(intrinsics_of_frame[0][0]->set_parameters(&first_solved[0], 4));
        REQUIRE(intrinsics_of_frame[0][3]->set_parameters(&last_solved[0], 4));
        double extrinsic_parameters[7] = {};
        optimisation::vertices::extrinsic::to_parameters(math::se3<double>(math::so3<double>::exp({ { 0.0, 0.01, 0.0 } }), { { 0.3, 0.005, 0.0 } }), extrinsic_parameters);
        REQUIRE(extrinsic_of_frame[3]->set_parameters(&extrinsic_parameters[0], 7));
        drift.apply();
        REQUIRE(following.refined);
        REQUIRE(following.chunks.size() == 3);
        REQUIRE((following.chunks[0].index == 0) && (following.chunks[1].index == 2) && (following.chunks[2].index == 5));
        REQUIRE(following.chunks[0].cameras[0].observations == 2);
        REQUIRE(following.chunks[1].cameras[0].observations == 0);
        REQUIRE((following.chunks[0].cameras.size() == 2) && (following.chunks[1].cameras.size() == 2) && (following.chunks[2].cameras.size() == 2));
        REQUIRE(following.chunks[2].cameras[1].observations == 1);
        REQUIRE(focal_of(frames.at(0).camera) == 505.0);
        REQUIRE(focal_of(frames.at(1).camera) == 505.0);
        REQUIRE(std::abs(focal_of(frames.at(2).camera) - 507.0) < 1.0e-9);
        REQUIRE(std::abs(focal_of(frames.at(2).camera, 1) - 517.0) < 1.0e-9);
        REQUIRE(focal_of(frames.at(5).camera) == focal_of(frames.at(2).camera));
        REQUIRE(focal_of(frames.at(3).camera) == 510.0);
        REQUIRE(focal_of(frames.at(4).camera) == focal);
        REQUIRE(focal_of(frames.at(0).views[0].camera) == focal);
        REQUIRE(std::abs(frames.at(5).views[0].reference_from_camera.translation()[1] - 0.002) < 1.0e-12);
        REQUIRE(std::abs(frames.at(5).views[0].reference_from_camera.rotation().log()[1] - 0.004) < 1.0e-12);
        REQUIRE(std::abs(following.chunks[1].cameras[1].reference_from_camera.translation()[1] - 0.002) < 1.0e-12);
        REQUIRE(frames.at(3).views[0].reference_from_camera.translation()[1] == 0.005);
        REQUIRE(frames.at(4).views[0].reference_from_camera.translation()[1] == 0.0);
        // The reference camera of a frame as it was before the solve.
        REQUIRE(focal_of(drift.camera_before(3, frames.at(3).camera)) == focal);
        REQUIRE(focal_of(drift.camera_before(4, frames.at(3).camera)) == 510.0);
        // Undone, the frames have what they had.
        drift.restore();
        REQUIRE(!following.refined);
        REQUIRE(following.chunks.empty());
        for (int frame_id = 0; frame_id < 6; ++frame_id) {
            REQUIRE(focal_of(frames.at(frame_id).camera) == focal);
            REQUIRE(frames.at(frame_id).views.empty() || (frames.at(frame_id).views[0].reference_from_camera.translation()[1] == 0.0));
        }
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    test_vertices_and_ties();
    test_focal_length_followed();
    test_focal_length_found();
    test_rig_followed(true);
    test_rig_followed(false);

    return EXIT_SUCCESS;
}
