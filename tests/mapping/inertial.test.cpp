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

#include "mapping/inertial.hpp"

#include "mapping/frame.hpp"
#include "mapping/rig.hpp"
#include "math/lie.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/prior.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/pose.hpp"
#include "sensor/imu/measurement.hpp"
#include "sensor/imu/preintegration.hpp"
#include "simulation/imu.hpp"
#include "simulation/scene.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static double norm(const math::matrix<double, 3, 1>& vector) {
    return std::sqrt(vector.get_length_squared());
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const sensor::imu::parameters noise;
    const simulation::imu_trajectory trajectory = simulation::imu_trajectory::generate(simulation::imu_trajectory::options());
    simulation::imu_sensor::options sensor_options;
    sensor_options.noise = false;
    sensor_options.random_walk = false;
    const simulation::imu_sensor::recording recording = simulation::imu_sensor::generate(trajectory, noise, sensor_options);
    // The camera, the rig's reference, with the imu turned and offset on it.
    const math::se3<double> camera_from_imu(math::so3<double>::exp({ { 0.4, -1.2, 0.3 } }), { { 0.05, 0.02, -0.03 } });
    mapping::rig rig;
    rig.add(1, mapping::rig::sensor_kind::camera, math::se3<double>::identity(), false);
    rig.add(7, mapping::rig::sensor_kind::imu, camera_from_imu, false);

    // A frame at a time posed as the camera truly is: world to camera, T_ci T_iw.
    const auto make_frame = [&](const int id, const double time) {
        mapping::frame frame;
        frame.id = id;
        const math::se3<double> pose = camera_from_imu * trajectory.world_to_imu(time);
        frame.rotation = pose.rotation().get_matrix();
        frame.translation = pose.translation();
        return frame;
    };

    mapping::inertial state;
    REQUIRE(!state.declared());
    size_t index = 0;
    REQUIRE(!state.index_in(rig, index));
    state.declare(7, noise);
    REQUIRE(state.declared());
    REQUIRE(state.sensor_id() == 7);
    REQUIRE(state.index_in(rig, index) && (index == 1));
    REQUIRE(!state.initialised());

    // Samples in time order only.
    for (size_t k = 0; k < 1201; ++k) {
        REQUIRE(state.add_sample(recording.samples[k]));
    }
    REQUIRE(!state.add_sample(recording.samples[5]));
    REQUIRE(state.samples().size() == 1201);

    // A frame between samples is preintegrated with the samples interpolated, and one a moment past the last sample with the last
    // one held; one further on, or before the first sample, is not.
    sensor::imu::preintegration preintegration;
    REQUIRE(state.preintegrate(1002500000ll, 2002500000ll, sensor::imu::bias(), preintegration));
    REQUIRE(std::abs(preintegration.get_delta_time() - 1.0) < 1e-12);
    REQUIRE(state.preintegrate(5000000000ll, 6008000000ll, sensor::imu::bias(), preintegration));
    REQUIRE(!state.preintegrate(5000000000ll, 6100000000ll, sensor::imu::bias(), preintegration));
    REQUIRE(!state.preintegrate(-1, 1000000000ll, sensor::imu::bias(), preintegration));
    long long timestamp = 0;
    REQUIRE(!state.frame_time(3, timestamp));
    state.set_frame_time(3, 1234);
    REQUIRE(state.frame_time(3, timestamp) && (timestamp == 1234));

    // Keyframes every half second: each preintegrated from the one before it; a new chain is tied to none before it.
    std::unordered_map<int, mapping::frame> frames;
    std::unordered_set<int> keyframe_ids;
    for (int id = 0; id < 8; ++id) {
        const long long time = 1000000000ll + (500000000ll * id);
        frames[id] = make_frame(id, 1.0e-9 * static_cast<double>(time));
        state.set_frame_time(id, time);
        state.add_keyframe(id, frames, rig);
        keyframe_ids.insert(id);
    }
    REQUIRE(state.keyframes().size() == 8);
    REQUIRE(!state.keyframes().at(0).linked);
    for (int id = 1; id < 8; ++id) {
        REQUIRE(state.keyframes().at(id).linked);
        REQUIRE(std::abs(state.keyframes().at(id).from_previous.get_delta_time() - 0.5) < 1e-12);
        REQUIRE(!state.keyframes().at(id).estimated);
    }
    REQUIRE(state.chain_window().size() == 8);

    // A culled keyframe's measurements merge into the next keyframe's.
    keyframe_ids.erase(3);
    state.synchronise(keyframe_ids, frames);
    REQUIRE(state.keyframes().count(3) == 0);
    REQUIRE(std::abs(state.keyframes().at(4).from_previous.get_delta_time() - 1.0) < 1e-12);
    REQUIRE(state.chain_window().size() == 7);

    // Estimated, with the true velocities and gravity, the keyframes predict a frame's pose; the inertial edges of an adjustment
    // agree with the truth, so a solve from slightly moved poses brings them back.
    const math::se3<double>& reference_from_imu = camera_from_imu;
    REQUIRE(!state.predict(4500000000ll, frames, rig, frames[0].rotation, frames[0].translation, frames[0].translation));
    state.set_gravity(trajectory.gravity());
    state.set_chain_initialised(state.chain());
    REQUIRE(state.initialised());
    for (auto& [id, keyframe] : state.keyframes()) {
        keyframe.velocity = trajectory.velocity(1.0e-9 * static_cast<double>(keyframe.timestamp));
        keyframe.estimated = true;
    }
    {
        const sensor::imu::preintegration::state at = mapping::inertial::state_at(frames[2], trajectory.velocity(2.0), reference_from_imu);
        REQUIRE(norm(at.position - trajectory.position(2.0)) < 1e-12);
        math::matrix<double, 3, 3> rotation;
        math::matrix<double, 3, 1> translation;
        mapping::inertial::pose_of(at, reference_from_imu, rotation, translation);
        REQUIRE(norm(translation - frames[2].translation) < 1e-12);
        math::matrix<double, 3, 1> velocity;
        REQUIRE(state.predict(4800000000ll, frames, rig, rotation, translation, velocity));
        const mapping::frame truth = make_frame(99, 4.8);
        REQUIRE(norm(translation - truth.translation) < 1e-3);
        REQUIRE(norm(velocity - trajectory.velocity(4.8)) < 1e-3);
    }
    {
        optimisation::factor_graph graph;
        std::unordered_map<int, optimisation::vertex*> poses;
        for (const auto& [id, keyframe] : state.keyframes()) {
            static_cast<void>(keyframe);
            const mapping::frame& frame = frames.at(id);
            const math::se3<double> pose(frame.rotation, frame.translation);
            const math::se3<double> moved = (id < 2) ? pose : (math::se3<double>::exp({ { 0.002, -0.001, 0.003, 0.01, -0.02, 0.015 } }) * pose);
            optimisation::vertex node{ optimisation::vertices::pose() };
            const double parameters[7] = { moved.translation()[0], moved.translation()[1], moved.translation()[2], moved.rotation().get_quaternion()[1], moved.rotation().get_quaternion()[2], moved.rotation().get_quaternion()[3], moved.rotation().get_quaternion()[0] };
            REQUIRE(node.set_parameters(&parameters[0], 7));
            node.set_fixed(id < 2);
            poses[id] = graph.add_vertex(static_cast<optimisation::vertex&&>(node));
        }
        const mapping::inertial::factors held = state.add_factors(graph, poses, frames, rig, false);
        REQUIRE(held.edges == 5);
        REQUIRE(held.extrinsic == nullptr);
        REQUIRE(held.velocities.size() == 5);
        REQUIRE(graph.solve(30, true) > 0);
        for (const auto& [id, node] : poses) {
            const double* const p = node->get_parameters();
            const math::matrix<double, 3, 1> translation = { { p[0], p[1], p[2] } };
            REQUIRE(norm(translation - frames.at(id).translation) < 5e-3);
        }
        state.read_factors(held, rig);
        for (const auto& [id, keyframe] : state.keyframes()) {
            REQUIRE(norm(keyframe.velocity - trajectory.velocity(1.0e-9 * static_cast<double>(keyframe.timestamp))) < 2e-2);
        }
    }

    // An imu whose extrinsic the rig refines gets an extrinsic vertex in the edges, which a solve from the truth leaves there and
    // the rig takes back; held, it gets none.
    {
        mapping::rig refined;
        refined.add(1, mapping::rig::sensor_kind::camera, math::se3<double>::identity(), false);
        refined.add(7, mapping::rig::sensor_kind::imu, camera_from_imu, true);
        optimisation::factor_graph graph;
        std::unordered_map<int, optimisation::vertex*> poses;
        for (const auto& [id, keyframe] : state.keyframes()) {
            static_cast<void>(keyframe);
            const math::se3<double> pose(frames.at(id).rotation, frames.at(id).translation);
            optimisation::vertex node{ optimisation::vertices::pose() };
            const double parameters[7] = { pose.translation()[0], pose.translation()[1], pose.translation()[2], pose.rotation().get_quaternion()[1], pose.rotation().get_quaternion()[2], pose.rotation().get_quaternion()[3], pose.rotation().get_quaternion()[0] };
            REQUIRE(node.set_parameters(&parameters[0], 7));
            node.set_fixed(id == 0);
            poses[id] = graph.add_vertex(static_cast<optimisation::vertex&&>(node));
        }
        const mapping::inertial::factors held = state.add_factors(graph, poses, frames, refined, false);
        REQUIRE(held.extrinsic != nullptr);
        REQUIRE(!held.extrinsic->is_fixed());
        static_cast<void>(graph.solve(10, true));
        state.read_factors(held, refined);
        const math::matrix<double, 6, 1> error = (refined.reference_from_sensor(1) * camera_from_imu.inverse()).log();
        REQUIRE(std::sqrt(error.get_length_squared()) < 0.02);
        optimisation::factor_graph held_graph;
        std::unordered_map<int, optimisation::vertex*> held_poses;
        for (const auto& [id, node] : poses) {
            held_poses[id] = held_graph.add_vertex(optimisation::vertex(*node));
        }
        REQUIRE(state.add_factors(held_graph, held_poses, frames, refined, true).extrinsic == nullptr);
    }

    // A new chain starts with an unlinked keyframe, and is not initialised.
    state.break_chain();
    frames[8] = make_frame(8, 5.2);
    state.set_frame_time(8, 5200000000ll);
    state.add_keyframe(8, frames, rig);
    REQUIRE(!state.keyframes().at(8).linked);
    REQUIRE(!state.chain_initialised(state.chain()));
    REQUIRE(state.chain_window().size() == 1);
    math::matrix<double, 3, 3> rotation;
    math::matrix<double, 3, 1> translation;
    math::matrix<double, 3, 1> velocity;
    REQUIRE(!state.predict(5300000000ll, frames, rig, rotation, translation, velocity));

    // The gyroscope's matrix. An imu with the EuRoC imu's noise whose gyroscope is scaled by about a percent and misaligned by
    // about half a degree, over thirty seconds of keyframes 0.4 s apart that vision poses to a milliradian and two millimetres: a
    // prior on each pose at the truth with that noise stands for the landmarks. Asked for, the matrix is a vertex of every
    // inertial edge, held in a window and until the initialisation has settled, and free in an adjustment of the whole map, where
    // it comes out as the truth's and the keyframes nearer the truth than with the gyroscope taken as it is.
    {
        sensor::imu::intrinsics true_intrinsics;
        true_intrinsics.gyroscope = { { { 0.010, 0.0087, -0.0052 }, { -0.0070, -0.012, 0.0061 }, { 0.0044, -0.0079, 0.008 } } };
        simulation::imu_sensor::options misaligned_options;
        misaligned_options.initial_bias.gyroscope = { { 0.003, -0.002, 0.004 } };
        misaligned_options.initial_bias.accelerometer = { { 0.03, -0.02, 0.04 } };
        misaligned_options.end = 32000000000ll;
        misaligned_options.intrinsics = true_intrinsics;
        const simulation::imu_sensor::recording misaligned = simulation::imu_sensor::generate(trajectory, noise, misaligned_options);
        constexpr static const int keyframe_count = 76;
        const auto keyframe_time = [](const int id) {
            return 1000000000ll + (400000000ll * id);
        };
        // What vision makes of the keyframes: the truth, each off by a milliradian and two millimetres an axis.
        std::unordered_map<int, mapping::frame> seen;
        simulation::random_stream random(29);
        for (int id = 0; id < keyframe_count; ++id) {
            seen[id] = make_frame(id, 1.0e-9 * static_cast<double>(keyframe_time(id)));
            if (id > 0) {
                const math::se3<double> off = math::se3<double>::exp({ { 1.0e-3 * random.gaussian(), 1.0e-3 * random.gaussian(), 1.0e-3 * random.gaussian(), 2.0e-3 * random.gaussian(), 2.0e-3 * random.gaussian(), 2.0e-3 * random.gaussian() } });
                const math::se3<double> pose = off * math::se3<double>(math::so3<double>(seen[id].rotation), seen[id].translation);
                seen[id].rotation = pose.rotation().get_matrix();
                seen[id].translation = pose.translation();
            }
        }
        math::matrix<double, 0, 0> vision_information = math::matrix<double, 0, 0>::zero(6, 6);
        for (size_t i = 0; i < 3; ++i) {
            vision_information[i][i] = 1.0 / (1.0e-3 * 1.0e-3);
            vision_information[3 + i][3 + i] = 1.0 / (2.0e-3 * 2.0e-3);
        }

        struct outcome final {
            double rotation_error = 0.0;
            double position_error = 0.0;
            sensor::imu::intrinsics estimate;
            size_t refinements = 0;
        };

        const auto run = [&](const bool refine) {
            mapping::inertial imu;
            imu.declare(7, noise);
            for (const sensor::imu::sample& sample : misaligned.samples) {
                REQUIRE(imu.add_sample(sample));
            }
            imu.set_gravity(trajectory.gravity());
            imu.set_chain_initialised(imu.chain());
            imu.gyroscope().set_refined(refine);
            std::unordered_map<int, mapping::frame> posed = seen;
            for (int id = 0; id < keyframe_count; ++id) {
                imu.set_frame_time(id, keyframe_time(id));
                imu.add_keyframe(id, posed, rig);
                mapping::inertial::keyframe_state& keyframe = imu.keyframes().at(id);
                keyframe.velocity = trajectory.velocity(1.0e-9 * static_cast<double>(keyframe.timestamp));
                // The first keyframe is held with its bias, the one the initialisation would leave it, which the rest start from.
                keyframe.bias = misaligned.biases.front();
                keyframe.estimated = true;
            }
            // Whether the edges of an adjustment took the matrix's vertex, and whether it was free.
            struct taken final {
                bool vertex = false;
                bool free = false;
            };
            const auto adjust = [&](const bool whole_map, const bool hold, const bool solve) {
                optimisation::factor_graph graph;
                std::unordered_map<int, optimisation::vertex*> poses;
                for (int id = 0; id < keyframe_count; ++id) {
                    const math::se3<double> pose(posed.at(id).rotation, posed.at(id).translation);
                    optimisation::vertex node{ optimisation::vertices::pose() };
                    const double parameters[7] = { pose.translation()[0], pose.translation()[1], pose.translation()[2], pose.rotation().get_quaternion()[1], pose.rotation().get_quaternion()[2], pose.rotation().get_quaternion()[3], pose.rotation().get_quaternion()[0] };
                    REQUIRE(node.set_parameters(&parameters[0], 7));
                    node.set_fixed(id == 0);
                    poses[id] = graph.add_vertex(static_cast<optimisation::vertex&&>(node));
                    if (id > 0) {
                        const math::se3<double> prior(seen.at(id).rotation, seen.at(id).translation);
                        const double prior_parameters[7] = { prior.translation()[0], prior.translation()[1], prior.translation()[2], prior.rotation().get_quaternion()[1], prior.rotation().get_quaternion()[2], prior.rotation().get_quaternion()[3], prior.rotation().get_quaternion()[0] };
                        optimisation::edge vision{ optimisation::edges::pose_prior() };
                        vision.set_observation(math::matrix<double, 0, 0>(7, 1, &prior_parameters[0]));
                        vision.set_information(vision_information);
                        REQUIRE(vision.add_vertex(poses[id]));
                        REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(vision)) != nullptr);
                    }
                }
                const mapping::inertial::factors factors = imu.add_factors(graph, poses, posed, rig, hold, whole_map);
                REQUIRE(factors.edges == static_cast<size_t>(keyframe_count - 1));
                taken result;
                result.vertex = factors.gyroscope != nullptr;
                result.free = result.vertex && !factors.gyroscope->is_fixed();
                if (!solve) {
                    return result;
                }
                REQUIRE(graph.solve(30, true) > 0);
                for (const auto& [id, node] : poses) {
                    const double* const p = node->get_parameters();
                    const math::se3<double> pose(math::so3<double>(p[6], p[3], p[4], p[5]), { { p[0], p[1], p[2] } });
                    posed.at(id).rotation = pose.rotation().get_matrix();
                    posed.at(id).translation = pose.translation();
                }
                mapping::rig read = rig;
                imu.read_factors(factors, read);
                return result;
            };
            if (!refine) {
                // Not asked for, the edges take no matrix.
                REQUIRE(!adjust(true, false, false).vertex);
            }
            else {
                // Held until the initialisation has settled, in a window, and where the calibrations are held.
                taken matrix = adjust(true, false, false);
                REQUIRE(matrix.vertex && !matrix.free);
                imu.gyroscope().set_settled(true);
                matrix = adjust(false, false, false);
                REQUIRE(matrix.vertex && !matrix.free);
                matrix = adjust(true, true, false);
                REQUIRE(matrix.vertex && !matrix.free);
                matrix = adjust(true, false, false);
                REQUIRE(matrix.vertex && matrix.free);
            }
            // Three adjustments of the whole map, as they come while a map grows: the measurements are integrated again at the
            // matrix and the biases the first one leaves.
            for (int round = 0; round < 3; ++round) {
                static_cast<void>(adjust(true, false, true));
            }
            outcome result;
            result.estimate = imu.gyroscope().estimate();
            result.refinements = imu.gyroscope().refinements();
            for (int id = 1; id < keyframe_count; ++id) {
                const mapping::frame truth = make_frame(id, 1.0e-9 * static_cast<double>(keyframe_time(id)));
                const math::se3<double> error = math::se3<double>(math::so3<double>(posed.at(id).rotation), posed.at(id).translation) * math::se3<double>(math::so3<double>(truth.rotation), truth.translation).inverse();
                result.rotation_error += error.rotation().log().get_length_squared();
                result.position_error += error.translation().get_length_squared();
            }
            result.rotation_error = std::sqrt(result.rotation_error / (keyframe_count - 1));
            result.position_error = std::sqrt(result.position_error / (keyframe_count - 1));
            if (refine) {
                // The measurements are integrated at the matrix from then on, the new ones and those it moved too far.
                REQUIRE(imu.keyframes().at(keyframe_count - 1).from_previous.get_linearisation_intrinsics().gyroscope != (math::matrix<double, 3, 3>::zero()));
                sensor::imu::preintegration fresh;
                REQUIRE(imu.preintegrate(keyframe_time(keyframe_count - 2), keyframe_time(keyframe_count - 1), sensor::imu::bias(), fresh));
                REQUIRE(fresh.get_linearisation_intrinsics().gyroscope == result.estimate.gyroscope);
            }
            return result;
        };
        const outcome held = run(false);
        const outcome refined = run(true);
        REQUIRE(held.refinements == 0);
        REQUIRE(held.estimate.gyroscope == (math::matrix<double, 3, 3>::zero()));
        REQUIRE(refined.refinements == 3);
        double largest = 0.0;
        for (size_t i = 0; i < 9; ++i) {
            largest = std::fmax(largest, std::abs(refined.estimate.gyroscope.data()[i] - true_intrinsics.gyroscope.data()[i]));
        }
        const math::matrix<double, 3, 3>& m = refined.estimate.gyroscope;
        std::printf("Gyroscope matrix: [%.5f %.5f %.5f; %.5f %.5f %.5f; %.5f %.5f %.5f], largest error %.5f; keyframes %.2f mrad and %.2f mm from the truth, %.2f mrad and %.2f mm with the matrix held at zero.\n", m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2], largest, 1.0e3 * refined.rotation_error, 1.0e3 * refined.position_error, 1.0e3 * held.rotation_error, 1.0e3 * held.position_error);
        std::fflush(stdout);
        // The entries are 0.004 to 0.012, and come out within 0.0008; the keyframes are 0.5 mrad and 1.3 mm from the truth, where
        // the gyroscope taken as it is leaves them 6.6 mrad and 7.0 mm from it.
        REQUIRE(largest < 1.0e-3);
        REQUIRE(refined.rotation_error < 1.0e-3);
        REQUIRE(refined.rotation_error < 0.2 * held.rotation_error);
        REQUIRE(refined.position_error < 0.5 * held.position_error);
    }
    return EXIT_SUCCESS;
}
