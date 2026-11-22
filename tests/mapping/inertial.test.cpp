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
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/pose.hpp"
#include "sensor/imu/measurement.hpp"
#include "simulation/imu.hpp"

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
    return EXIT_SUCCESS;
}
