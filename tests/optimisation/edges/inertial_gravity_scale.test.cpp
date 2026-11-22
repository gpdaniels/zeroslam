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

#include "optimisation/edges/inertial_gravity_scale.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/inertial.hpp"
#include "optimisation/edges/prior.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/gravity_direction.hpp"
#include "optimisation/vertices/imu_bias.hpp"
#include "optimisation/vertices/pose.hpp"
#include "optimisation/vertices/scale.hpp"
#include "optimisation/vertices/velocity.hpp"
#include "sensor/imu/measurement.hpp"
#include "sensor/imu/preintegration.hpp"
#include "simulation/imu.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

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

static double norm(const math::matrix<double, 3, 1>& vector) {
    return std::sqrt(vector.get_length_squared());
}

static optimisation::vertex make_pose(const math::se3<double>& pose) {
    optimisation::vertex vertex{ optimisation::vertices::pose() };
    const double parameters[7] = { pose.translation()[0], pose.translation()[1], pose.translation()[2], pose.rotation().get_quaternion()[1], pose.rotation().get_quaternion()[2], pose.rotation().get_quaternion()[3], pose.rotation().get_quaternion()[0] };
    REQUIRE(vertex.set_parameters(&parameters[0], 7));
    return vertex;
}

static optimisation::vertex make_velocity(const math::matrix<double, 3, 1>& velocity) {
    optimisation::vertex vertex{ optimisation::vertices::velocity() };
    REQUIRE(vertex.set_parameters(velocity.data(), 3));
    return vertex;
}

static optimisation::vertex make_bias(const sensor::imu::bias& bias) {
    optimisation::vertex vertex{ optimisation::vertices::imu_bias() };
    const double parameters[6] = { bias.gyroscope[0], bias.gyroscope[1], bias.gyroscope[2], bias.accelerometer[0], bias.accelerometer[1], bias.accelerometer[2] };
    REQUIRE(vertex.set_parameters(&parameters[0], 6));
    return vertex;
}

static optimisation::vertex make_direction(const math::so3<double>& rotation) {
    optimisation::vertex vertex{ optimisation::vertices::gravity_direction() };
    const double parameters[4] = { rotation.get_quaternion()[1], rotation.get_quaternion()[2], rotation.get_quaternion()[3], rotation.get_quaternion()[0] };
    REQUIRE(vertex.set_parameters(&parameters[0], 4));
    return vertex;
}

static optimisation::vertex make_scale(const double scale) {
    optimisation::vertex vertex{ optimisation::vertices::scale() };
    REQUIRE(vertex.set_parameters(&scale, 1));
    return vertex;
}

static math::matrix<double, 0, 0> numeric_jacobian(optimisation::edge& edge, const int slot) {
    optimisation::vertex* const vertex = edge.get_vertex(slot);
    const size_t rows = edge.get_residual().rows();
    const size_t columns = static_cast<size_t>(vertex->get_local_dimensions());
    const size_t parameter_count = static_cast<size_t>(vertex->get_parameter_dimensions());
    double backup[optimisation::vertex::maximum_parameters] = {};
    REQUIRE(vertex->get_parameters(&backup[0], parameter_count));
    math::matrix<double, 0, 0> result = math::matrix<double, 0, 0>::zero(rows, columns);
    const double step = 1e-6;
    for (size_t column = 0; column < columns; ++column) {
        double delta[optimisation::vertex::maximum_parameters] = {};
        delta[column] = step;
        vertex->plus(&delta[0]);
        edge.compute_residual();
        const math::matrix<double, 0, 0> forward = edge.get_residual();
        REQUIRE(vertex->set_parameters(&backup[0], parameter_count));
        delta[column] = -step;
        vertex->plus(&delta[0]);
        edge.compute_residual();
        const math::matrix<double, 0, 0> backward = edge.get_residual();
        REQUIRE(vertex->set_parameters(&backup[0], parameter_count));
        for (size_t row = 0; row < rows; ++row) {
            result[row][column] = (forward[row][0] - backward[row][0]) / (2.0 * step);
        }
    }
    edge.compute_residual();
    return result;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(optimisation::edges::inertial_gravity_scale) <= optimisation::edge::maximum_size, "The inertial gravity scale edge must fit.");

    const sensor::imu::parameters parameters;
    const simulation::imu_trajectory trajectory = simulation::imu_trajectory::generate(simulation::imu_trajectory::options());
    const math::se3<double> body_from_imu(math::so3<double>::exp({ { -0.2, 1.4, 0.1 } }), { { -0.04, 0.06, 0.02 } });
    sensor::imu::bias true_bias;
    true_bias.gyroscope = { { -0.01, 0.015, 0.02 } };
    true_bias.accelerometer = { { -0.05, 0.08, 0.1 } };

    // The map's world is turned from the gravity aligned world and its unit is a third of a metre: positions p_map = R_vg p / s and
    // rotations R_vg R, with the true scale s = 3 and gravity R_vg (0, 0, -g) in the map's world.
    const double true_scale = 3.0;
    const math::so3<double> map_from_gravity = math::so3<double>::exp({ { 0.7, -0.4, 2.0 } });
    const auto map_pose = [&](const double time) {
        const math::se3<double> world_from_body = trajectory.world_to_imu(time).inverse() * body_from_imu.inverse();
        const math::se3<double> map_from_body(map_from_gravity * world_from_body.rotation(), (1.0 / true_scale) * (map_from_gravity * world_from_body.translation()));
        return map_from_body.inverse();
    };
    const auto map_velocity = [&](const double time) {
        return map_from_gravity * trajectory.velocity(time);
    };

    simulation::imu_sensor::options sensor_options;
    sensor_options.random_walk = false;
    sensor_options.initial_bias = true_bias;
    const simulation::imu_sensor::recording recording = simulation::imu_sensor::generate(trajectory, parameters, sensor_options);

    // At scale one and with the direction's gravity, the residual is the inertial edge's; and the analytic jacobians agree with
    // central differences away from the truth.
    {
        sensor::imu::preintegration preintegration(parameters, true_bias);
        REQUIRE(preintegration.integrate(recording.samples, 3000000000ll, 3400000000ll));
        optimisation::vertex first_pose = make_pose(map_pose(3.0));
        optimisation::vertex first_velocity = make_velocity(map_velocity(3.0));
        optimisation::vertex bias = make_bias(true_bias);
        optimisation::vertex second_pose = make_pose(map_pose(3.4));
        optimisation::vertex second_velocity = make_velocity(map_velocity(3.4));
        optimisation::vertex direction = make_direction(map_from_gravity);
        optimisation::vertex scale = make_scale(true_scale);
        optimisation::edge edge{ optimisation::edges::inertial_gravity_scale(parameters.gravity, body_from_imu) };
        edge.set_observation(optimisation::edges::inertial::observation(preintegration));
        optimisation::vertex* const vertices[7] = { &first_pose, &first_velocity, &bias, &second_pose, &second_velocity, &direction, &scale };
        for (optimisation::vertex* const vertex : vertices) {
            REQUIRE(edge.add_vertex(vertex));
        }
        edge.compute_residual();
        for (size_t i = 0; i < 9; ++i) {
            REQUIRE(std::abs(edge.get_residual()[i][0]) < 3e-3);
        }

        const math::matrix<double, 3, 1> gravity = optimisation::edges::inertial_gravity_scale::gravity_from_direction(direction.get_parameters(), parameters.gravity);
        REQUIRE(norm(gravity - map_from_gravity * math::matrix<double, 3, 1>({ { 0.0, 0.0, -parameters.gravity } })) < 1e-12);
        const double one = 1.0;
        REQUIRE(scale.set_parameters(&one, 1));
        optimisation::edge plain{ optimisation::edges::inertial(gravity, body_from_imu) };
        plain.set_observation(edge.get_observation());
        for (int i = 0; i < 5; ++i) {
            REQUIRE(plain.add_vertex(vertices[i]));
        }
        plain.compute_residual();
        edge.compute_residual();
        for (size_t i = 0; i < 9; ++i) {
            REQUIRE(std::abs(plain.get_residual()[i][0] - edge.get_residual()[i][0]) < 1e-15);
        }

        const double scale_moved = 2.5;
        REQUIRE(scale.set_parameters(&scale_moved, 1));
        const double pose_moved[6] = { 0.03, -0.05, 0.02, 0.1, 0.05, -0.08 };
        first_pose.plus(&pose_moved[0]);
        const double velocity_moved[3] = { -0.1, 0.2, 0.05 };
        second_velocity.plus(&velocity_moved[0]);
        const double bias_moved[6] = { 0.005, 0.002, -0.004, 0.1, -0.05, 0.02 };
        bias.plus(&bias_moved[0]);
        const double direction_moved[2] = { 0.05, -0.03 };
        direction.plus(&direction_moved[0]);
        edge.compute_residual();
        edge.compute_jacobians();
        const size_t columns[7] = { 6, 3, 6, 6, 3, 2, 1 };
        for (int slot = 0; slot < 7; ++slot) {
            const math::matrix<double, 0, 0>& analytic = edge.get_jacobians()[static_cast<size_t>(slot)];
            REQUIRE(analytic.rows() == 9);
            REQUIRE(analytic.cols() == columns[slot]);
            const math::matrix<double, 0, 0> numeric = numeric_jacobian(edge, slot);
            for (size_t r = 0; r < 9; ++r) {
                for (size_t c = 0; c < analytic.cols(); ++c) {
                    REQUIRE(std::abs(analytic[r][c] - numeric[r][c]) < 1e-6 * (1.0 + std::abs(numeric[r][c])));
                }
            }
        }
    }

    // An inertial only optimisation of the fixed map poses, as a monocular inertial initialisation runs: from scale one, a gravity
    // direction 0.3 rad out, still velocities and zero biases, it finds the scale, gravity, the velocities and the gyroscope bias,
    // the accelerometer bias held at zero by a prior.
    {
        optimisation::factor_graph graph;
        const int keyframe_count = 11;
        const double period = 0.3;
        const double first_time = 2.0;
        std::vector<optimisation::vertex*> poses;
        std::vector<optimisation::vertex*> velocities;
        for (int k = 0; k < keyframe_count; ++k) {
            optimisation::vertex pose = make_pose(map_pose(first_time + period * k));
            pose.set_fixed(true);
            poses.push_back(graph.add_vertex(static_cast<optimisation::vertex&&>(pose)));
            velocities.push_back(graph.add_vertex(make_velocity(math::matrix<double, 3, 1>::zero())));
        }
        optimisation::vertex* const bias = graph.add_vertex(make_bias(sensor::imu::bias()));
        optimisation::vertex* const direction = graph.add_vertex(make_direction(map_from_gravity * math::so3<double>::exp({ { 0.2, -0.22, 0.5 } })));
        optimisation::vertex* const scale = graph.add_vertex(make_scale(1.0));
        for (int k = 0; (k + 1) < keyframe_count; ++k) {
            sensor::imu::preintegration preintegration(parameters);
            REQUIRE(preintegration.integrate(recording.samples, static_cast<long long int>(std::llround(1e9 * (first_time + period * k))), static_cast<long long int>(std::llround(1e9 * (first_time + period * (k + 1))))));
            optimisation::edge edge{ optimisation::edges::inertial_gravity_scale(parameters.gravity, body_from_imu) };
            edge.set_observation(optimisation::edges::inertial::observation(preintegration));
            math::matrix<double, 0, 0> information;
            REQUIRE(optimisation::edges::inertial::information(preintegration, information));
            edge.set_information(information);
            REQUIRE(edge.add_vertex(poses[static_cast<size_t>(k)]));
            REQUIRE(edge.add_vertex(velocities[static_cast<size_t>(k)]));
            REQUIRE(edge.add_vertex(bias));
            REQUIRE(edge.add_vertex(poses[static_cast<size_t>(k + 1)]));
            REQUIRE(edge.add_vertex(velocities[static_cast<size_t>(k + 1)]));
            REQUIRE(edge.add_vertex(direction));
            REQUIRE(edge.add_vertex(scale));
            REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(edge)) != nullptr);
        }
        optimisation::edge prior{ optimisation::edges::imu_bias_prior() };
        prior.set_observation(math::matrix<double, 0, 0>::zero(6, 1));
        prior.set_information(optimisation::edges::imu_bias_prior::information(1.0, 1e-5));
        REQUIRE(prior.add_vertex(bias));
        REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(prior)) != nullptr);

        REQUIRE(graph.solve(100, true) > 0);
        const double scale_estimate = scale->get_parameters()[0];
        const math::matrix<double, 3, 1> gravity = optimisation::edges::inertial_gravity_scale::gravity_from_direction(direction->get_parameters(), parameters.gravity);
        const math::matrix<double, 3, 1> true_gravity = map_from_gravity * math::matrix<double, 3, 1>({ { 0.0, 0.0, -parameters.gravity } });
        const double gravity_angle = std::acos(std::fmin(1.0, (gravity[0] * true_gravity[0] + gravity[1] * true_gravity[1] + gravity[2] * true_gravity[2]) / (parameters.gravity * parameters.gravity)));
        // The accelerometer bias held at zero shows up as a small error of the scale and of gravity.
        REQUIRE(std::abs(scale_estimate / true_scale - 1.0) < 0.04);
        REQUIRE(gravity_angle < 0.02);
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(std::abs(bias->get_parameters()[i] - true_bias.gyroscope[i]) < 2e-3);
            REQUIRE(std::abs(bias->get_parameters()[3 + i]) < 1e-4);
        }
        for (int k = 0; k < keyframe_count; ++k) {
            const math::matrix<double, 3, 1> velocity = { { velocities[static_cast<size_t>(k)]->get_parameters()[0], velocities[static_cast<size_t>(k)]->get_parameters()[1], velocities[static_cast<size_t>(k)]->get_parameters()[2] } };
            REQUIRE(norm(velocity - map_velocity(first_time + period * k)) < 0.05);
        }
    }

    return EXIT_SUCCESS;
}
