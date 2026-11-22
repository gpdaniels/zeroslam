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

#include "optimisation/edges/inertial.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/bias_random_walk.hpp"
#include "optimisation/edges/reprojection.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/extrinsic.hpp"
#include "optimisation/vertices/imu_bias.hpp"
#include "optimisation/vertices/point.hpp"
#include "optimisation/vertices/pose.hpp"
#include "optimisation/vertices/velocity.hpp"
#include "sensor/camera.hpp"
#include "sensor/imu/measurement.hpp"
#include "sensor/imu/preintegration.hpp"
#include "simulation/imu.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
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

static math::se3<double> read_pose(const optimisation::vertex& vertex) {
    const double* const parameters = vertex.get_parameters();
    return math::se3<double>(math::so3<double>(parameters[6], parameters[3], parameters[4], parameters[5]), { { parameters[0], parameters[1], parameters[2] } });
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

// The jacobian of an edge's residual with respect to one of its vertices' updates, by central differences.
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

    static_assert(sizeof(optimisation::edges::inertial) <= optimisation::edge::maximum_size, "The inertial edge must fit.");

    sensor::imu::parameters parameters;
    const simulation::imu_trajectory trajectory = simulation::imu_trajectory::generate(simulation::imu_trajectory::options());
    const math::matrix<double, 3, 1> gravity = trajectory.gravity();
    // The imu sits on the body (a camera) turned and offset.
    const math::se3<double> body_from_imu(math::so3<double>::exp({ { 0.1, -1.5, 0.2 } }), { { 0.05, -0.03, 0.1 } });
    const auto body_pose = [&trajectory, &body_from_imu](const double time) {
        // T_bw = T_bi * T_iw.
        return body_from_imu * trajectory.world_to_imu(time);
    };

    sensor::imu::bias true_bias;
    true_bias.gyroscope = { { 0.01, -0.02, 0.015 } };
    true_bias.accelerometer = { { 0.1, 0.05, -0.08 } };

    // The observation round trips the measurement, and the information inverts the covariance.
    {
        simulation::imu_sensor::options sensor_options;
        const simulation::imu_sensor::recording recording = simulation::imu_sensor::generate(trajectory, parameters, sensor_options);
        sensor::imu::preintegration preintegration(parameters, true_bias);
        REQUIRE(preintegration.integrate(recording.samples, 1000000000ll, 1700000000ll));
        const math::matrix<double, 0, 0> observation = optimisation::edges::inertial::observation(preintegration);
        REQUIRE(observation.rows() == static_cast<size_t>(optimisation::edges::inertial::observation_count));
        const optimisation::edges::inertial::measurement measurement = optimisation::edges::inertial::measurement::from_observation(observation);
        REQUIRE(measurement.delta_time == preintegration.get_delta_time());
        REQUIRE(measurement.delta_rotation == preintegration.get_delta_rotation());
        REQUIRE(measurement.delta_velocity == preintegration.get_delta_velocity());
        REQUIRE(measurement.delta_position == preintegration.get_delta_position());
        REQUIRE(measurement.linearisation.gyroscope == true_bias.gyroscope);
        REQUIRE(measurement.linearisation.accelerometer == true_bias.accelerometer);
        REQUIRE(measurement.rotation_gyroscope == preintegration.get_rotation_gyroscope_jacobian());
        REQUIRE(measurement.velocity_gyroscope == preintegration.get_velocity_gyroscope_jacobian());
        REQUIRE(measurement.velocity_accelerometer == preintegration.get_velocity_accelerometer_jacobian());
        REQUIRE(measurement.position_gyroscope == preintegration.get_position_gyroscope_jacobian());
        REQUIRE(measurement.position_accelerometer == preintegration.get_position_accelerometer_jacobian());

        math::matrix<double, 0, 0> information;
        REQUIRE(optimisation::edges::inertial::information(preintegration, information));
        for (size_t r = 0; r < 9; ++r) {
            for (size_t c = 0; c < 9; ++c) {
                double product = 0.0;
                for (size_t k = 0; k < 9; ++k) {
                    product += information[r][k] * preintegration.get_covariance()[k][c];
                }
                REQUIRE(std::abs(product - ((r == c) ? 1.0 : 0.0)) < 1e-6);
                REQUIRE(information[r][c] == information[c][r]);
            }
        }
        sensor::imu::preintegration empty(parameters);
        REQUIRE(!optimisation::edges::inertial::information(empty, information));
    }

    // The residual at the true states is the discretisation of the integration only, and the analytic jacobians agree with central
    // differences away from the truth and away from the linearisation point of the bias.
    {
        simulation::imu_sensor::options sensor_options;
        sensor_options.noise = false;
        sensor_options.random_walk = false;
        sensor_options.initial_bias = true_bias;
        const simulation::imu_sensor::recording recording = simulation::imu_sensor::generate(trajectory, parameters, sensor_options);
        const double first_time = 2.0;
        const double second_time = 2.6;
        sensor::imu::preintegration preintegration(parameters, true_bias);
        REQUIRE(preintegration.integrate(recording.samples, 2000000000ll, 2600000000ll));

        optimisation::vertex first_pose = make_pose(body_pose(first_time));
        optimisation::vertex first_velocity = make_velocity(trajectory.velocity(first_time));
        optimisation::vertex bias = make_bias(true_bias);
        optimisation::vertex second_pose = make_pose(body_pose(second_time));
        optimisation::vertex second_velocity = make_velocity(trajectory.velocity(second_time));
        optimisation::edge edge{ optimisation::edges::inertial(gravity, body_from_imu) };
        edge.set_observation(optimisation::edges::inertial::observation(preintegration));
        REQUIRE(edge.add_vertex(&first_pose));
        REQUIRE(edge.add_vertex(&first_velocity));
        REQUIRE(edge.add_vertex(&bias));
        REQUIRE(edge.add_vertex(&second_pose));
        REQUIRE(edge.add_vertex(&second_velocity));
        REQUIRE(!edge.add_vertex(&second_velocity));
        REQUIRE(edge.has_analytic_jacobians());
        REQUIRE(edge.is_measurable());
        edge.compute_residual();
        REQUIRE(edge.get_residual().rows() == 9);
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(std::abs(edge.get_residual()[i][0]) < 1e-5);
            REQUIRE(std::abs(edge.get_residual()[3 + i][0]) < 1e-4);
            REQUIRE(std::abs(edge.get_residual()[6 + i][0]) < 1e-4);
        }

        // Away from the truth.
        const double first_moved[6] = { 0.05, -0.1, 0.07, 0.3, -0.2, 0.1 };
        first_pose.plus(&first_moved[0]);
        const double second_moved[6] = { -0.08, 0.04, 0.12, -0.1, 0.25, -0.3 };
        second_pose.plus(&second_moved[0]);
        const double velocity_moved[3] = { 0.2, -0.3, 0.1 };
        first_velocity.plus(&velocity_moved[0]);
        second_velocity.plus(&velocity_moved[0]);
        const double bias_moved[6] = { 0.004, -0.006, 0.003, 0.05, -0.04, 0.06 };
        bias.plus(&bias_moved[0]);
        edge.compute_residual();
        edge.compute_jacobians();
        REQUIRE(edge.get_jacobians().size() == 5);
        const size_t columns[5] = { 6, 3, 6, 6, 3 };
        for (int slot = 0; slot < 5; ++slot) {
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

        // With the imu's place on the body an extrinsic vertex instead, the residual is the same at the same transform, and the
        // jacobians, the extrinsic's among them, agree with central differences with the extrinsic moved off.
        {
            optimisation::vertex extrinsic{ optimisation::vertices::extrinsic() };
            double extrinsic_parameters[optimisation::vertices::extrinsic::parameter_count];
            optimisation::vertices::extrinsic::to_parameters(body_from_imu, extrinsic_parameters);
            REQUIRE(extrinsic.set_parameters(&extrinsic_parameters[0], optimisation::vertices::extrinsic::parameter_count));
            optimisation::edge with_extrinsic{ optimisation::edges::inertial(gravity, math::se3<double>::identity(), true) };
            REQUIRE(with_extrinsic.required_vertices() == 6);
            with_extrinsic.set_observation(edge.get_observation());
            optimisation::vertex* const nodes[6] = { &first_pose, &first_velocity, &bias, &second_pose, &second_velocity, &extrinsic };
            for (optimisation::vertex* const node : nodes) {
                REQUIRE(with_extrinsic.add_vertex(node));
            }
            with_extrinsic.compute_residual();
            for (size_t i = 0; i < 9; ++i) {
                REQUIRE(std::abs(with_extrinsic.get_residual()[i][0] - edge.get_residual()[i][0]) < 1e-12);
            }
            const double extrinsic_moved[6] = { 0.02, -0.03, 0.01, 0.01, 0.02, -0.015 };
            extrinsic.plus(&extrinsic_moved[0]);
            with_extrinsic.compute_residual();
            with_extrinsic.compute_jacobians();
            for (int slot = 0; slot < 6; ++slot) {
                const math::matrix<double, 0, 0>& analytic = with_extrinsic.get_jacobians()[static_cast<size_t>(slot)];
                const math::matrix<double, 0, 0> numeric = numeric_jacobian(with_extrinsic, slot);
                REQUIRE(analytic.cols() == numeric.cols());
                for (size_t r = 0; r < 9; ++r) {
                    for (size_t c = 0; c < analytic.cols(); ++c) {
                        REQUIRE(std::abs(analytic[r][c] - numeric[r][c]) < 1e-6 * (1.0 + std::abs(numeric[r][c])));
                    }
                }
            }
        }

        // The state jacobians of the evaluation, gravity among them, by central differences too.
        const optimisation::edges::inertial::measurement measurement = optimisation::edges::inertial::measurement::from_observation(edge.get_observation());
        sensor::imu::preintegration::state first = optimisation::edges::inertial::imu_state(first_pose.get_parameters(), first_velocity.get_parameters(), body_from_imu);
        sensor::imu::preintegration::state second = optimisation::edges::inertial::imu_state(second_pose.get_parameters(), second_velocity.get_parameters(), body_from_imu);
        sensor::imu::bias estimate;
        estimate.gyroscope = { { bias.get_parameters()[0], bias.get_parameters()[1], bias.get_parameters()[2] } };
        estimate.accelerometer = { { bias.get_parameters()[3], bias.get_parameters()[4], bias.get_parameters()[5] } };
        double residual[9];
        optimisation::edges::inertial::state_jacobians blocks;
        optimisation::edges::inertial::evaluate(measurement, first, second, estimate, gravity, &residual[0], &blocks);
        for (size_t i = 0; i < 9; ++i) {
            REQUIRE(std::abs(residual[i] - edge.get_residual()[i][0]) < 1e-15);
        }
        const double step = 1e-6;
        for (size_t c = 0; c < 3; ++c) {
            math::matrix<double, 3, 1> direction = math::matrix<double, 3, 1>::zero();
            direction[c] = step;
            double forward[9];
            double backward[9];
            optimisation::edges::inertial::evaluate(measurement, first, second, estimate, gravity + direction, &forward[0], nullptr);
            optimisation::edges::inertial::evaluate(measurement, first, second, estimate, gravity - direction, &backward[0], nullptr);
            for (size_t r = 0; r < 9; ++r) {
                REQUIRE(std::abs(blocks.gravity[r][c] - (forward[r] - backward[r]) / (2.0 * step)) < 1e-6);
            }
            sensor::imu::preintegration::state turned = first;
            turned.rotation = first.rotation * math::so3<double>::exp(direction);
            optimisation::edges::inertial::evaluate(measurement, turned, second, estimate, gravity, &forward[0], nullptr);
            turned.rotation = first.rotation * math::so3<double>::exp(-1.0 * direction);
            optimisation::edges::inertial::evaluate(measurement, turned, second, estimate, gravity, &backward[0], nullptr);
            for (size_t r = 0; r < 9; ++r) {
                REQUIRE(std::abs(blocks.rotation_i[r][c] - (forward[r] - backward[r]) / (2.0 * step)) < 1e-6);
            }
            sensor::imu::preintegration::state moved = second;
            moved.position = second.position + direction;
            optimisation::edges::inertial::evaluate(measurement, first, moved, estimate, gravity, &forward[0], nullptr);
            moved.position = second.position - direction;
            optimisation::edges::inertial::evaluate(measurement, first, moved, estimate, gravity, &backward[0], nullptr);
            for (size_t r = 0; r < 9; ++r) {
                REQUIRE(std::abs(blocks.position_j[r][c] - (forward[r] - backward[r]) / (2.0 * step)) < 1e-6);
            }
        }
    }

    // A visual inertial bundle adjustment of a few keyframes and landmarks: a visual estimate of the right shape at the wrong scale,
    // with the velocities and the biases unknown, is brought to the metric trajectory and the biases by the imu, with gravity known
    // and the first pose fixed. Every solver strategy and precision gives the same result, the graph holding vertices other than
    // poses and landmarks taking the dense path.
    {
        const int keyframe_count = 9;
        const double keyframe_period = 0.5;
        const double first_time = 1.0;
        parameters.gyroscope_random_walk *= 0.01;
        parameters.accelerometer_random_walk *= 0.01;
        simulation::imu_sensor::options sensor_options;
        sensor_options.initial_bias = true_bias;
        const simulation::imu_sensor::recording recording = simulation::imu_sensor::generate(trajectory, parameters, sensor_options);

        const double camera_parameters[4] = { 400.0, 400.0, 320.0, 240.0 };
        const sensor::pinhole camera_model(&camera_parameters[0], 4);
        simulation::random_stream random(5);
        std::vector<math::se3<double>> poses;
        for (int k = 0; k < keyframe_count; ++k) {
            poses.push_back(body_pose(first_time + (keyframe_period * k)));
        }
        std::vector<math::matrix<double, 3, 1>> points;
        for (int k = 0; k < keyframe_count; ++k) {
            for (int p = 0; p < 12; ++p) {
                const double depth = random.uniform(3.0, 8.0);
                const math::matrix<double, 3, 1> in_camera = { { depth * random.uniform(-0.6, 0.6), depth * random.uniform(-0.45, 0.45), depth } };
                points.push_back(poses[static_cast<size_t>(k)].inverse() * in_camera);
            }
        }

        // The visual estimate: the true shape scaled by 0.6 about the first keyframe.
        const double wrong_scale = 0.6;
        const math::matrix<double, 3, 1> anchor = poses[0].inverse().translation();
        const auto shrink = [&anchor, wrong_scale](const math::matrix<double, 3, 1>& point) {
            return anchor + wrong_scale * (point - anchor);
        };

        math::se3<double> solutions[3][keyframe_count];
        double bias_solutions[3][6] = {};
        for (int strategy = 0; strategy < 3; ++strategy) {
            optimisation::factor_graph graph;
            if (strategy == 1) {
                graph.set_strategy(optimisation::factor_graph::strategy::square_root);
            }
            if (strategy == 2) {
                graph.set_strategy(optimisation::factor_graph::strategy::automatic);
                graph.set_square_root_parameter_threshold(1);
                graph.set_precision(optimisation::factor_graph::precision::single_precision);
            }
            std::vector<optimisation::vertex*> pose_vertices;
            std::vector<optimisation::vertex*> velocity_vertices;
            std::vector<optimisation::vertex*> bias_vertices;
            for (int k = 0; k < keyframe_count; ++k) {
                const math::se3<double> world_from_body = poses[static_cast<size_t>(k)].inverse();
                const math::se3<double> estimate = math::se3<double>(world_from_body.rotation(), shrink(world_from_body.translation())).inverse();
                optimisation::vertex pose = make_pose(estimate);
                pose.set_fixed(k == 0);
                pose_vertices.push_back(graph.add_vertex(static_cast<optimisation::vertex&&>(pose)));
                velocity_vertices.push_back(graph.add_vertex(make_velocity(math::matrix<double, 3, 1>::zero())));
                bias_vertices.push_back(graph.add_vertex(make_bias(sensor::imu::bias())));
            }
            for (int k = 0; (k + 1) < keyframe_count; ++k) {
                const long long int start = static_cast<long long int>(std::llround(1e9 * (first_time + keyframe_period * k)));
                const long long int end = static_cast<long long int>(std::llround(1e9 * (first_time + keyframe_period * (k + 1))));
                sensor::imu::preintegration preintegration(parameters);
                REQUIRE(preintegration.integrate(recording.samples, start, end));
                optimisation::edge inertial{ optimisation::edges::inertial(gravity, body_from_imu) };
                inertial.set_observation(optimisation::edges::inertial::observation(preintegration));
                math::matrix<double, 0, 0> information;
                REQUIRE(optimisation::edges::inertial::information(preintegration, information));
                inertial.set_information(information);
                REQUIRE(inertial.add_vertex(pose_vertices[static_cast<size_t>(k)]));
                REQUIRE(inertial.add_vertex(velocity_vertices[static_cast<size_t>(k)]));
                REQUIRE(inertial.add_vertex(bias_vertices[static_cast<size_t>(k)]));
                REQUIRE(inertial.add_vertex(pose_vertices[static_cast<size_t>(k + 1)]));
                REQUIRE(inertial.add_vertex(velocity_vertices[static_cast<size_t>(k + 1)]));
                REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(inertial)) != nullptr);

                optimisation::edge walk{ optimisation::edges::bias_random_walk() };
                walk.set_information(optimisation::edges::bias_random_walk::information(parameters, preintegration.get_delta_time()));
                REQUIRE(walk.add_vertex(bias_vertices[static_cast<size_t>(k)]));
                REQUIRE(walk.add_vertex(bias_vertices[static_cast<size_t>(k + 1)]));
                REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(walk)) != nullptr);
            }
            simulation::random_stream pixel_noise(9);
            int observation_count = 0;
            for (const math::matrix<double, 3, 1>& point : points) {
                optimisation::vertex landmark{ optimisation::vertices::point() };
                const math::matrix<double, 3, 1> estimate = shrink(point);
                REQUIRE(landmark.set_parameters(estimate.data(), 3));
                landmark.set_marginalised(true);
                optimisation::vertex* const landmark_vertex = graph.add_vertex(static_cast<optimisation::vertex&&>(landmark));
                for (int k = 0; k < keyframe_count; ++k) {
                    const math::matrix<double, 3, 1> in_camera = poses[static_cast<size_t>(k)] * point;
                    math::matrix<double, 2, 1> pixel;
                    if (!camera_model.project(in_camera.data(), pixel.data()) || (pixel[0] < 0.0) || (pixel[0] > 640.0) || (pixel[1] < 0.0) || (pixel[1] > 480.0)) {
                        continue;
                    }
                    const double observed[2] = { pixel[0] + 0.5 * pixel_noise.gaussian(), pixel[1] + 0.5 * pixel_noise.gaussian() };
                    optimisation::edge reprojection{ optimisation::edges::reprojection(sensor::camera::model<double>(camera_model)) };
                    reprojection.set_observation(math::matrix<double, 0, 0>(2, 1, &observed[0]));
                    reprojection.set_information(math::matrix<double, 0, 0>::identity(2, 2) * 4.0);
                    REQUIRE(reprojection.add_vertex(pose_vertices[static_cast<size_t>(k)]));
                    REQUIRE(reprojection.add_vertex(landmark_vertex));
                    REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(reprojection)) != nullptr);
                    ++observation_count;
                }
            }
            REQUIRE(observation_count > 200);

            REQUIRE(graph.solve(100, true) > 0);
            REQUIRE(!graph.get_diagnostics().used_square_root);
            REQUIRE(!graph.get_diagnostics().used_single_precision);
            for (int k = 0; k < keyframe_count; ++k) {
                solutions[strategy][k] = read_pose(*pose_vertices[static_cast<size_t>(k)]);
            }
            for (size_t i = 0; i < 6; ++i) {
                bias_solutions[strategy][i] = bias_vertices[0]->get_parameters()[i];
            }
        }

        double largest_position_error = 0.0;
        for (int k = 0; k < keyframe_count; ++k) {
            largest_position_error = std::fmax(largest_position_error, norm(solutions[0][k].inverse().translation() - poses[static_cast<size_t>(k)].inverse().translation()));
            for (int strategy = 1; strategy < 3; ++strategy) {
                REQUIRE(solutions[strategy][k] == solutions[0][k]);
            }
        }
        const double true_length = norm(poses.back().inverse().translation() - poses.front().inverse().translation());
        const double length = norm(solutions[0][keyframe_count - 1].inverse().translation() - solutions[0][0].inverse().translation());
        REQUIRE(std::abs(length / true_length - 1.0) < 0.01);
        REQUIRE(largest_position_error < 0.03);
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(std::abs(bias_solutions[0][i] - true_bias.gyroscope[i]) < 1e-3);
            REQUIRE(std::abs(bias_solutions[0][3 + i] - true_bias.accelerometer[i]) < 0.03);
            for (int strategy = 1; strategy < 3; ++strategy) {
                REQUIRE(bias_solutions[strategy][i] == bias_solutions[0][i]);
            }
        }
    }

    return EXIT_SUCCESS;
}
