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

#include "optimisation/inertial_initialisation.hpp"

#include "math/math.hpp"
#include "math/matrix_decomposition_cholesky.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/inertial.hpp"
#include "optimisation/edges/inertial_gravity_scale.hpp"
#include "optimisation/edges/prior.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/gravity_direction.hpp"
#include "optimisation/vertices/imu_bias.hpp"
#include "optimisation/vertices/pose.hpp"
#include "optimisation/vertices/scale.hpp"
#include "optimisation/vertices/velocity.hpp"

namespace {
    // Solves a dense symmetric positive definite system, false when it is not positive definite.
    bool solve_dense(const std::vector<double>& matrix, const std::vector<double>& right_hand_side, const int size, std::vector<double>& lower, std::vector<double>& solution) {
        lower.assign(static_cast<size_t>(size * size), 0.0);
        solution.assign(static_cast<size_t>(size), 0.0);
        if (!math::decompose_cholesky(matrix.data(), size, size, lower.data())) {
            return false;
        }
        return math::solve_cholesky(lower.data(), right_hand_side.data(), size, size, solution.data());
    }

    template <size_t rows, size_t columns>
    math::matrix<double, rows, columns> block_of(const math::matrix<double, 9, 9>& value, const size_t row, const size_t column) {
        return math::get_block<double, rows, columns>(value, row, column);
    }

    // The gyroscope bias that best explains the keyframes' relative rotations, with a prior holding it near zero.
    math::matrix<double, 3, 1> estimate_gyroscope_bias(const std::vector<math::so3<double>>& rotations, const std::vector<sensor::imu::preintegration>& preintegrations, const math::matrix<double, 3, 1>& start, const double deviation) {
        math::matrix<double, 3, 1> estimate = start;
        const double prior = 1.0 / (deviation * deviation);
        for (int iteration = 0; iteration < 10; ++iteration) {
            math::matrix<double, 3, 3> hessian = prior * math::matrix<double, 3, 3>::identity();
            math::matrix<double, 3, 1> gradient = prior * estimate;
            for (size_t k = 0; k < preintegrations.size(); ++k) {
                const sensor::imu::preintegration& preintegration = preintegrations[k];
                math::matrix<double, 3, 3> weight;
                if (!math::invert(block_of<3, 3>(preintegration.get_covariance(), 0, 0), weight)) {
                    continue;
                }
                const math::matrix<double, 3, 1> correction = preintegration.get_rotation_gyroscope_jacobian() * (estimate - preintegration.get_linearisation().gyroscope);
                const math::so3<double> delta = (preintegration.get_delta_rotation() * math::so3<double>::exp(correction)).normalised();
                const math::so3<double> error = (delta.inverse() * rotations[k].inverse() * rotations[k + 1]).normalised();
                const math::matrix<double, 3, 1> residual = error.log();
                const math::matrix<double, 3, 3> jacobian = -1.0 * (math::so3<double>::left_jacobian_inverse(-1.0 * residual) * math::transpose(error.get_matrix()) * math::so3<double>::left_jacobian(-1.0 * correction) * preintegration.get_rotation_gyroscope_jacobian());
                const math::matrix<double, 3, 3> weighted = math::transpose(jacobian) * weight;
                hessian = hessian + (weighted * jacobian);
                gradient = gradient + (weighted * residual);
            }
            math::matrix<double, 3, 3> inverse;
            if (!math::invert(hessian, inverse)) {
                break;
            }
            const math::matrix<double, 3, 1> step = -1.0 * (inverse * gradient);
            estimate = estimate + step;
            if (step.get_length_squared() < 1e-24) {
                break;
            }
        }
        return estimate;
    }

    optimisation::vertex make_vertex_pose(const math::se3<double>& pose) {
        optimisation::vertex vertex{ optimisation::vertices::pose() };
        const double parameters[7] = { pose.translation()[0], pose.translation()[1], pose.translation()[2], pose.rotation().get_quaternion()[1], pose.rotation().get_quaternion()[2], pose.rotation().get_quaternion()[3], pose.rotation().get_quaternion()[0] };
        static_cast<void>(vertex.set_parameters(&parameters[0], 7));
        vertex.set_fixed(true);
        return vertex;
    }

    optimisation::vertex make_vertex_velocity(const math::matrix<double, 3, 1>& velocity) {
        optimisation::vertex vertex{ optimisation::vertices::velocity() };
        static_cast<void>(vertex.set_parameters(velocity.data(), 3));
        return vertex;
    }

    optimisation::vertex make_vertex_bias(const sensor::imu::bias& bias) {
        optimisation::vertex vertex{ optimisation::vertices::imu_bias() };
        const double parameters[6] = { bias.gyroscope[0], bias.gyroscope[1], bias.gyroscope[2], bias.accelerometer[0], bias.accelerometer[1], bias.accelerometer[2] };
        static_cast<void>(vertex.set_parameters(&parameters[0], 6));
        return vertex;
    }

    optimisation::vertex make_vertex_direction(const math::so3<double>& rotation) {
        optimisation::vertex vertex{ optimisation::vertices::gravity_direction() };
        const double parameters[4] = { rotation.get_quaternion()[1], rotation.get_quaternion()[2], rotation.get_quaternion()[3], rotation.get_quaternion()[0] };
        static_cast<void>(vertex.set_parameters(&parameters[0], 4));
        return vertex;
    }

    optimisation::vertex make_vertex_scale(const double scale) {
        optimisation::vertex vertex{ optimisation::vertices::scale() };
        static_cast<void>(vertex.set_parameters(&scale, 1));
        return vertex;
    }
}

namespace optimisation {
    math::so3<double> inertial_initialisation::rotation_from_down(const math::matrix<double, 3, 1>& direction) {
        const double length = math::sqrt(direction.get_length_squared());
        if (!(length > 0.0)) {
            return math::so3<double>::identity();
        }
        const math::matrix<double, 3, 1> unit = (1.0 / length) * direction;
        // The axis (0, 0, -1) x unit and the angle between them.
        const math::matrix<double, 3, 1> axis = { { unit[1], -unit[0], 0.0 } };
        const double sine = math::sqrt(axis.get_length_squared());
        const double cosine = -unit[2];
        if (sine < 1e-12) {
            return (cosine > 0.0) ? math::so3<double>::identity() : math::so3<double>::exp({ { math::pi<double>(), 0.0, 0.0 } });
        }
        return math::so3<double>::exp((math::atan2(sine, cosine) / sine) * axis);
    }

    inertial_initialisation::result inertial_initialisation::estimate(const std::vector<math::se3<double>>& poses, const std::vector<sensor::imu::preintegration>& preintegrations, const options& settings) {
        result outcome;
        const size_t count = poses.size();
        if ((count < 4) || (preintegrations.size() + 1 != count)) {
            return outcome;
        }
        for (const sensor::imu::preintegration& preintegration : preintegrations) {
            if (!(preintegration.get_delta_time() > 0.0)) {
                return outcome;
            }
        }
        const double gravity_magnitude = preintegrations[0].get_parameters().gravity;
        const math::se3<double>& body_from_imu = settings.body_from_imu;

        // The bodies in the world, and the imu's rotation at each.
        std::vector<math::se3<double>> bodies;
        std::vector<math::so3<double>> rotations;
        for (const math::se3<double>& pose : poses) {
            bodies.push_back(pose.inverse());
            rotations.push_back((bodies.back().rotation() * body_from_imu.rotation()).normalised());
        }

        // 1. The gyroscope bias, then every preintegration integrated at it.
        std::vector<sensor::imu::preintegration> integrated = preintegrations;
        sensor::imu::bias bias = integrated[0].get_linearisation();
        bias.gyroscope = estimate_gyroscope_bias(rotations, integrated, bias.gyroscope, settings.gyroscope_bias_deviation);
        for (sensor::imu::preintegration& preintegration : integrated) {
            preintegration.reintegrate(bias);
        }

        // 2. The velocities, gravity and the scale, x = [v_0 ... v_n-1, g, s], from the linear equations of each preintegration in
        // the world: v_j - v_i - g dt = R_i dv and s (p_j - p_i) - v_i dt - g dt^2 / 2 = R_i dp - (R_wb_j - R_wb_i) p_bi, weighted
        // by the inverse covariance of dv and dp turned into the world.
        const int unknowns = static_cast<int>(3 * count) + 4;
        std::vector<double> normal(static_cast<size_t>(unknowns * unknowns), 0.0);
        std::vector<double> right_hand_side(static_cast<size_t>(unknowns), 0.0);
        for (size_t k = 0; k + 1 < count; ++k) {
            const sensor::imu::preintegration& preintegration = integrated[k];
            const double duration = preintegration.get_delta_time();
            const math::matrix<double, 3, 3> rotation = rotations[k].get_matrix();
            math::matrix<double, 6, 6> imu_weight;
            if (!math::invert(block_of<6, 6>(preintegration.get_covariance(), 3, 3), imu_weight)) {
                continue;
            }
            math::matrix<double, 6, 6> turn = math::matrix<double, 6, 6>::zero();
            math::set_block<double, 3, 3>(turn, 0, 0, rotation);
            math::set_block<double, 3, 3>(turn, 3, 3, rotation);
            const math::matrix<double, 6, 6> weight = turn * imu_weight * math::transpose(turn);

            // The coefficients of v_i, v_j, g and s, and the right hand side.
            math::matrix<double, 6, 10> coefficients = math::matrix<double, 6, 10>::zero();
            const math::matrix<double, 3, 1> displacement = bodies[k + 1].translation() - bodies[k].translation();
            const math::matrix<double, 3, 1> lever = (bodies[k + 1].rotation() * body_from_imu.translation()) - (bodies[k].rotation() * body_from_imu.translation());
            const math::matrix<double, 3, 1> velocity_target = rotation * preintegration.get_delta_velocity();
            const math::matrix<double, 3, 1> position_target = (rotation * preintegration.get_delta_position()) - lever;
            math::matrix<double, 6, 1> target;
            for (size_t i = 0; i < 3; ++i) {
                coefficients[i][i] = -1.0;
                coefficients[i][3 + i] = 1.0;
                coefficients[i][6 + i] = -duration;
                coefficients[3 + i][i] = -duration;
                coefficients[3 + i][6 + i] = -0.5 * duration * duration;
                coefficients[3 + i][9] = displacement[i];
                target[i] = velocity_target[i];
                target[3 + i] = position_target[i];
            }
            const size_t columns[10] = { 3 * k, (3 * k) + 1, (3 * k) + 2, 3 * (k + 1), (3 * (k + 1)) + 1, (3 * (k + 1)) + 2, 3 * count, (3 * count) + 1, (3 * count) + 2, (3 * count) + 3 };
            const math::matrix<double, 10, 6> weighted = math::transpose(coefficients) * weight;
            const math::matrix<double, 10, 10> block = weighted * coefficients;
            const math::matrix<double, 10, 1> projected = weighted * target;
            for (size_t a = 0; a < 10; ++a) {
                for (size_t b = 0; b < 10; ++b) {
                    normal[(columns[a] * static_cast<size_t>(unknowns)) + columns[b]] += block[a][b];
                }
                right_hand_side[columns[a]] += projected[a];
            }
        }
        std::vector<double> lower;
        std::vector<double> solution;
        std::vector<math::matrix<double, 3, 1>> velocities(count, math::matrix<double, 3, 1>::zero());
        math::so3<double> direction = math::so3<double>::identity();
        double scale = 1.0;
        const bool linear = solve_dense(normal, right_hand_side, unknowns, lower, solution);
        const math::matrix<double, 3, 1> linear_gravity = linear ? math::matrix<double, 3, 1>({ { solution[3 * count], solution[(3 * count) + 1], solution[(3 * count) + 2] } }) : math::matrix<double, 3, 1>::zero();
        const double linear_scale = linear ? solution[(3 * count) + 3] : 0.0;
        if (linear && (linear_scale > 0.0) && math::isfinite(linear_scale) && (linear_gravity.get_length_squared() > 0.0)) {
            outcome.linear_scale = linear_scale;
            scale = settings.fixed_scale ? 1.0 : linear_scale;
            direction = inertial_initialisation::rotation_from_down(linear_gravity);
            for (size_t k = 0; k < count; ++k) {
                velocities[k] = { { solution[3 * k], solution[(3 * k) + 1], solution[(3 * k) + 2] } };
            }
        }
        else {
            // ORB-SLAM3's start: gravity opposite the sum of the velocity deltas turned into the world, the velocities from the
            // visual displacements at scale one.
            math::matrix<double, 3, 1> down = math::matrix<double, 3, 1>::zero();
            for (size_t k = 0; k + 1 < count; ++k) {
                down = down - (rotations[k] * integrated[k].get_delta_velocity());
                velocities[k] = (1.0 / integrated[k].get_delta_time()) * (bodies[k + 1].translation() - bodies[k].translation());
            }
            velocities[count - 1] = velocities[count - 2];
            direction = inertial_initialisation::rotation_from_down(down);
        }

        // 3. The inertial only optimisation, again after integrating at the new bias while the bias moves past the thresholds.
        bool solved = false;
        for (int round = 0; round <= settings.reintegrations; ++round) {
            optimisation::factor_graph graph;
            std::vector<optimisation::vertex*> pose_vertices;
            std::vector<optimisation::vertex*> velocity_vertices;
            for (size_t k = 0; k < count; ++k) {
                pose_vertices.push_back(graph.add_vertex(make_vertex_pose(poses[k])));
                velocity_vertices.push_back(graph.add_vertex(make_vertex_velocity(velocities[k])));
            }
            optimisation::vertex* const bias_vertex = graph.add_vertex(make_vertex_bias(bias));
            optimisation::vertex* const direction_vertex = graph.add_vertex(make_vertex_direction(direction));
            optimisation::vertex scale_node = make_vertex_scale(scale);
            scale_node.set_fixed(settings.fixed_scale);
            optimisation::vertex* const scale_vertex = graph.add_vertex(static_cast<optimisation::vertex&&>(scale_node));
            std::vector<optimisation::edge*> inertial_edges;
            for (size_t k = 0; k + 1 < count; ++k) {
                optimisation::edge factor{ optimisation::edges::inertial_gravity_scale(gravity_magnitude, body_from_imu) };
                factor.set_observation(optimisation::edges::inertial::observation(integrated[k]));
                math::matrix<double, 0, 0> information;
                if (!optimisation::edges::inertial::information(integrated[k], information)) {
                    return outcome;
                }
                factor.set_information(information);
                static_cast<void>(factor.add_vertex(pose_vertices[k]));
                static_cast<void>(factor.add_vertex(velocity_vertices[k]));
                static_cast<void>(factor.add_vertex(bias_vertex));
                static_cast<void>(factor.add_vertex(pose_vertices[k + 1]));
                static_cast<void>(factor.add_vertex(velocity_vertices[k + 1]));
                static_cast<void>(factor.add_vertex(direction_vertex));
                static_cast<void>(factor.add_vertex(scale_vertex));
                inertial_edges.push_back(graph.add_edge(static_cast<optimisation::edge&&>(factor)));
            }
            optimisation::edge prior{ optimisation::edges::imu_bias_prior() };
            prior.set_observation(math::matrix<double, 0, 0>::zero(6, 1));
            const math::matrix<double, 0, 0> prior_information = optimisation::edges::imu_bias_prior::information(settings.gyroscope_bias_deviation, settings.accelerometer_bias_deviation);
            prior.set_information(prior_information);
            static_cast<void>(prior.add_vertex(bias_vertex));
            static_cast<void>(graph.add_edge(static_cast<optimisation::edge&&>(prior)));

            static_cast<void>(graph.solve(settings.iterations, true));
            outcome.cost = graph.get_current_chi();

            for (size_t k = 0; k < count; ++k) {
                const double* const values = velocity_vertices[k]->get_parameters();
                velocities[k] = { { values[0], values[1], values[2] } };
            }
            const double* const bias_values = bias_vertex->get_parameters();
            bias.gyroscope = { { bias_values[0], bias_values[1], bias_values[2] } };
            bias.accelerometer = { { bias_values[3], bias_values[4], bias_values[5] } };
            const double* const direction_values = direction_vertex->get_parameters();
            direction = math::so3<double>(direction_values[3], direction_values[0], direction_values[1], direction_values[2]).normalised();
            scale = scale_vertex->get_parameters()[0];
            solved = (scale > 0.0) && math::isfinite(scale);

            // The information of the free variables [v_0 ... v_n-1, b, gravity's direction, log s] at the estimate, from which the
            // deviations of the last three with the others marginalised.
            const int size = static_cast<int>(3 * count) + 9;
            const size_t bias_offset = 3 * count;
            const size_t direction_offset = bias_offset + 6;
            const size_t scale_offset = direction_offset + 2;
            std::vector<double> information(static_cast<size_t>(size * size), 0.0);
            for (size_t k = 0; k < inertial_edges.size(); ++k) {
                optimisation::edge* const factor = inertial_edges[k];
                factor->compute_residual();
                factor->compute_jacobians();
                const std::vector<math::matrix<double, 0, 0>>& jacobians = factor->get_jacobians();
                const math::matrix<double, 0, 0>& weight = factor->get_information();
                const int slots[5] = { 1, 2, 4, 5, 6 };
                const size_t offsets[5] = { 3 * k, bias_offset, 3 * (k + 1), direction_offset, scale_offset };
                for (size_t a = 0; a < 5; ++a) {
                    const math::matrix<double, 0, 0> weighted = math::transpose(jacobians[static_cast<size_t>(slots[a])]) * weight;
                    for (size_t b = 0; b < 5; ++b) {
                        const math::matrix<double, 0, 0> product = weighted * jacobians[static_cast<size_t>(slots[b])];
                        for (size_t r = 0; r < product.rows(); ++r) {
                            for (size_t c = 0; c < product.cols(); ++c) {
                                information[((offsets[a] + r) * static_cast<size_t>(size)) + offsets[b] + c] += product[r][c];
                            }
                        }
                    }
                }
            }
            for (size_t i = 0; i < 6; ++i) {
                information[((bias_offset + i) * static_cast<size_t>(size)) + bias_offset + i] += prior_information[i][i];
            }
            if (settings.fixed_scale) {
                // A held scale is known: its row stands alone.
                for (size_t i = 0; i < static_cast<size_t>(size); ++i) {
                    information[(scale_offset * static_cast<size_t>(size)) + i] = 0.0;
                    information[(i * static_cast<size_t>(size)) + scale_offset] = 0.0;
                }
                information[(scale_offset * static_cast<size_t>(size)) + scale_offset] = 1.0;
            }
            outcome.scale_deviation = math::inf<double>();
            outcome.gravity_deviation = math::inf<double>();
            std::vector<double> factor_lower(static_cast<size_t>(size * size), 0.0);
            if (math::decompose_cholesky(information.data(), size, size, factor_lower.data())) {
                double covariance[3][3] = {};
                bool inverted = true;
                for (size_t column = 0; column < 3; ++column) {
                    std::vector<double> unit(static_cast<size_t>(size), 0.0);
                    std::vector<double> column_values(static_cast<size_t>(size), 0.0);
                    unit[direction_offset + column] = 1.0;
                    inverted = inverted && math::solve_cholesky(factor_lower.data(), unit.data(), size, size, column_values.data());
                    for (size_t row = 0; row < 3; ++row) {
                        covariance[row][column] = column_values[direction_offset + row];
                    }
                }
                if (inverted) {
                    outcome.scale_deviation = settings.fixed_scale ? 0.0 : math::sqrt(math::max(covariance[2][2], 0.0));
                    // The larger eigenvalue of the direction's 2x2 block.
                    const double mean = 0.5 * (covariance[0][0] + covariance[1][1]);
                    const double difference = 0.5 * (covariance[0][0] - covariance[1][1]);
                    const double largest = mean + math::sqrt((difference * difference) + (covariance[0][1] * covariance[1][0]));
                    outcome.gravity_deviation = math::sqrt(math::max(largest, 0.0));
                }
            }

            if (!solved) {
                break;
            }
            bool moved = false;
            for (const sensor::imu::preintegration& preintegration : integrated) {
                moved = moved || preintegration.requires_reintegration(bias);
            }
            if (!moved) {
                break;
            }
            for (sensor::imu::preintegration& preintegration : integrated) {
                preintegration.reintegrate(bias);
            }
        }

        for (sensor::imu::preintegration& preintegration : integrated) {
            if ((preintegration.get_linearisation().gyroscope != bias.gyroscope) || (preintegration.get_linearisation().accelerometer != bias.accelerometer)) {
                preintegration.reintegrate(bias);
            }
        }
        outcome.success = solved;
        outcome.scale = scale;
        outcome.world_from_gravity = direction;
        outcome.gravity = direction * math::matrix<double, 3, 1>({ { 0.0, 0.0, -gravity_magnitude } });
        outcome.bias = bias;
        outcome.velocities = velocities;
        outcome.preintegrations = integrated;
        outcome.observable = solved && (outcome.scale_deviation < settings.maximum_scale_deviation) && (outcome.gravity_deviation < settings.maximum_gravity_deviation);
        return outcome;
    }
}
