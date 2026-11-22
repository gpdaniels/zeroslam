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

#include "optimisation/edges/rig_reprojection.hpp"

#include "math/lie.hpp"
#include "math/math.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/baseline.hpp"
#include "optimisation/edges/reprojection.hpp"
#include "optimisation/edges/reprojection_inverse_depth.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/extrinsic.hpp"
#include "optimisation/vertices/point.hpp"
#include "optimisation/vertices/pose.hpp"
#include "sensor/camera/model.hpp"
#include "sensor/camera/pinhole_radial_tangential.hpp"

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

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-8) {
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

namespace {
    void set_pose(optimisation::vertex& vertex, const math::se3<double>& pose) {
        const double parameters[7] = { pose.translation()[0], pose.translation()[1], pose.translation()[2], pose.rotation().get_quaternion()[1], pose.rotation().get_quaternion()[2], pose.rotation().get_quaternion()[3], pose.rotation().get_quaternion()[0] };
        REQUIRE(vertex.set_parameters(&parameters[0], 7));
    }

    void set_extrinsic(optimisation::vertex& vertex, const math::se3<double>& body_from_camera) {
        double parameters[7];
        optimisation::vertices::extrinsic::to_parameters(body_from_camera, parameters);
        REQUIRE(vertex.set_parameters(&parameters[0], 7));
    }

    // The analytic Jacobians against central differences through each vertex's own update.
    void check_jacobians(optimisation::edge& factor) {
        factor.compute_residual();
        factor.compute_jacobians();
        const std::vector<math::matrix<double, 0, 0>> analytic = factor.get_jacobians();
        const math::matrix<double, 0, 0> residual = factor.get_residual();
        for (size_t slot = 0; slot < factor.num_vertices(); ++slot) {
            optimisation::vertex* const node = factor.get_vertex(static_cast<int>(slot));
            const int dimensions = node->get_local_dimensions();
            REQUIRE(analytic[slot].rows() == 2);
            REQUIRE(analytic[slot].cols() == static_cast<size_t>(dimensions));
            double backup[optimisation::vertex::maximum_parameters] = {};
            REQUIRE(node->get_parameters(&backup[0], static_cast<size_t>(node->get_parameter_dimensions())));
            for (int d = 0; d < dimensions; ++d) {
                const double step = 1e-6;
                double delta[optimisation::vertex::maximum_parameters] = {};
                delta[d] = step;
                node->plus(&delta[0]);
                factor.compute_residual();
                const double plus[2] = { factor.get_residual()[0][0], factor.get_residual()[1][0] };
                REQUIRE(node->set_parameters(&backup[0], static_cast<size_t>(node->get_parameter_dimensions())));
                delta[d] = -step;
                node->plus(&delta[0]);
                factor.compute_residual();
                const double minus[2] = { factor.get_residual()[0][0], factor.get_residual()[1][0] };
                REQUIRE(node->set_parameters(&backup[0], static_cast<size_t>(node->get_parameter_dimensions())));
                for (size_t r = 0; r < 2; ++r) {
                    const double numeric = (plus[r] - minus[r]) / (2.0 * step);
                    REQUIRE(is_value_approx(analytic[slot][r][static_cast<size_t>(d)], numeric, 1e-5));
                }
            }
        }
        factor.compute_residual();
        REQUIRE(factor.get_residual()[0][0] == residual[0][0]);
        REQUIRE(factor.get_residual()[1][0] == residual[1][0]);
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(optimisation::edges::rig_reprojection) <= optimisation::edge::maximum_size, "The rig reprojection edge must fit.");

    const double camera_parameters[12] = { 458.0, 457.0, 367.7, 248.6, -0.28, 0.07, 0.0002, 0.00002, 0.0, 0.0, 0.0, 0.0 };
    const sensor::camera::pinhole_radial_tangential<double> camera_model(&camera_parameters[0], 12);
    const sensor::camera::model<double> camera(camera_model);
    const math::se3<double> pose(math::so3<double>::exp({ { -0.08, 0.12, 0.03 } }), { { -0.5, 0.2, 0.7 } });
    const math::se3<double> body_from_camera(math::so3<double>::exp({ { 0.01, -0.02, 0.015 } }), { { 0.11, -0.003, 0.0007 } });
    const math::matrix<double, 3, 1> world_point({ 1.2, -0.4, 6.0 });
    const math::se3<double> anchor_camera_from_world(math::so3<double>::exp({ { 0.1, -0.05, 0.2 } }), { { 0.4, -0.3, 0.2 } });
    const math::se3<double> anchor = anchor_camera_from_world.inverse();
    const math::matrix<double, 3, 1> anchored = anchor_camera_from_world * world_point;
    const double inverse_landmark[3] = { anchored[0] / anchored[2], anchored[1] / anchored[2], 1.0 / anchored[2] };

    {
        // The camera sees extrinsic^-1 * pose * landmark: a point placed where the camera projects it has no residual.
        const math::matrix<double, 3, 1> point_camera = body_from_camera.inverse() * (pose * world_point);
        double pixel[2];
        REQUIRE(camera_model.project(point_camera.data(), &pixel[0], nullptr, nullptr));
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        set_pose(pose_vertex, pose);
        optimisation::vertex extrinsic_vertex{ optimisation::vertices::extrinsic() };
        set_extrinsic(extrinsic_vertex, body_from_camera);
        for (int kind = 0; kind < 2; ++kind) {
            optimisation::vertex landmark_vertex{ optimisation::vertices::point() };
            REQUIRE(landmark_vertex.set_parameters((kind == 0) ? world_point.data() : &inverse_landmark[0], 3));
            optimisation::edge factor = (kind == 0) ? optimisation::edge{ optimisation::edges::rig_reprojection(camera) } : optimisation::edge{ optimisation::edges::rig_reprojection(camera, anchor.rotation().get_matrix(), anchor.translation()) };
            factor.set_observation(math::matrix<double, 0, 0>(2, 1, &pixel[0]));
            REQUIRE(factor.add_vertex(&pose_vertex));
            REQUIRE(factor.add_vertex(&landmark_vertex));
            REQUIRE(factor.add_vertex(&extrinsic_vertex));
            factor.compute_residual();
            REQUIRE(factor.is_measurable());
            REQUIRE(std::abs(factor.get_residual()[0][0]) < 1e-8);
            REQUIRE(std::abs(factor.get_residual()[1][0]) < 1e-8);
            // Away from the solution too, every block's Jacobian is the derivative of the residual through its vertex.
            const double observed[2] = { pixel[0] + 3.0, pixel[1] - 2.0 };
            factor.set_observation(math::matrix<double, 0, 0>(2, 1, &observed[0]));
            check_jacobians(factor);
        }
    }

    {
        // With the identity extrinsic the edge is the two vertex edge of its landmark.
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        set_pose(pose_vertex, pose);
        optimisation::vertex extrinsic_vertex{ optimisation::vertices::extrinsic() };
        set_extrinsic(extrinsic_vertex, math::se3<double>::identity());
        const double observed[2] = { 300.0, 250.0 };
        for (int kind = 0; kind < 2; ++kind) {
            optimisation::vertex landmark_vertex{ optimisation::vertices::point() };
            REQUIRE(landmark_vertex.set_parameters((kind == 0) ? world_point.data() : &inverse_landmark[0], 3));
            optimisation::edge rig = (kind == 0) ? optimisation::edge{ optimisation::edges::rig_reprojection(camera) } : optimisation::edge{ optimisation::edges::rig_reprojection(camera, anchor.rotation().get_matrix(), anchor.translation()) };
            optimisation::edge plain = (kind == 0) ? optimisation::edge{ optimisation::edges::reprojection(camera) } : optimisation::edge{ optimisation::edges::reprojection_inverse_depth(camera, anchor.rotation().get_matrix(), anchor.translation()) };
            rig.set_observation(math::matrix<double, 0, 0>(2, 1, &observed[0]));
            plain.set_observation(math::matrix<double, 0, 0>(2, 1, &observed[0]));
            rig.add_vertex(&pose_vertex);
            rig.add_vertex(&landmark_vertex);
            rig.add_vertex(&extrinsic_vertex);
            plain.add_vertex(&pose_vertex);
            plain.add_vertex(&landmark_vertex);
            rig.compute_residual();
            plain.compute_residual();
            rig.compute_jacobians();
            plain.compute_jacobians();
            for (size_t r = 0; r < 2; ++r) {
                REQUIRE(is_value_approx(rig.get_residual()[r][0], plain.get_residual()[r][0], 1e-12));
                for (size_t c = 0; c < 6; ++c) {
                    REQUIRE(is_value_approx(rig.get_jacobians()[0][r][c], plain.get_jacobians()[0][r][c], 1e-12));
                    // A step of the extrinsic is the opposite of one of the pose.
                    REQUIRE(is_value_approx(rig.get_jacobians()[2][r][c], -plain.get_jacobians()[0][r][c], 1e-12));
                }
                for (size_t c = 0; c < 3; ++c) {
                    REQUIRE(is_value_approx(rig.get_jacobians()[1][r][c], plain.get_jacobians()[1][r][c], 1e-12));
                }
            }
        }
    }

    {
        // A point behind the camera gives the behind camera residual of the two vertex edges, and is not measurable.
        optimisation::vertex pose_vertex{ optimisation::vertices::pose() };
        set_pose(pose_vertex, math::se3<double>::identity());
        optimisation::vertex extrinsic_vertex{ optimisation::vertices::extrinsic() };
        set_extrinsic(extrinsic_vertex, math::se3<double>(math::so3<double>::identity(), { { 0.0, 0.0, 1.0 } }));
        optimisation::vertex landmark_vertex{ optimisation::vertices::point() };
        const double behind[3] = { 0.1, 0.1, 0.5 };
        REQUIRE(landmark_vertex.set_parameters(&behind[0], 3));
        optimisation::edge factor{ optimisation::edges::rig_reprojection(camera) };
        const double observed[2] = { 300.0, 250.0 };
        factor.set_observation(math::matrix<double, 0, 0>(2, 1, &observed[0]));
        factor.add_vertex(&pose_vertex);
        factor.add_vertex(&landmark_vertex);
        factor.add_vertex(&extrinsic_vertex);
        factor.compute_residual();
        REQUIRE(!factor.is_measurable());
        REQUIRE(is_value_approx(factor.get_residual()[0][0], optimisation::edges::reprojection::behind_camera_residual(-0.5)));
        REQUIRE(factor.get_residual()[0][0] == factor.get_residual()[1][0]);
    }

    {
        // A rig of two cameras recovers the second camera's extrinsic from a perturbed start: the reference camera's
        // observations, through the identity, hold the gauge, both poses are free but the first, and the landmarks are free.
        const sensor::camera::model<double> pinhole(sensor::camera::pinhole_radial_tangential<double>(std::vector<double>{ 400.0, 400.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }.data(), 12));
        const math::se3<double> truth_extrinsic(math::so3<double>::exp({ { 0.0, 0.6, 0.0 } }), { { 0.3, 0.0, 0.05 } });
        std::vector<math::se3<double>> truth_poses;
        for (int i = 0; i < 4; ++i) {
            truth_poses.push_back(math::se3<double>(math::so3<double>::exp({ { 0.02 * i, -0.15 * i, 0.01 * i } }), { { 0.2 * i, -0.05 * i, 0.1 * i } }));
        }
        std::vector<math::matrix<double, 3, 1>> truth_points;
        for (int i = 0; i < 120; ++i) {
            const double x = -4.0 + (8.0 * static_cast<double>((i * 37) % 120) / 120.0);
            const double y = -2.0 + (4.0 * static_cast<double>((i * 53) % 120) / 120.0);
            const double z = 4.0 + (3.0 * static_cast<double>((i * 17) % 120) / 120.0);
            truth_points.push_back(math::matrix<double, 3, 1>({ x + (2.0 * static_cast<double>(i % 2)), y, z }));
        }
        for (const optimisation::factor_graph::strategy strategy : { optimisation::factor_graph::strategy::dense_schur, optimisation::factor_graph::strategy::square_root }) {
            optimisation::factor_graph graph;
            graph.set_strategy(strategy);
            std::vector<optimisation::vertex*> poses;
            for (size_t i = 0; i < truth_poses.size(); ++i) {
                optimisation::vertex node{ optimisation::vertices::pose() };
                const math::se3<double> start = (i == 0) ? truth_poses[i] : (math::se3<double>::exp({ { 0.01, -0.01, 0.005, 0.02, 0.01, -0.02 } }) * truth_poses[i]);
                set_pose(node, start);
                node.set_fixed(i == 0);
                poses.push_back(graph.add_vertex(static_cast<optimisation::vertex&&>(node)));
            }
            optimisation::vertex identity_node{ optimisation::vertices::extrinsic() };
            set_extrinsic(identity_node, math::se3<double>::identity());
            identity_node.set_fixed(true);
            optimisation::vertex* const identity = graph.add_vertex(static_cast<optimisation::vertex&&>(identity_node));
            optimisation::vertex extrinsic_node{ optimisation::vertices::extrinsic() };
            set_extrinsic(extrinsic_node, math::se3<double>::exp({ { 0.03, -0.04, 0.02, 0.02, -0.03, 0.01 } }) * truth_extrinsic);
            optimisation::vertex* const extrinsic = graph.add_vertex(static_cast<optimisation::vertex&&>(extrinsic_node));
            size_t observed_by_second = 0;
            for (size_t p = 0; p < truth_points.size(); ++p) {
                optimisation::vertex node{ optimisation::vertices::point() };
                const math::matrix<double, 3, 1> start = truth_points[p] + math::matrix<double, 3, 1>({ 0.05, -0.03, 0.08 });
                REQUIRE(node.set_parameters(start.data(), 3));
                node.set_marginalised(true);
                optimisation::vertex* const point = graph.add_vertex(static_cast<optimisation::vertex&&>(node));
                for (size_t i = 0; i < truth_poses.size(); ++i) {
                    for (int c = 0; c < 2; ++c) {
                        const math::se3<double>& body_from = (c == 0) ? math::se3<double>::identity() : truth_extrinsic;
                        const math::matrix<double, 3, 1> seen = body_from.inverse() * (truth_poses[i] * truth_points[p]);
                        double pixel[2];
                        if ((seen[2] <= 0.1) || !pinhole.project(seen.data(), &pixel[0]) || (pixel[0] < 0.0) || (pixel[0] > 640.0) || (pixel[1] < 0.0) || (pixel[1] > 480.0)) {
                            continue;
                        }
                        optimisation::edge factor{ optimisation::edges::rig_reprojection(pinhole) };
                        factor.set_observation(math::matrix<double, 0, 0>(2, 1, &pixel[0]));
                        factor.add_vertex(poses[i]);
                        factor.add_vertex(point);
                        factor.add_vertex((c == 0) ? identity : extrinsic);
                        REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(factor)) != nullptr);
                        observed_by_second += static_cast<size_t>(c);
                    }
                }
            }
            REQUIRE(observed_by_second > 100);
            // The map's scale is the rig's baseline, which no reprojection measures once the extrinsic is free, so its length is
            // held: what is recovered is the rotation and the direction of the translation.
            optimisation::edge length{ optimisation::edges::baseline(math::sqrt(truth_extrinsic.translation().get_length_squared())) };
            length.add_vertex(extrinsic);
            length.add_vertex(identity);
            length.set_information(math::matrix<double, 0, 0>::identity(1, 1) * 1.0e6);
            REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(length)) != nullptr);
            graph.solve(50, false);
            REQUIRE(graph.get_current_chi() < 1e-6);
            REQUIRE(graph.get_diagnostics().used_square_root == (strategy == optimisation::factor_graph::strategy::square_root));
            const math::se3<double> recovered = optimisation::vertices::extrinsic::from_parameters(extrinsic->get_parameters());
            const math::matrix<double, 6, 1> error = (recovered * truth_extrinsic.inverse()).log();
            for (size_t i = 0; i < 6; ++i) {
                REQUIRE(std::abs(error[i]) < 1e-6);
            }
        }
    }

    return EXIT_SUCCESS;
}
