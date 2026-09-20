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

#include "optimisation/factor_graph.hpp"

#include "core/random_pcg.hpp"
#include "geometry/plucker.hpp"
#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/baseline.hpp"
#include "optimisation/edges/line_reprojection.hpp"
#include "optimisation/edges/relative_similarity.hpp"
#include "optimisation/edges/reprojection.hpp"
#include "optimisation/loss.hpp"
#include "optimisation/losses/huber.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/line.hpp"
#include "optimisation/vertices/point.hpp"
#include "optimisation/vertices/pose.hpp"
#include "optimisation/vertices/similarity.hpp"
#include "sensor/camera/model.hpp"
#include "sensor/camera/pinhole.hpp"

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

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-8) {
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

class problem final {
public:
    optimisation::factor_graph graph;
    std::vector<optimisation::vertex*> cameras;
    std::vector<optimisation::vertex*> landmarks;
    std::vector<optimisation::vertex*> lines;
};

static void build_problem(problem& built, core::random_pcg& rng, const size_t camera_count, const size_t landmark_count, const bool baseline_edge, const bool similarity_vertex) {
    const sensor::camera::pinhole<double> camera_model(std::vector<double>{ 1.0, 1.0, 0.0, 0.0 }.data(), 4);
    std::vector<math::se3<double>> poses;
    for (size_t i = 0; i < camera_count; ++i) {
        const double angle = 0.05 * static_cast<double>(i);
        poses.push_back(math::se3<double>(math::so3<double>::exp({ { 0.0, angle, 0.0 } }), math::matrix<double, 3, 1>{ { 0.2 * static_cast<double>(i), 0.02 * static_cast<double>(i % 3), 0.0 } }));
    }
    std::vector<math::matrix<double, 3, 1>> points;
    for (size_t i = 0; i < landmark_count; ++i) {
        points.push_back(math::matrix<double, 3, 1>{ { rng.get_random(-1.0, 1.0) + (0.1 * static_cast<double>(camera_count)), rng.get_random(-1.0, 1.0), rng.get_random(3.0, 6.0) } });
    }
    for (size_t i = 0; i < camera_count; ++i) {
        optimisation::vertex camera_vertex{ optimisation::vertices::pose() };
        const double camera_pose[7] = { poses[i].translation()[0], poses[i].translation()[1], poses[i].translation()[2], poses[i].rotation().get_quaternion()[1], poses[i].rotation().get_quaternion()[2], poses[i].rotation().get_quaternion()[3], poses[i].rotation().get_quaternion()[0] };
        REQUIRE(camera_vertex.set_parameters(&camera_pose[0], 7));
        camera_vertex.set_fixed(i == 0);
        built.cameras.push_back(built.graph.add_vertex(static_cast<optimisation::vertex&&>(camera_vertex)));
    }
    for (size_t i = 0; i < landmark_count; ++i) {
        optimisation::vertex landmark_vertex{ optimisation::vertices::point() };
        const math::matrix<double, 3, 1> noisy = points[i] + math::matrix<double, 3, 1>{ { rng.get_random(-0.05, 0.05), rng.get_random(-0.05, 0.05), rng.get_random(-0.05, 0.05) } };
        REQUIRE(landmark_vertex.set_parameters(noisy.data(), 3));
        landmark_vertex.set_marginalised(true);
        built.landmarks.push_back(built.graph.add_vertex(static_cast<optimisation::vertex&&>(landmark_vertex)));
    }
    const optimisation::loss robust_loss{ optimisation::losses::huber(1.0) };
    for (size_t c = 0; c < camera_count; ++c) {
        for (size_t l = 0; l < landmark_count; ++l) {
            const math::matrix<double, 3, 1> in_camera = poses[c] * points[l];
            math::matrix<double, 2, 1> pixel;
            if (!camera_model.project(in_camera.data(), pixel.data())) {
                continue;
            }
            optimisation::edge edge{ optimisation::edges::reprojection(sensor::camera::model<double>(camera_model)) };
            edge.set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { pixel[0] + rng.get_random(-0.001, 0.001), pixel[1] + rng.get_random(-0.001, 0.001) } }.data()));
            edge.add_vertex(built.cameras[c]);
            edge.add_vertex(built.landmarks[l]);
            edge.set_loss(robust_loss);
            REQUIRE(built.graph.add_edge(static_cast<optimisation::edge&&>(edge)) != nullptr);
        }
    }
    if (baseline_edge) {
        optimisation::edge edge{ optimisation::edges::baseline(0.2) };
        edge.add_vertex(built.cameras[1]);
        edge.add_vertex(built.cameras[0]);
        edge.set_information(math::matrix<double, 0, 0>::identity(1, 1) * 1e4);
        REQUIRE(built.graph.add_edge(static_cast<optimisation::edge&&>(edge)) != nullptr);
    }
    if (similarity_vertex) {
        optimisation::vertex first{ optimisation::vertices::similarity() };
        optimisation::vertex second{ optimisation::vertices::similarity() };
        const double identity[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0 };
        const double moved[8] = { 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.1 };
        REQUIRE(first.set_parameters(&identity[0], 8));
        REQUIRE(second.set_parameters(&moved[0], 8));
        first.set_fixed(true);
        optimisation::vertex* const first_vertex = built.graph.add_vertex(static_cast<optimisation::vertex&&>(first));
        optimisation::vertex* const second_vertex = built.graph.add_vertex(static_cast<optimisation::vertex&&>(second));
        optimisation::edge edge{ optimisation::edges::relative_similarity() };
        edge.set_observation(math::matrix<double, 0, 0>(8, 1, &identity[0]));
        edge.add_vertex(first_vertex);
        edge.add_vertex(second_vertex);
        REQUIRE(built.graph.add_edge(static_cast<optimisation::edge&&>(edge)) != nullptr);
    }
}

static void add_lines(problem& built, core::random_pcg& rng, const size_t line_count, std::vector<geometry::plucker>& true_lines) {
    const sensor::camera::pinhole<double> camera_model(std::vector<double>{ 1.0, 1.0, 0.0, 0.0 }.data(), 4);
    const optimisation::loss robust_loss{ optimisation::losses::huber(1.0) };
    for (size_t i = 0; i < line_count; ++i) {
        const math::matrix<double, 3, 1> point_a{ { rng.get_random(-1.0, 1.0) + (0.1 * static_cast<double>(built.cameras.size())), rng.get_random(-1.0, 1.0), rng.get_random(3.0, 6.0) } };
        const math::matrix<double, 3, 1> point_b = point_a + math::matrix<double, 3, 1>{ { rng.get_random(-0.3, 0.3), (rng.get_random(0.0, 1.0) < 0.5) ? rng.get_random(-1.0, -0.7) : rng.get_random(0.7, 1.0), rng.get_random(-0.3, 0.3) } };
        geometry::plucker line;
        REQUIRE(geometry::plucker::from_points(point_a, point_b, line));
        true_lines.push_back(line);
        geometry::plucker noisy = line;
        REQUIRE(noisy.oplus(rng.get_random(-0.05, 0.05), rng.get_random(-0.05, 0.05), rng.get_random(-0.05, 0.05), rng.get_random(-0.05, 0.05)));
        optimisation::vertex line_vertex{ optimisation::vertices::line() };
        const double parameters[6] = { noisy.moment[0], noisy.moment[1], noisy.moment[2], noisy.direction[0], noisy.direction[1], noisy.direction[2] };
        REQUIRE(line_vertex.set_parameters(&parameters[0], 6));
        line_vertex.set_marginalised(true);
        built.lines.push_back(built.graph.add_vertex(static_cast<optimisation::vertex&&>(line_vertex)));
        for (size_t c = 0; c < built.cameras.size(); ++c) {
            const optimisation::vertex* const camera = built.cameras[c];
            const double* const camera_parameters = camera->get_parameters();
            const double quaternion[4] = { camera_parameters[6], camera_parameters[3], camera_parameters[4], camera_parameters[5] };
            const math::se3<double> pose(math::so3<double>(quaternion), math::matrix<double, 3, 1>{ { camera_parameters[0], camera_parameters[1], camera_parameters[2] } });
            double pixels[2][2];
            const math::matrix<double, 3, 1> in_camera_a = pose * point_a;
            const math::matrix<double, 3, 1> in_camera_b = pose * point_b;
            if (!camera_model.project(in_camera_a.data(), &pixels[0][0]) || !camera_model.project(in_camera_b.data(), &pixels[1][0])) {
                continue;
            }
            optimisation::edges::line_reprojection policy{ sensor::camera::model<double>(camera_model) };
            REQUIRE(policy.set_measured_segment(pixels[0][0] + rng.get_random(-0.001, 0.001), pixels[0][1] + rng.get_random(-0.001, 0.001), pixels[1][0] + rng.get_random(-0.001, 0.001), pixels[1][1] + rng.get_random(-0.001, 0.001)));
            optimisation::edge edge{ policy };
            edge.add_vertex(built.cameras[c]);
            edge.add_vertex(built.lines[i]);
            edge.set_loss(robust_loss);
            REQUIRE(built.graph.add_edge(static_cast<optimisation::edge&&>(edge)) != nullptr);
        }
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        optimisation::factor_graph factor_graph;
    }

    {
        optimisation::factor_graph factor_graph;
        std::vector<optimisation::vertex*> added;
        for (int i = 0; i < 64; ++i) {
            optimisation::vertex node{ optimisation::vertices::point() };
            const double location[3] = { static_cast<double>(i), 0.0, 0.0 };
            REQUIRE(node.set_parameters(&location[0], 3));
            optimisation::vertex* const stored = factor_graph.add_vertex(static_cast<optimisation::vertex&&>(node));
            REQUIRE(stored != nullptr);
            added.push_back(stored);
        }
        for (int i = 0; i < 64; ++i) {
            REQUIRE(added[static_cast<size_t>(i)]->get_parameters()[0] == static_cast<double>(i));
        }
        REQUIRE(factor_graph.add_vertex(optimisation::vertex()) == nullptr);
        REQUIRE(factor_graph.add_edge(optimisation::edge()) == nullptr);
        REQUIRE(factor_graph.remove_vertex(added[0]));
        REQUIRE(!factor_graph.remove_vertex(added[0]));
        REQUIRE(added[1]->get_parameters()[0] == 1.0);
    }

    {
        core::random_pcg rng;

        const sensor::camera::pinhole<double> camera_model(std::vector<double>{ 1.0, 1.0, 0.0, 0.0 }.data(), 4);

        optimisation::factor_graph factor_graph;

        std::vector<math::se3<double>> cameras{
            math::se3<double>::identity(),
            math::se3<double>(math::so3<double>::identity(), math::matrix<double, 3, 1>{ { 0.1, 0.0, 0.0 } }),
            math::se3<double>(math::so3<double>::identity(), math::matrix<double, 3, 1>{ { 0.0, 0.1, 0.0 } }),
            math::se3<double>(math::so3<double>::identity(), math::matrix<double, 3, 1>{ { 0.0, 0.0, 0.1 } }),
            math::se3<double>(math::so3<double>::identity(), math::matrix<double, 3, 1>{ { -0.1, 0.0, 0.0 } }),
            math::se3<double>(math::so3<double>::identity(), math::matrix<double, 3, 1>{ { 0.0, -0.1, 0.0 } }),
            math::se3<double>(math::so3<double>::identity(), math::matrix<double, 3, 1>{ { 0.0, 0.0, -0.1 } })
        };

        std::vector<math::matrix<double, 3, 1>> landmarks = {
            { { 0.0, 0.0, 1.0 } },
            { { 0.0, 1.0, 1.0 } },
            { { 1.0, 0.0, 1.0 } }
        };

        std::vector<optimisation::vertex*> camera_vertexes;
        std::vector<optimisation::vertex*> landmark_vertexes;

        for (size_t i = 0; i < cameras.size(); ++i) {
            optimisation::vertex camera_vertex{ optimisation::vertices::pose() };
            const double camera_pose[7] = { cameras[i].translation()[0], cameras[i].translation()[1], cameras[i].translation()[2], cameras[i].rotation().get_quaternion()[1], cameras[i].rotation().get_quaternion()[2], cameras[i].rotation().get_quaternion()[3], cameras[i].rotation().get_quaternion()[0] };
            REQUIRE(camera_vertex.set_parameters(&camera_pose[0], 7));
            camera_vertex.set_fixed(i == 0);
            camera_vertexes.push_back(factor_graph.add_vertex(static_cast<optimisation::vertex&&>(camera_vertex)));
            REQUIRE(camera_vertexes[i] != nullptr);
        }

        std::vector<math::matrix<double, 3, 1>> noisy_landmarks(landmarks.size());
        for (size_t i = 0; i < landmarks.size(); ++i) {
            optimisation::vertex landmark_vertex{ optimisation::vertices::point() };
            for (size_t j = 0; j < 3; ++j) {
                noisy_landmarks[i][j] = landmarks[i][j] + (static_cast<double>(static_cast<int>(rng.get_random_raw() % 10) - 5) * 0.01);
            }
            REQUIRE(landmark_vertex.set_parameters(noisy_landmarks[i].data(), 3));
            landmark_vertex.set_fixed(false);
            landmark_vertex.set_marginalised(true);
            landmark_vertexes.push_back(factor_graph.add_vertex(static_cast<optimisation::vertex&&>(landmark_vertex)));
        }

        size_t camera_id = 0;
        for (const math::se3<double>& camera : cameras) {
            size_t landmark_id = 0;
            for (const math::matrix<double, 3, 1>& landmark : landmarks) {
                math::matrix<double, 3, 1> world_point = camera * landmark;
                math::matrix<double, 2, 1> point;
                REQUIRE(camera_model.project(world_point.data(), point.data()));
                optimisation::edge m{ optimisation::edges::reprojection(sensor::camera::model<double>(camera_model)) };
                m.set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { point[0] + (static_cast<double>(static_cast<int>(rng.get_random_raw() % 10) - 5) * 0.0001), point[1] + (static_cast<double>(static_cast<int>(rng.get_random_raw() % 10) - 5) * 0.0001) } }.data()));
                m.add_vertex(camera_vertexes[camera_id]);
                m.add_vertex(landmark_vertexes[landmark_id]);
                REQUIRE(factor_graph.add_edge(static_cast<optimisation::edge&&>(m)) != nullptr);
                ++landmark_id;
            }
            ++camera_id;
        }

        double initialChi2 = factor_graph.get_current_chi();
        REQUIRE(factor_graph.solve(50, true));
        double error_noisy = 0;
        double error_optimised = 0;
        for (size_t i = 0; i < landmarks.size(); ++i) {
            math::matrix<double, 3, 1> result_landmark = { {
                landmark_vertexes[i]->get_parameters()[0],
                landmark_vertexes[i]->get_parameters()[1],
                landmark_vertexes[i]->get_parameters()[2],
            } };
            error_noisy += std::sqrt((landmarks[i] - noisy_landmarks[i]).get_length_squared());
            error_optimised += std::sqrt((landmarks[i] - result_landmark).get_length_squared());
        }

        REQUIRE(error_optimised < error_noisy);
        REQUIRE(!is_value_approx(factor_graph.get_current_chi(), initialChi2));
        REQUIRE(is_value_approx(factor_graph.get_current_chi(), 0.0, 1e-5));
        REQUIRE(factor_graph.get_current_chi() < initialChi2);
    }

    {
        const sensor::camera::pinhole<double> camera_model(std::vector<double>{ 1.0, 1.0, 0.0, 0.0 }.data(), 4);

        optimisation::factor_graph factor_graph;

        optimisation::vertex fixed_camera{ optimisation::vertices::pose() };
        const double fixed_camera_pose[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
        REQUIRE(fixed_camera.set_parameters(&fixed_camera_pose[0], 7));
        fixed_camera.set_fixed(true);
        REQUIRE(factor_graph.add_vertex(static_cast<optimisation::vertex&&>(fixed_camera)) != nullptr);

        optimisation::vertex free_camera{ optimisation::vertices::pose() };
        const double free_camera_pose[7] = { 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
        REQUIRE(free_camera.set_parameters(&free_camera_pose[0], 7));
        free_camera.set_fixed(false);
        optimisation::vertex* const free_camera_vertex = factor_graph.add_vertex(static_cast<optimisation::vertex&&>(free_camera));

        optimisation::vertex landmark{ optimisation::vertices::point() };
        const double landmark_location[3] = { 0.1, 0.2, 1.0 };
        REQUIRE(landmark.set_parameters(&landmark_location[0], 3));
        landmark.set_fixed(true);
        landmark.set_marginalised(true);
        optimisation::vertex* const landmark_vertex = factor_graph.add_vertex(static_cast<optimisation::vertex&&>(landmark));

        optimisation::edge edge{ optimisation::edges::reprojection(sensor::camera::model<double>(camera_model)) };
        edge.set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { 0.15, 0.25 } }.data()));
        edge.add_vertex(free_camera_vertex);
        edge.add_vertex(landmark_vertex);
        edge.set_information(-1.0 * math::matrix<double, 0, 0>::identity(2, 2));
        REQUIRE(factor_graph.add_edge(static_cast<optimisation::edge&&>(edge)) != nullptr);

        static_cast<void>(factor_graph.solve(10));

        const double* const optimised_parameters = free_camera_vertex->get_parameters();
        for (size_t i = 0; i < 7; ++i) {
            REQUIRE(std::isfinite(optimised_parameters[i]));
        }
    }

    {
        core::random_pcg rng;

        const sensor::camera::pinhole<double> camera_model(std::vector<double>{ 1.0, 1.0, 0.0, 0.0 }.data(), 4);

        optimisation::factor_graph factor_graph;

        std::vector<math::se3<double>> cameras{
            math::se3<double>::identity(),
            math::se3<double>(math::so3<double>::identity(), math::matrix<double, 3, 1>{ { 0.1, 0.0, 0.0 } }),
            math::se3<double>(math::so3<double>::identity(), math::matrix<double, 3, 1>{ { 0.0, 0.1, 0.0 } }),
            math::se3<double>(math::so3<double>::identity(), math::matrix<double, 3, 1>{ { 0.0, 0.0, 0.1 } }),
            math::se3<double>(math::so3<double>::identity(), math::matrix<double, 3, 1>{ { -0.1, 0.0, 0.0 } }),
            math::se3<double>(math::so3<double>::identity(), math::matrix<double, 3, 1>{ { 0.0, -0.1, 0.0 } }),
            math::se3<double>(math::so3<double>::identity(), math::matrix<double, 3, 1>{ { 0.0, 0.0, -0.1 } })
        };

        std::vector<math::matrix<double, 3, 1>> landmarks = {
            { { 0.0, 0.0, 1.0 } },
            { { 0.0, 1.0, 1.0 } },
            { { 1.0, 0.0, 1.0 } }
        };

        std::vector<optimisation::vertex*> camera_vertexes;
        std::vector<optimisation::vertex*> landmark_vertexes;

        for (size_t i = 0; i < cameras.size(); ++i) {
            optimisation::vertex camera_vertex{ optimisation::vertices::pose() };
            const double camera_pose[7] = { cameras[i].translation()[0], cameras[i].translation()[1], cameras[i].translation()[2], cameras[i].rotation().get_quaternion()[1], cameras[i].rotation().get_quaternion()[2], cameras[i].rotation().get_quaternion()[3], cameras[i].rotation().get_quaternion()[0] };
            REQUIRE(camera_vertex.set_parameters(&camera_pose[0], 7));
            camera_vertex.set_fixed(i == 0);
            camera_vertexes.push_back(factor_graph.add_vertex(static_cast<optimisation::vertex&&>(camera_vertex)));
            REQUIRE(camera_vertexes[i] != nullptr);
        }

        std::vector<math::matrix<double, 3, 1>> noisy_landmarks(landmarks.size());
        for (size_t i = 0; i < landmarks.size(); ++i) {
            optimisation::vertex landmark_vertex{ optimisation::vertices::point() };
            for (size_t j = 0; j < 3; ++j) {
                noisy_landmarks[i][j] = landmarks[i][j] + (static_cast<double>(static_cast<int>(rng.get_random_raw() % 10) - 5) * 0.01);
            }
            REQUIRE(landmark_vertex.set_parameters(noisy_landmarks[i].data(), 3));
            landmark_vertex.set_fixed(false);
            landmark_vertex.set_marginalised(true);
            landmark_vertexes.push_back(factor_graph.add_vertex(static_cast<optimisation::vertex&&>(landmark_vertex)));
        }

        size_t camera_id = 0;
        for (const math::se3<double>& camera : cameras) {
            size_t landmark_id = 0;
            for (const math::matrix<double, 3, 1>& landmark : landmarks) {
                math::matrix<double, 3, 1> world_point = camera * landmark;
                math::matrix<double, 2, 1> point;
                REQUIRE(camera_model.project(world_point.data(), point.data()));
                optimisation::edge m{ optimisation::edges::reprojection(sensor::camera::model<double>(camera_model)) };
                m.set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { point[0] + (static_cast<double>(static_cast<int>(rng.get_random_raw() % 10) - 5) * 0.0001), point[1] + (static_cast<double>(static_cast<int>(rng.get_random_raw() % 10) - 5) * 0.0001) } }.data()));
                m.add_vertex(camera_vertexes[camera_id]);
                m.add_vertex(landmark_vertexes[landmark_id]);
                REQUIRE(factor_graph.add_edge(static_cast<optimisation::edge&&>(m)) != nullptr);
                ++landmark_id;
            }
            ++camera_id;
        }

        double initialChi2 = factor_graph.get_current_chi();
        REQUIRE(factor_graph.solve(50, true));

        double error_noisy = 0;
        double error_optimised = 0;
        for (size_t i = 0; i < landmarks.size(); ++i) {
            math::matrix<double, 3, 1> result_landmark = { {
                landmark_vertexes[i]->get_parameters()[0],
                landmark_vertexes[i]->get_parameters()[1],
                landmark_vertexes[i]->get_parameters()[2],
            } };
            error_noisy += std::sqrt((landmarks[i] - noisy_landmarks[i]).get_length_squared());
            error_optimised += std::sqrt((landmarks[i] - result_landmark).get_length_squared());
        }

        REQUIRE(error_optimised < error_noisy);
        REQUIRE(!is_value_approx(factor_graph.get_current_chi(), initialChi2));
        REQUIRE(is_value_approx(factor_graph.get_current_chi(), 0.0, 1e-5));
        REQUIRE(factor_graph.get_current_chi() < initialChi2);
    }

    for (const bool baseline_edge : { false, true }) {
        core::random_pcg rng(0x5eed0066ull);
        problem built;
        build_problem(built, rng, 6, 25, baseline_edge, false);
        built.graph.set_conjugate_gradient_tolerance(1e-14);
        built.graph.set_conjugate_gradient_iteration_limit(100000);
        for (const double lambda : { 1e-6, 1e-2, 1.0, 1e2 }) {
            math::matrix<double, 0, 0> step_dense;
            built.graph.set_strategy(optimisation::factor_graph::strategy::dense_schur);
            REQUIRE(built.graph.compute_damped_step(lambda, step_dense));
            REQUIRE(!built.graph.get_diagnostics().used_square_root);
            math::matrix<double, 0, 0> step_square_root;
            built.graph.set_strategy(optimisation::factor_graph::strategy::square_root);
            REQUIRE(built.graph.compute_damped_step(lambda, step_square_root));
            REQUIRE(built.graph.get_diagnostics().used_square_root);
            REQUIRE(built.graph.get_diagnostics().landmark_blocks == 25);
            REQUIRE(built.graph.get_diagnostics().reduced_solves == 1);
            REQUIRE(built.graph.get_diagnostics().reduced_failures == 0);
            REQUIRE(step_dense.rows() == step_square_root.rows());
            double largest = 0.0;
            for (size_t i = 0; i < step_dense.rows(); ++i) {
                largest = std::max(largest, std::abs(step_dense[i][0]));
            }
            for (size_t i = 0; i < step_dense.rows(); ++i) {
                REQUIRE(std::isfinite(step_square_root[i][0]));
                REQUIRE(std::abs(step_dense[i][0] - step_square_root[i][0]) < 1e-7 * largest);
            }
        }
    }

    {
        double chi_dense = 0.0;
        double initial_chi = 0.0;
        {
            core::random_pcg rng(0x5eed0077ull);
            problem built;
            build_problem(built, rng, 8, 40, true, false);
            built.graph.set_strategy(optimisation::factor_graph::strategy::dense_schur);
            initial_chi = built.graph.get_current_chi();
            REQUIRE(built.graph.solve(50, true));
            chi_dense = built.graph.get_current_chi();
            REQUIRE(chi_dense < initial_chi);
        }
        {
            core::random_pcg rng(0x5eed0077ull);
            problem built;
            build_problem(built, rng, 8, 40, true, false);
            built.graph.set_strategy(optimisation::factor_graph::strategy::square_root);
            built.graph.set_conjugate_gradient_tolerance(1e-14);
            built.graph.set_conjugate_gradient_iteration_limit(100000);
            REQUIRE(is_value_approx(built.graph.get_current_chi(), initial_chi, 1e-15));
            REQUIRE(built.graph.solve(50, true));
            REQUIRE(built.graph.get_diagnostics().used_square_root);
            REQUIRE(!built.graph.get_diagnostics().used_single_precision);
            REQUIRE(is_value_approx(built.graph.get_current_chi(), chi_dense, 1e-6));
        }
        {
            core::random_pcg rng(0x5eed0077ull);
            problem built;
            build_problem(built, rng, 8, 40, true, false);
            built.graph.set_strategy(optimisation::factor_graph::strategy::square_root);
            built.graph.set_precision(optimisation::factor_graph::precision::single_precision);
            REQUIRE(built.graph.solve(50, true));
            REQUIRE(built.graph.get_diagnostics().used_single_precision);
            REQUIRE(built.graph.get_current_chi() < initial_chi);
            REQUIRE(is_value_approx(built.graph.get_current_chi(), chi_dense, 1e-2));
        }
    }

    {
        core::random_pcg rng(0x5eed0099ull);
        problem built;
        build_problem(built, rng, 6, 20, true, false);
        std::vector<geometry::plucker> true_lines;
        add_lines(built, rng, 12, true_lines);
        built.graph.set_conjugate_gradient_tolerance(1e-14);
        built.graph.set_conjugate_gradient_iteration_limit(100000);
        for (const double lambda : { 1e-6, 1e-2, 1.0, 1e2 }) {
            math::matrix<double, 0, 0> step_dense;
            built.graph.set_strategy(optimisation::factor_graph::strategy::dense_schur);
            REQUIRE(built.graph.compute_damped_step(lambda, step_dense));
            REQUIRE(!built.graph.get_diagnostics().used_square_root);
            math::matrix<double, 0, 0> step_square_root;
            built.graph.set_strategy(optimisation::factor_graph::strategy::square_root);
            REQUIRE(built.graph.compute_damped_step(lambda, step_square_root));
            REQUIRE(built.graph.get_diagnostics().used_square_root);
            REQUIRE(built.graph.get_diagnostics().landmark_blocks == 32);
            REQUIRE(built.graph.get_diagnostics().reduced_failures == 0);
            REQUIRE(step_dense.rows() == step_square_root.rows());
            REQUIRE(step_dense.rows() == static_cast<size_t>((5 * 6) + (20 * 3) + (12 * 4)));
            double largest = 0.0;
            for (size_t i = 0; i < step_dense.rows(); ++i) {
                largest = std::max(largest, std::abs(step_dense[i][0]));
            }
            for (size_t i = 0; i < step_dense.rows(); ++i) {
                REQUIRE(std::isfinite(step_square_root[i][0]));
                REQUIRE(std::abs(step_dense[i][0] - step_square_root[i][0]) < 1e-7 * largest);
            }
        }
    }

    {
        const auto line_error = [](const problem& built, const std::vector<geometry::plucker>& true_lines) {
            double error = 0.0;
            for (size_t l = 0; l < built.lines.size(); ++l) {
                const double* const parameters = built.lines[l]->get_parameters();
                for (size_t i = 0; i < 3; ++i) {
                    error += std::abs(parameters[i] - true_lines[l].moment[i]) + std::abs(parameters[3 + i] - true_lines[l].direction[i]);
                }
            }
            return error;
        };
        double chi_dense = 0.0;
        double initial_chi = 0.0;
        for (const int variant : { 0, 1, 2 }) {
            core::random_pcg rng(0x5eed00aaull);
            problem built;
            build_problem(built, rng, 8, 30, true, false);
            std::vector<geometry::plucker> true_lines;
            add_lines(built, rng, 16, true_lines);
            const double initial_error = line_error(built, true_lines);
            if (variant == 0) {
                built.graph.set_strategy(optimisation::factor_graph::strategy::dense_schur);
                initial_chi = built.graph.get_current_chi();
                REQUIRE(built.graph.solve(50, true));
                REQUIRE(!built.graph.get_diagnostics().used_square_root);
                chi_dense = built.graph.get_current_chi();
                REQUIRE(chi_dense < initial_chi);
            }
            else {
                built.graph.set_strategy(optimisation::factor_graph::strategy::square_root);
                if (variant == 2) {
                    built.graph.set_precision(optimisation::factor_graph::precision::single_precision);
                }
                else {
                    built.graph.set_conjugate_gradient_tolerance(1e-14);
                    built.graph.set_conjugate_gradient_iteration_limit(100000);
                }
                REQUIRE(is_value_approx(built.graph.get_current_chi(), initial_chi, 1e-15));
                REQUIRE(built.graph.solve(50, true));
                REQUIRE(built.graph.get_diagnostics().used_square_root);
                REQUIRE(built.graph.get_diagnostics().used_single_precision == (variant == 2));
                REQUIRE(is_value_approx(built.graph.get_current_chi(), chi_dense, (variant == 2) ? 1e-2 : 1e-6));
            }
            REQUIRE(line_error(built, true_lines) < 0.5 * initial_error);
            for (const optimisation::vertex* line : built.lines) {
                const double* const parameters = line->get_parameters();
                REQUIRE(is_value_approx(std::sqrt((parameters[3] * parameters[3]) + (parameters[4] * parameters[4]) + (parameters[5] * parameters[5])), 1.0, 1e-9));
            }
        }
    }

    {
        core::random_pcg rng(0x5eed0088ull);
        problem small;
        build_problem(small, rng, 4, 10, false, false);
        small.graph.set_strategy(optimisation::factor_graph::strategy::automatic);
        REQUIRE(small.graph.solve(3, true));
        REQUIRE(!small.graph.get_diagnostics().used_square_root);
        problem large;
        build_problem(large, rng, 4, 10, false, false);
        large.graph.set_strategy(optimisation::factor_graph::strategy::automatic);
        large.graph.set_square_root_parameter_threshold(0);
        REQUIRE(large.graph.solve(3, true));
        REQUIRE(large.graph.get_diagnostics().used_square_root);
        REQUIRE(large.graph.get_diagnostics().landmark_blocks == 10);
        problem mixed;
        build_problem(mixed, rng, 4, 10, false, true);
        mixed.graph.set_strategy(optimisation::factor_graph::strategy::square_root);
        REQUIRE(mixed.graph.solve(3, true));
        REQUIRE(!mixed.graph.get_diagnostics().used_square_root);
    }

    return EXIT_SUCCESS;
}
