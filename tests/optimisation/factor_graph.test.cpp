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
#include "math/matrix_decomposition_cholesky.hpp"
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
#include <limits>
#include <type_traits>
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

// A residual that can only be evaluated at the start of its vertex, as one outside its domain would be anywhere else.
class pinned_edge final {
public:
    constexpr static const char* name = "pinned";
    constexpr static const int residual_count = 1;
    constexpr static const int vertex_count = 1;

    double start;

    void compute_residual(const optimisation::edge& context, math::matrix<double, 0, 0>& residual) const {
        const double value = context.get_vertex(0)->get_parameters()[0];
        residual[0][0] = (value == this->start) ? (value - 1.0) : std::numeric_limits<double>::quiet_NaN();
    }

    void compute_jacobians(const optimisation::edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        static_cast<void>(context);
        const double values[3] = { 1.0, 0.0, 0.0 };
        jacobians[0] = math::matrix<double, 0, 0>(1, 3, &values[0]);
    }
};

// A linear residual of two points, A x + B y - t, which holds the same vertex in both slots in the tests, so that one edge adds
// two Hessian terms of the same block, one of them transposed.
class twice_edge final {
public:
    constexpr static const char* name = "twice";
    constexpr static const int residual_count = 2;
    constexpr static const int vertex_count = 2;

    constexpr static const double first[6] = { 1.0, 0.5, -0.2, 0.3, -1.0, 0.4 };
    constexpr static const double second[6] = { 0.2, 0.7, 1.1, -0.6, 0.1, 0.9 };
    constexpr static const double target[2] = { 0.3, -0.2 };

    void compute_residual(const optimisation::edge& context, math::matrix<double, 0, 0>& residual) const {
        const double* const x = context.get_vertex(0)->get_parameters();
        const double* const y = context.get_vertex(1)->get_parameters();
        for (size_t r = 0; r < 2; ++r) {
            double sum = -twice_edge::target[r];
            for (size_t c = 0; c < 3; ++c) {
                sum += (twice_edge::first[(r * 3) + c] * x[c]) + (twice_edge::second[(r * 3) + c] * y[c]);
            }
            residual[r][0] = sum;
        }
    }

    void compute_jacobians(const optimisation::edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        static_cast<void>(context);
        jacobians[0] = math::matrix<double, 0, 0>(2, 3, &twice_edge::first[0]);
        jacobians[1] = math::matrix<double, 0, 0>(2, 3, &twice_edge::second[0]);
    }
};

// The damped step of the whole linear system, assembled densely from the linearised edges and solved without the Schur complement.
static std::vector<double> dense_damped_step(const std::vector<optimisation::edge*>& edges, const size_t count, const double lambda) {
    std::vector<double> hessian(count * count, 0.0);
    std::vector<double> gradient(count, 0.0);
    for (const optimisation::edge* const factor : edges) {
        const size_t residuals = factor->get_residual().rows();
        const double weight = factor->robust_weight();
        for (size_t i = 0; i < factor->get_vertices().size(); ++i) {
            const optimisation::vertex* const first = factor->get_vertices()[i];
            if (first->is_fixed()) {
                continue;
            }
            const math::matrix<double, 0, 0>& jacobian_i = factor->get_jacobians()[i];
            for (size_t a = 0; a < jacobian_i.cols(); ++a) {
                double sum = 0.0;
                for (size_t r = 0; r < residuals; ++r) {
                    for (size_t c = 0; c < residuals; ++c) {
                        sum += jacobian_i[r][a] * weight * factor->get_information()[r][c] * factor->get_residual()[c][0];
                    }
                }
                gradient[static_cast<size_t>(first->get_ordering_id()) + a] -= sum;
            }
            for (size_t j = 0; j < factor->get_vertices().size(); ++j) {
                const optimisation::vertex* const second = factor->get_vertices()[j];
                if (second->is_fixed()) {
                    continue;
                }
                const math::matrix<double, 0, 0>& jacobian_j = factor->get_jacobians()[j];
                for (size_t a = 0; a < jacobian_i.cols(); ++a) {
                    for (size_t b = 0; b < jacobian_j.cols(); ++b) {
                        double sum = 0.0;
                        for (size_t r = 0; r < residuals; ++r) {
                            for (size_t c = 0; c < residuals; ++c) {
                                sum += jacobian_i[r][a] * weight * factor->get_information()[r][c] * jacobian_j[c][b];
                            }
                        }
                        hessian[((static_cast<size_t>(first->get_ordering_id()) + a) * count) + static_cast<size_t>(second->get_ordering_id()) + b] += sum;
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const double root = std::sqrt(std::max(hessian[(i * count) + i], 0.0));
        hessian[(i * count) + i] += lambda * (1.0 + root) * (1.0 + root);
    }
    std::vector<double> lower(count * count, 0.0);
    std::vector<double> step(count, 0.0);
    REQUIRE(math::decompose_cholesky(hessian.data(), static_cast<int>(count), static_cast<int>(count), lower.data()));
    REQUIRE(math::solve_cholesky(lower.data(), gradient.data(), static_cast<int>(count), static_cast<int>(count), step.data()));
    return step;
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

static size_t add_mixed_blocks(problem& built, core::random_pcg& rng) {
    const sensor::camera::pinhole<double> camera_model(std::vector<double>{ 1.0, 1.0, 0.0, 0.0 }.data(), 4);
    const optimisation::loss robust_loss{ optimisation::losses::huber(1.0) };
    const double correlated[4] = { 2.0e4, 0.5e4, 0.5e4, 1.0e4 };
    const math::matrix<double, 0, 0> information(2, 2, &correlated[0]);
    for (size_t l = 0; l < built.landmarks.size(); ++l) {
        built.landmarks[l]->set_fixed((l % 5) == 1);
        if ((l % 2) == 0) {
            for (optimisation::edge* const connected : built.graph.get_connected_edges(built.landmarks[l])) {
                connected->set_information(information);
            }
        }
    }
    constexpr static const size_t single_count = 6;
    for (size_t i = 0; i < single_count; ++i) {
        optimisation::vertex* const camera = built.cameras[1 + (i % (built.cameras.size() - 1))];
        const double* const camera_parameters = camera->get_parameters();
        const double quaternion[4] = { camera_parameters[6], camera_parameters[3], camera_parameters[4], camera_parameters[5] };
        const math::se3<double> pose(math::so3<double>(quaternion), math::matrix<double, 3, 1>{ { camera_parameters[0], camera_parameters[1], camera_parameters[2] } });
        const math::matrix<double, 3, 1> location{ { rng.get_random(-1.0, 1.0) + (0.1 * static_cast<double>(built.cameras.size())), rng.get_random(-1.0, 1.0), rng.get_random(3.0, 6.0) } };
        math::matrix<double, 2, 1> pixel;
        REQUIRE(camera_model.project((pose * location).data(), pixel.data()));
        optimisation::vertex landmark_vertex{ optimisation::vertices::point() };
        const math::matrix<double, 3, 1> noisy = location + math::matrix<double, 3, 1>{ { rng.get_random(-0.05, 0.05), rng.get_random(-0.05, 0.05), rng.get_random(-0.05, 0.05) } };
        REQUIRE(landmark_vertex.set_parameters(noisy.data(), 3));
        landmark_vertex.set_marginalised(true);
        landmark_vertex.set_fixed((i % 3) == 0);
        optimisation::vertex* const landmark = built.graph.add_vertex(static_cast<optimisation::vertex&&>(landmark_vertex));
        built.landmarks.push_back(landmark);
        optimisation::edge edge{ optimisation::edges::reprojection(sensor::camera::model<double>(camera_model)) };
        edge.set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { pixel[0] + rng.get_random(-0.001, 0.001), pixel[1] + rng.get_random(-0.001, 0.001) } }.data()));
        edge.add_vertex(camera);
        edge.add_vertex(landmark);
        edge.set_information(information);
        edge.set_loss(robust_loss);
        REQUIRE(built.graph.add_edge(static_cast<optimisation::edge&&>(edge)) != nullptr);
    }
    return single_count;
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

    static_assert(!std::is_copy_constructible<optimisation::factor_graph>::value, "The graph holds pointers into its own storage and must not be copied.");
    static_assert(!std::is_copy_assignable<optimisation::factor_graph>::value, "The graph holds pointers into its own storage and must not be copied.");
    static_assert(!std::is_move_constructible<optimisation::factor_graph>::value, "The graph holds pointers into its own storage and must not be moved.");
    static_assert(!std::is_move_assignable<optimisation::factor_graph>::value, "The graph holds pointers into its own storage and must not be moved.");

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

        // An edge must hold exactly the vertices its type takes, all of them vertices of this graph.
        const sensor::camera::pinhole<double> camera_model(std::vector<double>{ 1.0, 1.0, 0.0, 0.0 }.data(), 4);
        optimisation::vertex camera_vertex{ optimisation::vertices::pose() };
        const double camera_pose[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
        REQUIRE(camera_vertex.set_parameters(&camera_pose[0], 7));
        optimisation::vertex* const camera = factor_graph.add_vertex(static_cast<optimisation::vertex&&>(camera_vertex));
        REQUIRE(camera != nullptr);
        const auto make_edge = [&camera_model](optimisation::vertex* const first, optimisation::vertex* const second) {
            optimisation::edge factor{ optimisation::edges::reprojection(sensor::camera::model<double>(camera_model)) };
            factor.set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { 0.0, 0.0 } }.data()));
            if (first != nullptr) {
                REQUIRE(factor.add_vertex(first));
            }
            if (second != nullptr) {
                REQUIRE(factor.add_vertex(second));
            }
            return factor;
        };
        REQUIRE(factor_graph.add_edge(make_edge(camera, nullptr)) == nullptr);
        REQUIRE(factor_graph.add_edge(make_edge(nullptr, nullptr)) == nullptr);
        REQUIRE(factor_graph.add_edge(make_edge(camera, added[0])) == nullptr);
        optimisation::factor_graph other_graph;
        optimisation::vertex foreign_vertex{ optimisation::vertices::point() };
        const double foreign_location[3] = { 0.0, 0.0, 1.0 };
        REQUIRE(foreign_vertex.set_parameters(&foreign_location[0], 3));
        optimisation::vertex* const foreign = other_graph.add_vertex(static_cast<optimisation::vertex&&>(foreign_vertex));
        REQUIRE(foreign != nullptr);
        REQUIRE(factor_graph.add_edge(make_edge(camera, foreign)) == nullptr);
        optimisation::edge* const valid = factor_graph.add_edge(make_edge(camera, added[1]));
        REQUIRE(valid != nullptr);
        REQUIRE(factor_graph.get_connected_edges(camera).size() == 1);
        REQUIRE(factor_graph.get_connected_edges(added[1]).size() == 1);
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

    for (const int variant : { 0, 1, 2 }) {
        core::random_pcg rng(0x5eed0066ull);
        problem built;
        build_problem(built, rng, 6, 25, variant > 0, false);
        const size_t single_blocks = (variant == 2) ? add_mixed_blocks(built, rng) : 0;
        if (variant == 2) {
            static_cast<void>(built.graph.get_current_chi());
            size_t down_weighted = 0;
            size_t full_weight = 0;
            for (optimisation::vertex* const landmark : built.landmarks) {
                for (const optimisation::edge* const connected : built.graph.get_connected_edges(landmark)) {
                    down_weighted += (connected->robust_weight() < 1.0) ? 1u : 0u;
                    full_weight += (connected->robust_weight() == 1.0) ? 1u : 0u;
                }
            }
            REQUIRE(down_weighted > 0);
            REQUIRE(full_weight > 0);
        }
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
            REQUIRE(built.graph.get_diagnostics().landmark_blocks == static_cast<int>(25 + single_blocks));
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

    {
        core::random_pcg rng_marginalised(0x5eed0099ull);
        problem marginalised;
        build_problem(marginalised, rng_marginalised, 4, 10, false, false);
        core::random_pcg rng_general(0x5eed0099ull);
        problem general;
        build_problem(general, rng_general, 4, 10, false, false);
        general.landmarks[3]->set_marginalised(false);
        REQUIRE(marginalised.graph.solve(5, true));
        REQUIRE(general.graph.solve(5, true));
        for (size_t i = 0; i < marginalised.cameras.size(); ++i) {
            for (size_t p = 0; p < 7; ++p) {
                REQUIRE(is_value_approx(marginalised.cameras[i]->get_parameters()[p], general.cameras[i]->get_parameters()[p], 1e-6));
            }
        }
        for (size_t i = 0; i < marginalised.landmarks.size(); ++i) {
            for (size_t p = 0; p < 3; ++p) {
                REQUIRE(is_value_approx(marginalised.landmarks[i]->get_parameters()[p], general.landmarks[i]->get_parameters()[p], 1e-6));
            }
        }
        REQUIRE(general.graph.remove_vertex(general.landmarks[3]));
        REQUIRE(general.graph.remove_vertex(general.landmarks[4]));
        REQUIRE(general.graph.solve(5, true));
    }

    {
        core::random_pcg rng(0x5eed00aaull);
        problem unsupported;
        build_problem(unsupported, rng, 4, 10, false, false);
        unsupported.cameras[1]->set_marginalised(true);
        const double before = unsupported.landmarks[0]->get_parameters()[0];
        REQUIRE(unsupported.graph.solve(5, true) == 0);
        REQUIRE(unsupported.landmarks[0]->get_parameters()[0] == before);
    }

    {
        // Removing edges keeps the order of the others whatever the order of removal, so two graphs with the same edges
        // removed in different orders, one with extra edges added and removed in between, solve identically.
        core::random_pcg rng_first(0x5eed00ddull);
        core::random_pcg rng_second(0x5eed00ddull);
        problem first;
        problem second;
        build_problem(first, rng_first, 6, 25, true, false);
        build_problem(second, rng_second, 6, 25, true, false);
        std::vector<optimisation::edge*> first_removed;
        std::vector<optimisation::edge*> second_removed;
        for (size_t l = 0; l < first.landmarks.size(); l += 3) {
            first_removed.push_back(first.graph.get_connected_edges(first.landmarks[l]).front());
            second_removed.push_back(second.graph.get_connected_edges(second.landmarks[l]).front());
        }
        for (optimisation::edge* const factor : first_removed) {
            REQUIRE(first.graph.remove_edge(factor));
        }
        const sensor::camera::pinhole<double> camera_model(std::vector<double>{ 1.0, 1.0, 0.0, 0.0 }.data(), 4);
        std::vector<optimisation::edge*> extra;
        for (size_t i = 0; i < 10; ++i) {
            optimisation::edge factor{ optimisation::edges::reprojection(sensor::camera::model<double>(camera_model)) };
            factor.set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { 0.3, -0.2 } }.data()));
            REQUIRE(factor.add_vertex(second.cameras[1 + (i % 5)]));
            REQUIRE(factor.add_vertex(second.landmarks[i]));
            extra.push_back(second.graph.add_edge(static_cast<optimisation::edge&&>(factor)));
            REQUIRE(extra.back() != nullptr);
        }
        for (size_t i = second_removed.size(); i-- > 0;) {
            REQUIRE(second.graph.remove_edge(second_removed[i]));
            if (i < extra.size()) {
                REQUIRE(second.graph.remove_edge(extra[i]));
            }
        }
        for (optimisation::edge* const factor : extra) {
            static_cast<void>(second.graph.remove_edge(factor));
        }
        REQUIRE(!second.graph.remove_edge(second_removed[0]));
        REQUIRE(!first.graph.remove_edge(second_removed[0]));
        for (size_t l = 0; l < first.landmarks.size(); ++l) {
            REQUIRE(first.graph.get_connected_edges(first.landmarks[l]).size() == second.graph.get_connected_edges(second.landmarks[l]).size());
        }
        REQUIRE(first.graph.solve(20, true) == second.graph.solve(20, true));
        for (size_t c = 0; c < first.cameras.size(); ++c) {
            for (size_t p = 0; p < 7; ++p) {
                REQUIRE(first.cameras[c]->get_parameters()[p] == second.cameras[c]->get_parameters()[p]);
            }
        }
        for (size_t l = 0; l < first.landmarks.size(); ++l) {
            for (size_t p = 0; p < 3; ++p) {
                REQUIRE(first.landmarks[l]->get_parameters()[p] == second.landmarks[l]->get_parameters()[p]);
            }
        }
        // A removed vertex takes its edges with it.
        const size_t connected = second.graph.get_connected_edges(second.cameras[2]).size();
        REQUIRE(connected > 0);
        REQUIRE(second.graph.remove_vertex(second.landmarks[1]));
        REQUIRE(second.graph.get_connected_edges(second.landmarks[1]).empty());
        REQUIRE(second.graph.get_connected_edges(second.cameras[2]).size() == connected - 1);
        REQUIRE(second.graph.solve(5, true) > 0);
    }

    for (const int variant : { 0, 1, 2 }) {
        // The damped step of the Schur complement solved with the sparse block Cholesky equals the dense solve of the whole
        // system, with pair blocks from a baseline edge, fixed and correlated landmarks, lines, and a landmark kept general.
        core::random_pcg rng(0x5eed00ccull);
        problem built;
        build_problem(built, rng, 7, 30, true, false);
        static_cast<void>(add_mixed_blocks(built, rng));
        std::vector<geometry::plucker> true_lines;
        add_lines(built, rng, 8, true_lines);
        if (variant > 0) {
            built.landmarks[3]->set_marginalised(false);
            optimisation::edge twice{ twice_edge() };
            REQUIRE(twice.add_vertex(built.landmarks[3]));
            REQUIRE(twice.add_vertex(built.landmarks[3]));
            twice.set_information(math::matrix<double, 0, 0>::identity(2, 2) * 1e2);
            REQUIRE(built.graph.add_edge(static_cast<optimisation::edge&&>(twice)) != nullptr);
        }
        if (variant > 1) {
            built.cameras[2]->set_fixed(true);
            built.landmarks[5]->set_marginalised(false);
        }
        std::vector<optimisation::edge*> edges;
        for (const std::vector<optimisation::vertex*>* group : { &built.cameras, &built.landmarks, &built.lines }) {
            for (optimisation::vertex* const node : *group) {
                for (optimisation::edge* const connected : built.graph.get_connected_edges(node)) {
                    if (std::find(edges.begin(), edges.end(), connected) == edges.end()) {
                        edges.push_back(connected);
                    }
                }
            }
        }
        built.graph.set_strategy(optimisation::factor_graph::strategy::dense_schur);
        for (const double lambda : { 1e-6, 1e-2, 1.0 }) {
            math::matrix<double, 0, 0> step;
            REQUIRE(built.graph.compute_damped_step(lambda, step));
            REQUIRE(!built.graph.get_diagnostics().used_square_root);
            const std::vector<double> expected = dense_damped_step(edges, step.rows(), lambda);
            double largest = 0.0;
            for (const double value : expected) {
                largest = std::max(largest, std::abs(value));
            }
            REQUIRE(largest > 0.0);
            // Barely damped the system is conditioned near 1e8, so both solves carry errors near 1e-8 of the step there.
            const double tolerance = (lambda < 1e-3) ? 1e-6 : 1e-11;
            for (size_t i = 0; i < step.rows(); ++i) {
                REQUIRE(std::abs(step[i][0] - expected[i]) < tolerance * largest);
            }
        }
    }

    {
        // The gain ratio is relative to the cost: a tiny cost gets the ratio of a large one, which an absolute 1e-3 in the
        // predicted reduction pushed to about 0.
        REQUIRE(is_value_approx(optimisation::factor_graph::gain_ratio(1e-6, 0.5e-6, 0.5e-6), 1.0, 1e-9));
        REQUIRE(is_value_approx(optimisation::factor_graph::gain_ratio(1e6, 0.5e6, 0.5e6), 1.0, 1e-9));
        // A step the model does not predict to reduce the cost is rejected, including a cost increase over a negative
        // prediction, whose ratio was positive before.
        REQUIRE(optimisation::factor_graph::gain_ratio(1.0, 2.0, -0.1) == 0.0);
        REQUIRE(optimisation::factor_graph::gain_ratio(1.0, 0.5, 0.0) == 0.0);
        REQUIRE(!(optimisation::factor_graph::gain_ratio(1.0, 0.5, std::numeric_limits<double>::quiet_NaN()) > 0.0));
        // An infinite starting cost still takes a step to a finite one.
        REQUIRE(optimisation::factor_graph::gain_ratio(std::numeric_limits<double>::infinity(), 1.0, 1.0) > 0.0);
    }

    {
        // Every step of this edge leaves the domain of its residual, so the solve ends on rejected attempts, and the residual,
        // and the robust weight it gives, must describe the vertex after the last rollback, not the last rejected state.
        optimisation::factor_graph graph;
        optimisation::vertex node{ optimisation::vertices::point() };
        const double start[3] = { 5.0, 0.0, 0.0 };
        REQUIRE(node.set_parameters(&start[0], 3));
        optimisation::vertex* const stored = graph.add_vertex(static_cast<optimisation::vertex&&>(node));
        optimisation::edge factor{ pinned_edge{ start[0] } };
        REQUIRE(factor.add_vertex(stored));
        optimisation::edge* const added = graph.add_edge(static_cast<optimisation::edge&&>(factor));
        REQUIRE(added != nullptr);
        REQUIRE(graph.solve(10, true) == 0);
        REQUIRE(graph.get_diagnostics().rejected_attempts >= 10);
        REQUIRE(stored->get_parameters()[0] == start[0]);
        REQUIRE(added->get_residual()[0][0] == (start[0] - 1.0));
        REQUIRE(added->robust_weight() == 1.0);
        REQUIRE(graph.get_current_chi(false) == ((start[0] - 1.0) * (start[0] - 1.0)));
    }

    return EXIT_SUCCESS;
}
