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

#include "optimisation/edges/relative_similarity.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/similarity.hpp"

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

static math::sim3<double> make_similarity(double rx, double ry, double rz, double tx, double ty, double tz, double scale) {
    return math::sim3<double>(math::se3<double>(math::so3<double>::exp({ { rx, ry, rz } }), { { tx, ty, tz } }), scale);
}

static optimisation::vertex make_vertex(const math::sim3<double>& similarity, bool fixed) {
    optimisation::vertex vertex{ optimisation::vertices::similarity() };
    const double parameters[8] = { similarity.transformation().translation()[0], similarity.transformation().translation()[1], similarity.transformation().translation()[2], similarity.transformation().rotation().get_quaternion()[1], similarity.transformation().rotation().get_quaternion()[2], similarity.transformation().rotation().get_quaternion()[3], similarity.transformation().rotation().get_quaternion()[0], similarity.scale() };
    REQUIRE(vertex.set_parameters(&parameters[0], 8));
    vertex.set_fixed(fixed);
    return vertex;
}

static math::sim3<double> read_vertex(const optimisation::vertex& vertex) {
    const double* const parameters = vertex.get_parameters();
    return math::sim3<double>(math::se3<double>(math::so3<double>(parameters[6], parameters[3], parameters[4], parameters[5]), { { parameters[0], parameters[1], parameters[2] } }), parameters[7]);
}

static optimisation::edge make_edge(const math::sim3<double>& measurement, optimisation::vertex* first, optimisation::vertex* second) {
    optimisation::edge edge{ optimisation::edges::relative_similarity() };
    const double observation[8] = { measurement.transformation().translation()[0], measurement.transformation().translation()[1], measurement.transformation().translation()[2], measurement.transformation().rotation().get_quaternion()[1], measurement.transformation().rotation().get_quaternion()[2], measurement.transformation().rotation().get_quaternion()[3], measurement.transformation().rotation().get_quaternion()[0], measurement.scale() };
    edge.set_observation(math::matrix<double, 0, 0>(8, 1, &observation[0]));
    REQUIRE(edge.add_vertex(first));
    REQUIRE(edge.add_vertex(second));
    return edge;
}

static double distance(const math::sim3<double>& lhs, const math::sim3<double>& rhs) {
    return std::sqrt((lhs.inverse() * rhs).log().get_length_squared());
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(optimisation::edges::relative_similarity) <= optimisation::edge::maximum_size, "The relative similarity edge must fit.");

    {
        const math::sim3<double> first = make_similarity(0.1, -0.2, 0.3, 1.0, 2.0, 3.0, 1.5);
        const math::sim3<double> second = make_similarity(-0.3, 0.1, 0.2, -1.0, 0.5, 2.0, 0.8);
        optimisation::vertex first_vertex = make_vertex(first, false);
        optimisation::vertex second_vertex = make_vertex(second, false);

        optimisation::edge consistent = make_edge(first.inverse() * second, &first_vertex, &second_vertex);
        REQUIRE(consistent.get_residual().rows() == 7);
        consistent.compute_residual();
        for (size_t i = 0; i < 7; ++i) {
            REQUIRE(is_value_approx(consistent.get_residual()[i][0], 0.0));
        }
        REQUIRE(is_value_approx(consistent.chi2(), 0.0));

        const math::sim3<double> disagreement = make_similarity(0.05, 0.0, -0.02, 0.1, -0.2, 0.3, 1.1);
        optimisation::edge inconsistent = make_edge((first.inverse() * second) * disagreement, &first_vertex, &second_vertex);
        inconsistent.compute_residual();
        const math::matrix<double, 7, 1> expected = disagreement.inverse().log();
        for (size_t i = 0; i < 7; ++i) {
            REQUIRE(is_value_approx(inconsistent.get_residual()[i][0], expected[i]));
        }
        REQUIRE(inconsistent.chi2() > 0.0);

        inconsistent.compute_jacobians();
        REQUIRE(inconsistent.get_jacobians().size() == 2);
        for (size_t v = 0; v < 2; ++v) {
            const math::matrix<double, 0, 0>& jacobian = inconsistent.get_jacobians()[v];
            REQUIRE(jacobian.rows() == 7);
            REQUIRE(jacobian.cols() == 7);
            bool nonzero = false;
            for (size_t r = 0; r < 7; ++r) {
                for (size_t c = 0; c < 7; ++c) {
                    REQUIRE(std::isfinite(jacobian[r][c]));
                    nonzero = nonzero || (jacobian[r][c] != 0.0);
                }
            }
            REQUIRE(nonzero);
        }
    }

    {
        const math::sim3<double> first = make_similarity(-0.2, 0.4, -0.1, 2.0, -1.0, 0.5, 1.3);
        const math::sim3<double> second = make_similarity(0.3, 0.2, 0.5, -0.5, 1.5, 2.0, 0.7);
        optimisation::vertex first_vertex = make_vertex(first, false);
        optimisation::vertex second_vertex = make_vertex(second, false);

        const math::sim3<double> disagreement = make_similarity(0.05, -0.03, 0.04, 0.1, 0.2, -0.1, 1.1);
        optimisation::edge edge = make_edge((first.inverse() * second) * disagreement, &first_vertex, &second_vertex);

        edge.compute_residual();
        edge.compute_jacobians();
        REQUIRE(edge.get_jacobians().size() == 2);
        for (size_t v = 0; v < 2; ++v) {
            REQUIRE(edge.get_jacobians()[v].rows() == 7);
            REQUIRE(edge.get_jacobians()[v].cols() == 7);
        }

        const double delta = 1e-5;
        const double scalar = 1.0 / (2.0 * delta);
        double backup_parameters[8] = {};
        for (size_t v = 0; v < 2; ++v) {
            optimisation::vertex* const vertex = (v == 0) ? &first_vertex : &second_vertex;
            REQUIRE(vertex->get_parameters(&backup_parameters[0], 8));
            const math::matrix<double, 0, 0>& analytic = edge.get_jacobians()[v];
            for (size_t d = 0; d < 7; ++d) {
                double add_delta[8] = {};
                add_delta[d] = delta;
                vertex->plus(&add_delta[0]);
                edge.compute_residual();
                const math::matrix<double, 0, 0> error_addition = edge.get_residual();
                REQUIRE(vertex->set_parameters(&backup_parameters[0], 8));

                add_delta[d] = -delta;
                vertex->plus(&add_delta[0]);
                edge.compute_residual();
                const math::matrix<double, 0, 0> error_subtract = edge.get_residual();
                REQUIRE(vertex->set_parameters(&backup_parameters[0], 8));

                for (size_t r = 0; r < 7; ++r) {
                    const double numeric = scalar * (error_addition[r][0] - error_subtract[r][0]);
                    REQUIRE(is_value_approx(analytic[r][d], numeric, 1e-6));
                }
            }
        }
    }

    {
        const size_t keyframe_count = 8;
        const double radius = 2.0;
        const double two_pi = 2.0 * 3.14159265358979323846;

        std::vector<math::sim3<double>> truth(keyframe_count);
        for (size_t k = 0; k < keyframe_count; ++k) {
            const double angle = two_pi * static_cast<double>(k) / static_cast<double>(keyframe_count);
            truth[k] = make_similarity(0.0, 0.0, angle, radius * std::cos(angle), radius * std::sin(angle), 0.0, 1.0);
        }

        std::vector<math::sim3<double>> drifted(keyframe_count);
        drifted[0] = truth[0];
        const math::sim3<double> corruption = make_similarity(0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 1.05);
        for (size_t k = 0; k + 1 < keyframe_count; ++k) {
            drifted[k + 1] = drifted[k] * (truth[k].inverse() * truth[k + 1]) * corruption;
        }
        REQUIRE(drifted[keyframe_count - 1].scale() > 1.2);

        optimisation::factor_graph factor_graph;
        std::vector<optimisation::vertex*> vertices(keyframe_count);
        for (size_t k = 0; k < keyframe_count; ++k) {
            vertices[k] = factor_graph.add_vertex(make_vertex(drifted[k], k == 0));
            REQUIRE(vertices[k] != nullptr);
        }
        for (size_t k = 0; k + 1 < keyframe_count; ++k) {
            REQUIRE(factor_graph.add_edge(make_edge(truth[k].inverse() * truth[k + 1], vertices[k], vertices[k + 1])) != nullptr);
        }
        REQUIRE(factor_graph.add_edge(make_edge(truth[keyframe_count - 1].inverse() * truth[0], vertices[keyframe_count - 1], vertices[0])) != nullptr);

        double error_before = 0.0;
        for (size_t k = 0; k < keyframe_count; ++k) {
            error_before += distance(read_vertex(*vertices[k]), truth[k]);
        }
        const double chi2_before = factor_graph.get_current_chi();
        REQUIRE(chi2_before > 0.0);

        REQUIRE(factor_graph.solve(100, true));

        double error_after = 0.0;
        for (size_t k = 0; k < keyframe_count; ++k) {
            error_after += distance(read_vertex(*vertices[k]), truth[k]);
        }
        const double chi2_after = factor_graph.get_current_chi();
        REQUIRE(chi2_after < chi2_before);
        REQUIRE(chi2_after < 1e-9);
        REQUIRE(error_after < 0.1 * error_before);
        REQUIRE(std::abs(read_vertex(*vertices[keyframe_count - 1]).scale() - 1.0) < 1e-3);
        REQUIRE(distance(read_vertex(*vertices[0]), truth[0]) < 1e-12);
    }

    {
        const size_t keyframe_count = 5;
        std::vector<math::sim3<double>> truth(keyframe_count);
        for (size_t k = 0; k < keyframe_count; ++k) {
            const double t = static_cast<double>(k);
            truth[k] = make_similarity(0.01 * t, -0.02 * t, 0.05 * t, 0.5 * t, -0.3 * t, 0.1 * t, 1.0);
        }

        optimisation::factor_graph factor_graph;
        std::vector<optimisation::vertex*> vertices(keyframe_count);
        for (size_t k = 0; k < keyframe_count; ++k) {
            vertices[k] = factor_graph.add_vertex(make_vertex(truth[k], k == 0));
        }
        for (size_t k = 0; k + 1 < keyframe_count; ++k) {
            factor_graph.add_edge(make_edge(truth[k].inverse() * truth[k + 1], vertices[k], vertices[k + 1]));
        }

        REQUIRE(factor_graph.get_current_chi() < 1e-18);
        factor_graph.solve(10, true);
        for (size_t k = 0; k < keyframe_count; ++k) {
            REQUIRE(distance(read_vertex(*vertices[k]), truth[k]) < 1e-9);
        }
    }

    {
        std::vector<math::sim3<double>> truth(3);
        truth[0] = make_similarity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        truth[1] = make_similarity(0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 1.0);
        truth[2] = make_similarity(0.0, 0.0, 0.4, 2.0, 0.5, 0.0, 1.0);

        optimisation::factor_graph factor_graph;
        optimisation::vertex* first = factor_graph.add_vertex(make_vertex(truth[0], true));
        optimisation::vertex* middle = factor_graph.add_vertex(make_vertex(make_similarity(0.3, -0.1, 0.6, 1.7, 0.9, 0.4, 1.4), false));
        optimisation::vertex* last = factor_graph.add_vertex(make_vertex(truth[2], true));
        factor_graph.add_edge(make_edge(truth[0].inverse() * truth[1], first, middle));
        factor_graph.add_edge(make_edge(truth[1].inverse() * truth[2], middle, last));

        REQUIRE(factor_graph.solve(100, true));
        REQUIRE(distance(read_vertex(*middle), truth[1]) < 1e-6);
        REQUIRE(distance(read_vertex(*first), truth[0]) < 1e-15);
        REQUIRE(distance(read_vertex(*last), truth[2]) < 1e-15);
        REQUIRE(factor_graph.get_current_chi() < 1e-12);
    }

    return EXIT_SUCCESS;
}
