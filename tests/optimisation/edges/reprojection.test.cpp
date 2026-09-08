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

#include "optimisation/edges/reprojection.hpp"

#include "optimisation/edge.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/point.hpp"
#include "optimisation/vertices/pose.hpp"
#include "sensor/camera/model.hpp"
#include "sensor/camera/pinhole.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <utility>
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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(optimisation::edges::reprojection) <= optimisation::edge::maximum_size, "The reprojection edge must fit.");

    {
        const sensor::camera::pinhole<double> camera_model(std::vector<double>{ 1.0, 1.0, 0.0, 0.0 }.data(), 4);

        std::unique_ptr<optimisation::vertex> camera_vertex = std::make_unique<optimisation::vertex>(optimisation::vertices::pose());
        const double camera_parameters_pose[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
        camera_vertex->set_parameters(&camera_parameters_pose[0], 7);
        camera_vertex->set_fixed(false);

        {
            std::unique_ptr<optimisation::vertex> landmark_vertex = std::make_unique<optimisation::vertex>(optimisation::vertices::point());
            const double landmark_location[3] = { 0.0, 0.0, 1.0 };
            landmark_vertex->set_parameters(&landmark_location[0], 3);
            landmark_vertex->set_fixed(false);
            landmark_vertex->set_marginalised(true);

            std::unique_ptr<optimisation::edge> edge = std::make_unique<optimisation::edge>(optimisation::edges::reprojection(sensor::camera::model<double>(camera_model)));
            edge->set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { 0.1, -0.2 } }.data()));
            edge->add_vertex(camera_vertex.get());
            edge->add_vertex(landmark_vertex.get());

            edge->compute_residual();
            edge->compute_jacobians();

            REQUIRE(edge->get_residual()[0][0] == 0.1);
            REQUIRE(edge->get_residual()[1][0] == -0.2);

            const math::matrix<double, 0, 0>& jacobian_pose = edge->get_jacobians()[0];
            const double expected_pose[2][6] = {
                { 0.0, -1.0, 0.0, -1.0, 0.0, 0.0 },
                { 1.0, 0.0, 0.0, 0.0, -1.0, 0.0 }
            };
            for (size_t r = 0; r < 2; ++r) {
                for (size_t c = 0; c < 6; ++c) {
                    REQUIRE(jacobian_pose[r][c] == expected_pose[r][c]);
                }
            }

            const math::matrix<double, 0, 0>& jacobian_landmark = edge->get_jacobians()[1];
            const double expected_landmark[2][3] = {
                { -1.0, 0.0, 0.0 },
                { 0.0, -1.0, 0.0 }
            };
            for (size_t r = 0; r < 2; ++r) {
                for (size_t c = 0; c < 3; ++c) {
                    REQUIRE(jacobian_landmark[r][c] == expected_landmark[r][c]);
                }
            }
        }

        {
            std::unique_ptr<optimisation::vertex> landmark_vertex = std::make_unique<optimisation::vertex>(optimisation::vertices::point());
            const double landmark_location[3] = { 0.1, 0.2, -10.0 };
            landmark_vertex->set_parameters(&landmark_location[0], 3);
            landmark_vertex->set_fixed(false);
            landmark_vertex->set_marginalised(true);

            std::unique_ptr<optimisation::edge> edge = std::make_unique<optimisation::edge>(optimisation::edges::reprojection(sensor::camera::model<double>(camera_model)));
            edge->set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { 0.1, -0.2 } }.data()));
            edge->add_vertex(camera_vertex.get());
            edge->add_vertex(landmark_vertex.get());

            edge->compute_residual();

            REQUIRE(std::isfinite(edge->get_residual()[0][0]));
            REQUIRE(std::isfinite(edge->get_residual()[1][0]));
            REQUIRE(!is_value_approx(edge->get_residual()[0][0], 0.0));
            REQUIRE(!is_value_approx(edge->get_residual()[1][0], 0.0));

            const double chi2 = edge->chi2();
            REQUIRE(std::isfinite(chi2));
            REQUIRE(chi2 > 1.0);

            edge->compute_jacobians();
            const math::matrix<double, 0, 0>& jacobian_pose = edge->get_jacobians()[0];
            const math::matrix<double, 0, 0>& jacobian_landmark = edge->get_jacobians()[1];

            bool jacobian_pose_nonzero = false;
            for (size_t r = 0; r < 2; ++r) {
                for (size_t c = 0; c < 6; ++c) {
                    REQUIRE(std::isfinite(jacobian_pose[r][c]));
                    jacobian_pose_nonzero = jacobian_pose_nonzero || (jacobian_pose[r][c] != 0.0);
                }
            }
            bool jacobian_landmark_nonzero = false;
            for (size_t r = 0; r < 2; ++r) {
                for (size_t c = 0; c < 3; ++c) {
                    REQUIRE(std::isfinite(jacobian_landmark[r][c]));
                    jacobian_landmark_nonzero = jacobian_landmark_nonzero || (jacobian_landmark[r][c] != 0.0);
                }
            }
            REQUIRE(jacobian_pose_nonzero);
            REQUIRE(jacobian_landmark_nonzero);
            REQUIRE(jacobian_landmark[0][2] < 0.0);
            REQUIRE(jacobian_landmark[1][2] < 0.0);
        }
    }

    return EXIT_SUCCESS;
}
