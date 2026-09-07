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

#include "sensor/camera.hpp"

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
    if (std::isnan(lhs) && std::isnan(rhs))
        return true;
    if (std::isnan(lhs) != std::isnan(rhs))
        return false;
    if (std::isinf(lhs) != std::isinf(rhs))
        return false;
    if (std::signbit(lhs + epsilon) != std::signbit(rhs + epsilon))
        return false;
    if (std::isinf(lhs) && std::isinf(rhs))
        return true;
    return (std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon);
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        factor_graph::factor_graph factor_graph;
    }

    {
        class random_pcg final {
        private:
            unsigned long long int state;
            unsigned long long int increment;

        public:
            random_pcg()
                : state(0x853C49E6748FEA9Bull)
                , increment(0xDA3E39CB94B95BDBull) {
            }

            unsigned int get_random_raw() {
                // Save current state for output calculation.
                unsigned long long int state_previous = this->state;
                // Advance internal state.
                this->state = state_previous * 0x5851F42D4C957F2Dull + this->increment;
                // Calculate output function.
                unsigned int state_shift_xor_shift = static_cast<unsigned int>(((state_previous >> 18u) ^ state_previous) >> 27u);
                const int rotation = static_cast<int>(state_previous >> 59u);
                return (state_shift_xor_shift >> rotation) | (state_shift_xor_shift << ((-rotation) & 31));
            }
        };

        random_pcg rng;

        camera::pinhole camera_model(std::vector<double>{ 1.0, 1.0, 0.0, 0.0 }.data(), 4);
        // gtl::pinhole_arctangent<double> camera_model(std::vector<double>{1.0,1.0,0.0,0.0,0.5}.data(), 5);

        factor_graph::factor_graph factor_graph(true);

        std::vector<lie::se3<double>> cameras{
            lie::se3<double>::identity(),
            lie::se3<double>(lie::so3<double>::identity(), matrix::matrix<double, 3, 1>{ { 0.1, 0.0, 0.0 } }),
            lie::se3<double>(lie::so3<double>::identity(), matrix::matrix<double, 3, 1>{ { 0.0, 0.1, 0.0 } }),
            lie::se3<double>(lie::so3<double>::identity(), matrix::matrix<double, 3, 1>{ { 0.0, 0.0, 0.1 } }),
            lie::se3<double>(lie::so3<double>::identity(), matrix::matrix<double, 3, 1>{ { -0.1, 0.0, 0.0 } }),
            lie::se3<double>(lie::so3<double>::identity(), matrix::matrix<double, 3, 1>{ { 0.0, -0.1, 0.0 } }),
            lie::se3<double>(lie::so3<double>::identity(), matrix::matrix<double, 3, 1>{ { 0.0, 0.0, -0.1 } })
        };

        std::vector<matrix::matrix<double, 3, 1>> landmarks = {
            { { 0.0, 0.0, 1.0 } },
            { { 0.0, 1.0, 1.0 } },
            { { 1.0, 0.0, 1.0 } }
        };

        std::vector<std::unique_ptr<factor_graph::vertex_base>> camera_vertexes;
        std::vector<std::unique_ptr<factor_graph::vertex_base>> landmark_vertexes;
        std::vector<std::unique_ptr<factor_graph::edge_base>> observation_edges;

        double camera_parameters[4];
        camera_model.get_parameters(camera_parameters, 4);

        for (size_t i = 0; i < cameras.size(); ++i) {
            camera_vertexes.push_back(std::make_unique<factor_graph::vertex_pose>());
            camera_vertexes[i]->set_parameters(matrix::matrix<double, 0, 0>(7, 1, matrix::matrix<double, 7, 1>{ { cameras[i].translation()[0], cameras[i].translation()[1], cameras[i].translation()[2], cameras[i].rotation().get_quaternion()[1], cameras[i].rotation().get_quaternion()[2], cameras[i].rotation().get_quaternion()[3], cameras[i].rotation().get_quaternion()[0] } }.data()));
            camera_vertexes[i]->set_fixed(i == 0);
            factor_graph.add_vertex(camera_vertexes[i].get());
        }

        std::vector<matrix::matrix<double, 3, 1>> noisy_landmarks(landmarks.size());
        for (size_t i = 0; i < landmarks.size(); ++i) {
            landmark_vertexes.push_back(std::make_unique<factor_graph::vertex_point_xyz>());
            for (size_t j = 0; j < 3; ++j) {
                noisy_landmarks[i][j] = landmarks[i][j] + (static_cast<double>(static_cast<int>(rng.get_random_raw() % 10) - 5) * 0.01);
                auto parameters = landmark_vertexes[i]->get_parameters();
                parameters[j][0] = noisy_landmarks[i][j];
                landmark_vertexes[i]->set_parameters(parameters);
            }
            landmark_vertexes[i]->set_fixed(false);
            landmark_vertexes[i]->set_marginalised(true);
            factor_graph.add_vertex(landmark_vertexes[i].get());
            std::fprintf(stdout, "NOISY: % f % f % f\n", landmark_vertexes[i]->get_parameters()[0][0], landmark_vertexes[i]->get_parameters()[1][0], landmark_vertexes[i]->get_parameters()[2][0]);
        }

        int camera_id = 0;
        for (const lie::se3<double>& camera : cameras) {
            int landmark_id = 0;
            for (const matrix::matrix<double, 3, 1>& landmark : landmarks) {
                matrix::matrix<double, 3, 1> world_point = camera * landmark;
                matrix::matrix<double, 2, 1> point;
                REQUIRE(camera_model.project(world_point.data(), point.data()));
                camera::pinhole edge_camera(camera_parameters, 4);
                std::unique_ptr<factor_graph::edge_base> m = std::make_unique<factor_graph::edge_reprojection<camera::pinhole>>(edge_camera);
                m->set_observation(matrix::matrix<double, 0, 0>(2, 1, matrix::matrix<double, 2, 1>{ { point[0] + (static_cast<double>(static_cast<int>(rng.get_random_raw() % 10) - 5) * 0.0001), point[1] + (static_cast<double>(static_cast<int>(rng.get_random_raw() % 10) - 5) * 0.0001) } }.data()));
                m->add_vertex(camera_vertexes[static_cast<size_t>(camera_id)].get());
                m->add_vertex(landmark_vertexes[static_cast<size_t>(landmark_id)].get());
                factor_graph.add_edge(m.get());
                std::fprintf(stdout, "POINT [%d]: % f % f\n", camera_id, m->get_observation()[0][0], m->get_observation()[1][0]);
                observation_edges.emplace_back(std::move(m));
                ++landmark_id;
            }
            ++camera_id;
        }

        double initialChi2 = factor_graph.get_current_chi();
        // Exercise the relative-convergence criterion here.
        REQUIRE(factor_graph.solve(50, true));

        std::fprintf(stdout, "LANDMARK: ORIGINAL --> NOISY --> OPTIMISED\n");
        double error_noisy = 0;
        double error_optimised = 0;
        for (size_t i = 0; i < landmarks.size(); ++i) {
            matrix::matrix<double, 3, 1> result_landmark = { {
                landmark_vertexes[i]->get_parameters()[0][0],
                landmark_vertexes[i]->get_parameters()[1][0],
                landmark_vertexes[i]->get_parameters()[2][0],
            } };
            std::fprintf(stdout, "LANDMARK: %f %f %f --> %f %f %f --> %f %f %f\n", landmarks[i][0], landmarks[i][1], landmarks[i][2], noisy_landmarks[i][0], noisy_landmarks[i][1], noisy_landmarks[i][2], result_landmark[0], result_landmark[1], result_landmark[2]);
            error_noisy += std::sqrt((landmarks[i] - noisy_landmarks[i]).get_length_squared());
            error_optimised += std::sqrt((landmarks[i] - result_landmark).get_length_squared());
        }
        std::fprintf(stdout, "Error: %f --> %f (Chi2: %f --> %f)\n", error_noisy, error_optimised, initialChi2, factor_graph.get_current_chi());

        REQUIRE(error_optimised < error_noisy);
        REQUIRE(!is_value_approx(factor_graph.get_current_chi(), initialChi2));
        REQUIRE(is_value_approx(factor_graph.get_current_chi(), 0.0, 1e-5));
        REQUIRE(factor_graph.get_current_chi() < initialChi2);
    }

    {
        // Construct a graph whose reduced pose system is not positive definite, forcing the Cholesky solver to fail.
        // The solve should treat failed attempts as bad steps and complete without corrupting the vertex states.
        camera::pinhole camera_model(std::vector<double>{ 1.0, 1.0, 0.0, 0.0 }.data(), 4);
        double camera_parameters[4];
        camera_model.get_parameters(camera_parameters, 4);

        factor_graph::factor_graph factor_graph(true);

        std::unique_ptr<factor_graph::vertex_base> fixed_camera_vertex = std::make_unique<factor_graph::vertex_pose>();
        fixed_camera_vertex->set_parameters(matrix::matrix<double, 0, 0>(7, 1, matrix::matrix<double, 7, 1>{ { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 } }.data()));
        fixed_camera_vertex->set_fixed(true);
        factor_graph.add_vertex(fixed_camera_vertex.get());

        std::unique_ptr<factor_graph::vertex_base> free_camera_vertex = std::make_unique<factor_graph::vertex_pose>();
        free_camera_vertex->set_parameters(matrix::matrix<double, 0, 0>(7, 1, matrix::matrix<double, 7, 1>{ { 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 } }.data()));
        free_camera_vertex->set_fixed(false);
        factor_graph.add_vertex(free_camera_vertex.get());

        std::unique_ptr<factor_graph::vertex_base> landmark_vertex = std::make_unique<factor_graph::vertex_point_xyz>();
        landmark_vertex->set_parameters(matrix::matrix<double, 0, 0>(3, 1, matrix::matrix<double, 3, 1>{ { 0.1, 0.2, 1.0 } }.data()));
        landmark_vertex->set_fixed(true);
        landmark_vertex->set_marginalised(true);
        factor_graph.add_vertex(landmark_vertex.get());

        camera::pinhole edge_camera(camera_parameters, 4);
        std::unique_ptr<factor_graph::edge_base> edge = std::make_unique<factor_graph::edge_reprojection<camera::pinhole>>(edge_camera);
        edge->set_observation(matrix::matrix<double, 0, 0>(2, 1, matrix::matrix<double, 2, 1>{ { 0.15, 0.25 } }.data()));
        edge->add_vertex(free_camera_vertex.get());
        edge->add_vertex(landmark_vertex.get());
        // A negative definite information matrix makes the pose Hessian block negative definite until the damping grows large enough.
        edge->set_information(-1.0 * matrix::matrix<double, 0, 0>::identity(2, 2));
        factor_graph.add_edge(edge.get());

        static_cast<void>(factor_graph.solve(10));

        const matrix::matrix<double, 0, 0>& optimised_parameters = free_camera_vertex->get_parameters();
        REQUIRE(optimised_parameters.rows() == 7);
        REQUIRE(optimised_parameters.cols() == 1);
        for (size_t i = 0; i < 7; ++i) {
            REQUIRE(std::isfinite(optimised_parameters[i][0]));
        }
    }

    {
        // A landmark behind the camera must not produce a zero residual/jacobian, and the normal (in front) case must be completely unaffected by the `behind_camera_penalty`.
        camera::pinhole camera_model(std::vector<double>{ 1.0, 1.0, 0.0, 0.0 }.data(), 4);
        double camera_parameters[4];
        camera_model.get_parameters(camera_parameters, 4);

        std::unique_ptr<factor_graph::vertex_base> camera_vertex = std::make_unique<factor_graph::vertex_pose>();
        camera_vertex->set_parameters(matrix::matrix<double, 0, 0>(7, 1, matrix::matrix<double, 7, 1>{ { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 } }.data()));
        camera_vertex->set_fixed(false);

        // Normal case: landmark in front of the camera (Z = 1 > 0).
        {
            std::unique_ptr<factor_graph::vertex_base> landmark_vertex = std::make_unique<factor_graph::vertex_point_xyz>();
            landmark_vertex->set_parameters(matrix::matrix<double, 0, 0>(3, 1, matrix::matrix<double, 3, 1>{ { 0.0, 0.0, 1.0 } }.data()));
            landmark_vertex->set_fixed(false);
            landmark_vertex->set_marginalised(true);

            camera::pinhole edge_camera(camera_parameters, 4);
            std::unique_ptr<factor_graph::edge_base> edge = std::make_unique<factor_graph::edge_reprojection<camera::pinhole>>(edge_camera);
            edge->set_observation(matrix::matrix<double, 0, 0>(2, 1, matrix::matrix<double, 2, 1>{ { 0.1, -0.2 } }.data()));
            edge->add_vertex(camera_vertex.get());
            edge->add_vertex(landmark_vertex.get());

            edge->compute_residual();
            edge->compute_jacobians();

            REQUIRE(edge->get_residual()[0][0] == 0.1);
            REQUIRE(edge->get_residual()[1][0] == -0.2);

            const matrix::matrix<double, 0, 0>& jacobian_pose = edge->get_jacobians()[0];
            const double expected_pose[2][6] = {
                { 0.0, -1.0, 0.0, -1.0, 0.0, 0.0 },
                { 1.0, 0.0, 0.0, 0.0, -1.0, 0.0 }
            };
            for (size_t r = 0; r < 2; ++r) {
                for (size_t c = 0; c < 6; ++c) {
                    REQUIRE(jacobian_pose[r][c] == expected_pose[r][c]);
                }
            }

            const matrix::matrix<double, 0, 0>& jacobian_landmark = edge->get_jacobians()[1];
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
            std::unique_ptr<factor_graph::vertex_base> landmark_vertex = std::make_unique<factor_graph::vertex_point_xyz>();
            landmark_vertex->set_parameters(matrix::matrix<double, 0, 0>(3, 1, matrix::matrix<double, 3, 1>{ { 0.1, 0.2, -10.0 } }.data()));
            landmark_vertex->set_fixed(false);
            landmark_vertex->set_marginalised(true);

            camera::pinhole edge_camera(camera_parameters, 4);
            std::unique_ptr<factor_graph::edge_base> edge = std::make_unique<factor_graph::edge_reprojection<camera::pinhole>>(edge_camera);
            edge->set_observation(matrix::matrix<double, 0, 0>(2, 1, matrix::matrix<double, 2, 1>{ { 0.1, -0.2 } }.data()));
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
            const matrix::matrix<double, 0, 0>& jacobian_pose = edge->get_jacobians()[0];
            const matrix::matrix<double, 0, 0>& jacobian_landmark = edge->get_jacobians()[1];

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
