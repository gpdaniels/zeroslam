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
#include "math/lie.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/reprojection.hpp"
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

    return EXIT_SUCCESS;
}
