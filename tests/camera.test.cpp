/*
Copyright (C) 2025 Geoffrey Daniels. https://gpdaniels.com/

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

#include "camera.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>

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
        camera::pinhole pinhole;
        double parameters[4]{};
        pinhole.get_parameters(&parameters[0], 4);
        REQUIRE(is_value_approx(parameters[0], 1.0f));
        REQUIRE(is_value_approx(parameters[1], 1.0f));
        REQUIRE(is_value_approx(parameters[2], 0.5f));
        REQUIRE(is_value_approx(parameters[3], 0.5f));
        const double parameters_new[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
        pinhole.set_parameters(&parameters_new[0], 4);
        pinhole.get_parameters(&parameters[0], 4);
        REQUIRE(is_value_approx(parameters_new[0], parameters[0]));
        REQUIRE(is_value_approx(parameters_new[1], parameters[1]));
        REQUIRE(is_value_approx(parameters_new[2], parameters[2]));
        REQUIRE(is_value_approx(parameters_new[3], parameters[3]));
    }

    {
        camera::pinhole pinhole;
        const double parameters_new[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
        pinhole.set_parameters(&parameters_new[0], 4);
        double parameters[4];
        pinhole.get_parameters(&parameters[0], 4);
        REQUIRE(is_value_approx(parameters_new[0], parameters[0]));
        REQUIRE(is_value_approx(parameters_new[1], parameters[1]));
        REQUIRE(is_value_approx(parameters_new[2], parameters[2]));
        REQUIRE(is_value_approx(parameters_new[3], parameters[3]));
    }

    {
        {
            camera::pinhole pinhole;
            const double world_point[3] = { 0, 0, 1 };
            const double image_point_expected[2] = { 0.5f, 0.5f };
            double image_point[2]{};
            REQUIRE(pinhole.project(&world_point[0], &image_point[0]));
            REQUIRE(is_value_approx(image_point[0], image_point_expected[0]));
            REQUIRE(is_value_approx(image_point[1], image_point_expected[1]));
        }

        {
            camera::pinhole pinhole;
            const double world_point[3] = { 0, 0, 1 };
            const double image_point_expected[2] = { 0.5f, 0.5f };
            const double jacobian_projection_expected[2][3] = { { 1, 0, 0 }, { 0, 1, 0 } };
            double image_point[2]{};
            double jacobian_projection[2 * 3]{};
            REQUIRE(pinhole.project(&world_point[0], &image_point[0], &jacobian_projection[0]));
            REQUIRE(is_value_approx(image_point[0], image_point_expected[0]));
            REQUIRE(is_value_approx(image_point[1], image_point_expected[1]));
            for (size_t j = 0; j < 2; ++j) {
                for (size_t k = 0; k < 3; ++k) {
                    REQUIRE(is_value_approx(jacobian_projection[j * 3 + k], jacobian_projection_expected[j][k]));
                }
            }
        }
        {
            camera::pinhole pinhole;
            const double world_point[3] = { 0, 0, 1 };
            const double image_point_expected[2] = { 0.5f, 0.5f };
            const double jacobian_parameter_expected[2][pinhole.parameter_count] = { { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
            double image_point[2]{};
            double jacobian_parameter[2 * pinhole.parameter_count]{};
            REQUIRE(pinhole.project(&world_point[0], &image_point[0], nullptr, &jacobian_parameter[0]));
            REQUIRE(is_value_approx(image_point[0], image_point_expected[0]));
            REQUIRE(is_value_approx(image_point[1], image_point_expected[1]));
            for (size_t j = 0; j < 2; ++j) {
                for (size_t k = 0; k < pinhole.parameter_count; ++k) {
                    REQUIRE(is_value_approx(jacobian_parameter[j * pinhole.parameter_count + k], jacobian_parameter_expected[j][k]));
                }
            }
        }
        {
            camera::pinhole pinhole;
            const double world_point[3] = { -1, 1, 2 };
            const double image_point_expected[2] = { 0.0f, 1.0f };
            const double jacobian_projection_expected[2][3] = { { 0.5, 0, 0.25 }, { 0, 0.5, -0.25 } };
            const double jacobian_parameter_expected[2][pinhole.parameter_count] = { { -0.5f, 0, 1, 0 }, { 0, 0.5f, 0, 1 } };
            double image_point[2]{};
            double jacobian_projection[2 * 3]{};
            double jacobian_parameter[2 * pinhole.parameter_count]{};
            REQUIRE(pinhole.project(&world_point[0], &image_point[0], &jacobian_projection[0], &jacobian_parameter[0]));
            REQUIRE(is_value_approx(image_point[0], image_point_expected[0]));
            REQUIRE(is_value_approx(image_point[1], image_point_expected[1]));
            for (size_t j = 0; j < 2; ++j) {
                for (size_t k = 0; k < 3; ++k) {
                    REQUIRE(is_value_approx(jacobian_projection[j * 3 + k], jacobian_projection_expected[j][k]));
                }
                for (size_t k = 0; k < pinhole.parameter_count; ++k) {
                    REQUIRE(is_value_approx(jacobian_parameter[j * pinhole.parameter_count + k], jacobian_parameter_expected[j][k]));
                }
            }
        }
    }

    {
        {
            camera::pinhole pinhole;
            const double image_point[2] = { 0.5, 0.5 };
            const double world_ray_expected[3] = { 0, 0, 1 };
            double world_ray[3]{};
            REQUIRE(pinhole.unproject(&image_point[0], &world_ray[0]));
            REQUIRE(is_value_approx(world_ray[0], world_ray_expected[0]));
            REQUIRE(is_value_approx(world_ray[1], world_ray_expected[1]));
            REQUIRE(is_value_approx(world_ray[2], world_ray_expected[2]));
        }
        {
            camera::pinhole pinhole;
            const double image_point[2] = { 0.5, 0.5 };
            const double world_ray_expected[3] = { 0, 0, 1 };
            double jacobian_unprojection_expected[3][2] = { { 1, 0 }, { 0, 1 }, { 0, 0 } };
            double world_ray[3]{};
            double jacobian_unprojection[3 * 2]{};
            REQUIRE(pinhole.unproject(&image_point[0], &world_ray[0], &jacobian_unprojection[0]));
            REQUIRE(is_value_approx(world_ray[0], world_ray_expected[0]));
            REQUIRE(is_value_approx(world_ray[1], world_ray_expected[1]));
            REQUIRE(is_value_approx(world_ray[2], world_ray_expected[2]));
            for (size_t j = 0; j < 3; ++j) {
                for (size_t k = 0; k < 2; ++k) {
                    REQUIRE(is_value_approx(jacobian_unprojection[j * 2 + k], jacobian_unprojection_expected[j][k]));
                }
            }
        }
    }

    {
        camera::pinhole pinhole;
        const double parameters_new[4] = { 525, 525, 640 / 2, 480 / 2 };
        pinhole.set_parameters(&parameters_new[0], 4);
        double parameters[4];
        pinhole.get_parameters(&parameters[0], 4);
        REQUIRE(is_value_approx(parameters_new[0], parameters[0]));
        REQUIRE(is_value_approx(parameters_new[1], parameters[1]));
        REQUIRE(is_value_approx(parameters_new[2], parameters[2]));
        REQUIRE(is_value_approx(parameters_new[3], parameters[3]));
        const double image_point_expected[2] = { 640 / 4, 3 * (480 / 4) };
        const double world_ray_expected[3] = {
            (image_point_expected[0] - parameters_new[2]) / parameters_new[0],
            (image_point_expected[1] - parameters_new[3]) / parameters_new[1],
            1
        };
        double world_ray[3]{};
        REQUIRE(pinhole.unproject(&image_point_expected[0], &world_ray[0]));
        REQUIRE(is_value_approx(world_ray[0], world_ray_expected[0]));
        REQUIRE(is_value_approx(world_ray[1], world_ray_expected[1]));
        REQUIRE(is_value_approx(world_ray[2], world_ray_expected[2]));
        double image_point[2]{};
        REQUIRE(pinhole.project(&world_ray[0], &image_point[0]));
        REQUIRE(is_value_approx(image_point[0], image_point_expected[0]));
        REQUIRE(is_value_approx(image_point[1], image_point_expected[1]));
    }

    {
        const double parameters[4] = { 525, 525, 640 / 2, 480 / 2 };
        const camera::pinhole pinhole(&parameters[0], 4);
        const double image_point_expected[2] = { 640 / 4, 3 * (480 / 4) };
        const double world_ray_expected[3] = {
            ((image_point_expected[0] - parameters[2]) / parameters[0]) * 0.25,
            ((image_point_expected[1] - parameters[3]) / parameters[1]) * 0.25,
            0.25
        };
        double image_point[2]{};
        REQUIRE(pinhole.project(&world_ray_expected[0], &image_point[0]));
        REQUIRE(is_value_approx(image_point[0], image_point_expected[0]));
        REQUIRE(is_value_approx(image_point[1], image_point_expected[1]));
    }

    return EXIT_SUCCESS;
}