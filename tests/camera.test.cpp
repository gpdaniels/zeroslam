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

    {
        const double parameters[4] = { 525, 525, 320, 240 };
        camera::pinhole pinhole(&parameters[0], 4);
        
        const double step = 1e-6;
        
        const double test_points[5][3] = {
            { 1.0, 2.0, 5.0 },
            { -1.0, 1.0, 3.0 },
            { 0.5, -0.5, 2.0 },
            { 2.0, 2.0, 10.0 },
            { -2.0, -1.0, 4.0 }
        };
        
        for (size_t t = 0; t < 5; ++t) {
            const double* point = test_points[t];
            
            double projected[2] = { 0.0, 0.0 };
            double jacobian_analytical[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            
            REQUIRE(pinhole.project(point, projected, jacobian_analytical, nullptr));
            
            double jacobian_numerical[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            for (size_t i = 0; i < 3; ++i) {
                double point_plus[3] = { point[0], point[1], point[2] };
                point_plus[i] += step;
                double projected_plus[2] = { 0.0, 0.0 };
                REQUIRE(pinhole.project(point_plus, projected_plus));
                
                double point_minus[3] = { point[0], point[1], point[2] };
                point_minus[i] -= step;
                double projected_minus[2] = { 0.0, 0.0 };
                REQUIRE(pinhole.project(point_minus, projected_minus));
                
                jacobian_numerical[0 * 3 + i] = (projected_plus[0] - projected_minus[0]) / (2.0 * step);
                jacobian_numerical[1 * 3 + i] = (projected_plus[1] - projected_minus[1]) / (2.0 * step);
            }
            
            for (size_t r = 0; r < 2; ++r) {
                for (size_t c = 0; c < 3; ++c) {
                    const double analytical = jacobian_analytical[r * 3 + c];
                    const double numerical = jacobian_numerical[r * 3 + c];
                    const double relative_error = (numerical != 0.0) ? std::abs((analytical - numerical) / numerical) : std::abs(analytical - numerical);
                    REQUIRE(relative_error < 1e-5);
                }
            }
        }
        
        for (size_t t = 0; t < 5; ++t) {
            const double* point = test_points[t];
            
            double projected[2] = { 0.0, 0.0 };
            double jacobian_analytical[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            
            REQUIRE(pinhole.project(point, projected, nullptr, jacobian_analytical));
            
            double jacobian_numerical[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            for (size_t i = 0; i < 4; ++i) {
                double params_plus[4] = { parameters[0], parameters[1], parameters[2], parameters[3] };
                params_plus[i] += step;
                camera::pinhole pinhole_plus(params_plus, 4);
                double projected_plus[2] = { 0.0, 0.0 };
                REQUIRE(pinhole_plus.project(point, projected_plus));
                
                double params_minus[4] = { parameters[0], parameters[1], parameters[2], parameters[3] };
                params_minus[i] -= step;
                camera::pinhole pinhole_minus(params_minus, 4);
                double projected_minus[2] = { 0.0, 0.0 };
                REQUIRE(pinhole_minus.project(point, projected_minus));
                
                jacobian_numerical[0 * 4 + i] = (projected_plus[0] - projected_minus[0]) / (2.0 * step);
                jacobian_numerical[1 * 4 + i] = (projected_plus[1] - projected_minus[1]) / (2.0 * step);
            }
            
            for (size_t r = 0; r < 2; ++r) {
                for (size_t c = 0; c < 4; ++c) {
                    const double analytical = jacobian_analytical[r * 4 + c];
                    const double numerical = jacobian_numerical[r * 4 + c];
                    const double relative_error = (std::abs(numerical) > 1e-10) ? std::abs((analytical - numerical) / numerical) : std::abs(analytical - numerical);
                    REQUIRE(relative_error < 1e-5);
                }
            }
        }
        
        {
            const double image_point[2] = { 320, 240 };
            double ray[3] = { 0.0, 0.0, 0.0 };
            double jacobian_analytical[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            
            REQUIRE(pinhole.unproject(image_point, ray, jacobian_analytical));
            
            double jacobian_numerical[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            for (size_t i = 0; i < 2; ++i) {
                double img_plus[2] = { image_point[0], image_point[1] };
                img_plus[i] += step;
                double ray_plus[3] = { 0.0, 0.0, 0.0 };
                REQUIRE(pinhole.unproject(img_plus, ray_plus));
                
                double img_minus[2] = { image_point[0], image_point[1] };
                img_minus[i] -= step;
                double ray_minus[3] = { 0.0, 0.0, 0.0 };
                REQUIRE(pinhole.unproject(img_minus, ray_minus));
                
                jacobian_numerical[0 * 2 + i] = (ray_plus[0] - ray_minus[0]) / (2.0 * step);
                jacobian_numerical[1 * 2 + i] = (ray_plus[1] - ray_minus[1]) / (2.0 * step);
                jacobian_numerical[2 * 2 + i] = (ray_plus[2] - ray_minus[2]) / (2.0 * step);
            }
            
            for (size_t r = 0; r < 3; ++r) {
                for (size_t c = 0; c < 2; ++c) {
                    const double analytical = jacobian_analytical[r * 2 + c];
                    const double numerical = jacobian_numerical[r * 2 + c];
                    const double relative_error = (std::abs(numerical) > 1e-10) ? std::abs((analytical - numerical) / numerical) : std::abs(analytical - numerical);
                    REQUIRE(relative_error < 1e-5);
                }
            }
        }
    }

    return EXIT_SUCCESS;
}