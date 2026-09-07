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

#include "sensor/camera/pinhole_radial_tangential.hpp"

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
        const double pinhole_parameters[4] = { 458.654, 457.296, 367.215, 248.375 };
        const double radial_tangential_parameters[12] = { 458.654, 457.296, 367.215, 248.375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        const sensor::camera::pinhole<double> pinhole(&pinhole_parameters[0], 4);
        const sensor::camera::pinhole_radial_tangential<double> radial_tangential(&radial_tangential_parameters[0], 12);
        for (const double x : { -1.5, -0.3, 0.0, 0.7, 1.2 }) {
            for (const double y : { -0.9, 0.1, 0.8 }) {
                for (const double z : { 0.5, 2.0, 10.0 }) {
                    const double point[3] = { x, y, z };
                    double pinhole_pixel[2];
                    double radial_tangential_pixel[2];
                    double pinhole_jacobian[6];
                    double radial_tangential_jacobian[6];
                    REQUIRE(pinhole.project(&point[0], &pinhole_pixel[0], &pinhole_jacobian[0]));
                    REQUIRE(radial_tangential.project(&point[0], &radial_tangential_pixel[0], &radial_tangential_jacobian[0]));
                    REQUIRE(pinhole_pixel[0] == radial_tangential_pixel[0]);
                    REQUIRE(pinhole_pixel[1] == radial_tangential_pixel[1]);
                    for (int index = 0; index < 6; ++index) {
                        REQUIRE(pinhole_jacobian[index] == radial_tangential_jacobian[index]);
                    }
                    double pinhole_ray[3];
                    double radial_tangential_ray[3];
                    double pinhole_unprojection[6];
                    double radial_tangential_unprojection[6];
                    REQUIRE(pinhole.unproject(&pinhole_pixel[0], &pinhole_ray[0], &pinhole_unprojection[0]));
                    REQUIRE(radial_tangential.unproject(&pinhole_pixel[0], &radial_tangential_ray[0], &radial_tangential_unprojection[0]));
                    for (int index = 0; index < 3; ++index) {
                        REQUIRE(pinhole_ray[index] == radial_tangential_ray[index]);
                    }
                    for (int index = 0; index < 6; ++index) {
                        REQUIRE(pinhole_unprojection[index] == radial_tangential_unprojection[index]);
                    }
                }
            }
        }
    }
    {
        const double parameters[12] = { 458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0, 0.0, 0.0, 0.0 };
        const sensor::camera::pinhole_radial_tangential<double> model(&parameters[0], 12);
        for (int pixel_y = 10; pixel_y < 480; pixel_y += 47) {
            for (int pixel_x = 10; pixel_x < 752; pixel_x += 53) {
                const double pixel[2] = { static_cast<double>(pixel_x), static_cast<double>(pixel_y) };
                double ray[3];
                REQUIRE(model.unproject(&pixel[0], &ray[0]));
                REQUIRE(ray[2] == 1.0);
                double reprojected[2];
                REQUIRE(model.project(&ray[0], &reprojected[0]));
                REQUIRE(is_value_approx(reprojected[0], pixel[0], 1e-7));
                REQUIRE(is_value_approx(reprojected[1], pixel[1], 1e-7));
            }
        }
        for (const double x : { -0.55, -0.2, 0.0, 0.3, 0.6 }) {
            for (const double y : { -0.35, 0.05, 0.4 }) {
                const double point[3] = { x, y, 1.0 };
                double pixel[2];
                REQUIRE(model.project(&point[0], &pixel[0]));
                double ray[3];
                REQUIRE(model.unproject(&pixel[0], &ray[0]));
                REQUIRE(is_value_approx(ray[0], x, 1e-8));
                REQUIRE(is_value_approx(ray[1], y, 1e-8));
            }
        }
    }
    {
        const double parameters[12] = { 458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.011, -0.004, 0.002, -0.001 };
        const sensor::camera::pinhole_radial_tangential<double> model(&parameters[0], 12);
        const double point[3] = { 0.35, -0.22, 1.4 };

        {
            double pixel[2];
            double jacobian[2 * 3];
            REQUIRE(model.project(&point[0], &pixel[0], &jacobian[0]));
            const double step = 1e-7;
            for (int axis = 0; axis < 3; ++axis) {
                double forward[3] = { point[0], point[1], point[2] };
                double backward[3] = { point[0], point[1], point[2] };
                forward[axis] += step;
                backward[axis] -= step;
                double pixel_forward[2];
                double pixel_backward[2];
                REQUIRE(model.project(&forward[0], &pixel_forward[0]));
                REQUIRE(model.project(&backward[0], &pixel_backward[0]));
                for (int row = 0; row < 2; ++row) {
                    const double numeric = (pixel_forward[row] - pixel_backward[row]) / (2.0 * step);
                    REQUIRE(is_value_approx(jacobian[row * 3 + axis], numeric, 1e-5));
                }
            }
        }

        {
            double pixel[2];
            double jacobian[2 * 12];
            REQUIRE(model.project(&point[0], &pixel[0], nullptr, &jacobian[0]));
            for (int parameter = 0; parameter < 12; ++parameter) {
                const double step = (parameter < 4) ? 1e-4 : 1e-7;
                double forward_parameters[12];
                double backward_parameters[12];
                REQUIRE(model.get_parameters(&forward_parameters[0], 12));
                REQUIRE(model.get_parameters(&backward_parameters[0], 12));
                forward_parameters[parameter] += step;
                backward_parameters[parameter] -= step;
                const sensor::camera::pinhole_radial_tangential<double> forward_model(&forward_parameters[0], 12);
                const sensor::camera::pinhole_radial_tangential<double> backward_model(&backward_parameters[0], 12);
                double pixel_forward[2];
                double pixel_backward[2];
                REQUIRE(forward_model.project(&point[0], &pixel_forward[0]));
                REQUIRE(backward_model.project(&point[0], &pixel_backward[0]));
                for (int row = 0; row < 2; ++row) {
                    const double numeric = (pixel_forward[row] - pixel_backward[row]) / (2.0 * step);
                    REQUIRE(is_value_approx(jacobian[row * 12 + parameter], numeric, 1e-5));
                }
            }
        }

        {
            double pixel[2];
            REQUIRE(model.project(&point[0], &pixel[0]));
            double ray[3];
            double jacobian[3 * 2];
            REQUIRE(model.unproject(&pixel[0], &ray[0], &jacobian[0]));
            const double step = 1e-5;
            for (int axis = 0; axis < 2; ++axis) {
                double forward[2] = { pixel[0], pixel[1] };
                double backward[2] = { pixel[0], pixel[1] };
                forward[axis] += step;
                backward[axis] -= step;
                double ray_forward[3];
                double ray_backward[3];
                REQUIRE(model.unproject(&forward[0], &ray_forward[0]));
                REQUIRE(model.unproject(&backward[0], &ray_backward[0]));
                for (int row = 0; row < 3; ++row) {
                    const double numeric = (ray_forward[row] - ray_backward[row]) / (2.0 * step);
                    REQUIRE(is_value_approx(jacobian[row * 2 + axis], numeric, 1e-5));
                }
            }
        }
    }
    {
        const double parameters[12] = { 400.0, 410.0, 320.0, 240.0, -0.2, 0.05, 0.001, -0.002, 0.01, 0.003, -0.001, 0.0005 };
        const sensor::camera::pinhole_radial_tangential<double> model(&parameters[0], 12);
        double recovered[12];
        REQUIRE(model.get_parameters(&recovered[0], 12));
        for (int index = 0; index < 12; ++index) {
            REQUIRE(recovered[index] == parameters[index]);
        }
        const double point_at_camera[3] = { 0.1, 0.1, 0.0 };
        double pixel[2];
        REQUIRE(model.project(&point_at_camera[0], &pixel[0]) == false);
        const double point_behind_camera[3] = { 0.1, 0.1, -1.0 };
        REQUIRE(model.project(&point_behind_camera[0], &pixel[0]) == false);
    }

    return EXIT_SUCCESS;
}
