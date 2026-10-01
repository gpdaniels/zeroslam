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

static void require_jacobians_match_differences(const double* const parameters, const double* const point) {
    const sensor::camera::pinhole_radial_tangential<double> model(&parameters[0], 12);
    double pixel[2];
    double jacobian_projection[2 * 3];
    double jacobian_parameters[2 * 12];
    REQUIRE(model.project(&point[0], &pixel[0], &jacobian_projection[0], &jacobian_parameters[0]));
    for (int axis = 0; axis < 3; ++axis) {
        const double step = 1e-7;
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
            REQUIRE(is_value_approx(jacobian_projection[row * 3 + axis], numeric, 1e-5));
        }
    }
    for (int parameter = 0; parameter < 12; ++parameter) {
        const double step = (parameter < 4) ? 1e-4 : 1e-7;
        double forward_parameters[12];
        double backward_parameters[12];
        for (int index = 0; index < 12; ++index) {
            forward_parameters[index] = parameters[index];
            backward_parameters[index] = parameters[index];
        }
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
            REQUIRE(is_value_approx(jacobian_parameters[row * 12 + parameter], numeric, 1e-5));
        }
    }
    double ray[3];
    double jacobian_unprojection[3 * 2];
    REQUIRE(model.unproject(&pixel[0], &ray[0], &jacobian_unprojection[0]));
    for (int axis = 0; axis < 2; ++axis) {
        const double step = 1e-5;
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
            REQUIRE(is_value_approx(jacobian_unprojection[row * 2 + axis], numeric, 1e-5));
        }
    }
}

// Every pixel of the image unprojects and the ray reprojects onto it, and no point in front of the camera is rejected.
static void require_whole_image_unaffected(const double* const parameters, const int width, const int height) {
    const sensor::camera::pinhole_radial_tangential<double> model(&parameters[0], 12);
    for (int pixel_y = 0; pixel_y <= height; pixel_y += 8) {
        for (int pixel_x = 0; pixel_x <= width; pixel_x += 8) {
            const double pixel[2] = { static_cast<double>(pixel_x), static_cast<double>(pixel_y) };
            double ray[3];
            REQUIRE(model.unproject(&pixel[0], &ray[0]));
            double reprojected[2];
            REQUIRE(model.project(&ray[0], &reprojected[0]));
            REQUIRE(is_value_approx(reprojected[0], pixel[0], 1e-7));
            REQUIRE(is_value_approx(reprojected[1], pixel[1], 1e-7));
        }
    }
    for (const double radius : { 1.0, 2.0, 10.0, 100.0, 1000.0 }) {
        for (int direction = 0; direction < 16; ++direction) {
            const double angle = static_cast<double>(direction) * 0.39269908169872414;
            const double point[3] = { radius * std::cos(angle), radius * std::sin(angle), 1.0 };
            double pixel[2];
            REQUIRE(model.project(&point[0], &pixel[0]));
        }
    }
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

    {
        // The shipped calibrations never fold, TUM freiburg 1 and 2 and EuRoC with the importers' principal points.
        const double freiburg1[12] = { 517.3, 516.5, 319.1, 255.8, 0.2624, -0.9531, -0.0054, 0.0026, 1.1633, 0.0, 0.0, 0.0 };
        const double freiburg2[12] = { 520.9, 521.0, 325.6, 250.2, 0.2312, -0.7849, -0.0033, -0.0001, 0.9172, 0.0, 0.0, 0.0 };
        const double euroc[12] = { 458.654, 457.296, 367.715, 248.875, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0, 0.0, 0.0, 0.0 };
        require_whole_image_unaffected(&freiburg1[0], 640, 480);
        require_whole_image_unaffected(&freiburg2[0], 640, 480);
        require_whole_image_unaffected(&euroc[0], 752, 480);
        const double point[3] = { 0.35, -0.22, 1.4 };
        require_jacobians_match_differences(&freiburg1[0], &point[0]);
        require_jacobians_match_differences(&euroc[0], &point[0]);
    }

    {
        // The derivative of r * (1 - 0.3 * r^2) is 1 - 0.9 * r^2, so the distortion folds back at r = sqrt(1 / 0.9).
        const double parameters[12] = { 400.0, 400.0, 320.5, 240.5, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        const sensor::camera::pinhole_radial_tangential<double> model(&parameters[0], 12);
        const double fold = std::sqrt(1.0 / 0.9);
        for (int direction = 0; direction < 16; ++direction) {
            const double angle = static_cast<double>(direction) * 0.39269908169872414;
            const double inside[3] = { 0.999 * fold * std::cos(angle), 0.999 * fold * std::sin(angle), 1.0 };
            const double outside[3] = { 1.001 * fold * std::cos(angle), 1.001 * fold * std::sin(angle), 1.0 };
            double pixel[2];
            REQUIRE(model.project(&inside[0], &pixel[0]));
            REQUIRE(!model.project(&outside[0], &pixel[0]));
        }
        // Folded back, a point far outside the field of view would land inside the 640 by 480 image.
        const double folded[3] = { 1.6, 0.0, 1.0 };
        REQUIRE((320.5 + (400.0 * 1.6 * (1.0 - (0.3 * 1.6 * 1.6)))) < 640.0);
        double folded_pixel[2];
        REQUIRE(!model.project(&folded[0], &folded_pixel[0]));
        // The fold reaches a distorted radius of 2 / 3 * fold, a pixel further out has no ray in front of the fold.
        const double beyond[2] = { 320.5 + 300.0, 240.5 };
        REQUIRE((300.0 / 400.0) > ((2.0 / 3.0) * fold));
        double beyond_ray[3];
        REQUIRE(!model.unproject(&beyond[0], &beyond_ray[0]));
        // Unprojection used to converge past the fold, this pixel gave a ray at (1.90, 1.70) on the opposite side.
        const double opposite[2] = { -396.0, -400.0 };
        double opposite_ray[3];
        REQUIRE(!model.unproject(&opposite[0], &opposite_ray[0]));
        int unprojected = 0;
        for (int pixel_y = -400; pixel_y <= 880; pixel_y += 16) {
            for (int pixel_x = -400; pixel_x <= 1040; pixel_x += 16) {
                const double pixel[2] = { static_cast<double>(pixel_x), static_cast<double>(pixel_y) };
                double ray[3];
                if (!model.unproject(&pixel[0], &ray[0])) {
                    continue;
                }
                ++unprojected;
                REQUIRE(((ray[0] * ray[0]) + (ray[1] * ray[1])) <= (fold * fold));
                double reprojected[2];
                REQUIRE(model.project(&ray[0], &reprojected[0]));
                REQUIRE(is_value_approx(reprojected[0], pixel[0], 1e-6));
                REQUIRE(is_value_approx(reprojected[1], pixel[1], 1e-6));
            }
        }
        REQUIRE(unprojected > 0);
        for (int i = -9; i <= 9; ++i) {
            for (int j = -9; j <= 9; ++j) {
                const double point[3] = { static_cast<double>(i) * 0.1 * fold, static_cast<double>(j) * 0.1 * fold, 1.0 };
                if (((point[0] * point[0]) + (point[1] * point[1])) > (0.81 * fold * fold)) {
                    continue;
                }
                double pixel[2];
                REQUIRE(model.project(&point[0], &pixel[0]));
                double ray[3];
                REQUIRE(model.unproject(&pixel[0], &ray[0]));
                REQUIRE(is_value_approx(ray[0], point[0], 1e-8));
                REQUIRE(is_value_approx(ray[1], point[1], 1e-8));
            }
        }
        const double point[3] = { 0.55, -0.4, 0.9 };
        require_jacobians_match_differences(&parameters[0], &point[0]);
    }

    {
        // With k4 = -0.5 the rational denominator 1 - 0.5 * r^2 reaches zero at r = sqrt(2), past it points are mirrored.
        const double parameters[12] = { 400.0, 400.0, 320.5, 240.5, 0.1, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0 };
        const sensor::camera::pinhole_radial_tangential<double> model(&parameters[0], 12);
        const double pole = std::sqrt(2.0);
        const double inside[3] = { 0.0, 0.999 * pole, 1.0 };
        const double outside[3] = { 0.0, 1.001 * pole, 1.0 };
        double pixel[2];
        REQUIRE(model.project(&inside[0], &pixel[0]));
        REQUIRE(!model.project(&outside[0], &pixel[0]));
        const double point[3] = { -0.3, 0.45, 1.1 };
        require_jacobians_match_differences(&parameters[0], &point[0]);
    }

    {
        // A pincushion that folds at r = 1, its distorted points near the fold lie past it, so undistortion must start inside.
        const double parameters[12] = { 400.0, 400.0, 320.5, 240.5, 0.5, -0.5, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0 };
        const sensor::camera::pinhole_radial_tangential<double> model(&parameters[0], 12);
        const double outside[3] = { 1.001, 0.0, 1.0 };
        double pixel[2];
        REQUIRE(!model.project(&outside[0], &pixel[0]));
        int started_past_fold = 0;
        for (int direction = 0; direction < 32; ++direction) {
            const double angle = static_cast<double>(direction) * 0.19634954084936207;
            const double point[3] = { 0.97 * std::cos(angle), 0.97 * std::sin(angle), 1.0 };
            REQUIRE(model.project(&point[0], &pixel[0]));
            const double distorted_x = (pixel[0] - 320.5) / 400.0;
            const double distorted_y = (pixel[1] - 240.5) / 400.0;
            started_past_fold += (((distorted_x * distorted_x) + (distorted_y * distorted_y)) > 1.0) ? 1 : 0;
            double ray[3];
            REQUIRE(model.unproject(&pixel[0], &ray[0]));
            REQUIRE(is_value_approx(ray[0], point[0], 1e-6));
            REQUIRE(is_value_approx(ray[1], point[1], 1e-6));
        }
        REQUIRE(started_past_fold > 0);
    }

    {
        const float parameters[12] = { 400.0f, 400.0f, 320.5f, 240.5f, -0.3f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        const sensor::camera::pinhole_radial_tangential<float> model(&parameters[0], 12);
        const float inside[3] = { 1.04f, 0.0f, 1.0f };
        const float outside[3] = { 1.07f, 0.0f, 1.0f };
        float pixel[2];
        REQUIRE(model.project(&inside[0], &pixel[0]));
        REQUIRE(!model.project(&outside[0], &pixel[0]));
    }

    return EXIT_SUCCESS;
}
