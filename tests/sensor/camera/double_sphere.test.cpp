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

#include "sensor/camera/double_sphere.hpp"

#include "sensor/camera/extended_unified.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-8) {
    if (std::isnan(lhs) || std::isnan(rhs)) {
        return false;
    }
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

// The angle between two rays of any length.
static double angle_between(const double* const lhs, const double* const rhs) {
    const double cross[3] = { (lhs[1] * rhs[2]) - (lhs[2] * rhs[1]), (lhs[2] * rhs[0]) - (lhs[0] * rhs[2]), (lhs[0] * rhs[1]) - (lhs[1] * rhs[0]) };
    const double dot = (lhs[0] * rhs[0]) + (lhs[1] * rhs[1]) + (lhs[2] * rhs[2]);
    return std::atan2(std::sqrt((cross[0] * cross[0]) + (cross[1] * cross[1]) + (cross[2] * cross[2])), dot);
}

// The analytic jacobians of project (point and parameters) and unproject against central differences.
template <typename model_type>
static void require_jacobians(const double* const parameters, const double* const point, const double tolerance = 1e-6) {
    constexpr static const size_t count = model_type::parameter_count;
    const model_type model(&parameters[0], count);
    double pixel[2];
    double jacobian_projection[2 * 3];
    double jacobian_parameters[2 * count];
    REQUIRE(model.project(&point[0], &pixel[0], &jacobian_projection[0], &jacobian_parameters[0]));
    double pixel_alone[2];
    REQUIRE(model.project(&point[0], &pixel_alone[0]));
    REQUIRE(pixel_alone[0] == pixel[0]);
    REQUIRE(pixel_alone[1] == pixel[1]);
    const double length = std::sqrt((point[0] * point[0]) + (point[1] * point[1]) + (point[2] * point[2]));
    for (int axis = 0; axis < 3; ++axis) {
        const double step = 1e-6 * length;
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
            REQUIRE(is_value_approx(jacobian_projection[row * 3 + axis], numeric, tolerance));
        }
    }
    for (size_t parameter = 0; parameter < count; ++parameter) {
        const double step = (parameter < 4) ? 1e-4 : 1e-7;
        double forward_parameters[count];
        double backward_parameters[count];
        for (size_t index = 0; index < count; ++index) {
            forward_parameters[index] = parameters[index];
            backward_parameters[index] = parameters[index];
        }
        forward_parameters[parameter] += step;
        backward_parameters[parameter] -= step;
        model_type forward_model;
        model_type backward_model;
        REQUIRE(forward_model.set_parameters(&forward_parameters[0], count));
        REQUIRE(backward_model.set_parameters(&backward_parameters[0], count));
        double pixel_forward[2];
        double pixel_backward[2];
        REQUIRE(forward_model.project(&point[0], &pixel_forward[0]));
        REQUIRE(backward_model.project(&point[0], &pixel_backward[0]));
        for (size_t row = 0; row < 2; ++row) {
            const double numeric = (pixel_forward[row] - pixel_backward[row]) / (2.0 * step);
            REQUIRE(is_value_approx(jacobian_parameters[row * count + parameter], numeric, tolerance));
        }
    }
    double ray[3];
    double jacobian_unprojection[3 * 2];
    REQUIRE(model.unproject(&pixel[0], &ray[0], &jacobian_unprojection[0]));
    double ray_alone[3];
    REQUIRE(model.unproject(&pixel[0], &ray_alone[0]));
    for (int row = 0; row < 3; ++row) {
        REQUIRE(ray_alone[row] == ray[row]);
    }
    REQUIRE(angle_between(&ray[0], &point[0]) < 1e-9);
    for (int axis = 0; axis < 2; ++axis) {
        const double step = 1e-4;
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
            REQUIRE(is_value_approx(jacobian_unprojection[row * 2 + axis], numeric, tolerance));
        }
    }
}

// Rays in every direction up to the given angle from the axis project, unproject onto themselves, and reproject onto the
// same pixel; with a rejection angle (below a half turn), rays from just past it out to a half turn are rejected.
template <typename model_type>
static void require_round_trips(const model_type& model, const double maximum_angle, const double rejection_angle, const double tolerance = 1e-9) {
    for (int ring = 0; ring <= 40; ++ring) {
        const double theta = maximum_angle * static_cast<double>(ring) / 40.0;
        for (int direction = 0; direction < 24; ++direction) {
            const double phi = 0.2617993877991494 * static_cast<double>(direction) + 0.1;
            for (const double distance : { 0.25, 3.0 }) {
                const double point[3] = { distance * std::sin(theta) * std::cos(phi), distance * std::sin(theta) * std::sin(phi), distance * std::cos(theta) };
                double pixel[2];
                REQUIRE(model.project(&point[0], &pixel[0]));
                REQUIRE(std::isfinite(pixel[0]) && std::isfinite(pixel[1]));
                double ray[3];
                REQUIRE(model.unproject(&pixel[0], &ray[0]));
                REQUIRE(angle_between(&ray[0], &point[0]) < tolerance);
                double reprojected[2];
                REQUIRE(model.project(&ray[0], &reprojected[0]));
                REQUIRE(std::abs(reprojected[0] - pixel[0]) < 1e3 * tolerance);
                REQUIRE(std::abs(reprojected[1] - pixel[1]) < 1e3 * tolerance);
            }
        }
    }
    const double origin[3] = { 0.0, 0.0, 0.0 };
    double origin_pixel[2];
    REQUIRE(!model.project(&origin[0], &origin_pixel[0]));
    if (!(rejection_angle < 3.141592653589793)) {
        return;
    }
    for (int ring = 1; ring <= 20; ++ring) {
        const double theta = rejection_angle + ((3.141592653589793 - rejection_angle) * static_cast<double>(ring) / 20.0) - 1e-6;
        if (!(theta > rejection_angle + 1e-7)) {
            continue;
        }
        for (int direction = 0; direction < 24; ++direction) {
            const double phi = 0.2617993877991494 * static_cast<double>(direction) + 0.1;
            const double point[3] = { std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta) };
            double pixel[2];
            REQUIRE(!model.project(&point[0], &pixel[0]));
        }
    }
}

// Every pixel that unprojects gives a ray that projects back onto it.
template <typename model_type>
static int require_pixels_round_trip(const model_type& model, const double centre_x, const double centre_y, const double extent, const double tolerance = 1e-6) {
    int unprojected = 0;
    for (int j = -40; j <= 40; ++j) {
        for (int i = -40; i <= 40; ++i) {
            const double pixel[2] = { centre_x + (extent * static_cast<double>(i) / 40.0), centre_y + (extent * static_cast<double>(j) / 40.0) };
            double ray[3];
            if (!model.unproject(&pixel[0], &ray[0])) {
                continue;
            }
            ++unprojected;
            REQUIRE(std::isfinite(ray[0]) && std::isfinite(ray[1]) && std::isfinite(ray[2]));
            double reprojected[2];
            REQUIRE(model.project(&ray[0], &reprojected[0]));
            REQUIRE(std::abs(reprojected[0] - pixel[0]) < tolerance);
            REQUIRE(std::abs(reprojected[1] - pixel[1]) < tolerance);
        }
    }
    return unprojected;
}

// The parameters read back as they were set, a wrong count or a non-finite value is refused and leaves the model as it was.
template <typename model_type>
static void require_parameters(const double* const parameters) {
    constexpr static const size_t count = model_type::parameter_count;
    model_type model;
    REQUIRE(model.get_parameter_count() == count);
    REQUIRE(model.set_parameters(&parameters[0], count));
    REQUIRE(!model.set_parameters(&parameters[0], count - 1));
    REQUIRE(!model.set_parameters(nullptr, count));
    double recovered[count];
    REQUIRE(!model.get_parameters(&recovered[0], count + 1));
    REQUIRE(model.get_parameters(&recovered[0], count));
    for (size_t index = 0; index < count; ++index) {
        REQUIRE(recovered[index] == parameters[index]);
    }
    for (size_t index = 0; index < count; ++index) {
        double changed[count];
        for (size_t other = 0; other < count; ++other) {
            changed[other] = parameters[other];
        }
        changed[index] = std::nan("");
        REQUIRE(!model.set_parameters(&changed[0], count));
    }
    double negative[count];
    for (size_t index = 0; index < count; ++index) {
        negative[index] = parameters[index];
    }
    negative[0] = -negative[0];
    REQUIRE(!model.set_parameters(&negative[0], count));
    REQUIRE(model.get_parameters(&recovered[0], count));
    for (size_t index = 0; index < count; ++index) {
        REQUIRE(recovered[index] == parameters[index]);
    }
    double pinhole[4];
    REQUIRE(!model.get_pinhole(&pinhole[0], 3));
    REQUIRE(model.get_pinhole(&pinhole[0], 4));
    REQUIRE(pinhole[2] == parameters[2]);
    REQUIRE(pinhole[3] == parameters[3]);
}

// The pinhole agrees with the model to first order at the principal point: the pixel of a ray a small angle off the axis.
template <typename model_type>
static void require_pinhole_at_centre(const model_type& model) {
    double pinhole[4];
    REQUIRE(model.get_pinhole(&pinhole[0], 4));
    for (const double offset : { -1e-4, 1e-4 }) {
        const double point_x[3] = { offset, 0.0, 1.0 };
        const double point_y[3] = { 0.0, offset, 1.0 };
        double pixel_x[2];
        double pixel_y[2];
        REQUIRE(model.project(&point_x[0], &pixel_x[0]));
        REQUIRE(model.project(&point_y[0], &pixel_y[0]));
        REQUIRE(is_value_approx(pixel_x[0], pinhole[0] * offset + pinhole[2], 1e-6));
        REQUIRE(is_value_approx(pixel_y[1], pinhole[1] * offset + pinhole[3], 1e-6));
    }
    const double axis[3] = { 0.0, 0.0, 2.0 };
    double centre[2];
    REQUIRE(model.project(&axis[0], &centre[0]));
    REQUIRE(is_value_approx(centre[0], pinhole[2], 1e-12));
    REQUIRE(is_value_approx(centre[1], pinhole[3], 1e-12));
}

// The single precision model agrees with the double one.
template <typename float_model_type, typename double_model_type>
static void require_single_precision(const double* const parameters, const double maximum_angle) {
    constexpr static const size_t count = double_model_type::parameter_count;
    float single_parameters[count];
    for (size_t index = 0; index < count; ++index) {
        single_parameters[index] = static_cast<float>(parameters[index]);
    }
    const float_model_type single(&single_parameters[0], count);
    double rounded[count];
    for (size_t index = 0; index < count; ++index) {
        rounded[index] = static_cast<double>(single_parameters[index]);
    }
    const double_model_type reference(&rounded[0], count);
    for (int ring = 0; ring <= 8; ++ring) {
        const double theta = maximum_angle * static_cast<double>(ring) / 8.0;
        for (int direction = 0; direction < 8; ++direction) {
            const double phi = 0.7853981633974483 * static_cast<double>(direction) + 0.2;
            const double point[3] = { std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta) };
            const float single_point[3] = { static_cast<float>(point[0]), static_cast<float>(point[1]), static_cast<float>(point[2]) };
            double pixel[2];
            float single_pixel[2];
            float single_jacobian[6];
            float single_parameter_jacobian[2 * count];
            REQUIRE(reference.project(&point[0], &pixel[0]));
            REQUIRE(single.project(&single_point[0], &single_pixel[0], &single_jacobian[0], &single_parameter_jacobian[0]));
            REQUIRE(std::abs(static_cast<double>(single_pixel[0]) - pixel[0]) < 0.05);
            REQUIRE(std::abs(static_cast<double>(single_pixel[1]) - pixel[1]) < 0.05);
            float single_ray[3];
            float single_unprojection[6];
            REQUIRE(single.unproject(&single_pixel[0], &single_ray[0], &single_unprojection[0]));
            const double ray[3] = { static_cast<double>(single_ray[0]), static_cast<double>(single_ray[1]), static_cast<double>(single_ray[2]) };
            REQUIRE(angle_between(&ray[0], &point[0]) < 1e-3);
        }
    }
}

// The bound w2 of the valid region z > -w2 * d1.
static double valid_bound(const double xi, const double alpha) {
    const double w1 = (alpha > 0.5) ? ((1.0 - alpha) / alpha) : (alpha / (1.0 - alpha));
    return (w1 + xi) / std::sqrt((2.0 * w1 * xi) + (xi * xi) + 1.0);
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    using model_type = sensor::camera::double_sphere<double>;
    // The paper's calibration of the EuRoC fisheye style lens (xi -0.2, alpha 0.56), one with xi above zero, alpha below a half.
    const double paper[6] = { 349.7, 349.6, 365.6, 249.9, -0.2, 0.56 };
    const double positive_xi[6] = { 300.0, 310.0, 320.5, 240.5, 0.4, 0.35 };

    require_parameters<model_type>(&paper[0]);
    require_parameters<model_type>(&positive_xi[0]);
    {
        model_type model;
        const double bad[3][6] = { { 300.0, 310.0, 320.5, 240.5, -1.0, 0.5 }, { 300.0, 310.0, 320.5, 240.5, 0.1, 1.1 }, { 300.0, 310.0, 320.5, 240.5, 0.1, -0.1 } };
        for (const auto& parameters : bad) {
            REQUIRE(!model.set_parameters(&parameters[0], 6));
        }
    }

    {
        // The reference formula of Usenko, Demmel and Cremers.
        for (const auto* const parameters : { &paper[0], &positive_xi[0] }) {
            const model_type model(parameters, 6);
            for (const double x : { -0.8, -0.2, 0.0, 0.4, 1.1 }) {
                for (const double y : { -0.6, 0.0, 0.35 }) {
                    const double z = 0.9;
                    const double point[3] = { x, y, z };
                    const double d1 = std::sqrt((x * x) + (y * y) + (z * z));
                    const double shifted = (parameters[4] * d1) + z;
                    const double d2 = std::sqrt((x * x) + (y * y) + (shifted * shifted));
                    const double denominator = (parameters[5] * d2) + ((1.0 - parameters[5]) * shifted);
                    double pixel[2];
                    REQUIRE(model.project(&point[0], &pixel[0]));
                    REQUIRE(is_value_approx(pixel[0], (parameters[0] * x / denominator) + parameters[2], 1e-12));
                    REQUIRE(is_value_approx(pixel[1], (parameters[1] * y / denominator) + parameters[3], 1e-12));
                }
            }
        }
    }

    {
        // With xi = 0 it is the extended unified model with beta = 1.
        const double sphere[6] = { 300.0, 310.0, 320.5, 240.5, 0.0, 0.6 };
        const double extended_parameters[6] = { 300.0, 310.0, 320.5, 240.5, 0.6, 1.0 };
        const model_type model(&sphere[0], 6);
        const sensor::camera::extended_unified<double> extended(&extended_parameters[0], 6);
        for (const double theta : { 0.0, 0.4, 1.0, 1.6, 2.0 }) {
            for (const double phi : { 0.3, 2.0, 4.0 }) {
                const double point[3] = { std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta) };
                double pixel[2];
                double expected[2];
                REQUIRE(model.project(&point[0], &pixel[0]));
                REQUIRE(extended.project(&point[0], &expected[0]));
                REQUIRE(is_value_approx(pixel[0], expected[0], 1e-12));
                REQUIRE(is_value_approx(pixel[1], expected[1], 1e-12));
                double ray[3];
                double expected_ray[3];
                REQUIRE(model.unproject(&pixel[0], &ray[0]));
                REQUIRE(extended.unproject(&pixel[0], &expected_ray[0]));
                REQUIRE(angle_between(&ray[0], &expected_ray[0]) < 1e-12);
            }
        }
    }

    for (const auto* const parameters : { &paper[0], &positive_xi[0] }) {
        // The valid region z > -w2 * d1, and with alpha above a half the pixels inside r^2 < 1 / (2 * alpha - 1).
        const model_type model(parameters, 6);
        const double limit = std::acos(-valid_bound(parameters[4], parameters[5]));
        REQUIRE(limit > 1.5707963267948966);
        require_round_trips(model, limit - 0.1, limit, 1e-8);
        REQUIRE(require_pixels_round_trip(model, parameters[2], parameters[3], 1500.0, 1e-5) > 0);
        require_pinhole_at_centre(model);
        double pinhole[4];
        REQUIRE(model.get_pinhole(&pinhole[0], 4));
        REQUIRE(is_value_approx(pinhole[0], parameters[0] / (1.0 + parameters[4]), 1e-15));
    }
    {
        const model_type model(&paper[0], 6);
        const double rim = 1.0 / std::sqrt((2.0 * paper[5]) - 1.0);
        const double outside[2] = { paper[2] + (paper[0] * rim * 1.001), paper[3] };
        double ray[3];
        REQUIRE(!model.unproject(&outside[0], &ray[0]));
        // The paper's lens sees past 90 degrees.
        const double sideways[3] = { 1.0, 0.0, -0.05 };
        double pixel[2];
        REQUIRE(model.project(&sideways[0], &pixel[0]));
        REQUIRE(model.unproject(&pixel[0], &ray[0]));
        REQUIRE(ray[2] < 0.0);
        REQUIRE(angle_between(&ray[0], &sideways[0]) < 1e-10);
    }

    {
        const double points[6][3] = { { 0.3, -0.2, 1.4 }, { 1e-9, -2e-9, 1.0 }, { 0.0, 0.0, 2.0 }, { 2.5, 1.1, 1.0 }, { -0.4, 0.9, -0.1 }, { 0.1, 0.05, 30.0 } };
        for (const auto& point : points) {
            require_jacobians<model_type>(&paper[0], &point[0]);
            require_jacobians<model_type>(&positive_xi[0], &point[0]);
        }
    }

    require_single_precision<sensor::camera::double_sphere<float>, model_type>(&paper[0], 1.6);

    return EXIT_SUCCESS;
}
