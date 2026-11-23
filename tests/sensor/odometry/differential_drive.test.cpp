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

#include "sensor/odometry/differential_drive.hpp"

#include "sensor/odometry/sample.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-12) {
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

static void check_jacobians(const double (&parameters)[3], const double (&inputs)[2]) {
    const sensor::odometry::differential_drive model;
    double distance = 0.0;
    double heading = 0.0;
    double jacobian_inputs[4] = {};
    double jacobian_parameters[6] = {};
    REQUIRE(model.motion(&parameters[0], &inputs[0], distance, heading, &jacobian_inputs[0], &jacobian_parameters[0]));
    const double step = 1e-6;
    for (size_t i = 0; i < 2; ++i) {
        double plus[2] = { inputs[0], inputs[1] };
        double minus[2] = { inputs[0], inputs[1] };
        plus[i] += step;
        minus[i] -= step;
        double distance_plus = 0.0;
        double heading_plus = 0.0;
        double distance_minus = 0.0;
        double heading_minus = 0.0;
        REQUIRE(model.motion(&parameters[0], &plus[0], distance_plus, heading_plus, nullptr, nullptr));
        REQUIRE(model.motion(&parameters[0], &minus[0], distance_minus, heading_minus, nullptr, nullptr));
        REQUIRE(is_value_approx(jacobian_inputs[i], (distance_plus - distance_minus) / (2.0 * step), 1e-7));
        REQUIRE(is_value_approx(jacobian_inputs[2 + i], (heading_plus - heading_minus) / (2.0 * step), 1e-7));
    }
    for (size_t i = 0; i < 3; ++i) {
        double plus[3] = { parameters[0], parameters[1], parameters[2] };
        double minus[3] = { parameters[0], parameters[1], parameters[2] };
        plus[i] += step;
        minus[i] -= step;
        double distance_plus = 0.0;
        double heading_plus = 0.0;
        double distance_minus = 0.0;
        double heading_minus = 0.0;
        REQUIRE(model.motion(&plus[0], &inputs[0], distance_plus, heading_plus, nullptr, nullptr));
        REQUIRE(model.motion(&minus[0], &inputs[0], distance_minus, heading_minus, nullptr, nullptr));
        REQUIRE(is_value_approx(jacobian_parameters[i], (distance_plus - distance_minus) / (2.0 * step), 1e-7));
        REQUIRE(is_value_approx(jacobian_parameters[3 + i], (heading_plus - heading_minus) / (2.0 * step), 1e-7));
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const sensor::odometry::differential_drive model;
    const double parameters[3] = { 0.1, 0.102, 0.5 };

    {
        REQUIRE(std::strcmp(sensor::odometry::differential_drive::parameter_name(sensor::odometry::differential_drive::left_radius), "left_radius") == 0);
        REQUIRE(std::strcmp(sensor::odometry::differential_drive::parameter_name(sensor::odometry::differential_drive::right_radius), "right_radius") == 0);
        REQUIRE(std::strcmp(sensor::odometry::differential_drive::parameter_name(sensor::odometry::differential_drive::track_width), "track_width") == 0);
        REQUIRE(std::strcmp(sensor::odometry::differential_drive::parameter_name(3), "") == 0);
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(!sensor::odometry::differential_drive::parameter_is_angle(i));
        }
    }

    // The inputs are the two wheels' rotations, with the slip of each over the distance it rolled.
    {
        sensor::odometry::sample from;
        from.rotation[0] = 1.0;
        from.rotation[1] = -1.0;
        sensor::odometry::sample to;
        to.rotation[0] = 11.0;
        to.rotation[1] = 4.0;
        double inputs[2] = {};
        model.inputs(from, to, &inputs[0]);
        REQUIRE(is_value_approx(inputs[0], 10.0));
        REQUIRE(is_value_approx(inputs[1], 5.0));
        sensor::odometry::noise noise;
        noise.slip = 0.03;
        double variances[2] = {};
        model.input_variances(&parameters[0], &inputs[0], noise, &variances[0]);
        REQUIRE(is_value_approx(variances[0] * 0.1 * 0.1, 0.03 * 0.03 * 1.0));
        REQUIRE(is_value_approx(variances[1] * 0.102 * 0.102, 0.03 * 0.03 * 0.51));
    }

    // Equal distances drive straight, opposite ones turn on the spot, and otherwise the axle's middle moves on the arc between.
    {
        double distance = 0.0;
        double heading = 0.0;
        const double straight[2] = { 10.0, 10.0 * 0.1 / 0.102 };
        REQUIRE(model.motion(&parameters[0], &straight[0], distance, heading, nullptr, nullptr));
        REQUIRE(is_value_approx(distance, 1.0));
        REQUIRE(is_value_approx(heading, 0.0));
        check_jacobians(parameters, straight);

        const double spin[2] = { -5.0, 5.0 * 0.1 / 0.102 };
        REQUIRE(model.motion(&parameters[0], &spin[0], distance, heading, nullptr, nullptr));
        REQUIRE(is_value_approx(distance, 0.0));
        REQUIRE(is_value_approx(heading, 1.0 / 0.5));
        check_jacobians(parameters, spin);

        // Wheels rolling 1 m and 1.5 m: the middle rolls 1.25 m and turns by 0.5 / 0.5 = 1 rad, on an arc of radius 1.25 m whose
        // two wheels sit 0.25 m to either side, at radii 1 m and 1.5 m.
        const double arc[2] = { 10.0, 1.5 / 0.102 };
        REQUIRE(model.motion(&parameters[0], &arc[0], distance, heading, nullptr, nullptr));
        REQUIRE(is_value_approx(distance, 1.25));
        REQUIRE(is_value_approx(heading, 1.0));
        REQUIRE(is_value_approx((distance / heading) - 0.25, 1.0));
        REQUIRE(is_value_approx((distance / heading) + 0.25, 1.5));
        check_jacobians(parameters, arc);
    }

    {
        double distance = 0.0;
        double heading = 0.0;
        const double inputs[2] = { 1.0, 1.0 };
        const double no_left[3] = { 0.0, 0.1, 0.5 };
        const double no_right[3] = { 0.1, -0.1, 0.5 };
        const double no_track[3] = { 0.1, 0.1, 0.0 };
        REQUIRE(!model.motion(&no_left[0], &inputs[0], distance, heading, nullptr, nullptr));
        REQUIRE(!model.motion(&no_right[0], &inputs[0], distance, heading, nullptr, nullptr));
        REQUIRE(!model.motion(&no_track[0], &inputs[0], distance, heading, nullptr, nullptr));
    }

    return EXIT_SUCCESS;
}
