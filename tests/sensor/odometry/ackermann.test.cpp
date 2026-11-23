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

#include "sensor/odometry/ackermann.hpp"

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

// The analytic Jacobians of the motion by the inputs and by the parameters against central differences.
static void check_jacobians(const sensor::odometry::ackermann& model, const double (&parameters)[4], const double (&inputs)[2]) {
    double distance = 0.0;
    double heading = 0.0;
    double jacobian_inputs[4] = {};
    double jacobian_parameters[8] = {};
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
    for (size_t i = 0; i < 4; ++i) {
        double plus[4] = { parameters[0], parameters[1], parameters[2], parameters[3] };
        double minus[4] = { parameters[0], parameters[1], parameters[2], parameters[3] };
        plus[i] += step;
        minus[i] -= step;
        double distance_plus = 0.0;
        double heading_plus = 0.0;
        double distance_minus = 0.0;
        double heading_minus = 0.0;
        REQUIRE(model.motion(&plus[0], &inputs[0], distance_plus, heading_plus, nullptr, nullptr));
        REQUIRE(model.motion(&minus[0], &inputs[0], distance_minus, heading_minus, nullptr, nullptr));
        REQUIRE(is_value_approx(jacobian_parameters[i], (distance_plus - distance_minus) / (2.0 * step), 1e-7));
        REQUIRE(is_value_approx(jacobian_parameters[4 + i], (heading_plus - heading_minus) / (2.0 * step), 1e-7));
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const double parameters[4] = { 0.3, 2.7, 1.1, 0.02 };

    // The parameters are named, and only the steering offset is an angle.
    {
        REQUIRE(std::strcmp(sensor::odometry::ackermann::parameter_name(sensor::odometry::ackermann::radius), "radius") == 0);
        REQUIRE(std::strcmp(sensor::odometry::ackermann::parameter_name(sensor::odometry::ackermann::wheelbase), "wheelbase") == 0);
        REQUIRE(std::strcmp(sensor::odometry::ackermann::parameter_name(sensor::odometry::ackermann::steering_scale), "steering_scale") == 0);
        REQUIRE(std::strcmp(sensor::odometry::ackermann::parameter_name(sensor::odometry::ackermann::steering_offset), "steering_offset") == 0);
        REQUIRE(std::strcmp(sensor::odometry::ackermann::parameter_name(4), "") == 0);
        REQUIRE(!sensor::odometry::ackermann::parameter_is_angle(sensor::odometry::ackermann::radius));
        REQUIRE(sensor::odometry::ackermann::parameter_is_angle(sensor::odometry::ackermann::steering_offset));
    }

    // The inputs are the wheel's rotation and the mean steering reading, and their variances the slip over the distance rolled
    // and the reading's.
    {
        const sensor::odometry::ackermann model;
        REQUIRE(model.get_measured_wheel() == sensor::odometry::ackermann::measured_wheel::rear_axle);
        sensor::odometry::sample from;
        from.rotation[0] = 2.0;
        from.steering = 0.1;
        sensor::odometry::sample to;
        to.rotation[0] = 7.0;
        to.steering = 0.3;
        double inputs[2] = {};
        model.inputs(from, to, &inputs[0]);
        REQUIRE(is_value_approx(inputs[0], 5.0));
        REQUIRE(is_value_approx(inputs[1], 0.2));

        sensor::odometry::noise noise;
        noise.slip = 0.02;
        noise.steering = 0.01;
        double variances[2] = {};
        model.input_variances(&parameters[0], &inputs[0], noise, &variances[0]);
        // The wheel rolled 1.5 m, so the distance's variance is 0.02^2 * 1.5, and the rotation's that over the radius squared.
        REQUIRE(is_value_approx(variances[0] * parameters[0] * parameters[0], 0.02 * 0.02 * 1.5));
        REQUIRE(is_value_approx(variances[1], 0.01 * 0.01));
    }

    // A rear axle wheel moves the rear axle's centre by the distance it rolls, on an arc of curvature tan(delta) / L.
    {
        const sensor::odometry::ackermann model(sensor::odometry::ackermann::measured_wheel::rear_axle);
        const double inputs[2] = { 5.0, 0.2 };
        double distance = 0.0;
        double heading = 0.0;
        REQUIRE(model.motion(&parameters[0], &inputs[0], distance, heading, nullptr, nullptr));
        const double angle = (1.1 * 0.2) + 0.02;
        REQUIRE(is_value_approx(distance, 1.5));
        REQUIRE(is_value_approx(heading, 1.5 * std::tan(angle) / 2.7));
        check_jacobians(model, parameters, inputs);

        // Reversing reverses the turn, and the opposite steering turns the other way.
        const double reverse[2] = { -5.0, 0.2 };
        double reverse_distance = 0.0;
        double reverse_heading = 0.0;
        REQUIRE(model.motion(&parameters[0], &reverse[0], reverse_distance, reverse_heading, nullptr, nullptr));
        REQUIRE(is_value_approx(reverse_distance, -distance));
        REQUIRE(is_value_approx(reverse_heading, -heading));
        check_jacobians(model, parameters, reverse);

        // The steering reading that cancels the offset drives straight.
        const double straight[2] = { 5.0, -0.02 / 1.1 };
        REQUIRE(model.motion(&parameters[0], &straight[0], distance, heading, nullptr, nullptr));
        REQUIRE(is_value_approx(heading, 0.0));
        check_jacobians(model, parameters, straight);
    }

    // A front wheel rolls along its own heading, on an arc of radius L / sin(delta) about the same centre as the rear axle's arc
    // of radius L / tan(delta), so both turn by the same angle and the rear axle travels cos(delta) of the front wheel's distance.
    {
        const sensor::odometry::ackermann model(sensor::odometry::ackermann::measured_wheel::front_wheel);
        REQUIRE(model.get_measured_wheel() == sensor::odometry::ackermann::measured_wheel::front_wheel);
        const double inputs[2] = { 5.0, 0.6 };
        double distance = 0.0;
        double heading = 0.0;
        REQUIRE(model.motion(&parameters[0], &inputs[0], distance, heading, nullptr, nullptr));
        const double angle = (1.1 * 0.6) + 0.02;
        const double front_radius = 2.7 / std::sin(angle);
        const double rear_radius = 2.7 / std::tan(angle);
        REQUIRE(is_value_approx(heading, 1.5 / front_radius));
        REQUIRE(is_value_approx(distance, heading * rear_radius));
        check_jacobians(model, parameters, inputs);

        // Close to a right angle the front wheel still gives a finite motion.
        const double sharp[2] = { 1.0, 1.25 };
        REQUIRE(model.motion(&parameters[0], &sharp[0], distance, heading, nullptr, nullptr));
        REQUIRE(std::isfinite(distance) && std::isfinite(heading));
        check_jacobians(model, parameters, sharp);
    }

    // A radius or wheelbase that is not positive, or a steering angle past the limit, is refused.
    {
        const sensor::odometry::ackermann model;
        const double inputs[2] = { 1.0, 0.1 };
        double distance = 0.0;
        double heading = 0.0;
        const double no_radius[4] = { 0.0, 2.7, 1.0, 0.0 };
        REQUIRE(!model.motion(&no_radius[0], &inputs[0], distance, heading, nullptr, nullptr));
        const double negative_wheelbase[4] = { 0.3, -2.7, 1.0, 0.0 };
        REQUIRE(!model.motion(&negative_wheelbase[0], &inputs[0], distance, heading, nullptr, nullptr));
        const double too_far[2] = { 1.0, 1.5 };
        const double unit[4] = { 0.3, 2.7, 1.0, 0.0 };
        REQUIRE(!model.motion(&unit[0], &too_far[0], distance, heading, nullptr, nullptr));
        const double not_a_number[2] = { 1.0, std::nan("") };
        REQUIRE(!model.motion(&unit[0], &not_a_number[0], distance, heading, nullptr, nullptr));
    }

    return EXIT_SUCCESS;
}
