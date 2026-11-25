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

#include "sensor/imu/measurement.hpp"

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

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-12) {
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // A default sample, bias, intrinsics and set of parameters.
    {
        const sensor::imu::sample sample;
        REQUIRE(sample.timestamp == 0);
        const sensor::imu::bias bias;
        const sensor::imu::parameters parameters;
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(sample.angular_velocity[i] == 0.0);
            REQUIRE(sample.linear_acceleration[i] == 0.0);
            REQUIRE(bias.gyroscope[i] == 0.0);
            REQUIRE(bias.accelerometer[i] == 0.0);
            // A gyroscope is taken as it is unless its matrix says otherwise.
            for (size_t j = 0; j < 3; ++j) {
                REQUIRE(sensor::imu::intrinsics().gyroscope[i][j] == 0.0);
            }
        }
        REQUIRE(parameters.gyroscope_noise_density > 0.0);
        REQUIRE(parameters.accelerometer_noise_density > 0.0);
        REQUIRE(parameters.gyroscope_random_walk > 0.0);
        REQUIRE(parameters.accelerometer_random_walk > 0.0);
        REQUIRE(parameters.rate > 0.0);
        REQUIRE(is_value_approx(parameters.gravity, 9.81));
    }

    // Interpolation between two samples, inside and outside them, and between two at the same timestamp.
    {
        sensor::imu::sample first;
        first.timestamp = 1000;
        first.angular_velocity = { { 1.0, 2.0, 3.0 } };
        first.linear_acceleration = { { -1.0, 0.0, 9.0 } };
        sensor::imu::sample second;
        second.timestamp = 2000;
        second.angular_velocity = { { 3.0, 2.0, -1.0 } };
        second.linear_acceleration = { { 1.0, 4.0, 10.0 } };

        const sensor::imu::sample start = sensor::imu::sample::interpolate(first, second, 1000);
        const sensor::imu::sample middle = sensor::imu::sample::interpolate(first, second, 1250);
        const sensor::imu::sample end = sensor::imu::sample::interpolate(first, second, 2000);
        const sensor::imu::sample beyond = sensor::imu::sample::interpolate(first, second, 3000);
        REQUIRE(middle.timestamp == 1250);
        REQUIRE(beyond.timestamp == 3000);
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(start.angular_velocity[i] == first.angular_velocity[i]);
            REQUIRE(start.linear_acceleration[i] == first.linear_acceleration[i]);
            REQUIRE(end.angular_velocity[i] == second.angular_velocity[i]);
            REQUIRE(end.linear_acceleration[i] == second.linear_acceleration[i]);
            REQUIRE(is_value_approx(middle.angular_velocity[i], 0.75 * first.angular_velocity[i] + 0.25 * second.angular_velocity[i]));
            REQUIRE(is_value_approx(middle.linear_acceleration[i], 0.75 * first.linear_acceleration[i] + 0.25 * second.linear_acceleration[i]));
            REQUIRE(is_value_approx(beyond.angular_velocity[i], 2.0 * second.angular_velocity[i] - first.angular_velocity[i]));
        }

        const sensor::imu::sample same = sensor::imu::sample::interpolate(first, first, 5000);
        REQUIRE(same.timestamp == 5000);
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(same.angular_velocity[i] == first.angular_velocity[i]);
        }
    }

    // The discrete variances of the noise model.
    {
        sensor::imu::parameters parameters;
        parameters.gyroscope_noise_density = 0.01;
        parameters.accelerometer_noise_density = 0.1;
        parameters.gyroscope_random_walk = 0.001;
        parameters.accelerometer_random_walk = 0.02;
        parameters.rate = 400.0;
        REQUIRE(is_value_approx(parameters.gyroscope_sample_variance(), 0.01 * 0.01 * 400.0));
        REQUIRE(is_value_approx(parameters.accelerometer_sample_variance(), 0.1 * 0.1 * 400.0));
        REQUIRE(is_value_approx(parameters.gyroscope_walk_variance(2.0), 0.001 * 0.001 * 2.0));
        REQUIRE(is_value_approx(parameters.accelerometer_walk_variance(0.5), 0.02 * 0.02 * 0.5));
        REQUIRE(parameters.gyroscope_walk_variance(0.0) == 0.0);
    }

    return 0;
}
