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

#include "sensor/odometry/sample.hpp"

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

    // Interpolation is linear in time in every reading, and clamped to the two samples.
    {
        sensor::odometry::sample from;
        from.timestamp = 1000;
        from.rotation[0] = 1.0;
        from.rotation[1] = -2.0;
        from.steering = 0.1;
        sensor::odometry::sample to;
        to.timestamp = 2000;
        to.rotation[0] = 3.0;
        to.rotation[1] = 2.0;
        to.steering = -0.3;

        const sensor::odometry::sample quarter = sensor::odometry::sample::interpolate(from, to, 1250);
        REQUIRE(quarter.timestamp == 1250);
        REQUIRE(is_value_approx(quarter.rotation[0], 1.5));
        REQUIRE(is_value_approx(quarter.rotation[1], -1.0));
        REQUIRE(is_value_approx(quarter.steering, 0.0));

        const sensor::odometry::sample before = sensor::odometry::sample::interpolate(from, to, 500);
        REQUIRE(before.timestamp == 1000);
        REQUIRE(before.rotation[0] == 1.0);
        const sensor::odometry::sample after = sensor::odometry::sample::interpolate(from, to, 2500);
        REQUIRE(after.timestamp == 2000);
        REQUIRE(after.rotation[1] == 2.0);
        const sensor::odometry::sample exact = sensor::odometry::sample::interpolate(from, to, 2000);
        REQUIRE(exact.steering == to.steering);
    }

    // Ticks are a fraction of a revolution.
    {
        REQUIRE(is_value_approx(sensor::odometry::sample::rotation_from_ticks(1024, 1024.0), 2.0 * 3.14159265358979323846));
        REQUIRE(is_value_approx(sensor::odometry::sample::rotation_from_ticks(-256, 1024.0), -0.5 * 3.14159265358979323846));
        REQUIRE(sensor::odometry::sample::rotation_from_ticks(0, 100.0) == 0.0);
    }

    // A wrapping counter is carried across its wrap in either direction, and a counter that does not wrap is left alone.
    {
        const long long int modulus = 65536;
        long long int unwrapped = 65530;
        unwrapped = sensor::odometry::sample::unwrap_ticks(unwrapped, 4, modulus);
        REQUIRE(unwrapped == 65540);
        unwrapped = sensor::odometry::sample::unwrap_ticks(unwrapped, 65534, modulus);
        REQUIRE(unwrapped == 65534);
        unwrapped = sensor::odometry::sample::unwrap_ticks(unwrapped, 65534, modulus);
        REQUIRE(unwrapped == 65534);

        long long int backwards = 3;
        backwards = sensor::odometry::sample::unwrap_ticks(backwards, 65533, modulus);
        REQUIRE(backwards == -3);
        backwards = sensor::odometry::sample::unwrap_ticks(backwards, 10, modulus);
        REQUIRE(backwards == 10);

        REQUIRE(sensor::odometry::sample::unwrap_ticks(123, 456, 0) == 456);
    }

    // The default noise is positive in every field.
    {
        const sensor::odometry::noise noise;
        REQUIRE(noise.slip > 0.0);
        REQUIRE(noise.steering > 0.0);
        REQUIRE(noise.lateral > 0.0);
        REQUIRE(noise.heading > 0.0);
        REQUIRE(noise.vertical > 0.0);
        REQUIRE(noise.tilt > 0.0);
        REQUIRE(noise.minimum_translation > 0.0);
        REQUIRE(noise.minimum_rotation > 0.0);
    }

    return EXIT_SUCCESS;
}
