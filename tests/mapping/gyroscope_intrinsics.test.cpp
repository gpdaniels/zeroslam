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

#include "mapping/gyroscope_intrinsics.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edges/gyroscope_intrinsics_prior.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/gyroscope_intrinsics.hpp"
#include "sensor/imu/measurement.hpp"
#include "sensor/imu/preintegration.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

// A measurement that turns at a constant angular velocity for a time, in steps of 5 ms.
static sensor::imu::preintegration turning(const math::matrix<double, 3, 1>& angular_velocity, const double duration) {
    sensor::imu::preintegration measurement{ sensor::imu::parameters() };
    const int steps = static_cast<int>(std::lround(duration / 0.005));
    for (int k = 0; k < steps; ++k) {
        measurement.integrate(angular_velocity, { { 0.0, 0.0, 9.81 } }, duration / steps);
    }
    return measurement;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // The excitation is the sum of the squared turns about the least turned axis, each turn less the mean rate's share.
    {
        mapping::gyroscope_intrinsics::excitation none;
        REQUIRE(none.count() == 0);
        REQUIRE(none.weakest() == 0.0);
        // Two measurements are too few to tell a turn from a rate.
        none.add(turning({ { 0.5, 0.0, 0.0 } }, 0.4));
        none.add(turning({ { 0.0, -0.5, 0.0 } }, 0.4));
        REQUIRE(none.count() == 2);
        REQUIRE(none.weakest() == 0.0);

        // Turns of 0.2, 0.1 and 0.05 rad to either side about x, y and z, half a second each: the sums of squares are 0.08, 0.02
        // and 0.005 rad^2, the mean rate nothing, and the least turned axis z.
        mapping::gyroscope_intrinsics::excitation about_axes;
        const double turns[3] = { 0.2, 0.1, 0.05 };
        for (size_t axis = 0; axis < 3; ++axis) {
            for (const double sign : { 1.0, -1.0 }) {
                math::matrix<double, 3, 1> rate = math::matrix<double, 3, 1>::zero();
                rate[axis] = sign * turns[axis] / 0.5;
                about_axes.add(turning(rate, 0.5));
            }
        }
        REQUIRE(about_axes.count() == 6);
        REQUIRE(std::abs(about_axes.weakest() - 0.005) < 1e-9);

        // The same turns about axes that are not the imu's give the same.
        mapping::gyroscope_intrinsics::excitation about_others;
        const math::so3<double> frame = math::so3<double>::exp({ { 0.3, -0.7, 0.5 } });
        for (size_t axis = 0; axis < 3; ++axis) {
            for (const double sign : { 1.0, -1.0 }) {
                math::matrix<double, 3, 1> rate = math::matrix<double, 3, 1>::zero();
                rate[axis] = sign * turns[axis] / 0.5;
                about_others.add(turning(frame * rate, 0.5));
            }
        }
        REQUIRE(std::abs(about_others.weakest() - 0.005) < 1e-9);

        // Turns about two axes only show nothing of the third's column.
        mapping::gyroscope_intrinsics::excitation planar;
        for (int k = 0; k < 20; ++k) {
            planar.add(turning({ { 0.4 * std::sin(0.9 * k), 0.3 * std::cos(1.7 * k), 0.0 } }, 0.5));
        }
        REQUIRE(planar.weakest() < 1e-12);

        // A steady turn about every axis is the gyroscope bias's: it shows nothing, however long.
        mapping::gyroscope_intrinsics::excitation steady;
        for (int k = 0; k < 20; ++k) {
            steady.add(turning({ { 0.2, -0.3, 0.25 } }, 0.25 + (0.05 * (k % 4))));
        }
        REQUIRE(steady.weakest() < 1e-9);
        // The same with a turn to either side about each axis on top is what the turns on top give.
        for (size_t axis = 0; axis < 3; ++axis) {
            for (const double sign : { 1.0, -1.0 }) {
                math::matrix<double, 3, 1> rate = { { 0.2, -0.3, 0.25 } };
                rate[axis] += sign * 0.4;
                steady.add(turning(rate, 0.5));
            }
        }
        REQUIRE(steady.weakest() > 0.05);
        REQUIRE(steady.weakest() < 0.09);
    }

    // The matrix is zero and taken by no edge until it is asked for, and is free only in an adjustment of the whole map that does
    // not hold the calibrations, once the initialisation has settled and the measurements turn enough.
    mapping::gyroscope_intrinsics gyroscope;
    REQUIRE(!gyroscope.refined());
    REQUIRE(!gyroscope.settled());
    REQUIRE(!gyroscope.active());
    REQUIRE(gyroscope.refinements() == 0);
    REQUIRE(gyroscope.estimate().gyroscope == (math::matrix<double, 3, 3>::zero()));
    mapping::gyroscope_intrinsics::excitation enough;
    mapping::gyroscope_intrinsics::excitation little;
    for (size_t axis = 0; axis < 3; ++axis) {
        for (const double sign : { 1.0, -1.0 }) {
            math::matrix<double, 3, 1> rate = math::matrix<double, 3, 1>::zero();
            rate[axis] = sign;
            // Half a radian to either side is 0.5 rad^2 about each axis, and a tenth of a radian 0.02 rad^2.
            enough.add(turning(rate, 0.5));
            little.add(turning(rate, 0.1));
        }
    }
    REQUIRE(enough.weakest() > mapping::gyroscope_intrinsics::excitation_minimum);
    REQUIRE(little.weakest() < mapping::gyroscope_intrinsics::excitation_minimum);
    REQUIRE(little.weakest() > 0.0);
    REQUIRE(!gyroscope.frees(true, false, enough));
    gyroscope.set_settled(true);
    REQUIRE(gyroscope.settled());
    REQUIRE(!gyroscope.frees(true, false, enough));
    gyroscope.set_refined(true);
    REQUIRE(gyroscope.refined());
    REQUIRE(gyroscope.active());
    REQUIRE(gyroscope.frees(true, false, enough));
    REQUIRE(!gyroscope.frees(false, false, enough));
    REQUIRE(!gyroscope.frees(true, true, enough));
    REQUIRE(!gyroscope.frees(true, false, little));
    REQUIRE(!gyroscope.frees(true, false, mapping::gyroscope_intrinsics::excitation()));
    gyroscope.set_settled(false);
    REQUIRE(!gyroscope.frees(true, false, enough));
    gyroscope.set_settled(true);

    // The vertex holds the estimate, free or held, and a free one's prior is the paper's.
    sensor::imu::intrinsics intrinsics;
    intrinsics.gyroscope = { { { 0.011, -0.006, 0.004 }, { 0.007, -0.009, -0.005 }, { -0.003, 0.008, 0.012 } } };
    {
        const optimisation::vertex held = gyroscope.vertex(false);
        REQUIRE(std::string_view(held.name()) == "gyroscope_intrinsics");
        REQUIRE(held.is_fixed());
        for (size_t i = 0; i < 9; ++i) {
            REQUIRE(held.get_parameters()[i] == 0.0);
        }
        mapping::gyroscope_intrinsics given = gyroscope;
        given.set_estimate(intrinsics);
        REQUIRE(given.refinements() == 0);
        const optimisation::vertex free = given.vertex(true);
        REQUIRE(!free.is_fixed());
        REQUIRE(optimisation::vertices::gyroscope_intrinsics::from_parameters(free.get_parameters()).gyroscope == intrinsics.gyroscope);

        // With nothing but its prior a matrix goes to zero, which is then read as the estimate.
        optimisation::factor_graph graph;
        optimisation::vertex* const node = graph.add_vertex(given.vertex(true));
        given.add_prior(graph, node);
        REQUIRE(graph.solve(10) > 0);
        given.read(*node);
        REQUIRE(given.refinements() == 1);
        for (size_t i = 0; i < 9; ++i) {
            REQUIRE(std::abs(given.estimate().gyroscope.data()[i]) < 1e-9);
        }
        // A matrix that is not finite is not taken.
        double broken[9] = { 0.001, 0.0, 0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0, 0.0 };
        optimisation::vertex bad{ optimisation::vertices::gyroscope_intrinsics() };
        REQUIRE(bad.set_parameters(&broken[0], 9));
        const sensor::imu::intrinsics before = given.estimate();
        given.read(bad);
        REQUIRE(given.refinements() == 1);
        REQUIRE(given.estimate().gyroscope == before.gyroscope);
        broken[4] = -0.002;
        REQUIRE(bad.set_parameters(&broken[0], 9));
        given.read(bad);
        REQUIRE(given.refinements() == 2);
        REQUIRE(given.estimate().gyroscope[0][0] == 0.001);
        REQUIRE(given.estimate().gyroscope[1][1] == -0.002);
    }

    // A matrix that was refined stays in the edges when it is no longer asked for, held.
    {
        mapping::gyroscope_intrinsics kept;
        kept.set_estimate(intrinsics);
        REQUIRE(!kept.refined());
        REQUIRE(kept.active());
        kept.set_settled(true);
        REQUIRE(!kept.frees(true, false, enough));
    }

    // A measurement is integrated again at the estimate when it was integrated at another matrix, and left alone otherwise.
    {
        sensor::imu::bias bias;
        bias.gyroscope = { { 0.002, -0.001, 0.003 } };
        sensor::imu::preintegration measurement(sensor::imu::parameters(), bias);
        for (int k = 0; k < 100; ++k) {
            measurement.integrate({ { 0.3 * std::sin(0.1 * k), -0.5, 0.8 * std::cos(0.07 * k) } }, { { 1.0, -2.0, 9.0 } }, 0.005);
        }
        const sensor::imu::preintegration original = measurement;
        gyroscope.align(measurement);
        REQUIRE(measurement.get_delta_velocity() == original.get_delta_velocity());
        REQUIRE(measurement.get_linearisation_intrinsics().gyroscope == (math::matrix<double, 3, 3>::zero()));
        mapping::gyroscope_intrinsics given = gyroscope;
        given.set_estimate(intrinsics);
        given.align(measurement);
        REQUIRE(measurement.get_linearisation_intrinsics().gyroscope == intrinsics.gyroscope);
        REQUIRE(measurement.get_linearisation().gyroscope == bias.gyroscope);
        REQUIRE(measurement.get_steps().size() == original.get_steps().size());
        sensor::imu::preintegration expected = original;
        expected.reintegrate(bias, intrinsics);
        REQUIRE(measurement.get_delta_velocity() == expected.get_delta_velocity());
        REQUIRE(measurement.get_delta_rotation().get_quaternion() == expected.get_delta_rotation().get_quaternion());
        REQUIRE(!(measurement.get_delta_velocity() == original.get_delta_velocity()));
    }

    return EXIT_SUCCESS;
}
