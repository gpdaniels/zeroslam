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

#include "sensor/odometry/model.hpp"

#include "sensor/odometry/ackermann.hpp"
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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // An empty model is invalid and does nothing.
    {
        const sensor::odometry::model none;
        REQUIRE(!none.is_valid());
        REQUIRE(none.get_kind() == sensor::odometry::model::kind::none);
        REQUIRE(none.parameter_count() == 0);
        REQUIRE(none.input_count() == 0);
        REQUIRE(std::strcmp(none.name(), "") == 0);
        REQUIRE(std::strcmp(none.parameter_name(0), "") == 0);
        REQUIRE(!none.parameter_is_angle(0));
        const double parameters[1] = { 1.0 };
        const double inputs[2] = { 1.0, 1.0 };
        double distance = 0.0;
        double heading = 0.0;
        REQUIRE(!none.motion(&parameters[0], &inputs[0], distance, heading, nullptr, nullptr));
    }

    // Each model forwards to its kinematics, and two models are equal when the kinematics are.
    {
        const sensor::odometry::model rear(sensor::odometry::ackermann(sensor::odometry::ackermann::measured_wheel::rear_axle));
        const sensor::odometry::model front(sensor::odometry::ackermann(sensor::odometry::ackermann::measured_wheel::front_wheel));
        const sensor::odometry::model differential(sensor::odometry::differential_drive{});
        REQUIRE(rear.is_valid() && front.is_valid() && differential.is_valid());
        REQUIRE(rear.get_kind() == sensor::odometry::model::kind::ackermann);
        REQUIRE(differential.get_kind() == sensor::odometry::model::kind::differential_drive);
        REQUIRE(std::strcmp(rear.name(), "ackermann") == 0);
        REQUIRE(std::strcmp(differential.name(), "differential_drive") == 0);
        REQUIRE(rear.parameter_count() == 4);
        REQUIRE(differential.parameter_count() == 3);
        REQUIRE(rear.input_count() == 2);
        REQUIRE(differential.input_count() == 2);
        REQUIRE(std::strcmp(rear.parameter_name(3), "steering_offset") == 0);
        REQUIRE(std::strcmp(differential.parameter_name(2), "track_width") == 0);
        REQUIRE(rear.parameter_is_angle(3));
        REQUIRE(!differential.parameter_is_angle(2));
        REQUIRE(front.get_ackermann().get_measured_wheel() == sensor::odometry::ackermann::measured_wheel::front_wheel);
        REQUIRE(rear == rear);
        REQUIRE(rear != front);
        REQUIRE(rear != differential);
        REQUIRE(differential == sensor::odometry::model(sensor::odometry::differential_drive{}));

        sensor::odometry::sample from;
        sensor::odometry::sample to;
        to.rotation[0] = 4.0;
        to.rotation[1] = 6.0;
        to.steering = 0.2;
        double inputs[2] = {};
        rear.inputs(from, to, &inputs[0]);
        REQUIRE((inputs[0] == 4.0) && (inputs[1] == 0.1));
        differential.inputs(from, to, &inputs[0]);
        REQUIRE((inputs[0] == 4.0) && (inputs[1] == 6.0));

        const double differential_parameters[3] = { 0.1, 0.1, 0.5 };
        double distance = 0.0;
        double heading = 0.0;
        REQUIRE(differential.motion(&differential_parameters[0], &inputs[0], distance, heading, nullptr, nullptr));
        REQUIRE(std::abs(distance - 0.5) < 1e-12);
        REQUIRE(std::abs(heading - 0.4) < 1e-12);
        sensor::odometry::noise noise;
        double variances[2] = {};
        differential.input_variances(&differential_parameters[0], &inputs[0], noise, &variances[0]);
        REQUIRE((variances[0] > 0.0) && (variances[1] > variances[0]));
    }

    // A calibration takes the model's parameter count, and its mask the free parameters.
    {
        const sensor::odometry::model rear(sensor::odometry::ackermann{});
        const double values[4] = { 0.3, 2.7, 1.0, 0.01 };
        sensor::odometry::calibration calibration(rear, &values[0], 4);
        REQUIRE(calibration.count == 4);
        REQUIRE(calibration.values[1] == 2.7);
        REQUIRE(calibration.free_mask() == 0u);
        REQUIRE(calibration.free_count() == 0);
        calibration.free[sensor::odometry::ackermann::radius] = true;
        calibration.free[sensor::odometry::ackermann::steering_offset] = true;
        REQUIRE(calibration.free_mask() == 9u);
        REQUIRE(calibration.free_count() == 2);
        // A flag past the model's parameters is not counted.
        calibration.free[6] = true;
        REQUIRE(calibration.free_mask() == 9u);

        const sensor::odometry::calibration empty;
        REQUIRE(empty.count == 0);
        REQUIRE(empty.free_mask() == 0u);
    }

    return EXIT_SUCCESS;
}
