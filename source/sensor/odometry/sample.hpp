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

#pragma once
#ifndef ZEROSLAM_SENSOR_ODOMETRY_SAMPLE_HPP
#define ZEROSLAM_SENSOR_ODOMETRY_SAMPLE_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace sensor::odometry {
    // One reading of a wheel odometer: the cumulative rotation of each measured wheel, in radians, and the steering reading, at a
    // timestamp in nanoseconds. An Ackermann vehicle measures one wheel (rotation[0]) and the steering, a differential drive its
    // left (rotation[0]) and right (rotation[1]) wheels. The rotations are cumulative so that a sample lost in transit loses no
    // distance, and encoder ticks are turned into them with rotation_from_ticks, after unwrap_ticks for a counter that wraps.
    class sample final {
    public:
        constexpr static const size_t maximum_wheels = 2;

    public:
        long long int timestamp = 0;
        double rotation[maximum_wheels] = { 0.0, 0.0 };
        double steering = 0.0;

    public:
        // The sample at a time between two samples, every reading linear in time. The time is clamped to the two samples.
        static sample interpolate(const sample& from, const sample& to, const long long int time);

        // The rotation in radians of a wheel that has turned the given encoder ticks.
        static double rotation_from_ticks(const long long int ticks, const double ticks_per_revolution);

        // The count of a wrapping encoder counter (modulus ticks, such as 65536 for a 16 bit counter) carried on from the previous
        // unwrapped count, taking the change since the previous reading to be the one of least magnitude.
        static long long int unwrap_ticks(const long long int previous_unwrapped, const long long int reading, const long long int modulus);
    };

    // The noise of a wheel odometer, as standard deviations. The wheel and the lateral slip and the heading noise accumulate with
    // the distance travelled, as random walks: their variance grows by the square of the density for every metre, so that the
    // covariance of a preintegrated interval does not depend on the rate of the samples. The steering reading has white noise.
    // The vertical and tilt densities are the departures from planar motion the odometer cannot see (the ground's slope and bumps,
    // the suspension), and the minimums floor the variance of any interval, a vehicle standing still included.
    class noise final {
    public:
        // Wheel travel, metres after one metre of wheel travel.
        double slip = 0.01;
        // Steering reading, in the units of the reading.
        double steering = 0.002;
        // Sideways slip of the reference point, metres after one metre.
        double lateral = 0.005;
        // Heading, radians after one metre.
        double heading = 0.002;
        // Height out of the plane of motion, metres after one metre.
        double vertical = 0.01;
        // Roll and pitch out of the plane of motion, radians after one metre.
        double tilt = 0.005;
        // The least standard deviation of a translation (metres) and of a rotation (radians) of any interval.
        double minimum_translation = 0.001;
        double minimum_rotation = 0.0005;
    };
}

namespace sensor::odometry {
    inline sample sample::interpolate(const sample& from, const sample& to, const long long int time) {
        if ((time <= from.timestamp) || (to.timestamp <= from.timestamp)) {
            return from;
        }
        if (time >= to.timestamp) {
            return to;
        }
        const double fraction = static_cast<double>(time - from.timestamp) / static_cast<double>(to.timestamp - from.timestamp);
        sample result;
        result.timestamp = time;
        for (size_t wheel = 0; wheel < sample::maximum_wheels; ++wheel) {
            result.rotation[wheel] = from.rotation[wheel] + (fraction * (to.rotation[wheel] - from.rotation[wheel]));
        }
        result.steering = from.steering + (fraction * (to.steering - from.steering));
        return result;
    }

    inline double sample::rotation_from_ticks(const long long int ticks, const double ticks_per_revolution) {
        return (2.0 * 3.14159265358979323846 * static_cast<double>(ticks)) / ticks_per_revolution;
    }

    inline long long int sample::unwrap_ticks(const long long int previous_unwrapped, const long long int reading, const long long int modulus) {
        if (modulus <= 0) {
            return reading;
        }
        const long long int previous_reading = ((previous_unwrapped % modulus) + modulus) % modulus;
        long long int change = (((reading - previous_reading) % modulus) + modulus) % modulus;
        if (change > (modulus / 2)) {
            change -= modulus;
        }
        return previous_unwrapped + change;
    }
}

#endif // ZEROSLAM_SENSOR_ODOMETRY_SAMPLE_HPP
