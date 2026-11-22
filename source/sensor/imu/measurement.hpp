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
#ifndef ZEROSLAM_SENSOR_IMU_MEASUREMENT_HPP
#define ZEROSLAM_SENSOR_IMU_MEASUREMENT_HPP

#include "math/matrix.hpp"

namespace sensor::imu {
    // One sample of an imu in its own frame, as measured: the angular velocity (rad/s) and the specific force, the acceleration
    // less gravity (m/s^2), at a timestamp in nanoseconds.
    class sample final {
    public:
        long long int timestamp = 0;
        math::matrix<double, 3, 1> angular_velocity = math::matrix<double, 3, 1>::zero();
        math::matrix<double, 3, 1> linear_acceleration = math::matrix<double, 3, 1>::zero();

    public:
        // The sample a straight line between two samples gives at a timestamp, which may lie outside them. Two samples at the
        // same timestamp give the first.
        static sample interpolate(const sample& first, const sample& second, const long long int timestamp) {
            sample result;
            result.timestamp = timestamp;
            const long long int span = second.timestamp - first.timestamp;
            const double fraction = (span == 0) ? 0.0 : (static_cast<double>(timestamp - first.timestamp) / static_cast<double>(span));
            result.angular_velocity = first.angular_velocity + fraction * (second.angular_velocity - first.angular_velocity);
            result.linear_acceleration = first.linear_acceleration + fraction * (second.linear_acceleration - first.linear_acceleration);
            return result;
        }
    };

    // The slowly changing offsets of the gyroscope (rad/s) and the accelerometer (m/s^2), subtracted from their measurements.
    class bias final {
    public:
        math::matrix<double, 3, 1> gyroscope = math::matrix<double, 3, 1>::zero();
        math::matrix<double, 3, 1> accelerometer = math::matrix<double, 3, 1>::zero();
    };

    // The noise model of an imu, in the continuous time densities its calibration publishes (as Kalibr and the EuRoC datasets do),
    // and the magnitude of gravity. Each measurement is the true value plus its bias plus white noise of the noise density, and each
    // bias wanders as the integral of white noise of the random walk density. The defaults are those of the ADIS16448 of the EuRoC
    // MAV datasets.
    class parameters final {
    public:
        // Gyroscope white noise, rad / s / sqrt(Hz).
        double gyroscope_noise_density = 1.6968e-04;
        // Accelerometer white noise, m / s^2 / sqrt(Hz).
        double accelerometer_noise_density = 2.0000e-03;
        // Gyroscope bias random walk, rad / s^2 / sqrt(Hz).
        double gyroscope_random_walk = 1.9393e-05;
        // Accelerometer bias random walk, m / s^3 / sqrt(Hz).
        double accelerometer_random_walk = 3.0000e-03;
        // The sample rate, Hz.
        double rate = 200.0;
        // The magnitude of gravity, m/s^2.
        double gravity = 9.81;

    public:
        // The variance of each axis of one sample's noise, the density squared over the sample period.
        double gyroscope_sample_variance() const {
            return this->gyroscope_noise_density * this->gyroscope_noise_density * this->rate;
        }

        double accelerometer_sample_variance() const {
            return this->accelerometer_noise_density * this->accelerometer_noise_density * this->rate;
        }

        // The variance of each axis of the change of a bias over a duration in seconds.
        double gyroscope_walk_variance(const double duration) const {
            return this->gyroscope_random_walk * this->gyroscope_random_walk * duration;
        }

        double accelerometer_walk_variance(const double duration) const {
            return this->accelerometer_random_walk * this->accelerometer_random_walk * duration;
        }
    };
}

#endif // ZEROSLAM_SENSOR_IMU_MEASUREMENT_HPP
