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
#ifndef ZEROSLAM_SENSOR_ODOMETRY_ACKERMANN_HPP
#define ZEROSLAM_SENSOR_ODOMETRY_ACKERMANN_HPP

#include "sensor/odometry/sample.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace sensor::odometry {
    // The kinematic bicycle model of a car-like (Ackermann steered) vehicle. Its reference point, the origin of the odometry
    // frame, is the centre of the rear axle, with x forward, y left and z up, and it moves on arcs of curvature tan(delta) / L for
    // the steering angle delta of the virtual front wheel and the wheelbase L. The steering angle is the reading through a gain and
    // an offset, delta = scale * reading + offset.
    //
    // The distance comes from one wheel's rotation times its radius. A wheel on the rear axle (or the mean of the two rear wheels,
    // or the drive shaft) moves with the reference point. A front wheel rolls along its own heading, so the reference point moves
    // that distance times cos(delta) and turns by that distance times sin(delta) / L, which holds as the steering angle nears a
    // right angle, where the rear axle form's tan(delta) does not.
    class ackermann final {
    public:
        enum class measured_wheel {
            rear_axle,
            front_wheel
        };

        enum parameter : size_t {
            radius = 0,
            wheelbase = 1,
            steering_scale = 2,
            steering_offset = 3
        };

        constexpr static const char* name = "ackermann";
        constexpr static const size_t parameter_count = 4;
        // The inputs of an increment: the measured wheel's rotation over it, and the mean of its two steering readings.
        constexpr static const size_t input_count = 2;
        // A steering angle beyond this (radians) from straight ahead is refused, as the rear axle form's curvature diverges.
        constexpr static const double maximum_steering_angle = 1.4;

    private:
        measured_wheel wheel;

    public:
        explicit ackermann(const measured_wheel measured = measured_wheel::rear_axle);

    public:
        measured_wheel get_measured_wheel() const;

        static const char* parameter_name(const size_t index);

        // Whether the parameter is an angle, rather than a length or a gain, so that a change in it is judged absolutely.
        static bool parameter_is_angle(const size_t index);

        // The inputs of the increment between two samples.
        void inputs(const sample& from, const sample& to, double* const values) const;

        // The variances of the inputs of an increment: the wheel's slip over the distance it rolled and the steering reading's.
        void input_variances(const double* const parameters, const double* const values, const noise& model, double* const variances) const;

        // The arc length travelled by the reference point over an increment and its change of heading, with their Jacobians,
        // row major, by the inputs (2 x input_count) and by the parameters (2 x parameter_count), each when not null. Fails on a
        // radius or wheelbase that is not positive or a steering angle beyond the maximum.
        bool motion(const double* const parameters, const double* const values, double& distance, double& heading, double* const jacobian_inputs, double* const jacobian_parameters) const;
    };
}

#endif // ZEROSLAM_SENSOR_ODOMETRY_ACKERMANN_HPP
