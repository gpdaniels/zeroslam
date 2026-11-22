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
#ifndef ZEROSLAM_SENSOR_ODOMETRY_DIFFERENTIAL_DRIVE_HPP
#define ZEROSLAM_SENSOR_ODOMETRY_DIFFERENTIAL_DRIVE_HPP

#include "sensor/odometry/sample.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace sensor::odometry {
    // A differential drive: two wheels on one axle, each with its own radius, the track width between their contact points. The
    // reference point, the origin of the odometry frame, is the middle of the axle, with x forward, y left (towards the left
    // wheel) and z up. Over an increment the left and right wheels roll r_l * phi_l and r_r * phi_r, the reference point moves
    // their mean and turns by their difference over the track width.
    class differential_drive final {
    public:
        enum parameter : size_t {
            left_radius = 0,
            right_radius = 1,
            track_width = 2
        };

        constexpr static const char* name = "differential_drive";
        constexpr static const size_t parameter_count = 3;
        // The inputs of an increment: the left and the right wheel's rotation over it.
        constexpr static const size_t input_count = 2;

    public:
        static const char* parameter_name(const size_t index);

        static bool parameter_is_angle(const size_t index);

        void inputs(const sample& from, const sample& to, double* const values) const;

        void input_variances(const double* const parameters, const double* const values, const noise& model, double* const variances) const;

        // The arc length travelled by the reference point over an increment and its change of heading, with their Jacobians,
        // row major, by the inputs (2 x input_count) and by the parameters (2 x parameter_count), each when not null. Fails on a
        // radius or track width that is not positive.
        bool motion(const double* const parameters, const double* const values, double& distance, double& heading, double* const jacobian_inputs, double* const jacobian_parameters) const;
    };
}

#endif // ZEROSLAM_SENSOR_ODOMETRY_DIFFERENTIAL_DRIVE_HPP
