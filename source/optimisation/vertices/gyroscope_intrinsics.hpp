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
#ifndef ZEROSLAM_OPTIMISATION_VERTICES_GYROSCOPE_INTRINSICS_HPP
#define ZEROSLAM_OPTIMISATION_VERTICES_GYROSCOPE_INTRINSICS_HPP

#include "sensor/imu/measurement.hpp"

namespace optimisation::vertices {
    // The matrix M that corrects a gyroscope's scale and the alignment of its axes, w = (I + M) * w_measured - b_g, after Van
    // Meter and Paul, "SLAM Dunk: A Recipe for Long-Horizon Egocentric SLAM in the Wild" (2026): its nine entries row by row,
    // updated by adding to them. One vertex holds the matrix of an imu for a whole run, shared by every inertial edge.
    class gyroscope_intrinsics final {
    public:
        constexpr static const char* name = "gyroscope_intrinsics";
        constexpr static const int parameter_count = 9;
        constexpr static const int local_count = 9;

    public:
        void plus(double* const parameters, const double* const delta) const;

        // The parameters of an imu's intrinsics, and the intrinsics of the parameters.
        static void to_parameters(const sensor::imu::intrinsics& intrinsics, double (&parameters)[parameter_count]);
        static sensor::imu::intrinsics from_parameters(const double* const parameters);
    };
}

#endif // ZEROSLAM_OPTIMISATION_VERTICES_GYROSCOPE_INTRINSICS_HPP
