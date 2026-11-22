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
#ifndef ZEROSLAM_OPTIMISATION_VERTICES_EXTRINSIC_HPP
#define ZEROSLAM_OPTIMISATION_VERTICES_EXTRINSIC_HPP

#include "math/lie.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace optimisation::vertices {
    // The pose of a sensor on a rig, sensor to body: p_body = R * p_sensor + t, held as [tx ty tz qx qy qz qw] like a pose
    // vertex. Every sensor declares its extrinsic with one, a camera's reprojection edges and the factors of other sensors
    // alike, and a fixed vertex holds a calibrated extrinsic still. The update is on the left, exp(delta) * T, so a step is
    // taken in the body's frame, delta = [omega upsilon] as math::se3::exp takes it.
    class extrinsic final {
    public:
        constexpr static const char* name = "extrinsic";
        constexpr static const int parameter_count = 7;
        constexpr static const int local_count = 6;

    public:
        void plus(double* const parameters, const double* const delta) const;

        // The parameters of a sensor to body transform, and the transform of the parameters.
        static void to_parameters(const math::se3<double>& body_from_sensor, double (&parameters)[parameter_count]);
        static math::se3<double> from_parameters(const double* const parameters);
    };
}

#endif // ZEROSLAM_OPTIMISATION_VERTICES_EXTRINSIC_HPP
