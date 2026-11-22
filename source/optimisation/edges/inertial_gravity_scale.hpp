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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_INERTIAL_GRAVITY_SCALE_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_INERTIAL_GRAVITY_SCALE_HPP

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edge.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace optimisation::edges {
    // The inertial edge of an up to scale map whose world is not aligned with gravity, for initialising a monocular inertial
    // system, as ORB-SLAM3's inertial only optimisation does (Campos, Elvira, Gomez, Montiel and Tardos, TRO 2021).
    //
    // The vertices are the body pose at i (world to body, in the map's units), the imu's velocity at i (metric, in the map's world
    // axes), the imu bias (gyroscope then accelerometer, usually one vertex shared by every edge of the window), the body pose at j,
    // the imu's velocity at j, the direction of gravity in the map's world (a gravity_direction vertex, R_wg) and the scale that
    // makes the map metric (a scale vertex). The imu's position is s p_wb + R_wb p_bi and gravity is R_wg (0, 0, -g); otherwise the
    // residual, the observation and the information are those of the inertial edge.
    class inertial_gravity_scale final {
    public:
        constexpr static const char* name = "inertial_gravity_scale";
        constexpr static const int residual_count = 9;
        constexpr static const int vertex_count = 7;

    private:
        double gravity_magnitude;
        math::se3<double> body_from_imu;

    public:
        explicit inertial_gravity_scale(const double gravity, const math::se3<double>& imu_on_body = math::se3<double>::identity());

    public:
        // Gravity in the world from the parameters of a gravity_direction vertex.
        static math::matrix<double, 3, 1> gravity_from_direction(const double* const direction, const double gravity);

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;

        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_INERTIAL_GRAVITY_SCALE_HPP
