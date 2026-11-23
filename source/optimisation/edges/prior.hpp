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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_PRIOR_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_PRIOR_HPP

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
    // Priors on one vertex, the prior value being the edge's observation and its weight the edge's information.

    // A prior on a pose vertex (world to body), the observation being the prior pose's parameters [t, q (x, y, z, w)]. The residual
    // is Log(T * T_prior^-1), (rotation, translation), in the tangent space of the pose vertex's update exp(delta) * T.
    class pose_prior final {
    public:
        constexpr static const char* name = "pose_prior";
        constexpr static const int residual_count = 6;
        constexpr static const int vertex_count = 1;

    public:
        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;
        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };

    // A prior on a velocity vertex, the residual being v - v_prior.
    class velocity_prior final {
    public:
        constexpr static const char* name = "velocity_prior";
        constexpr static const int residual_count = 3;
        constexpr static const int vertex_count = 1;

    public:
        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;
        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };

    // A prior on an imu_bias vertex, the residual being b - b_prior, gyroscope then accelerometer. A monocular inertial
    // initialisation holds the biases near zero with one, the accelerometer's tightly while the motion cannot tell it from gravity.
    class imu_bias_prior final {
    public:
        constexpr static const char* name = "imu_bias_prior";
        constexpr static const int residual_count = 6;
        constexpr static const int vertex_count = 1;

    public:
        // The information of standard deviations of the gyroscope bias (rad/s) and the accelerometer bias (m/s^2).
        static math::matrix<double, 0, 0> information(const double gyroscope_deviation, const double accelerometer_deviation);

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;
        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_PRIOR_HPP
