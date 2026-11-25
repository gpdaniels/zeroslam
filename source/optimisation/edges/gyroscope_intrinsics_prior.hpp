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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_GYROSCOPE_INTRINSICS_PRIOR_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_GYROSCOPE_INTRINSICS_PRIOR_HPP

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
    // The prior that holds a gyroscope's matrix (a gyroscope_intrinsics vertex) where the motion does not show it: a zero mean
    // Gaussian on each of its nine entries, after Van Meter and Paul, "SLAM Dunk: A Recipe for Long-Horizon Egocentric SLAM in
    // the Wild" (2026). The residual is the matrix itself, row by row, and the information the inverse of the prior's variance.
    class gyroscope_intrinsics_prior final {
    public:
        constexpr static const char* name = "gyroscope_intrinsics_prior";
        constexpr static const int residual_count = 9;
        constexpr static const int vertex_count = 1;

        // The paper's standard deviation of each entry: a scale error of 0.2%, a misalignment of a ninth of a degree.
        constexpr static const double default_deviation = 0.002;

    public:
        // The information of a standard deviation of each entry.
        static math::matrix<double, 0, 0> information(const double deviation = default_deviation);

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;

        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_GYROSCOPE_INTRINSICS_PRIOR_HPP
