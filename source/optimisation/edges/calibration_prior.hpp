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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_CALIBRATION_PRIOR_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_CALIBRATION_PRIOR_HPP

#include "math/matrix.hpp"
#include "optimisation/edge.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace optimisation::edges {
    // A prior on the values of a calibration vertex: the observation is the prior values, one per value of the vertex, and the
    // residual of each free value its difference from its prior, of each held value zero. The information is diagonal, from
    // information(), with nothing on a value that has no prior.
    class calibration_prior final {
    public:
        constexpr static const char* name = "calibration_prior";
        constexpr static const int residual_count = 8;
        constexpr static const int vertex_count = 1;

    public:
        static math::matrix<double, 0, 0> observation(const double* const values, const size_t count);
        // The information of standard deviations of the values, each zero or less for no prior.
        static math::matrix<double, 0, 0> information(const double* const standard_deviations, const size_t count);

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;
        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_CALIBRATION_PRIOR_HPP
