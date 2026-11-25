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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_INTRINSICS_RATIO_PRIOR_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_INTRINSICS_RATIO_PRIOR_HPP

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
    // A Gaussian prior on the ratio of two parameters of a camera's intrinsics vertex (vertices::intrinsics), such as the
    // aspect ratio of its pixels, the horizontal focal length over the vertical one: the residual is the first parameter
    // less the ratio times the second, in the first parameter's units, weighted by the edge's information. It ties two
    // parameters that are each free, softly, where a local dimension of the vertex that both share ties them rigidly.
    class intrinsics_ratio_prior final {
    public:
        constexpr static const char* name = "intrinsics_ratio_prior";
        constexpr static const int residual_count = 1;
        constexpr static const int vertex_count = 1;

    private:
        int first_index;
        int second_index;
        double ratio;

    public:
        // The prior that the first parameter is the ratio times the second.
        intrinsics_ratio_prior(const int first_index_value, const int second_index_value, const double ratio_value);

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;

        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_INTRINSICS_RATIO_PRIOR_HPP
