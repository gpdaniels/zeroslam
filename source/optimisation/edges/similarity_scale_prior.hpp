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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_SIMILARITY_SCALE_PRIOR_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_SIMILARITY_SCALE_PRIOR_HPP

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
    // Holds the scale of a similarity vertex at a target, as the difference of their logarithms. With a large information it
    // turns a similarity into a rigid transform for a map whose scale is metric, where a loop measures no change of scale.
    class similarity_scale_prior final {
    public:
        constexpr static const char* name = "similarity_scale_prior";
        constexpr static const int residual_count = 1;
        constexpr static const int vertex_count = 1;

    private:
        double target_scale;

    public:
        explicit similarity_scale_prior(const double target_scale_value);

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;

        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_SIMILARITY_SCALE_PRIOR_HPP
