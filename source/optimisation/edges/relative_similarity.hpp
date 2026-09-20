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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_RELATIVE_SIMILARITY_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_RELATIVE_SIMILARITY_HPP

#include "math/matrix.hpp"
#include "optimisation/edge.hpp"

namespace optimisation::edges {
    class relative_similarity final {
    public:
        constexpr static const char* name = "relative_similarity";
        constexpr static const int residual_count = 7;
        constexpr static const int vertex_count = 2;

    public:
        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;
        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_RELATIVE_SIMILARITY_HPP
