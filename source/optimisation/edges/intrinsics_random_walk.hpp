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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_INTRINSICS_RANDOM_WALK_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_INTRINSICS_RANDOM_WALK_HPP

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
    // The random walk of one parameter of a camera between two instants: the vertices are the camera's intrinsics
    // (vertices::intrinsics) at i and at j, and the residual is the parameter at j less the parameter at i, whose
    // information is the inverse of the walk's variance over the time between them. One edge per parameter, as
    // intrinsics_prior is, so the walk does not depend on how many parameters a model has and which are free; a parameter
    // a vertex holds fixed has a zero jacobian there, and one its local dimension moves in proportion to another, the
    // parameter's share of the step.
    class intrinsics_random_walk final {
    public:
        constexpr static const char* name = "intrinsics_random_walk";
        constexpr static const int residual_count = 1;
        constexpr static const int vertex_count = 2;

    private:
        int parameter_index;

    public:
        explicit intrinsics_random_walk(const int parameter_index_value);

        // The information of a walk whose standard deviation is given over a reference duration, over another duration.
        static math::matrix<double, 0, 0> information(const double sigma, const double reference_duration, const double duration);

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;

        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_INTRINSICS_RANDOM_WALK_HPP
