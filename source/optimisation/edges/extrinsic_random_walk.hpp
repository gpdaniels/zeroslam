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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_EXTRINSIC_RANDOM_WALK_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_EXTRINSIC_RANDOM_WALK_HPP

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
    // The random walk of a sensor's extrinsic between two instants: the vertices are the extrinsic (vertices::extrinsic,
    // sensor to body, updated on the left in the body's frame) at i and at j, and the residual is how far the sensor has
    // turned between them, Log(R_i^T * R_j) in the sensor's frame at i, followed by how far it has moved on the body,
    // t_j - t_i. The information is the inverse of the walk's variance over the time between them, the rotation's then the
    // translation's.
    class extrinsic_random_walk final {
    public:
        constexpr static const char* name = "extrinsic_random_walk";
        constexpr static const int residual_count = 6;
        constexpr static const int vertex_count = 2;

    public:
        // The information of a walk whose standard deviations, of the rotation (radians) and of the translation (metres), are
        // given over a reference duration, over another duration; a standard deviation of zero gives no information.
        static math::matrix<double, 0, 0> information(const double rotation_sigma, const double translation_sigma, const double reference_duration, const double duration);

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;

        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_EXTRINSIC_RANDOM_WALK_HPP
