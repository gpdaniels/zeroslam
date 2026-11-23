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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_DEPTH_INVERSE_DEPTH_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_DEPTH_INVERSE_DEPTH_HPP

#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "sensor/depth.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace optimisation::edges {
    // The depth edge (see optimisation::edges::depth) of a landmark held as an anchored inverse depth, the point vertex
    // holding the bearing (a, b) and inverse depth rho in the anchor camera as reprojection_inverse_depth does. With q = rho
    // times the point in the observing camera, the predicted inverse depth is rho / q_z (axial) or rho / |q| (radial), which
    // stays smooth through a landmark at infinity.
    class depth_inverse_depth final {
    public:
        constexpr static const char* name = "depth_inverse_depth";
        constexpr static const int residual_count = 1;
        constexpr static const int vertex_count = 2;

        // The measure of the scaled point below which the bearing is taken to point behind the camera, where the residual is
        // the measured inverse depth and the Jacobians are zero.
        constexpr static const double minimum_scaled_depth = 1.0e-9;

    private:
        sensor::depth::measure_kind measure;
        math::matrix<double, 3, 3> anchor_rotation;
        math::matrix<double, 3, 1> anchor_translation;

    private:
        void compute_scaled_point(const edge& context, math::matrix<double, 3, 1>& q, math::matrix<double, 3, 3>& rotation_combined, math::matrix<double, 3, 1>& rho_column, double& rho) const;

    public:
        depth_inverse_depth(const sensor::depth::measure_kind measure_value, const math::matrix<double, 3, 3>& anchor_rotation_value, const math::matrix<double, 3, 1>& anchor_translation_value);

        // Whether the bearing points in front of the observing camera, so that the residual compares two inverse depths.
        bool measurable(const edge& context) const;

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;

        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_DEPTH_INVERSE_DEPTH_HPP
