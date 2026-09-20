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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_REPROJECTION_INVERSE_DEPTH_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_REPROJECTION_INVERSE_DEPTH_HPP

#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "sensor/camera/model.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace optimisation::edges {
    class reprojection_inverse_depth final {
    public:
        constexpr static const char* name = "reprojection_inverse_depth";
        constexpr static const int residual_count = 2;
        constexpr static const int vertex_count = 2;

    private:
        sensor::camera::model<double> camera;
        math::matrix<double, 3, 3> anchor_rotation;
        math::matrix<double, 3, 1> anchor_translation;
        constexpr static const double behind_camera_penalty = 0.3;

    private:
        void compute_scaled_point(const edge& context, math::matrix<double, 3, 1>& q, math::matrix<double, 3, 3>& rotation_combined, math::matrix<double, 3, 1>& rho_column, double& rho) const;

    public:
        reprojection_inverse_depth(const sensor::camera::model<double>& camera_model, const math::matrix<double, 3, 3>& anchor_rotation_value, const math::matrix<double, 3, 1>& anchor_translation_value);
        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;
        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_REPROJECTION_INVERSE_DEPTH_HPP
