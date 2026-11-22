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
    // The reprojection of an anchored inverse depth landmark (vertices::point holding the bearing (x / z, y / z) and the
    // inverse depth in its anchor frame) into the frame of a pose. As reprojection, made with_intrinsics it takes a third
    // vertex, the camera's vertices::intrinsics, after the pose and the landmark.
    class reprojection_inverse_depth final {
    public:
        constexpr static const char* name = "reprojection_inverse_depth";
        constexpr static const int residual_count = 2;
        // The vertices without the optional ones, get_vertex_count gives those of an edge.
        constexpr static const int vertex_count = 2;

    private:
        sensor::camera::model<double> camera;
        math::matrix<double, 3, 3> anchor_rotation;
        math::matrix<double, 3, 1> anchor_translation;
        // The slot of the intrinsics vertex, -1 when the camera is held fixed.
        int intrinsics_slot;

    private:
        void compute_scaled_point(const edge& context, math::matrix<double, 3, 1>& q, math::matrix<double, 3, 3>& rotation_combined, math::matrix<double, 3, 1>& rho_column, double& rho) const;

    public:
        reprojection_inverse_depth(const sensor::camera::model<double>& camera_model, const math::matrix<double, 3, 3>& anchor_rotation_value, const math::matrix<double, 3, 1>& anchor_translation_value, const bool with_intrinsics = false);

        int get_vertex_count() const;

        // Whether the camera projects the point where the vertices put it, so that the residual is a distance in the image and
        // not the behind camera residual. A negative inverse depth is no obstacle, as in compute_residual.
        bool measurable(const edge& context) const;

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;
        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_REPROJECTION_INVERSE_DEPTH_HPP
