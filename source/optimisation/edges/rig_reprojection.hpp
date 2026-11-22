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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_RIG_REPROJECTION_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_RIG_REPROJECTION_HPP

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

namespace {
    using size_t = decltype(sizeof(0));
}

namespace optimisation::edges {
    // The reprojection of a landmark into one camera of a rig. The vertices are, in this order, the pose of the rig's body
    // (world to body, as vertices::pose), the landmark (a world point as reprojection takes it, or an anchored inverse depth
    // as reprojection_inverse_depth takes it) and the camera's extrinsic (camera to body, vertices::extrinsic), fixed when
    // the calibration is held: the camera sees extrinsic^-1 * pose * landmark. With the identity extrinsic the residual and
    // the pose and landmark Jacobians are those of the two vertex edges.
    //
    // The camera point and its derivatives by each parameter block are composed apart from the projection (compose), so a
    // block that is a constant here, the camera's intrinsics, can become a vertex of its own by adding its slot and the
    // camera model's Jacobian by its parameters, without touching the others.
    class rig_reprojection final {
    public:
        constexpr static const char* name = "rig_reprojection";
        constexpr static const int residual_count = 2;
        constexpr static const int vertex_count = 3;

        constexpr static const int pose_slot = 0;
        constexpr static const int landmark_slot = 1;
        constexpr static const int extrinsic_slot = 2;

        enum class landmark_kind {
            point,
            inverse_depth
        };

    private:
        sensor::camera::model<double> camera;
        landmark_kind landmark;
        math::matrix<double, 3, 3> anchor_rotation;
        math::matrix<double, 3, 1> anchor_translation;

    private:
        // The point the camera sees, scaled by the inverse depth for an inverse depth landmark (which the projection does not
        // see), and when asked its derivatives by the pose, the landmark and the extrinsic.
        void compose(const edge& context, math::matrix<double, 3, 1>& point_camera, math::matrix<double, 3, 6>* jacobian_pose, math::matrix<double, 3, 3>* jacobian_landmark, math::matrix<double, 3, 6>* jacobian_extrinsic) const;

    public:
        // A world point landmark.
        explicit rig_reprojection(const sensor::camera::model<double>& camera_model);

        // An inverse depth landmark anchored in a frame whose camera to world transform this is, as mapping::point holds it.
        rig_reprojection(const sensor::camera::model<double>& camera_model, const math::matrix<double, 3, 3>& anchor_rotation_value, const math::matrix<double, 3, 1>& anchor_translation_value);

        // Whether the camera projects the point where the vertices put it, as reprojection::measurable.
        bool measurable(const edge& context) const;

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;

        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_RIG_REPROJECTION_HPP
