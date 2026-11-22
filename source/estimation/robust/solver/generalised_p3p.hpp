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
#ifndef ZEROSLAM_ESTIMATION_ROBUST_SOLVER_GENERALISED_P3P_HPP
#define ZEROSLAM_ESTIMATION_ROBUST_SOLVER_GENERALISED_P3P_HPP

#include "math/lie.hpp"
#include "math/matrix.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace estimation::robust::solver {
    // The pose of a rig of cameras, a generalised camera, from points each seen by one of its cameras. Every camera with
    // enough points proposes the rig poses of its own perspective three point consensus (solver::p3p), carried to the rig by
    // its extrinsic, and the proposal most of all the points agree with is kept: the minimal samples are central, so each
    // proposal is a sound pose, and the scoring is generalised, so a camera that sees a few points or none still has its say
    // and a rig whose reference camera faces a blank wall is posed by the others.
    template <typename type>
    class generalised_p3p final {
    public:
        class correspondence final {
        public:
            // The point in its camera's image plane at unit depth.
            type bearing[2];
            // The point in the world.
            type point[3];
            // The camera that saw it, an index into the extrinsics.
            size_t camera;
        };

        class model final {
        public:
            // World to the rig's reference frame.
            type rotation[3][3];
            type translation[3];
            // The camera whose points proposed it.
            size_t camera;
        };

        // A camera proposes poses once it sees this many points.
        constexpr static const size_t camera_points_minimum = 6;

    public:
        // The camera_from_reference transforms give each camera's pose on the rig, camera from reference. The residual is one
        // minus the cosine of the angle between the observed and reprojected bearings, as for solver::p3p; inliers holds
        // data_size entries.
        static bool solve(
            const correspondence* const correspondences,
            const size_t data_size,
            const math::se3<type>* const camera_from_reference,
            const size_t camera_count,
            size_t* const inliers,
            size_t& inliers_size,
            model& best_model,
            const float residual_threshold
        );
    };

    extern template class generalised_p3p<double>;
}

#endif // ZEROSLAM_ESTIMATION_ROBUST_SOLVER_GENERALISED_P3P_HPP
