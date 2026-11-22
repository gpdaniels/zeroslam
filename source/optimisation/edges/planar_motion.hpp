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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_PLANAR_MOTION_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_PLANAR_MOTION_HPP

#include "math/matrix.hpp"
#include "optimisation/edge.hpp"

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
    // A soft constraint that a ground vehicle moves on a plane: between two body poses (world to body) the odometry frame, through
    // the extrinsic (body to odometry), stays in the plane of the first, its z axis along the first's and its origin at the
    // first's height. The vertices are the two body poses and the extrinsic, as for the odometry edge, which holds the same three
    // rows between the two ends of each interval; this edge holds them between any two keyframes, such as the ends of a loop or
    // keyframes whose odometry was lost. The residual is the first two components of the end's z axis in the start's odometry
    // frame (about the pitch and minus the roll) and the end's height.
    class planar_motion final {
    public:
        constexpr static const char* name = "planar_motion";
        constexpr static const int residual_count = 3;
        constexpr static const int vertex_count = 3;

    public:
        // The information of the constraint between two poses the given distance apart, for the standard deviations of a tilt
        // (radians) and a height (metres) per square root of a metre and their floors.
        static math::matrix<double, 0, 0> information(const double distance, const double tilt_density, const double vertical_density, const double minimum_tilt, const double minimum_vertical);

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;
        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_PLANAR_MOTION_HPP
