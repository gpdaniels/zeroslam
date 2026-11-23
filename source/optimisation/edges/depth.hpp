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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_DEPTH_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_DEPTH_HPP

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
    // A measured depth of a world point (the point vertex holds x y z) seen from a pose, as the difference of the inverse
    // depths, measured less predicted. The inverse depth is what a structured light or stereo camera measures with a constant
    // error, its disparity being proportional to it, so with the quadratic noise model the residual's weight is near constant
    // and the residual is smooth out to a point at infinity. The observation is the measured depth in metres; the measure says
    // whether it is the z of the point in the camera frame or its distance from the camera centre (see sensor::depth).
    class depth final {
    public:
        constexpr static const char* name = "depth";
        constexpr static const int residual_count = 1;
        constexpr static const int vertex_count = 2;

        // The predicted depth below which a point is taken to be behind or at the camera, where the residual is the
        // measured inverse depth (the point as if at infinity) and the Jacobians are zero.
        constexpr static const double minimum_depth = 1.0e-6;

    private:
        sensor::depth::measure_kind measure;

    public:
        explicit depth(const sensor::depth::measure_kind measure_value);

        // Whether the pose puts the point in front of the camera, so that the residual compares two depths.
        bool measurable(const edge& context) const;

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;

        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_DEPTH_HPP
