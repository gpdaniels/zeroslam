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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_REPROJECTION_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_REPROJECTION_HPP

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
    class reprojection final {
    public:
        constexpr static const char* name = "reprojection";
        constexpr static const int residual_count = 2;
        constexpr static const int vertex_count = 2;

        // Every reprojection edge gives a point it cannot project, such as one behind the camera, the same residual in both
        // rows: the slope times the depth behind the camera, which pulls the point back in front and costs little just behind
        // it. A residual bounded away from zero made every such observation an outlier and a step in the cost that LM would
        // not cross, which lost tracking on TUM fr1 and EuRoC sequences, so the pipeline keeps this weak pull.
        constexpr static const double behind_camera_slope = 0.3;

    private:
        sensor::camera::model<double> camera;

    public:
        explicit reprojection(const sensor::camera::model<double>& camera_model);

        static double behind_camera_residual(const double depth);

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;

        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_REPROJECTION_HPP
