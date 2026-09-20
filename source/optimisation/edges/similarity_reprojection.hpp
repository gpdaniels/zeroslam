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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_SIMILARITY_REPROJECTION_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_SIMILARITY_REPROJECTION_HPP

#include "math/lie.hpp"
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
    class similarity_reprojection final {
    public:
        constexpr static const char* name = "similarity_reprojection";
        constexpr static const int residual_count = 2;
        constexpr static const int vertex_count = 1;

    private:
        sensor::camera::model<double> camera;
        math::matrix<double, 3, 1> location;
        math::se3<double> observer;
        bool inverted;

        constexpr static const double behind_camera_penalty = 0.3;

    public:
        similarity_reprojection(const sensor::camera::model<double>& camera_model, const math::matrix<double, 3, 1>& point_location, const math::se3<double>& observer_pose, const bool apply_inverse);

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;

        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_SIMILARITY_REPROJECTION_HPP
