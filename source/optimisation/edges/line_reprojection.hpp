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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_LINE_REPROJECTION_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_LINE_REPROJECTION_HPP

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
    class line_reprojection final {
    public:
        constexpr static const char* name = "line_reprojection";
        constexpr static const int residual_count = 2;
        constexpr static const int vertex_count = 2;

    private:
        sensor::camera::model<double> camera;
        double measured_normalised[2][2] = { { 0.0, 0.0 }, { 0.0, 0.0 } };
        bool measured_valid = false;
        double focal = 1.0;
        constexpr static const double minimum_line_norm = 1.0e-9;

    private:
        void compute_camera_line(const edge& context, math::matrix<double, 3, 1>& moment_camera, math::matrix<double, 3, 1>& direction_camera, math::matrix<double, 3, 3>& rotation, math::matrix<double, 3, 1>& translation) const;

    public:
        explicit line_reprojection(const sensor::camera::model<double>& camera_model);

        bool set_measured_segment(const double x1, const double y1, const double x2, const double y2);

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;
        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_LINE_REPROJECTION_HPP
