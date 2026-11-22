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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_GRAVITY_ALIGNMENT_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_GRAVITY_ALIGNMENT_HPP

#include "math/matrix.hpp"
#include "optimisation/edge.hpp"

namespace optimisation::edges {
    // Holds the direction of gravity as a frame sees it, its roll and pitch, and leaves its turn about gravity and its position
    // free: a pose graph of an inertial map whose every vertex has one is the four degree of freedom graph a visual inertial loop
    // closure corrects (ORB-SLAM3), the imu having made the roll and pitch observable. The vertex is a pose or a similarity, whose
    // parameters hold [t, q (x, y, z, w), ...]; the residual is R d - observed, with R the vertex's rotation, or its inverse when
    // the edge is made so, d the unit direction of gravity in the frame R turns from, and the observation the direction it is to
    // be turned to. The jacobian is found by central differences.
    class gravity_alignment final {
    public:
        constexpr static const char* name = "gravity_alignment";
        constexpr static const int residual_count = 3;
        constexpr static const int vertex_count = 1;

    private:
        double direction[3];
        bool inverse;

    public:
        gravity_alignment(const math::matrix<double, 3, 1>& gravity_direction, const bool inverse_rotation);

        // The direction the vertex's rotation turns gravity to now, the observation that holds it there.
        math::matrix<double, 3, 1> turned(const double* const parameters) const;

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_GRAVITY_ALIGNMENT_HPP
