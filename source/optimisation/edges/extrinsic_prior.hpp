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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_EXTRINSIC_PRIOR_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_EXTRINSIC_PRIOR_HPP

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
    // A prior on a sensor extrinsic held by an extrinsic vertex (vertices::extrinsic, sensor to body T_bs, updated on the left in
    // the body's frame). The observation is the prior as an extrinsic vertex's parameters, and the residual is taken on the
    // inverse, body to sensor T_sb: the rotation error Log(R_sb * R_sb_prior^T) followed by the translation error
    // t_sb - t_sb_prior, both in the sensor's frame whatever frame the vertex steps in, so that the information can hold each
    // sensor axis apart: an odometer's planar motion leaves the height of its frame above the plane unobservable, which is the translation
    // error's third row alone, and a tight prior there with loose ones elsewhere holds it without biasing what the motion shows.
    class extrinsic_prior final {
    public:
        constexpr static const char* name = "extrinsic_prior";
        constexpr static const int residual_count = 6;
        constexpr static const int vertex_count = 1;

    public:
        // The observation of a prior transform, and the information of standard deviations of the three rotation errors
        // (radians) and the three translation errors (metres), each zero for none.
        static math::matrix<double, 0, 0> observation(const double* const pose_parameters);
        static math::matrix<double, 0, 0> information(const double* const standard_deviations);

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;
        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_EXTRINSIC_PRIOR_HPP
