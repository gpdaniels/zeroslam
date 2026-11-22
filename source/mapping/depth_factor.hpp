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
#ifndef ZEROSLAM_MAPPING_DEPTH_FACTOR_HPP
#define ZEROSLAM_MAPPING_DEPTH_FACTOR_HPP

#include "mapping/frame.hpp"
#include "mapping/point.hpp"
#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/loss.hpp"
#include "optimisation/relative_bundle_adjustment.hpp"
#include "optimisation/vertex.hpp"

namespace mapping {
    // What a map with metric scale does with the depth a frame's depth image measured of a landmark: the factor it adds to
    // an adjustment beside the observation's reprojection, the information that adds to the landmark, whether a measurement
    // agrees with the map, and where a depth puts a new landmark. The depth is a sample of the frame's sensor::depth at the
    // observation, with the frame's noise model and measure.
    class depth_factor final {
    public:
        // The 95% chi-squared bound on one degree of freedom, for a depth measurement against the map, and the robust loss's
        // threshold in units of its standard deviation.
        constexpr static const double inlier_bound_squared = 3.841;

    public:
        // The depth the frame's depth image measured at an image position, zero when it has none there.
        static double sample(const mapping::frame& frame, const double x, const double y);

        // The measured depth as a factor between the frame's pose vertex and the landmark's point vertex (added in that order),
        // for the landmark's parameterisation, weighted by the inverse of the variance of the measured inverse depth, with a
        // Huber loss so that a depth off at an occluding edge, or a depth image registered to the camera a few pixels off,
        // loses its weight without taking the bearing with it.
        static optimisation::edge make_edge(const mapping::frame& frame, const mapping::point& landmark, const double depth);

        // The same for a point vertex that holds a world point (x y z), whatever the landmark's parameterisation.
        static optimisation::edge make_edge(const mapping::frame& frame, const double depth);

        // Adds the measured depth's information about the landmark's parameters, J^T w J, as the frame's pose has it.
        static void add_information(const mapping::frame& frame, const mapping::point& landmark, const double depth, math::matrix<double, 3, 3>& information);

        // Adds a solved factor's information about its landmark, J^T w J weighted by its robust loss.
        static void add_information(optimisation::edge& factor, math::matrix<double, 3, 3>& information);

        // The depth the frame's pose puts the landmark at, as its depth image measures depth, false when the landmark is at
        // infinity or not in front of the frame.
        static bool predicted(const mapping::frame& frame, const mapping::point& landmark, double& value);

        // Whether a measured depth agrees with where the frame's pose puts the landmark, within the bound on the inverse depths.
        static bool agrees(const mapping::frame& frame, const mapping::point& landmark, const double depth);

        // Gives an observation of the relative adjustment the measured depth, with the inverse depth's deviation.
        static void relative_observation(const mapping::frame& frame, const double depth, optimisation::relative_bundle_adjustment::observation& measured);

        // The world point a depth measured at an image position puts a landmark at, through the frame's camera and pose.
        static bool location(const mapping::frame& frame, const double x, const double y, const double depth, math::matrix<double, 3, 1>& world, double& axial_depth);
    };
}

#endif // ZEROSLAM_MAPPING_DEPTH_FACTOR_HPP
