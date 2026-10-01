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
#ifndef ZEROSLAM_OPTIMISATION_RELATIVE_BUNDLE_ADJUSTMENT_HPP
#define ZEROSLAM_OPTIMISATION_RELATIVE_BUNDLE_ADJUSTMENT_HPP

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "sensor/camera.hpp"

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

namespace optimisation {
    // Bundle adjustment over a graph of relative transforms, after Sibley, Mei, Reid and Newman, "Adaptive Relative Bundle
    // Adjustment" (RSS 2009). There is no world frame: a transform maps its child frame's coordinates into its parent's,
    // a landmark is an inverse depth bearing in its base frame, and an observation is predicted through the chain of
    // transforms from the observing frame to that base frame. Only the active transforms and landmarks are solved for,
    // the landmarks eliminated by the Schur complement, and every other estimate is held fixed.
    class relative_bundle_adjustment final {
    public:
        // X_parent = scale * R * X_child + t, updated as estimate * exp(delta) for the delta solved for.
        class transform final {
        public:
            math::sim3<double> estimate = math::sim3<double>::identity();
            // A similarity also solves for the scale between its frames, as a loop between two monocular estimates needs;
            // any other transform keeps its scale.
            bool similarity = false;
            bool active = false;
        };

        // The bearing (x / z, y / z) and inverse depth 1 / z of the landmark in its base frame.
        class landmark final {
        public:
            double parameters[3] = { 0.0, 0.0, 0.0 };
            bool active = false;
        };

        class step final {
        public:
            int transform = -1;
            bool inverse = false;
        };

        // The base frame's coordinates are taken into the observing frame by T(path[0]) * T(path[1]) * ..., each step being
        // its transform or that transform's inverse. An empty path observes the landmark in its own base frame.
        class observation final {
        public:
            int landmark = -1;
            int camera = -1;
            double pixel[2] = { 0.0, 0.0 };
            double sigma = 1.0;
            std::vector<step> path;
        };

        // Holds a transform's translation length, which fixes the scale of a monocular map when no fixed transform does.
        class length_prior final {
        public:
            int transform = -1;
            double length = 0.0;
            double information = 0.0;
        };

        class summary final {
        public:
            double initial_cost = 0.0;
            double final_cost = 0.0;
            int iterations = 0;
            int accepted = 0;
            int active_transforms = 0;
            int active_landmarks = 0;
        };

        // sqrt(5.991), the 95% bound of a two dimensional reprojection error in units of its sigma.
        constexpr static const double default_huber = 2.4476519;

        std::vector<sensor::model> cameras;
        std::vector<transform> transforms;
        std::vector<landmark> landmarks;
        std::vector<observation> observations;
        std::vector<length_prior> priors;

    public:
        // Levenberg-Marquardt over the active transforms and landmarks with a robust (Huber) cost.
        summary solve(const int rounds, const double huber_delta = relative_bundle_adjustment::default_huber);

        // The base frame coordinates of an observation's landmark in its observing frame, scaled by the inverse depth.
        math::matrix<double, 3, 1> point_in_frame(const observation& measured) const;

        // The pixel the observation is predicted at under the current estimates, false when that falls behind its camera.
        bool predict(const observation& measured, double (&pixel)[2]) const;

        // The residual observed - predicted as the solver linearises it, with the jacobians of the prediction by each step's
        // transform, in path order and by all seven parameters (omega, upsilon, sigma) of its update, and by the landmark.
        void linearise(const observation& measured, double (&residual)[2], std::vector<math::matrix<double, 2, 7>>& step_jacobians, math::matrix<double, 2, 3>& landmark_jacobian) const;

        // The robust cost of every observation and prior under the current estimates.
        double cost(const double huber_delta = relative_bundle_adjustment::default_huber) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_RELATIVE_BUNDLE_ADJUSTMENT_HPP
