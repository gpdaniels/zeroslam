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
#ifndef ZEROSLAM_OPTIMISATION_INERTIAL_INITIALISATION_HPP
#define ZEROSLAM_OPTIMISATION_INERTIAL_INITIALISATION_HPP

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "sensor/imu/measurement.hpp"
#include "sensor/imu/preintegration.hpp"

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
    // The initialisation of a monocular inertial system from a window of keyframes: their body poses from vision, right up to a
    // scale and in a world that is not aligned with gravity, and the imu preintegrated between each keyframe and the next. It
    // estimates the direction of gravity in the map's world, the scale that makes the map metric, the imu's velocity at each
    // keyframe and one bias for the window, in three steps:
    //   1. The gyroscope bias from the rotations alone, which the scale and gravity do not touch, by Gauss-Newton.
    //   2. Gravity, the scale and the velocities in closed form, linear with the accelerometer bias held at the linearisation point
    //      (after Mur-Artal and Tardos, "Visual-Inertial Monocular SLAM With Map Reuse", RA-L 2017, and VINS-Mono's alignment),
    //      as the starting point of
    //   3. the maximum a posteriori inertial only optimisation of ORB-SLAM3 (Campos, Elvira, Gomez, Montiel and Tardos, "Inertial-
    //      Only Optimization for Visual-Inertial Initialization", ICRA 2020): the poses fixed, the velocities, the bias, gravity's
    //      direction and the scale free, with priors on the biases, the preintegrations integrated again whenever the bias moves
    //      past their thresholds.
    // Whether the motion of the window held the estimate is judged from its uncertainty: the standard deviations of the log of the
    // scale and of gravity's direction, from the information of the optimisation with the velocities and the bias marginalised.
    class inertial_initialisation final {
    public:
        class options final {
        public:
            // The imu's place on the body, the identity when the body frame is the imu's.
            math::se3<double> body_from_imu = math::se3<double>::identity();
            // The standard deviations of the priors that hold the biases near zero, rad/s and m/s^2. The defaults are ORB-SLAM3's for
            // the first monocular initialisation, which holds the accelerometer bias at zero as a short window of motion cannot tell
            // it from gravity and the scale; a later refinement over a longer window can free it.
            double gyroscope_bias_deviation = 0.1;
            double accelerometer_bias_deviation = 1e-5;
            int iterations = 100;
            // How many times the preintegrations may be integrated again at the new bias, and the optimisation run again.
            int reintegrations = 3;
            // The largest standard deviations of the log scale and of gravity's direction (rad) for the estimate to be observable.
            double maximum_scale_deviation = 0.05;
            double maximum_gravity_deviation = 0.035;
        };

        class result final {
        public:
            // The optimisation ran and gave a positive, finite scale.
            bool success = false;
            // And the uncertainties are below the options' thresholds.
            bool observable = false;
            // Metres per unit of the map.
            double scale = 1.0;
            // The rotation from a gravity aligned frame, with gravity (0, 0, -g), into the map's world, and gravity in the map's world.
            math::so3<double> world_from_gravity = math::so3<double>::identity();
            math::matrix<double, 3, 1> gravity = math::matrix<double, 3, 1>::zero();
            sensor::imu::bias bias;
            // The imu's velocity at each keyframe, metric, in the map's world axes.
            std::vector<math::matrix<double, 3, 1>> velocities;
            // The preintegrations integrated at the estimated bias.
            std::vector<sensor::imu::preintegration> preintegrations;
            // The standard deviation of the log of the scale (the scale's relative deviation) and the largest of gravity's direction.
            double scale_deviation = 0.0;
            double gravity_deviation = 0.0;
            // The scale of the closed form estimate, zero when it failed.
            double linear_scale = 0.0;
            double cost = 0.0;
        };

    public:
        // The keyframes' body poses, world to body, and the preintegration from each keyframe to the next. At least four keyframes.
        static result estimate(const std::vector<math::se3<double>>& poses, const std::vector<sensor::imu::preintegration>& preintegrations, const options& settings);

        // The rotation that takes (0, 0, -1) onto a direction, the shortest one.
        static math::so3<double> rotation_from_down(const math::matrix<double, 3, 1>& direction);
    };
}

#endif // ZEROSLAM_OPTIMISATION_INERTIAL_INITIALISATION_HPP
