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
#ifndef ZEROSLAM_MAPPING_CALIBRATION_HPP
#define ZEROSLAM_MAPPING_CALIBRATION_HPP

#include "math/matrix.hpp"
#include "sensor/camera.hpp"
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

namespace mapping {
    // The calibration of the camera the frames are taken with, which the adjustments may refine.
    //
    // The caller gives a camera with every frame. While it gives the same one, the frames are processed with the refined
    // camera instead; a camera other than the last one given is a new calibration, which starts again from it. Which
    // parameters are refined is chosen by what each is (sensor::camera::parameter_role), so any model whose parameters
    // say so can be refined; a model that does not say can only be refined whole. Below the level that refines everything
    // the focal lengths move together, keeping the pixels' aspect ratio, which is the sensor's and rarely in doubt: a
    // motion in a plane, a vehicle's or the circle of a rendered scene, observes the focal length along the plane and
    // leaves the other free to stretch the scene across it. Each free parameter has a Gaussian prior
    // on the given calibration, wide enough that the observations decide where they observe it well, and holding it where
    // they hardly do: the principal point against a turn, a focal length against the depth of a narrow scene, the
    // higher order distortion against everything.
    //
    // A monocular map does not observe the focal length until it has seen the scene turn. From a translation alone the
    // image of a scene of little depth relief is that of a scene stretched along the view, seen by a camera with its focal
    // length stretched as much (in the affine limit only f / Z is seen), and the two view geometry, the triangulation and
    // the first adjustments build the map in whatever focal length they are given. Turning breaks that tie, as the stretch
    // of one view is not the stretch of another, so the refinement waits for enough keyframes and for their views to span
    // an angle before it starts.
    class calibration final {
    public:
        enum class level {
            off,
            focal,
            focal_centre,
            intrinsics,
            all
        };

        // What an adjustment needs to refine the camera, and what it gives back. Every frame whose camera has exactly the
        // parameters of the request's camera shares one intrinsics vertex, with the free parameters and the priors given.
        class refinement final {
        public:
            sensor::camera::model<double> camera;
            // The local dimension of the intrinsics vertex that updates each parameter, -1 for one held fixed (see
            // optimisation::vertices::intrinsics).
            std::vector<int> free_parameters;
            std::vector<double> prior_values;
            // The prior's standard deviation per parameter, zero for no prior.
            std::vector<double> prior_sigmas;
            // Whether the adjustment refined the camera and kept the result, and the parameters it ended with.
            bool refined = false;
            std::vector<double> parameters;
        };

        // The keyframes the map holds, and the angle their views span, before the refinement starts.
        constexpr static const size_t keyframes_minimum = 10;
        constexpr static const double view_spread_degrees_minimum = 10.0;

        // The priors' standard deviations: a focal length's as a fraction of itself, the principal point's as a fraction of
        // the mean focal length, and the distortion's as values of the normalised coordinates' coefficients.
        constexpr static const double focal_length_prior_fraction = 0.1;
        constexpr static const double principal_point_prior_fraction = 0.05;
        constexpr static const double distortion_prior_sigma = 0.1;
        constexpr static const double distortion_higher_prior_sigma = 0.01;

    private:
        level refined_level = level::off;
        bool given_valid = false;
        sensor::model given_camera;
        sensor::model current_camera;
        size_t refinement_count = 0;
        bool started = false;

    public:
        void set_level(const level value);

        level get_level() const;

        // The camera to process a frame with, given the one the caller gave with it.
        const sensor::model& camera_for(const sensor::model& given);

        // The camera as refined so far, and as last given; both the given camera until a frame has been.
        const sensor::model& get_camera() const;

        const sensor::model& get_given_camera() const;

        size_t get_refinements() const;

        // Whether the refinement has started.
        bool has_started() const;

        // Whether an adjustment of a map of this many keyframes, whose views span this angle, may refine the camera. Once it
        // has, every later adjustment that is asked to may, even if culling takes the keyframes back under the bounds.
        bool ready(const size_t keyframes, const double view_spread_degrees) const;

        // The largest angle between the optical axis of any of the given poses (world to camera rotations) and their mean.
        static double view_spread_degrees(const std::vector<math::matrix<double, 3, 3>>& rotations);

        // The local dimension updating each parameter of the camera that the level asks to refine, -1 for the others, and
        // false when there are none to refine: the level is off, the camera has no parameters, or it does not say what they
        // are and the level is not all.
        bool free_parameters(const sensor::camera::model<double>& camera, std::vector<int>& free) const;

        // A request to refine the current camera, false when there is nothing to refine.
        bool make_refinement(refinement& request) const;

        // Takes an adjustment's result: the refined camera when it refined and kept it.
        void update(const refinement& request);
    };
}

#endif // ZEROSLAM_MAPPING_CALIBRATION_HPP
