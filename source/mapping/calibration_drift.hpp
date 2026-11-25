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
#ifndef ZEROSLAM_MAPPING_CALIBRATION_DRIFT_HPP
#define ZEROSLAM_MAPPING_CALIBRATION_DRIFT_HPP

#include "mapping/calibration.hpp"
#include "math/lie.hpp"
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

namespace mapping {
    // The calibration of a rig's cameras followed over time, after J. Van Meter and A. Paul, "SLAM Dunk: A Recipe for
    // Long-Horizon Egocentric SLAM in the Wild", 2026 (https://www.alphaxiv.org/abs/2611.slam-dunk). A calibration held
    // constant over a long recording leaves an error that the map cannot adjust away: the focal lengths of a wearable
    // device's cameras rise as it warms, and the cameras move on its frame. The paper's model is implemented as described:
    // - Time is cut into chunks, five seconds each, and every frame belongs to the chunk its time falls in.
    // - Each chunk has its own focal lengths of each camera, and its own extrinsic of each camera but the reference, which
    //   stays the gauge the others are posed on. The principal points and the distortion stay as given, and so do the
    //   extrinsics of the sensors that are not cameras.
    // - A random walk ties each chunk to the one before it, so the calibration may drift and may not jump: a focal length
    //   by 0.035 pixels in 240 over five seconds, an extrinsic's rotation by 0.001 radians and its translation by 0.5 mm.
    //   The walk's variance grows with the time walked, so chunks of another length, and chunks that are skipped, are
    //   walked across at the same rate.
    // - Each chunk's focal lengths have a prior on the calibration given, of 0.5 pixels in 240 times the square root of
    //   the number of chunks, which together weigh as one such prior on their mean, and a prior on their ratio, the
    //   aspect ratio of the pixels, of 0.05 pixels in 240. Each chunk's extrinsic keeps the distance from the reference
    //   it was declared at to 0.5 mm, which is a metric rig's scale.
    // The focal lengths' walk and priors are held here as fractions of the focal length, the paper's pixels over the 240
    // pixel focal length of its cameras, so that they mean the same at any resolution.
    //
    // The chunks are solved in the adjustments of the whole map (mapping::calibration_drift_refinement); every other
    // adjustment holds each frame's calibration where its chunk has it. A new chunk starts from the one before it, which
    // is also what the frames that follow are tracked with until an adjustment has solved it. Only frames that come with
    // a time are followed: without one a frame is in no chunk, and keeps the calibration it was given.
    class calibration_drift final {
    public:
        struct settings final {
            bool enabled = false;
            // The length of a chunk of time.
            double chunk_seconds = 5.0;
            // The standard deviations of the walks over walk_seconds: a focal length's as a fraction of itself, an
            // extrinsic's rotation in radians and its translation in metres. A focal length walk of zero holds the focal
            // lengths as given, and either of the extrinsic's at zero holds the extrinsics.
            double focal_walk = 0.035 / 240.0;
            double rotation_walk = 0.001;
            double translation_walk = 0.0005;
        };

        // The time the walks' standard deviations are given over.
        constexpr static const double walk_seconds = 5.0;
        // The prior on the mean of a focal length over the chunks, and on the ratio of the focal lengths in each chunk, as
        // fractions of the focal length.
        constexpr static const double focal_prior_fraction = 0.5 / 240.0;
        constexpr static const double aspect_prior_fraction = 0.05 / 240.0;
        // The prior on the length of a baseline in each chunk, in metres.
        constexpr static const double baseline_sigma = 0.0005;

        constexpr static const double chunk_seconds_minimum = 0.001;
        constexpr static const double chunk_seconds_maximum = 86400.0;

        // One camera's calibration in one chunk of time.
        struct camera_state final {
            sensor::model camera;
            // The camera's pose on the reference camera, the identity for the reference itself.
            math::se3<double> reference_from_camera = math::se3<double>::identity();
            // The observations that held it in the last adjustment that followed the calibration; with none, it is where
            // the walk between the chunks around it leaves it, or still the calibration its chunk started from.
            size_t observations = 0;
        };

        // One chunk of time: its place in the sequence of chunks, counted from the first frame's, and its cameras, the
        // reference first, in the rig's order (see mapping::rig::camera).
        struct chunk final {
            long long index = 0;
            std::vector<camera_state> cameras;
        };

        // What an adjustment of the whole map needs to follow the calibration, and what it gives back.
        class request final {
        public:
            struct camera final {
                // The camera as given, which the focal lengths' priors are on.
                sensor::model given;
                bool follow_focal = false;
                bool follow_extrinsic = false;
                // The distance from the reference the camera was declared at, held when it is a metric baseline.
                double baseline = 0.0;
            };

            settings walk;
            // The prior on the mean of each focal length over the chunks, as a fraction of it.
            double focal_prior_fraction = calibration_drift::focal_prior_fraction;
            std::vector<camera> cameras;

            // The chunks, ascending: as they are held when the request is made, each with every camera, and once the
            // adjustment has followed the calibration and kept the result, which refined then says, as it left them.
            bool refined = false;
            std::vector<chunk> chunks;
        };

    private:
        settings configuration;
        // The cameras as the first frame followed came with them, and the time of that frame, which the chunks count from.
        std::vector<sensor::model> given_cameras;
        long long origin = 0;
        long long chunk_duration = 0;
        // The chunks that have held a frame, ascending.
        std::vector<chunk> held_chunks;
        size_t refinement_count = 0;

    public:
        // The settings; the length of a chunk is taken when the first frame is followed and kept from then on.
        void set(const settings& value);

        const settings& get() const;

        // Takes a frame at its time, in nanoseconds, with the cameras it came with, the reference first, and each camera's
        // pose on the reference as the rig has it now. Returns the frame's chunk and leaves that chunk's cameras and poses
        // in the arguments; a chunk no frame has been in before starts from the newest. Returns -1, the arguments as they
        // were, for a frame that is not followed: the following is off, the frame comes before the first one followed, or
        // with other cameras than it did.
        long long take_frame(const long long timestamp, std::vector<sensor::model>& cameras, std::vector<math::se3<double>>& extrinsics);

        // The chunks that have held a frame, ascending, and the newest of them, null when there is none.
        const std::vector<chunk>& chunks() const;
        const chunk* newest() const;

        // The time a chunk starts at and how long a chunk is, in nanoseconds; zero until a frame is followed.
        long long chunk_start(const long long index) const;
        long long chunk_nanoseconds() const;

        size_t get_refinements() const;

        // Whether an adjustment of a map of this many keyframes, whose views span this angle, may follow the calibration:
        // as for a constant calibration (see mapping::calibration::ready), a focal length is not observed until the scene
        // has been seen to turn, and once it has been followed every later adjustment may.
        bool ready(const size_t keyframes, const double view_spread_degrees) const;

        // A request to follow the calibration of the chunks, false when there is nothing to follow. The level of the constant
        // calibration says how far the given focal lengths are trusted: off, to the paper's prior; otherwise only to the
        // wide prior of the constant refinement (mapping::calibration::focal_length_prior_fraction), which the chunks then
        // share. The baselines are the distances from the reference the cameras were declared at, the reference's first.
        bool make_request(const calibration::level level, const bool extrinsics_held, const std::vector<double>& baselines, request& following) const;

        // Takes an adjustment's result. A chunk the result leaves out, the map holding no frame of it, goes the way along
        // between the result's chunks around it, or with the nearest of them when it is before the first or after the last.
        void update(const request& following);

        // The parameters of the camera that are focal lengths, none for a camera that does not say what its parameters are.
        static std::vector<size_t> focal_parameters(const sensor::model& camera);

        // A camera and an extrinsic a fraction of the way from one to another: where a walk between two chunks leaves a
        // chunk between them that nothing else holds.
        static sensor::model camera_between(const sensor::model& before, const sensor::model& after, const double fraction);
        static math::se3<double> extrinsic_between(const math::se3<double>& before, const math::se3<double>& after, const double fraction);
    };
}

#endif // ZEROSLAM_MAPPING_CALIBRATION_DRIFT_HPP
