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
#ifndef ZEROSLAM_MAPPING_ODOMETRY_HPP
#define ZEROSLAM_MAPPING_ODOMETRY_HPP

#include "mapping/rig.hpp"
#include "math/lie.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "sensor/odometry/model.hpp"
#include "sensor/odometry/preintegration.hpp"
#include "sensor/odometry/sample.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace mapping {
    // The wheel odometry of a map: the sensor's model, noise and calibration, its samples, the time of every frame, and the
    // motions preintegrated between frames. The odometer is a sensor of the rig (mapping::rig, sensor_kind::odometry), its
    // extrinsic the odometry frame's pose in the reference camera's frame, refined through the rig's extrinsic vertex when the
    // rig asks for it. An adjustment takes an odometry edge between each two consecutive frames it holds, all sharing one
    // calibration vertex and the extrinsic vertex, with priors on both, and the motion between two frames is integrated once
    // and kept: integrated again when the calibration moves past what its first order correction holds, and merged from the
    // motions on either side of a keyframe culled between them. Nothing is used until the odometer is configured.
    class odometry final {
    public:
        class settings final {
        public:
            int sensor_id = 0;
            sensor::odometry::model kinematics;
            sensor::odometry::noise noise;
            // The declared calibration: its values, which are free, and the prior deviation of each (zero for none).
            sensor::odometry::calibration calibration;
            // The deviations of the prior on the extrinsic, rotation (radians) then translation (metres), each about an axis of
            // the odometry frame, zero for none. Planar motion leaves the height above the plane (the last) unobservable.
            double extrinsic_deviations[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        };

        // The vertices an adjustment took, for reading back.
        class graph_vertices final {
        public:
            optimisation::vertex* calibration = nullptr;
            optimisation::vertex* extrinsic = nullptr;
            size_t sensor_index = 0;
            size_t edges = 0;
        };

        // Frames further apart than this (seconds) take no odometry edge: tracking was lost between them, and the odometer's
        // drift over the gap is no measurement of the map.
        constexpr static const double maximum_interval = 10.0;
        // A motion may run past the newest sample by this much (seconds), the samples carried on at their last speed: a frame
        // usually comes before the sample after it. Such a motion is not kept, and is integrated again when that sample comes.
        constexpr static const double maximum_extrapolation = 0.1;
        // An initialisation over less than this distance (metres) is not scaled by the odometry.
        constexpr static const double minimum_initialisation_distance = 0.02;

    private:
        bool enabled = false;
        settings configuration;
        double values[sensor::odometry::model::maximum_parameters] = {};
        double values_before[sensor::odometry::model::maximum_parameters] = {};
        std::vector<sensor::odometry::sample> samples;
        std::unordered_map<int, long long int> frame_times;
        std::unordered_map<long long int, sensor::odometry::preintegration> motions;
        bool extrinsic_prior_set = false;
        math::se3<double> extrinsic_prior;

    public:
        // Takes the odometer's settings, its calibration estimate starting at the declared values. False, leaving the odometer
        // unconfigured, for a model that is not valid or a calibration of another size.
        bool configure(const settings& configuration_value);
        bool is_enabled() const;
        const settings& get_settings() const;

        // The calibration as the adjustments have it, a value per parameter of the model.
        const double* calibration() const;
        size_t parameter_count() const;

        // Takes a sample, false for one that is not after the newest.
        bool add_sample(const sensor::odometry::sample& value);
        size_t sample_count() const;

        // The time of a frame, which the motions between frames are integrated between.
        void set_frame_time(const int frame_id, const long long int timestamp);
        bool frame_time(const int frame_id, long long int& timestamp) const;

        // The motion of the odometry frame between two frames, integrated at the current calibration: false when the samples
        // do not reach both frames' times, past the newest by at most the extrapolation allowed.
        bool motion(const int from_frame_id, const int to_frame_id, sensor::odometry::preintegration& result);

        // The motion of the reference camera between two frames that the odometry predicts, to from from (world to camera
        // pose of the second times the inverse of the first's), through the odometer's extrinsic in the rig.
        bool camera_motion(const int from_frame_id, const int to_frame_id, const mapping::rig& rig, math::se3<double>& result);

        // Scales the translation of a two view initialisation, the second frame's pose relative to the first's (world to camera
        // of the second times the inverse of the first's), to the distance the odometry says the camera moved, so that the map
        // starts in metres. False, leaving it, when the odometry does not reach both frames or saw too little motion.
        bool scale_initialisation(const int anchor_frame_id, const int frame_id, const mapping::rig& rig, math::matrix<double, 3, 1>& translation);

        // Adds the odometry between each two consecutive frames of those the adjustment holds, the calibration vertex and the
        // extrinsic's, with their priors. No edge is added when the odometer is not configured or not a sensor of the rig.
        graph_vertices add_to_graph(optimisation::factor_graph& graph, const std::unordered_map<int, optimisation::vertex*>& frame_vertices, const mapping::rig& rig);

        // Takes the calibration and the extrinsic the adjustment left, keeping what they were for restore().
        void read_from_graph(const graph_vertices& vertices, mapping::rig& rig);
        void restore(const graph_vertices& vertices);

    private:
        static long long int key(const int from_frame_id, const int to_frame_id);
        bool integrate(const long long int start, const long long int end, sensor::odometry::preintegration& result, bool& covered) const;
        bool find_sensor(const mapping::rig& rig, size_t& index) const;
    };
}

#endif // ZEROSLAM_MAPPING_ODOMETRY_HPP
