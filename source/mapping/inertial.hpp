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
#ifndef ZEROSLAM_MAPPING_INERTIAL_HPP
#define ZEROSLAM_MAPPING_INERTIAL_HPP

#include "mapping/frame.hpp"
#include "mapping/rig.hpp"
#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "sensor/imu/measurement.hpp"
#include "sensor/imu/preintegration.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace mapping {
    // The imu of a visual inertial map: its samples, the time of each frame, and for each keyframe the imu's velocity and bias and
    // the measurements preintegrated from the keyframe before it. The map's frames are posed by the reference camera, which is the
    // body the inertial edges take (optimisation::edges::inertial), the imu sitting on it at the rig's extrinsic.
    //
    // Keyframes are chained in time within a run of tracking: a run that starts again after tracking is lost (a new submap) starts
    // a new chain, its first keyframe tied to nothing before it. A chain is initialised once the window of its keyframes has given
    // the direction of gravity, the scale where the map has none, the velocities and the biases; from then on its keyframes carry
    // a velocity and a bias and every adjustment of two consecutive ones holds them with an inertial edge and a random walk edge.
    class inertial final {
    public:
        struct keyframe_state final {
            long long timestamp = 0;
            size_t chain = 0;
            // Whether the measurements from the keyframe before it in its chain are preintegrated.
            bool linked = false;
            // Whether the velocity and the bias are estimated, the chain being initialised.
            bool estimated = false;
            math::matrix<double, 3, 1> velocity = math::matrix<double, 3, 1>::zero();
            sensor::imu::bias bias;
            sensor::imu::preintegration from_previous;
        };

        // The vertices an adjustment holds for the imu, read back after it.
        struct factors final {
            std::vector<std::pair<int, optimisation::vertex*>> velocities;
            std::vector<std::pair<int, optimisation::vertex*>> biases;
            optimisation::vertex* extrinsic = nullptr;
            size_t edges = 0;
        };

        // A sample may be held over this many sample periods past the last one, for a frame that arrives a moment before the
        // sample at its own time.
        constexpr static const double hold_periods = 3.0;
        // Samples older than this before the newest keyframe are dropped.
        constexpr static const long long sample_history = 2000000000ll;

    private:
        bool declared_ = false;
        int sensor_id_ = 0;
        sensor::imu::parameters noise_;
        std::vector<sensor::imu::sample> samples_;
        std::map<int, long long> frame_times_;
        std::map<int, keyframe_state> keyframes_;
        std::map<size_t, bool> chain_initialised_;
        size_t chain_ = 0;
        math::matrix<double, 3, 1> gravity_ = math::matrix<double, 3, 1>::zero();
        bool gravity_known_ = false;

    public:
        void declare(const int sensor_id, const sensor::imu::parameters& noise);
        bool declared() const;
        int sensor_id() const;
        const sensor::imu::parameters& noise() const;

        // The index of the imu in the rig, false when it is not there.
        bool index_in(const mapping::rig& rig, size_t& index) const;

        // Takes a sample, false when it does not follow the last one in time.
        bool add_sample(const sensor::imu::sample& sample);
        const std::vector<sensor::imu::sample>& samples() const;

        void set_frame_time(const int frame_id, const long long timestamp);
        bool frame_time(const int frame_id, long long& timestamp) const;

        // Preintegrates the samples between two timestamps at a bias, holding the last sample a moment past it if need be.
        bool preintegrate(const long long start, const long long end, const sensor::imu::bias& linearisation, sensor::imu::preintegration& result) const;

        // Adds a keyframe to the current chain, preintegrated from the one before it in the chain and, once the chain is
        // initialised, with the velocity and bias carried to it from there.
        void add_keyframe(const int frame_id, const std::unordered_map<int, mapping::frame>& frames, const mapping::rig& rig);

        // Starts a new chain: the next keyframe is tied to none before it.
        void break_chain();
        size_t chain() const;

        // Drops the keyframes no longer among the map's, the measurements of each merged into the next keyframe of its chain.
        void synchronise(const std::unordered_set<int>& keyframe_ids, const std::unordered_map<int, mapping::frame>& frames);

        std::map<int, keyframe_state>& keyframes();
        const std::map<int, keyframe_state>& keyframes() const;

        // The keyframes of the current chain that are linked one to the next, ending at the newest, ascending.
        std::vector<int> chain_window() const;

        bool chain_initialised(const size_t chain) const;
        void set_chain_initialised(const size_t chain);
        // Whether any chain is initialised, the map then being in metres with gravity known.
        bool initialised() const;

        const math::matrix<double, 3, 1>& gravity() const;
        void set_gravity(const math::matrix<double, 3, 1>& gravity_world);

        // The imu's state at a frame (its pose the reference camera's, world to camera) and the reverse.
        static sensor::imu::preintegration::state state_at(const mapping::frame& frame, const math::matrix<double, 3, 1>& velocity, const math::se3<double>& reference_from_imu);
        static void pose_of(const sensor::imu::preintegration::state& state, const math::se3<double>& reference_from_imu, math::matrix<double, 3, 3>& rotation, math::matrix<double, 3, 1>& translation);

        // The pose and velocity at a timestamp predicted from the newest estimated keyframe of the current chain, false when there
        // is none or the samples do not reach it.
        bool predict(const long long timestamp, const std::unordered_map<int, mapping::frame>& frames, const mapping::rig& rig, math::matrix<double, 3, 3>& rotation, math::matrix<double, 3, 1>& translation, math::matrix<double, 3, 1>& velocity) const;

        // Adds the velocities, biases and inertial edges of the estimated keyframes among the posed ones, and of the keyframe
        // before each, held where its pose is held or not posed in the adjustment. The extrinsic is refined where the rig asks for
        // it, unless held.
        factors add_factors(optimisation::factor_graph& graph, const std::unordered_map<int, optimisation::vertex*>& poses, const std::unordered_map<int, mapping::frame>& frames, const mapping::rig& rig, const bool hold_extrinsic) const;

        // Takes the velocities, biases and extrinsic an adjustment left, integrating again what the new biases move too far.
        void read_factors(const factors& held, mapping::rig& rig);
    };
}

#endif // ZEROSLAM_MAPPING_INERTIAL_HPP
