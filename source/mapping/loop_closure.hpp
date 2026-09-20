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
#ifndef ZEROSLAM_MAPPING_LOOP_CLOSURE_HPP
#define ZEROSLAM_MAPPING_LOOP_CLOSURE_HPP

#include "estimation/correspondence_3d_3d.hpp"
#include "feature/descriptor/binary.hpp"
#include "mapping/covisibility.hpp"
#include "mapping/place_recognition.hpp"
#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "sensor/camera.hpp"

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
    class loop_closure final {
    public:
        class record final {
        public:
            int landmark_id;
            feature::descriptor::binary<256> descriptor;
            math::matrix<double, 3, 1> location;
            float pixel_x;
            float pixel_y;
        };

        class correspondence final {
        public:
            int landmark_id;
            int recorded_landmark_id;
        };

        class result final {
        public:
            bool found;
            int keyframe_id;
            size_t correspondences;
            size_t inliers;
            math::sim3<double> correction;
            math::sim3<double> relative;
            std::vector<correspondence> matches;
        };

        constexpr static const size_t max_candidates = 20;
        constexpr static const int min_keyframe_gap = 30;
        constexpr static const int max_covisible_landmarks = 15;
        constexpr static const double max_covisible_fraction = 0.5;
        constexpr static const double covisible_loop_min_translation = 0.25;
        constexpr static const double covisible_loop_min_relative_translation = 0.1;
        constexpr static const float match_hamming_maximum = 50.0f;
        constexpr static const float match_ratio = 0.8f;
        constexpr static const size_t gms_minimum_pairs = 40;
        constexpr static const double inlier_radius_fraction = 0.05;
        constexpr static const double guided_search_radius = 10.0;
        constexpr static const float guided_hamming_maximum = 80.0f;
        constexpr static const int refine_rounds = 10;
        constexpr static const double reprojection_inlier_bound_squared = 9.210;
        constexpr static const size_t min_inliers = 15;
        constexpr static const double min_inlier_fraction = 0.5;
        constexpr static const double max_scale_ratio = 2.0;

    private:
        class keyframe final {
        public:
            math::se3<double> pose;
            sensor::model camera;
            std::vector<record> records;
        };

        place_recognition recognition;
        std::unordered_map<int, keyframe> keyframes;
        float hamming_scale = 1.0f;

        bool covisible_revisit_loop(const int keyframe_id, const math::se3<double>& pose, const sensor::model& camera, const record* const keyframe_records, const int candidate_id, const keyframe& candidate, const std::vector<estimation::correspondence_3d_3d<double>>& correspondences, const std::vector<correspondence>& pairs, const std::vector<std::pair<size_t, size_t>>& pair_records, result& outcome) const;

    public:
        loop_closure();

    public:
        result detect(const int keyframe_id, const math::se3<double>& pose, const sensor::model& camera, const covisibility& graph, const record* const keyframe_records, const size_t keyframe_records_size, const int submap_start_id = 0) const;

        void add_keyframe(const int keyframe_id, const math::se3<double>& pose, const sensor::model& camera, const record* const keyframe_records, const size_t keyframe_records_size);

        void remove_keyframe(const int keyframe_id);

        size_t num_keyframes() const;

        void set_hamming_scale(const float scale);

        std::vector<int> recall(const feature::descriptor::binary<256>* const descriptors, const size_t descriptors_size, const size_t max_recalled) const;

        const record* records_of(const int keyframe_id, size_t& records_size) const;
    };
}

#endif // ZEROSLAM_MAPPING_LOOP_CLOSURE_HPP
