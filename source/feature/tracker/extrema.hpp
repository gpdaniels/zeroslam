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
#ifndef ZEROSLAM_FEATURE_TRACKER_EXTREMA_HPP
#define ZEROSLAM_FEATURE_TRACKER_EXTREMA_HPP

#include "feature/descriptor/binary.hpp"
#include "feature/detector/curvature_extrema.hpp"
#include "feature/point.hpp"
#include "feature/tracker/dominant_flow.hpp"
#include "feature/tracker/tracker.hpp"
#include "image/image.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdint>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::tracker {
    class extrema final {
    public:
        struct options final {
            detector::curvature_extrema::options detection;
            int flow_scale = 6;
            float flow_match_threshold = 64.0f;
            float flow_ratio = 0.75f;
            float flow_sigma = 2.0f;
            size_t flow_maximum_features = 500;
            float climb_lambda = 0.3f;
            float climb_sigma = 2.5f;
            int climb_iteration_limit = 100;
            float maximum_correction = 8.0f;
            bool climb_bidirectional = false;
            float bidirectional_threshold = 3.0f;
            float minimum_spawn_distance = 4.0f;
            int maximum_missed = 0;
            size_t maximum_tracks = 20000;
            int descriptor_border = 25;
        };

        struct diagnostics final {
            size_t detected_extrema = 0;
            size_t flow_matches = 0;
            bool flow_fallback = false;
            dominant_flow::affine flow;
            size_t tracked = 0;
            size_t lost = 0;
            size_t spawned = 0;
            size_t bidirectional_rejections = 0;
            std::int64_t response_scale = 1;
        };

    private:
        struct entry final {
            tracker::track track;
            int sign;
        };

        options settings;
        std::vector<entry> entries;
        int next_id;
        bool has_previous;
        std::vector<feature::point> small_points_previous;
        std::vector<descriptor::binary<256>> small_descriptors_previous;
        dominant_flow::affine flow_previous;
        std::vector<std::int64_t> kappa_previous;
        std::int64_t response_scale_previous;
        int width_previous;
        int height_previous;
        diagnostics last_diagnostics;

    private:
        static void observe(tracker::track& existing, int frame_id, float x, float y);

        static void hill_climb(
            const std::int64_t* __restrict const kappa,
            int width,
            int height,
            int border,
            double sign,
            double scale,
            double lambda,
            double sigma_squared,
            float predicted_x,
            float predicted_y,
            int iteration_limit,
            int& cell_x,
            int& cell_y
        );

        static dominant_flow::affine small_to_full(const dominant_flow::affine& small, float ratio_x, float ratio_y);

    public:
        extrema();
        explicit extrema(const options& opts);

        void update(int frame_id, const image::image& image_level0);

        void set_options(const options& opts);
        const options& get_options() const;

        const diagnostics& get_diagnostics() const;

        int sign_of(int track_id) const;

        std::vector<tracker::track*> active_tracks();
        std::vector<tracker::track*> all_tracks();
        tracker::track* find(int track_id);
        size_t size() const;
    };
}

#endif // ZEROSLAM_FEATURE_TRACKER_EXTREMA_HPP
