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
#ifndef ZEROSLAM_FEATURE_TRACKER_TRACKER_HPP
#define ZEROSLAM_FEATURE_TRACKER_TRACKER_HPP

#include "core/coordinates.hpp"
#include "feature/descriptor/binary.hpp"
#include "feature/point.hpp"
#include "feature/tracker/dominant_flow.hpp"
#include "feature/tracker/patch_flow.hpp"
#include "feature/tracker/wavelet_flow.hpp"
#include "image/pyramid.hpp"
#include "image/quaternion_wavelet.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <memory>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::tracker {
    class tracker final {
    public:
        struct observation final {
            int frame_id;
            core::pixel_centre x;
            core::pixel_centre y;
        };

        struct anchoring final {
            patch_flow::anchor anchor;
            patch_flow::state warp;
        };

        struct wavelet_anchoring final {
            wavelet_flow::anchor anchor;
            float displacement_x = 0.0f;
            float displacement_y = 0.0f;
        };

        struct track final {
            int id;
            int landmark_id;
            bool active;
            core::pixel_centre x;
            core::pixel_centre y;
            int octave;
            int start_frame_id;
            int last_frame_id;
            int length;
            int missed;
            int outliers;
            int outlier_frame_id;
            descriptor::binary<256> descriptor;
            std::vector<observation> history;
            std::unique_ptr<anchoring> anchored;
            std::unique_ptr<wavelet_anchoring> wavelet_anchored;
        };

        enum class association_kind {
            optical_flow,
            descriptor,
            both
        };

        enum class flow_kind {
            intensity,
            wavelet
        };

        enum class wavelet_seed_kind {
            rest,
            klt,
            klt_fallback
        };

        struct options final {
            association_kind association = association_kind::both;
            flow_kind flow = flow_kind::intensity;
            int wavelet_half_window = 2;
            int wavelet_levels = 6;
            bool wavelet_robust = false;
            bool wavelet_undecimated = false;
            wavelet_seed_kind wavelet_seed = wavelet_seed_kind::rest;
            float match_radius = 12.0f;
            float match_search_radius = 60.0f;
            unsigned int match_hamming = 80;
            float match_ratio = 0.9f;
            float drift_radius = 1.5f;
            unsigned int drift_hamming = 0;
            int klt_half_window = 7;
            bool klt_predict = false;
            int klt_half_window_blur = 0;
            float blur_carry_ratio = 0.7f;
            int klt_iterations = 30;
            float klt_min_eigenvalue = 1e-3f;
            float klt_fb_threshold = 1.0f;
            float klt_max_error = 40.0f;
            bool klt_damped_steps = false;
            bool anchored_patches = false;
            patch_flow::model_kind anchor_model = patch_flow::model_kind::translation;
            float anchor_refresh_error = 0.0f;
            float min_spawn_distance = 8.0f;
            float collision_distance = 2.0f;
            float reacquire_radius = 20.0f;
            unsigned int reacquire_hamming = 50;
            int max_missed = 5;
        };

        constexpr static const int octave_match_window = 1;

    private:
        options settings;
        std::vector<track> track_list;
        image::pyramid pyramid_previous;
        image::quaternion_wavelet wavelet_previous;
        dominant_flow::affine flow_previous;
        float carry_ratio_previous;
        bool has_previous;
        int next_id;

    private:
        static float distance_squared(float ax, float ay, float bx, float by);

        void observe(track& existing, int frame_id, float x, float y);

        patch_flow::options anchor_options() const;

        wavelet_flow::options wavelet_options() const;

        void build_anchors(const image::pyramid& pyramid);

        void build_wavelet_anchors(const image::quaternion_wavelet& transform);

        void discard_anchors();

    public:
        tracker();

        explicit tracker(const options& opts);

        void set_options(const options& opts);

        const options& get_options() const;

        void update(
            int frame_id,
            const image::pyramid& pyramid,
            const std::vector<feature::point>& keypoints,
            const std::vector<descriptor::binary<256>>& descriptors
        );

        bool near_existing_track(float x, float y) const;

        void prune_collisions();

        track& spawn(int frame_id, const feature::point& keypoint, const descriptor::binary<256>& descriptor);

        const std::vector<track>& tracks() const;

        std::vector<track*> active_tracks();

        std::vector<track*> all_tracks();

        track* find(int track_id);
    };
}

#endif // ZEROSLAM_FEATURE_TRACKER_TRACKER_HPP
