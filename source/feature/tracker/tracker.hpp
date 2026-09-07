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

#include "feature/descriptor/binary.hpp"
#include "feature/point.hpp"
#include "image/pyramid.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::tracker {
    class tracker final {
    public:
        struct observation final {
            int frame_id;
            float x;
            float y;
        };

        struct track final {
            int id;
            int landmark_id;
            bool active;
            float x;
            float y;
            int octave;
            int start_frame_id;
            int last_frame_id;
            int length;
            int missed;
            descriptor::binary<256> descriptor;
            std::vector<observation> history;
        };

        struct options final {
            int klt_half_window = 7;
            int klt_iterations = 30;
            float klt_min_eigenvalue = 1e-3f;
            float klt_fb_threshold = 1.0f;
            float klt_max_error = 40.0f;
            float min_spawn_distance = 8.0f;
            unsigned int reacquire_hamming = 50;
            float reacquire_radius = 20.0f;
            int max_missed = 5;
        };

        constexpr static const int octave_match_window = 1;

    private:
        options settings;
        std::vector<track> track_list;
        image::pyramid pyramid_previous;
        bool has_previous;
        int next_id;

    private:
        static float distance_squared(float ax, float ay, float bx, float by);

        bool near_existing_track(float x, float y) const;

        void spawn(int frame_id, const feature::point& keypoint, const descriptor::binary<256>& descriptor);

        void observe(track& existing, int frame_id, float x, float y);

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

        const std::vector<track>& tracks() const;

        std::vector<track*> active_tracks();

        std::vector<track*> all_tracks();

        track* find(int track_id);
    };
}

#endif // ZEROSLAM_FEATURE_TRACKER_TRACKER_HPP
