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
#ifndef ZEROSLAM_FEATURE_TRACKER_LINE_HPP
#define ZEROSLAM_FEATURE_TRACKER_LINE_HPP

#include "feature/detector/elsed.hpp"

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

namespace feature::tracker {
    class line final {
    public:
        struct observation final {
            int frame_id;
            float x1;
            float y1;
            float x2;
            float y2;
        };

        struct track final {
            int id;
            int landmark_id;
            bool active;
            float x1;
            float y1;
            float x2;
            float y2;
            int start_frame_id;
            int last_frame_id;
            int length;
            int missed;
            std::vector<observation> history;
        };

        struct options final {
            float min_length = 20.0f;
            float match_angle_tolerance = 10.0f;
            float match_midpoint_distance = 30.0f;
            float match_overlap = 0.3f;
            int max_missed = 5;
        };

    private:
        options settings;
        std::vector<track> track_list;
        int next_id;

    private:
        static float midpoint_distance_squared(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2);
        static float angle_degrees(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2);
        static float overlap_fraction(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2);
        void spawn(int frame_id, const detector::elsed::segment& segment);
        void observe(track& existing, int frame_id, const detector::elsed::segment& segment);

    public:
        line();
        explicit line(const options& opts);

        void set_options(const options& opts);
        const options& get_options() const;

        void update(int frame_id, const std::vector<detector::elsed::segment>& segments);

        const std::vector<track>& tracks() const;
        std::vector<track*> active_tracks();
        std::vector<track*> all_tracks();
        track* find(int track_id);
    };
}

#endif // ZEROSLAM_FEATURE_TRACKER_LINE_HPP
