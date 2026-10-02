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

        // The polarity is the sign of the intensity step across the segment, left side (looking from the first endpoint to
        // the second) minus right side, or 0 when the step is too weak to tell.
        struct track final {
            int id;
            int landmark_id;
            bool active;
            float x1;
            float y1;
            float x2;
            float y2;
            int polarity;
            int start_frame_id;
            int last_frame_id;
            int length;
            int missed;
            std::vector<observation> history;
        };

        // Where a track's segment is expected in the frame being matched, as the motion of the points around it predicts.
        struct prediction final {
            bool valid;
            float x1;
            float y1;
            float x2;
            float y2;
        };

        struct options final {
            float min_length = 20.0f;
            float match_angle_tolerance = 10.0f;
            float match_midpoint_distance = 30.0f;
            float match_overlap = 0.3f;
            // With a prediction, a detection must lie along it: within this angle and this mean distance of its endpoints
            // from the predicted line.
            float predicted_angle_tolerance = 5.0f;
            float predicted_distance = 5.0f;
            int max_missed = 5;
        };

        constexpr static const float polarity_offset = 2.0f;
        constexpr static const float polarity_step_minimum = 4.0f;

    private:
        options settings;
        std::vector<track> track_list;
        int next_id;

    private:
        static float midpoint_distance_squared(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2);
        static float angle_degrees(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2);
        static float overlap_fraction(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2);
        void spawn(int frame_id, const detector::elsed::segment& segment, int polarity);
        void observe(track& existing, int frame_id, const detector::elsed::segment& segment, int polarity);

    public:
        line();
        explicit line(const options& opts);

        void set_options(const options& opts);
        const options& get_options() const;

        void update(int frame_id, const std::vector<detector::elsed::segment>& segments);

        // Matches each track with a valid prediction (predictions[i] for tracks()[i]) along it and the rest by proximity,
        // and moves a predicted track that matches nothing to its prediction; polarities[d] is detection d's polarity, and a
        // detection of the opposite polarity to a track never continues it.
        void update(int frame_id, const std::vector<detector::elsed::segment>& segments, const std::vector<prediction>& predictions, const std::vector<int>& polarities);

        // The polarity of a segment in an image (see track), from the intensity step polarity_offset pixels either side of it.
        static int polarity(const unsigned char* __restrict const data, const int width, const int height, const int stride, const float x1, const float y1, const float x2, const float y2);

        // Whether the second segment, of the given polarity, continues the first: the polarity read along the first one's direction.
        static int aligned_polarity(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2, int polarity);

        static float angle_between(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2);

        // The mean distance of the second segment's endpoints from the first one's line.
        static float line_distance(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2);

        // The share of the second segment that projects onto the first.
        static float overlap(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2);

        const std::vector<track>& tracks() const;
        std::vector<track*> active_tracks();
        std::vector<track*> all_tracks();
        track* find(int track_id);
    };
}

#endif // ZEROSLAM_FEATURE_TRACKER_LINE_HPP
