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

#include "feature/tracker/line.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static inline bool is_value_approx(float lhs, float rhs, float epsilon = 1e-4f) {
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

namespace {
    feature::detector::elsed::segment make_segment(float x1, float y1, float x2, float y2) {
        feature::detector::elsed::segment segment;
        segment.x1 = x1;
        segment.y1 = y1;
        segment.x2 = x2;
        segment.y2 = y2;
        const float dx = x2 - x1;
        const float dy = y2 - y1;
        segment.length = std::sqrt((dx * dx) + (dy * dy));
        segment.response = 0.0f;
        segment.support = 0;
        return segment;
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    feature::tracker::line::options opts;
    opts.max_missed = 5;
    feature::tracker::line tracker(opts);

    {
        std::vector<feature::detector::elsed::segment> segments;
        segments.push_back(make_segment(100.0f, 100.0f, 300.0f, 100.0f));
        segments.push_back(make_segment(200.0f, 50.0f, 200.0f, 250.0f));
        tracker.update(0, segments);
        const std::vector<feature::tracker::line::track>& tracks = tracker.tracks();
        REQUIRE(tracks.size() == 2);
        REQUIRE(tracks[0].id == 0);
        REQUIRE(tracks[1].id == 1);
        REQUIRE(tracks[0].active);
        REQUIRE(tracks[1].active);
        REQUIRE(tracks[0].length == 1);
        REQUIRE(tracks[0].start_frame_id == 0);
        REQUIRE(tracks[0].landmark_id == -1);
    }

    {
        std::vector<feature::detector::elsed::segment> segments;
        segments.push_back(make_segment(105.0f, 103.0f, 305.0f, 103.0f));
        segments.push_back(make_segment(205.0f, 53.0f, 205.0f, 253.0f));
        tracker.update(1, segments);
        const std::vector<feature::tracker::line::track>& tracks = tracker.tracks();
        REQUIRE(tracks.size() == 2);
        REQUIRE(tracks[0].active);
        REQUIRE(tracks[1].active);
        REQUIRE(tracks[0].length == 2);
        REQUIRE(tracks[1].length == 2);
        REQUIRE(tracks[0].last_frame_id == 1);
        REQUIRE(is_value_approx(tracks[0].x1, 105.0f));
        REQUIRE(is_value_approx(tracks[0].y1, 103.0f));
        REQUIRE(is_value_approx(tracks[0].x2, 305.0f));
        REQUIRE(is_value_approx(tracks[0].y2, 103.0f));
        REQUIRE(tracks[0].history.size() == 2);
        REQUIRE(tracks[0].history[0].frame_id == 0);
        REQUIRE(tracks[0].history[1].frame_id == 1);
    }

    {
        std::vector<feature::detector::elsed::segment> segments;
        segments.push_back(make_segment(207.0f, 55.0f, 207.0f, 255.0f));
        segments.push_back(make_segment(400.0f, 400.0f, 450.0f, 400.0f));
        tracker.update(2, segments);
        const std::vector<feature::tracker::line::track>& tracks = tracker.tracks();
        REQUIRE(tracks.size() == 3);
        REQUIRE(tracks[0].id == 0);
        REQUIRE(tracks[0].active == false);
        REQUIRE(tracks[0].missed == 1);
        REQUIRE(tracks[1].id == 1);
        REQUIRE(tracks[1].active);
        REQUIRE(tracks[1].length == 3);
        REQUIRE(tracks[1].last_frame_id == 2);
        REQUIRE(tracks[2].id == 2);
        REQUIRE(tracks[2].active);
        REQUIRE(tracks[2].length == 1);
    }

    {
        for (int frame_id = 3; frame_id <= 6; ++frame_id) {
            std::vector<feature::detector::elsed::segment> segments;
            segments.push_back(make_segment(209.0f, 57.0f, 209.0f, 257.0f));
            tracker.update(frame_id, segments);
            const std::vector<feature::tracker::line::track>& tracks = tracker.tracks();
            REQUIRE(tracks.size() == 3);
            REQUIRE(tracks[0].id == 0);
            REQUIRE(tracks[0].active == false);
            REQUIRE(tracks[0].missed == frame_id - 1);
            REQUIRE(tracks[1].id == 1);
            REQUIRE(tracks[1].active);
            REQUIRE(tracks[2].id == 2);
            REQUIRE(tracks[2].active == false);
        }

        {
            std::vector<feature::detector::elsed::segment> segments;
            segments.push_back(make_segment(211.0f, 59.0f, 211.0f, 259.0f));
            tracker.update(7, segments);
            const std::vector<feature::tracker::line::track>& tracks = tracker.tracks();
            REQUIRE(tracks.size() == 2);
            REQUIRE(tracks[0].id == 1);
            REQUIRE(tracks[1].id == 2);
            REQUIRE(tracks[1].missed == 5);
        }

        {
            std::vector<feature::detector::elsed::segment> segments;
            segments.push_back(make_segment(213.0f, 61.0f, 213.0f, 261.0f));
            tracker.update(8, segments);
            const std::vector<feature::tracker::line::track>& tracks = tracker.tracks();
            REQUIRE(tracks.size() == 1);
            REQUIRE(tracks[0].id == 1);
            REQUIRE(tracks[0].active);
            REQUIRE(tracks[0].length == 9);
            REQUIRE(tracker.find(0) == nullptr);
            REQUIRE(tracker.find(1) != nullptr);
        }
    }

    {
        std::vector<feature::detector::elsed::segment> segments;
        segments.push_back(make_segment(154.0f, 106.0f, 264.0f, 204.0f));
        tracker.update(9, segments);
        const std::vector<feature::tracker::line::track>& tracks = tracker.tracks();
        REQUIRE(tracks.size() == 2);
        REQUIRE(tracks[0].id == 1);
        REQUIRE(tracks[0].active == false);
        REQUIRE(tracks[1].active);
        REQUIRE(tracks[1].id == 3);
    }

    return EXIT_SUCCESS;
}
