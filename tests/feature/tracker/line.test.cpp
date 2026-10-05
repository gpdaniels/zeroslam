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
        // The defaults, the options handed to the constructor, and a later change are all read back as set.
        const feature::tracker::line defaulted;
        REQUIRE(defaulted.get_options().min_length == 20.0f);
        REQUIRE(defaulted.get_options().max_missed == 5);
        REQUIRE(tracker.get_options().max_missed == 5);
        REQUIRE(tracker.get_options().match_overlap == 0.3f);
        feature::tracker::line::options changed;
        changed.min_length = 40.0f;
        changed.match_angle_tolerance = 2.5f;
        changed.match_midpoint_distance = 12.0f;
        changed.match_overlap = 0.75f;
        changed.predicted_angle_tolerance = 1.5f;
        changed.predicted_distance = 3.0f;
        changed.max_missed = 2;
        tracker.set_options(changed);
        const feature::tracker::line::options& read_back = tracker.get_options();
        REQUIRE(read_back.min_length == 40.0f);
        REQUIRE(read_back.match_angle_tolerance == 2.5f);
        REQUIRE(read_back.match_midpoint_distance == 12.0f);
        REQUIRE(read_back.match_overlap == 0.75f);
        REQUIRE(read_back.predicted_angle_tolerance == 1.5f);
        REQUIRE(read_back.predicted_distance == 3.0f);
        REQUIRE(read_back.max_missed == 2);
        tracker.set_options(opts);
        REQUIRE(tracker.get_options().max_missed == 5);
    }

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

    {
        // Polarity: a step from dark (left of x = 40) to bright, read along a downward segment on the step, sees the bright
        // side on its left; the reversed segment sees it on its right, and a flat image has none.
        constexpr static const int width = 80;
        constexpr static const int height = 60;
        std::vector<unsigned char> step(static_cast<size_t>(width * height));
        std::vector<unsigned char> flat(static_cast<size_t>(width * height), static_cast<unsigned char>(90));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                step[static_cast<size_t>((y * width) + x)] = static_cast<unsigned char>((x < 40) ? 40 : 200);
            }
        }
        const int down = feature::tracker::line::polarity(step.data(), width, height, width, 40.0f, 10.0f, 40.0f, 50.0f);
        const int up = feature::tracker::line::polarity(step.data(), width, height, width, 40.0f, 50.0f, 40.0f, 10.0f);
        REQUIRE((down != 0) && (up == -down));
        REQUIRE(feature::tracker::line::polarity(flat.data(), width, height, width, 40.0f, 10.0f, 40.0f, 50.0f) == 0);
        REQUIRE(feature::tracker::line::aligned_polarity(40.0f, 10.0f, 40.0f, 50.0f, 40.0f, 50.0f, 40.0f, 10.0f, up) == down);
        REQUIRE(feature::tracker::line::aligned_polarity(40.0f, 10.0f, 40.0f, 50.0f, 41.0f, 12.0f, 41.0f, 48.0f, down) == down);
        REQUIRE(is_value_approx(feature::tracker::line::line_distance(0.0f, 0.0f, 100.0f, 0.0f, 10.0f, 3.0f, 60.0f, -5.0f), 4.0f));
        REQUIRE(is_value_approx(feature::tracker::line::overlap(0.0f, 0.0f, 100.0f, 0.0f, 50.0f, 1.0f, 150.0f, 1.0f), 0.5f));
    }

    {
        // Two parallel lines 12 px apart move 25 px across them: proximity alone takes the nearer, wrong one, while the
        // prediction keeps each track on its own line, and a prediction coasts a track that finds nothing.
        feature::tracker::line::options parallel_options;
        parallel_options.min_length = 10.0f;
        feature::tracker::line by_proximity(parallel_options);
        feature::tracker::line by_prediction(parallel_options);
        std::vector<feature::detector::elsed::segment> first;
        first.push_back(make_segment(100.0f, 50.0f, 100.0f, 150.0f));
        by_proximity.update(0, first);
        by_prediction.update(0, first, std::vector<feature::tracker::line::prediction>(), std::vector<int>(1, 1));
        REQUIRE(by_prediction.tracks().size() == 1);
        REQUIRE(by_prediction.tracks()[0].polarity == 1);
        std::vector<feature::detector::elsed::segment> second;
        second.push_back(make_segment(113.0f, 50.0f, 113.0f, 150.0f));
        second.push_back(make_segment(125.0f, 50.0f, 125.0f, 150.0f));
        by_proximity.update(1, second);
        REQUIRE(by_proximity.tracks()[0].active);
        REQUIRE(is_value_approx(by_proximity.tracks()[0].x1, 113.0f));
        const std::vector<feature::tracker::line::prediction> shifted(1, feature::tracker::line::prediction{ true, 125.0f, 52.0f, 125.0f, 148.0f });
        const std::vector<int> both_bright(2, 1);
        by_prediction.update(1, second, shifted, both_bright);
        REQUIRE(by_prediction.tracks()[0].active);
        REQUIRE(by_prediction.tracks()[0].length == 2);
        REQUIRE(is_value_approx(by_prediction.tracks()[0].x1, 125.0f));

        // The predicted line, but of the opposite polarity: the track does not take it and coasts to its prediction.
        std::vector<feature::detector::elsed::segment> third;
        third.push_back(make_segment(150.0f, 50.0f, 150.0f, 150.0f));
        const std::vector<feature::tracker::line::prediction> moved(by_prediction.tracks().size(), feature::tracker::line::prediction{ true, 150.0f, 50.0f, 150.0f, 150.0f });
        by_prediction.update(2, third, moved, std::vector<int>(1, -1));
        const feature::tracker::line::track& coasted = by_prediction.tracks()[0];
        REQUIRE(!coasted.active);
        REQUIRE(coasted.missed == 1);
        REQUIRE(coasted.length == 2);
        REQUIRE(is_value_approx(coasted.x1, 150.0f) && is_value_approx(coasted.y2, 150.0f));
        // The line at 113 px spawned a track in the frame before; the detection both refused spawns a track of its own.
        REQUIRE(by_prediction.tracks().size() == 3);
        REQUIRE(!by_prediction.tracks()[1].active);
        REQUIRE(by_prediction.tracks()[2].active);
        REQUIRE(by_prediction.tracks()[2].polarity == -1);
    }

    return EXIT_SUCCESS;
}
