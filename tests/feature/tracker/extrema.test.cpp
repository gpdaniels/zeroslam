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

#include "feature/tracker/extrema.hpp"

#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

namespace {
    constexpr const int frame_width = 480;
    constexpr const int frame_height = 360;

    struct wave final {
        float frequency_x;
        float frequency_y;
        float phase;
        float amplitude;
    };

    std::vector<wave> make_waves(unsigned long long seed) {
        std::vector<wave> waves;
        unsigned long long state = seed;
        const auto next_unit = [&state]() {
            state += 0x9E3779B97F4A7C15ull;
            unsigned long long z = state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            z = z ^ (z >> 31);
            return static_cast<float>(static_cast<double>(z >> 11) / 9007199254740992.0);
        };
        for (int i = 0; i < 30; ++i) {
            const float wavelength = 8.0f + 22.0f * next_unit();
            const float direction = 2.0f * 3.14159265f * next_unit();
            wave created;
            created.frequency_x = (2.0f * 3.14159265f / wavelength) * math::cos(direction);
            created.frequency_y = (2.0f * 3.14159265f / wavelength) * math::sin(direction);
            created.phase = 2.0f * 3.14159265f * next_unit();
            created.amplitude = 12.0f + 24.0f * next_unit();
            waves.push_back(created);
        }
        return waves;
    }

    unsigned char field(const std::vector<wave>& waves, const float x, const float y) {
        float value = 128.0f;
        for (const wave& component : waves) {
            value += component.amplitude * math::cos(component.frequency_x * x + component.frequency_y * y + component.phase);
        }
        return static_cast<unsigned char>(math::max(0.0f, math::min(255.0f, value)) + 0.5f);
    }

    std::vector<unsigned char> sample_frame(
        const std::vector<wave>& waves,
        const float a00,
        const float a01,
        const float a10,
        const float a11,
        const float offset_x,
        const float offset_y
    ) {
        std::vector<unsigned char> frame(static_cast<size_t>(frame_width) * static_cast<size_t>(frame_height));
        for (int y = 0; y < frame_height; ++y) {
            for (int x = 0; x < frame_width; ++x) {
                const float source_x = a00 * static_cast<float>(x) + a01 * static_cast<float>(y) + offset_x;
                const float source_y = a10 * static_cast<float>(x) + a11 * static_cast<float>(y) + offset_y;
                frame[static_cast<size_t>(y) * static_cast<size_t>(frame_width) + static_cast<size_t>(x)] = field(waves, source_x, source_y);
            }
        }
        return frame;
    }

    feature::tracker::extrema::options test_options() {
        feature::tracker::extrema::options settings;
        settings.detection.quantile = 0.8f;
        settings.flow_scale = 3;
        return settings;
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const std::vector<wave> waves = make_waves(42);

    {
        feature::tracker::extrema tracker(test_options());
        const std::vector<unsigned char> frame_data = sample_frame(waves, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        image::image frame(frame_height, frame_width, const_cast<unsigned char*>(frame_data.data()));
        tracker.update(0, frame);
        const feature::tracker::extrema::diagnostics& diagnostics = tracker.get_diagnostics();
        REQUIRE(diagnostics.spawned > 100);
        REQUIRE(diagnostics.spawned == tracker.size());
        std::vector<feature::tracker::tracker::track*> active = tracker.active_tracks();
        REQUIRE(active.size() == diagnostics.spawned);
        const float spacing_squared = 4.0f * 4.0f;
        for (size_t i = 0; i < active.size(); ++i) {
            REQUIRE(active[i]->length == 1);
            REQUIRE(tracker.sign_of(active[i]->id) != 0);
            for (size_t j = i + 1; j < active.size(); ++j) {
                const float du = active[i]->x - active[j]->x;
                const float dv = active[i]->y - active[j]->y;
                REQUIRE(du * du + dv * dv >= spacing_squared * 0.99f);
            }
        }
    }

    {
        feature::tracker::extrema::options settings = test_options();
        settings.maximum_tracks = 25;
        feature::tracker::extrema tracker(settings);
        const std::vector<unsigned char> frame_data = sample_frame(waves, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        image::image frame(frame_height, frame_width, const_cast<unsigned char*>(frame_data.data()));
        tracker.update(0, frame);
        REQUIRE(tracker.size() == 25);
    }

    {
        const int shift_x = 12;
        const int shift_y = 9;
        feature::tracker::extrema tracker(test_options());
        const std::vector<unsigned char> frame_a = sample_frame(waves, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        const std::vector<unsigned char> frame_b = sample_frame(waves, 1.0f, 0.0f, 0.0f, 1.0f, -static_cast<float>(shift_x), -static_cast<float>(shift_y));
        image::image image_a(frame_height, frame_width, const_cast<unsigned char*>(frame_a.data()));
        image::image image_b(frame_height, frame_width, const_cast<unsigned char*>(frame_b.data()));
        tracker.update(0, image_a);
        const size_t spawned = tracker.size();
        REQUIRE(spawned > 100);
        std::vector<int> ids;
        std::vector<float> xs;
        std::vector<float> ys;
        for (feature::tracker::tracker::track* track : tracker.active_tracks()) {
            ids.push_back(track->id);
            xs.push_back(track->x);
            ys.push_back(track->y);
        }
        tracker.update(1, image_b);
        const feature::tracker::extrema::diagnostics& diagnostics = tracker.get_diagnostics();
        REQUIRE(!diagnostics.flow_fallback);
        REQUIRE(math::abs(diagnostics.flow.bx - static_cast<float>(shift_x)) < 1.0f);
        REQUIRE(math::abs(diagnostics.flow.by - static_cast<float>(shift_y)) < 1.0f);
        REQUIRE(math::abs(diagnostics.flow.a00 - 1.0f) < 0.02f);
        REQUIRE(math::abs(diagnostics.flow.a11 - 1.0f) < 0.02f);
        size_t interior = 0;
        size_t surviving = 0;
        size_t exact = 0;
        for (size_t i = 0; i < ids.size(); ++i) {
            const float expected_x = xs[i] + static_cast<float>(shift_x);
            const float expected_y = ys[i] + static_cast<float>(shift_y);
            if ((xs[i] < 20.0f) || (expected_x >= static_cast<float>(frame_width) - 20.0f) ||
                (ys[i] < 20.0f) || (expected_y >= static_cast<float>(frame_height) - 20.0f)) {
                continue;
            }
            ++interior;
            feature::tracker::tracker::track* track = tracker.find(ids[i]);
            if ((track == nullptr) || (!track->active)) {
                continue;
            }
            ++surviving;
            if ((math::abs(track->x - expected_x) < 1.0e-3f) && (math::abs(track->y - expected_y) < 1.0e-3f)) {
                ++exact;
            }
            REQUIRE(track->length == 2);
            REQUIRE(track->history.size() == 2);
            REQUIRE(track->history[1].frame_id == 1);
        }
        REQUIRE(interior > 50);
        REQUIRE(surviving * 10 >= interior * 8);
        REQUIRE(exact * 10 >= surviving * 9);
    }

    {
        const float angle = 0.5f * 3.14159265f / 180.0f;
        const float scale = 1.01f;
        const float a00 = scale * math::cos(angle);
        const float a01 = -scale * math::sin(angle);
        const float a10 = scale * math::sin(angle);
        const float a11 = scale * math::cos(angle);
        const float offset_x = -3.4f;
        const float offset_y = -2.6f;
        feature::tracker::extrema tracker(test_options());
        const std::vector<unsigned char> frame_a = sample_frame(waves, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        const std::vector<unsigned char> frame_b = sample_frame(waves, a00, a01, a10, a11, offset_x, offset_y);
        image::image image_a(frame_height, frame_width, const_cast<unsigned char*>(frame_a.data()));
        image::image image_b(frame_height, frame_width, const_cast<unsigned char*>(frame_b.data()));
        tracker.update(0, image_a);
        std::vector<int> ids;
        std::vector<float> xs;
        std::vector<float> ys;
        for (feature::tracker::tracker::track* track : tracker.active_tracks()) {
            ids.push_back(track->id);
            xs.push_back(track->x);
            ys.push_back(track->y);
        }
        REQUIRE(ids.size() > 100);
        tracker.update(1, image_b);
        REQUIRE(!tracker.get_diagnostics().flow_fallback);
        const float determinant = a00 * a11 - a01 * a10;
        size_t checked = 0;
        size_t within = 0;
        for (size_t i = 0; i < ids.size(); ++i) {
            const float relative_x = xs[i] - offset_x;
            const float relative_y = ys[i] - offset_y;
            const float expected_x = (a11 * relative_x - a01 * relative_y) / determinant;
            const float expected_y = (-a10 * relative_x + a00 * relative_y) / determinant;
            if ((expected_x < 20.0f) || (expected_x >= static_cast<float>(frame_width) - 20.0f) ||
                (expected_y < 20.0f) || (expected_y >= static_cast<float>(frame_height) - 20.0f) ||
                (xs[i] < 20.0f) || (xs[i] >= static_cast<float>(frame_width) - 20.0f) ||
                (ys[i] < 20.0f) || (ys[i] >= static_cast<float>(frame_height) - 20.0f)) {
                continue;
            }
            feature::tracker::tracker::track* track = tracker.find(ids[i]);
            if ((track == nullptr) || (!track->active)) {
                continue;
            }
            ++checked;
            const float error_x = track->x - expected_x;
            const float error_y = track->y - expected_y;
            if (error_x * error_x + error_y * error_y < 0.75f * 0.75f) {
                ++within;
            }
        }
        REQUIRE(checked > 50);
        REQUIRE(within * 10 >= checked * 8);
    }

    {
        feature::tracker::extrema tracker(test_options());
        const std::vector<unsigned char> frame_a = sample_frame(waves, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        const std::vector<unsigned char> frame_b = sample_frame(waves, 1.0f, 0.0f, 0.0f, 1.0f, 40.0f, 0.0f);
        image::image image_a(frame_height, frame_width, const_cast<unsigned char*>(frame_a.data()));
        image::image image_b(frame_height, frame_width, const_cast<unsigned char*>(frame_b.data()));
        tracker.update(0, image_a);
        std::vector<int> leaving_ids;
        for (feature::tracker::tracker::track* track : tracker.active_tracks()) {
            if (track->x < 30.0f) {
                leaving_ids.push_back(track->id);
            }
        }
        REQUIRE(!leaving_ids.empty());
        tracker.update(1, image_b);
        for (const int id : leaving_ids) {
            REQUIRE(tracker.find(id) == nullptr);
        }
        REQUIRE(tracker.get_diagnostics().lost >= leaving_ids.size());
    }

    {
        const int shift_x = 12;
        const int shift_y = 9;
        feature::tracker::extrema::options settings = test_options();
        settings.climb_bidirectional = true;
        feature::tracker::extrema tracker(settings);
        const std::vector<unsigned char> frame_a = sample_frame(waves, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        const std::vector<unsigned char> frame_b = sample_frame(waves, 1.0f, 0.0f, 0.0f, 1.0f, -static_cast<float>(shift_x), -static_cast<float>(shift_y));
        image::image image_a(frame_height, frame_width, const_cast<unsigned char*>(frame_a.data()));
        image::image image_b(frame_height, frame_width, const_cast<unsigned char*>(frame_b.data()));
        tracker.update(0, image_a);
        std::vector<int> ids;
        std::vector<float> xs;
        std::vector<float> ys;
        for (feature::tracker::tracker::track* track : tracker.active_tracks()) {
            ids.push_back(track->id);
            xs.push_back(track->x);
            ys.push_back(track->y);
        }
        tracker.update(1, image_b);
        size_t interior = 0;
        size_t surviving = 0;
        size_t exact = 0;
        for (size_t i = 0; i < ids.size(); ++i) {
            const float expected_x = xs[i] + static_cast<float>(shift_x);
            const float expected_y = ys[i] + static_cast<float>(shift_y);
            if ((xs[i] < 20.0f) || (expected_x >= static_cast<float>(frame_width) - 20.0f) ||
                (ys[i] < 20.0f) || (expected_y >= static_cast<float>(frame_height) - 20.0f)) {
                continue;
            }
            ++interior;
            feature::tracker::tracker::track* track = tracker.find(ids[i]);
            if ((track == nullptr) || (!track->active)) {
                continue;
            }
            ++surviving;
            if ((math::abs(track->x - expected_x) < 1.0e-3f) && (math::abs(track->y - expected_y) < 1.0e-3f)) {
                ++exact;
            }
        }
        REQUIRE(interior > 50);
        REQUIRE(surviving * 10 >= interior * 7);
        REQUIRE(exact * 10 >= surviving * 9);
    }

    {
        feature::tracker::extrema first(test_options());
        feature::tracker::extrema second(test_options());
        const std::vector<unsigned char> frame_a = sample_frame(waves, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        const std::vector<unsigned char> frame_b = sample_frame(waves, 1.0f, 0.0f, 0.0f, 1.0f, -4.0f, -6.0f);
        image::image image_a(frame_height, frame_width, const_cast<unsigned char*>(frame_a.data()));
        image::image image_b(frame_height, frame_width, const_cast<unsigned char*>(frame_b.data()));
        first.update(0, image_a);
        first.update(1, image_b);
        second.update(0, image_a);
        second.update(1, image_b);
        REQUIRE(first.size() == second.size());
        std::vector<feature::tracker::tracker::track*> tracks_first = first.active_tracks();
        std::vector<feature::tracker::tracker::track*> tracks_second = second.active_tracks();
        REQUIRE(tracks_first.size() == tracks_second.size());
        for (size_t i = 0; i < tracks_first.size(); ++i) {
            REQUIRE(tracks_first[i]->id == tracks_second[i]->id);
            REQUIRE(tracks_first[i]->x == tracks_second[i]->x);
            REQUIRE(tracks_first[i]->y == tracks_second[i]->y);
            REQUIRE(tracks_first[i]->length == tracks_second[i]->length);
            REQUIRE(first.sign_of(tracks_first[i]->id) == second.sign_of(tracks_second[i]->id));
        }
    }

    std::printf("All extrema tracker tests passed.\n");
    return 0;
}
