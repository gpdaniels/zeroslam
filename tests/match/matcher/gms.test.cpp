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

#include "match/matcher/gms.hpp"

#include "core/random_pcg.hpp"

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

static void test_filter(const float margin) {
    constexpr static const size_t true_count = 1500;
    constexpr static const size_t false_count = 500;
    core::random_pcg random(0x6d5u);
    std::vector<feature::point> lhs;
    std::vector<feature::point> rhs;
    std::vector<match::pair> matches;
    for (size_t i = 0; i < true_count; ++i) {
        const float x = static_cast<float>(random.get_random(20.0, 620.0));
        const float y = static_cast<float>(random.get_random(20.0, 460.0));
        const float dx = x - 320.0f;
        const float dy = y - 240.0f;
        const float moved_x = 320.0f + (0.9986f * dx) - (0.0523f * dy) + 25.0f + static_cast<float>(random.get_random(-0.5, 0.5));
        const float moved_y = 240.0f + (0.0523f * dx) + (0.9986f * dy) - 12.0f + static_cast<float>(random.get_random(-0.5, 0.5));
        lhs.push_back(feature::point{ x, y, 0.0f, 0.0f, 0 });
        rhs.push_back(feature::point{ moved_x, moved_y, 0.0f, 0.0f, 0 });
        matches.push_back(match::pair{ lhs.size() - 1, rhs.size() - 1, 0.0f });
    }
    for (size_t i = 0; i < false_count; ++i) {
        lhs.push_back(feature::point{ static_cast<float>(random.get_random(0.0, 640.0)), static_cast<float>(random.get_random(0.0, 480.0)), 0.0f, 0.0f, 0 });
        rhs.push_back(feature::point{ static_cast<float>(random.get_random(0.0, 640.0)), static_cast<float>(random.get_random(0.0, 480.0)), 0.0f, 0.0f, 0 });
        matches.push_back(match::pair{ lhs.size() - 1, rhs.size() - 1, 0.0f });
    }
    match::matcher::gms::options settings;
    settings.margin = margin;
    const size_t kept = match::matcher::gms::filter(lhs.data(), 640.0f, 480.0f, rhs.data(), 640.0f, 480.0f, matches.data(), matches.size(), settings);
    size_t kept_true = 0;
    size_t kept_false = 0;
    for (size_t m = 0; m < kept; ++m) {
        kept_true += (matches[m].lhs_index < true_count);
        kept_false += (matches[m].lhs_index >= true_count);
    }
    REQUIRE(kept_true * 10 >= true_count * 8);
    REQUIRE(kept_false * 20 <= false_count);
}

class kept_counts final {
public:
    size_t true_matches;
    size_t false_matches;
};

// True matches turned by angle_degrees and scaled about the image centre, kept only when they land in the right image, then false matches.
static kept_counts filter_transformed(const double angle_degrees, const double scale, const size_t true_count, const size_t false_count, const unsigned long long int seed, const match::matcher::gms::options& settings) {
    core::random_pcg random(seed);
    const double angle = angle_degrees * 3.14159265358979323846 / 180.0;
    const double turn_cos = std::cos(angle) * scale;
    const double turn_sin = std::sin(angle) * scale;
    std::vector<feature::point> lhs;
    std::vector<feature::point> rhs;
    std::vector<match::pair> matches;
    while (lhs.size() < true_count) {
        const double dx = random.get_random(0.0, 640.0) - 320.0;
        const double dy = random.get_random(0.0, 480.0) - 240.0;
        const double moved_x = 320.0 + (turn_cos * dx) - (turn_sin * dy) + 15.0 + random.get_random(-0.5, 0.5);
        const double moved_y = 240.0 + (turn_sin * dx) + (turn_cos * dy) - 10.0 + random.get_random(-0.5, 0.5);
        if ((moved_x < 0.0) || (moved_x >= 640.0) || (moved_y < 0.0) || (moved_y >= 480.0)) {
            continue;
        }
        lhs.push_back(feature::point{ static_cast<float>(320.0 + dx), static_cast<float>(240.0 + dy), 0.0f, 0.0f, 0 });
        rhs.push_back(feature::point{ static_cast<float>(moved_x), static_cast<float>(moved_y), 0.0f, 0.0f, 0 });
        matches.push_back(match::pair{ lhs.size() - 1, rhs.size() - 1, 0.0f });
    }
    for (size_t i = 0; i < false_count; ++i) {
        lhs.push_back(feature::point{ static_cast<float>(random.get_random(0.0, 640.0)), static_cast<float>(random.get_random(0.0, 480.0)), 0.0f, 0.0f, 0 });
        rhs.push_back(feature::point{ static_cast<float>(random.get_random(0.0, 640.0)), static_cast<float>(random.get_random(0.0, 480.0)), 0.0f, 0.0f, 0 });
        matches.push_back(match::pair{ lhs.size() - 1, rhs.size() - 1, 0.0f });
    }
    const size_t kept = match::matcher::gms::filter(lhs.data(), 640.0f, 480.0f, rhs.data(), 640.0f, 480.0f, matches.data(), matches.size(), settings);
    kept_counts counts{ 0, 0 };
    for (size_t m = 0; m < kept; ++m) {
        counts.true_matches += (matches[m].lhs_index < true_count) ? 1u : 0u;
        counts.false_matches += (matches[m].lhs_index >= true_count) ? 1u : 0u;
    }
    return counts;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);
    test_filter(0.0f);
    test_filter(0.1f);
    REQUIRE(match::matcher::gms::filter(nullptr, 640.0f, 480.0f, nullptr, 640.0f, 480.0f, nullptr, 0) == 0);

    // In-plane rotation, at a match count near the pipeline's minimum and at a few hundred: most true matches kept by default, few false.
    {
        const match::matcher::gms::options defaults;
        match::matcher::gms::options unrotated;
        unrotated.rotation = false;
        const double angles[4] = { 0.0, 45.0, 90.0, 180.0 };
        const size_t true_counts[2] = { 60, 300 };
        for (const size_t true_count : true_counts) {
            const size_t false_count = true_count / 3;
            const size_t false_allowed = (true_count < 100) ? (false_count * 2 / 5) : (false_count * 3 / 20);
            for (const double angle : angles) {
                for (unsigned long long int seed = 1; seed <= 3; ++seed) {
                    const kept_counts turned = filter_transformed(angle, 1.0, true_count, false_count, seed, defaults);
                    REQUIRE(turned.true_matches * 4 >= true_count * 3);
                    REQUIRE(turned.false_matches <= false_allowed);
                    if (angle >= 90.0) {
                        // Without the rotation patterns a quarter or half turn loses nearly every true match.
                        const kept_counts fixed = filter_transformed(angle, 1.0, true_count, false_count, seed, unrotated);
                        REQUIRE(fixed.true_matches * 5 <= true_count);
                    }
                }
            }
        }
    }

    // A change of scale needs the scale hypotheses, and unrelated points barely pass.
    {
        const match::matcher::gms::options defaults;
        match::matcher::gms::options scaled;
        scaled.scale = true;
        const size_t true_counts[2] = { 60, 300 };
        for (const size_t true_count : true_counts) {
            const size_t false_count = true_count / 3;
            for (unsigned long long int seed = 1; seed <= 3; ++seed) {
                const kept_counts zoom_default = filter_transformed(0.0, 2.0, true_count, false_count, seed, defaults);
                const kept_counts zoom_scaled = filter_transformed(0.0, 2.0, true_count, false_count, seed, scaled);
                const kept_counts shrink_scaled = filter_transformed(0.0, 0.5, true_count, false_count, seed, scaled);
                REQUIRE(zoom_scaled.true_matches * 20 >= true_count * 17);
                REQUIRE(zoom_scaled.true_matches > zoom_default.true_matches);
                REQUIRE(zoom_scaled.false_matches * 5 <= false_count);
                REQUIRE(shrink_scaled.true_matches * 20 >= true_count * 17);
                REQUIRE(shrink_scaled.false_matches * 5 <= false_count);
                const kept_counts unrelated = filter_transformed(0.0, 1.0, 0, true_count, seed, defaults);
                REQUIRE(unrelated.false_matches * 10 <= true_count);
            }
        }
    }

    return EXIT_SUCCESS;
}
