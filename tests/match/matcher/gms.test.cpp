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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);
    test_filter(0.0f);
    test_filter(0.1f);
    REQUIRE(match::matcher::gms::filter(nullptr, 640.0f, 480.0f, nullptr, 640.0f, 480.0f, nullptr, 0) == 0);
    return EXIT_SUCCESS;
}
