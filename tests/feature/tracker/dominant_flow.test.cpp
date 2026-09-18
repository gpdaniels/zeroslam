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

#include "feature/tracker/dominant_flow.hpp"

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
    float next_random_unit(unsigned long long& state) {
        state += 0x9E3779B97F4A7C15ull;
        unsigned long long z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z = z ^ (z >> 31);
        return static_cast<float>(static_cast<double>(z >> 11) / 9007199254740992.0);
    }

    struct pair_set {
        std::vector<float> from_x;
        std::vector<float> from_y;
        std::vector<float> to_x;
        std::vector<float> to_y;
    };

    pair_set make_pairs(const feature::tracker::dominant_flow::affine& truth, const size_t count, unsigned long long& state) {
        pair_set pairs;
        for (size_t i = 0; i < count; ++i) {
            const float x = 100.0f * next_random_unit(state);
            const float y = 100.0f * next_random_unit(state);
            float mapped_x;
            float mapped_y;
            truth.apply(x, y, mapped_x, mapped_y);
            pairs.from_x.push_back(x);
            pairs.from_y.push_back(y);
            pairs.to_x.push_back(mapped_x);
            pairs.to_y.push_back(mapped_y);
        }
        return pairs;
    }

    bool close(const float lhs, const float rhs, const float tolerance) {
        return math::abs(lhs - rhs) <= tolerance;
    }

    bool affine_close(const feature::tracker::dominant_flow::affine& lhs, const feature::tracker::dominant_flow::affine& rhs, const float linear_tolerance, const float translation_tolerance) {
        return close(lhs.a00, rhs.a00, linear_tolerance) && close(lhs.a01, rhs.a01, linear_tolerance) &&
               close(lhs.a10, rhs.a10, linear_tolerance) && close(lhs.a11, rhs.a11, linear_tolerance) &&
               close(lhs.bx, rhs.bx, translation_tolerance) && close(lhs.by, rhs.by, translation_tolerance);
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        feature::tracker::dominant_flow::affine transform;
        float x;
        float y;
        transform.apply(3.0f, -7.0f, x, y);
        REQUIRE(x == 3.0f);
        REQUIRE(y == -7.0f);
        transform.a00 = 2.0f;
        transform.a01 = 0.5f;
        transform.bx = 10.0f;
        transform.apply(4.0f, 2.0f, x, y);
        REQUIRE(x == 2.0f * 4.0f + 0.5f * 2.0f + 10.0f);
        REQUIRE(y == 2.0f);
    }

    {
        const float x[2] = { 0.0f, 1.0f };
        const float y[2] = { 0.0f, 1.0f };
        feature::tracker::dominant_flow::affine flow;
        flow.bx = 123.0f;
        REQUIRE(!feature::tracker::dominant_flow::fit(x, y, x, y, 2, feature::tracker::dominant_flow::options(), flow));
        REQUIRE(flow.bx == 123.0f);
    }

    {
        const float from_x[4] = { 0.0f, 1.0f, 2.0f, 3.0f };
        const float from_y[4] = { 0.0f, 1.0f, 2.0f, 3.0f };
        const float to_x[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
        const float to_y[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
        feature::tracker::dominant_flow::affine flow;
        REQUIRE(!feature::tracker::dominant_flow::fit(from_x, from_y, to_x, to_y, 4, feature::tracker::dominant_flow::options(), flow));
    }

    {
        unsigned long long state = 11;
        feature::tracker::dominant_flow::affine truth;
        truth.bx = 6.25f;
        truth.by = -3.5f;
        for (const size_t count : { size_t(3), size_t(10), size_t(100) }) {
            const pair_set pairs = make_pairs(truth, count, state);
            feature::tracker::dominant_flow::affine flow;
            REQUIRE(feature::tracker::dominant_flow::fit(pairs.from_x.data(), pairs.from_y.data(), pairs.to_x.data(), pairs.to_y.data(), count, feature::tracker::dominant_flow::options(), flow));
            REQUIRE(affine_close(flow, truth, 1.0e-4f, 1.0e-3f));
        }

        feature::tracker::dominant_flow::affine general;
        general.a00 = 0.98f;
        general.a01 = -0.12f;
        general.a10 = 0.11f;
        general.a11 = 1.02f;
        general.bx = -8.0f;
        general.by = 4.75f;
        const pair_set pairs = make_pairs(general, 50, state);
        feature::tracker::dominant_flow::affine flow;
        REQUIRE(feature::tracker::dominant_flow::fit(pairs.from_x.data(), pairs.from_y.data(), pairs.to_x.data(), pairs.to_y.data(), pairs.from_x.size(), feature::tracker::dominant_flow::options(), flow));
        REQUIRE(affine_close(flow, general, 1.0e-4f, 1.0e-3f));
    }

    {
        unsigned long long state = 99;
        feature::tracker::dominant_flow::affine truth;
        truth.a00 = 1.01f;
        truth.a01 = 0.05f;
        truth.a10 = -0.04f;
        truth.a11 = 0.99f;
        truth.bx = 12.0f;
        truth.by = -7.0f;
        pair_set pairs = make_pairs(truth, 100, state);
        for (size_t i = 0; i < pairs.to_x.size(); ++i) {
            pairs.to_x[i] += 0.4f * (next_random_unit(state) - 0.5f);
            pairs.to_y[i] += 0.4f * (next_random_unit(state) - 0.5f);
        }
        for (size_t i = 0; i < 30; ++i) {
            const size_t index = static_cast<size_t>(next_random_unit(state) * 100.0f);
            pairs.to_x[index] += 30.0f + 40.0f * next_random_unit(state);
            pairs.to_y[index] -= 25.0f + 40.0f * next_random_unit(state);
        }
        feature::tracker::dominant_flow::affine flow;
        feature::tracker::dominant_flow::options settings;
        settings.iterations = 20;
        REQUIRE(feature::tracker::dominant_flow::fit(pairs.from_x.data(), pairs.from_y.data(), pairs.to_x.data(), pairs.to_y.data(), pairs.from_x.size(), settings, flow));
        REQUIRE(affine_close(flow, truth, 0.01f, 0.5f));
        float probe_x;
        float probe_y;
        flow.apply(50.0f, 50.0f, probe_x, probe_y);
        float truth_x;
        float truth_y;
        truth.apply(50.0f, 50.0f, truth_x, truth_y);
        REQUIRE(close(probe_x, truth_x, 0.75f));
        REQUIRE(close(probe_y, truth_y, 0.75f));
    }

    {
        feature::tracker::dominant_flow::affine transform;
        transform.a00 = 0.98f;
        transform.a01 = -0.12f;
        transform.a10 = 0.11f;
        transform.a11 = 1.02f;
        transform.bx = -8.0f;
        transform.by = 4.75f;
        feature::tracker::dominant_flow::affine inverse;
        REQUIRE(transform.invert(inverse));
        for (const float x : { -3.0f, 0.0f, 17.5f }) {
            for (const float y : { 2.0f, -40.25f }) {
                float forward_x;
                float forward_y;
                transform.apply(x, y, forward_x, forward_y);
                float round_trip_x;
                float round_trip_y;
                inverse.apply(forward_x, forward_y, round_trip_x, round_trip_y);
                REQUIRE(math::abs(round_trip_x - x) < 1.0e-4f);
                REQUIRE(math::abs(round_trip_y - y) < 1.0e-4f);
            }
        }
        feature::tracker::dominant_flow::affine singular;
        singular.a00 = 1.0f;
        singular.a01 = 2.0f;
        singular.a10 = 2.0f;
        singular.a11 = 4.0f;
        feature::tracker::dominant_flow::affine untouched;
        untouched.bx = 123.0f;
        REQUIRE(!singular.invert(untouched));
        REQUIRE(untouched.bx == 123.0f);
    }

    {
        unsigned long long state = 5;
        feature::tracker::dominant_flow::affine truth;
        truth.bx = 2.0f;
        const pair_set pairs = make_pairs(truth, 25, state);
        feature::tracker::dominant_flow::affine first;
        feature::tracker::dominant_flow::affine second;
        REQUIRE(feature::tracker::dominant_flow::fit(pairs.from_x.data(), pairs.from_y.data(), pairs.to_x.data(), pairs.to_y.data(), pairs.from_x.size(), feature::tracker::dominant_flow::options(), first));
        REQUIRE(feature::tracker::dominant_flow::fit(pairs.from_x.data(), pairs.from_y.data(), pairs.to_x.data(), pairs.to_y.data(), pairs.from_x.size(), feature::tracker::dominant_flow::options(), second));
        REQUIRE(first.a00 == second.a00);
        REQUIRE(first.a01 == second.a01);
        REQUIRE(first.a10 == second.a10);
        REQUIRE(first.a11 == second.a11);
        REQUIRE(first.bx == second.bx);
        REQUIRE(first.by == second.by);
    }

    std::printf("All dominant flow tests passed.\n");
    return 0;
}
