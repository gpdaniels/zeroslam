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

#include "core/sort.hpp"

#include "feature/point.hpp"

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

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-8) {
    if (std::isnan(lhs) && std::isnan(rhs))
        return true;
    if (std::isnan(lhs) != std::isnan(rhs))
        return false;
    if (std::isinf(lhs) != std::isinf(rhs))
        return false;
    if (std::signbit(lhs + epsilon) != std::signbit(rhs + epsilon))
        return false;
    if (std::isinf(lhs) && std::isinf(rhs))
        return true;
    return (std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon);
}

static inline bool is_value_approx(float lhs, float rhs, double epsilon = 1e-8) {
    return is_value_approx(static_cast<double>(lhs), static_cast<double>(rhs), epsilon);
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        feature::point features[9] = {
            { -1, -1, 5, 0, 0 },
            { 0, -1, 6, 0, 0 },
            { +1, -1, 7, 0, 0 },
            { -1, 0, 4, 0, 0 },
            { 0, 0, 1, 0, 0 },
            { +1, 0, 8, 0, 0 },
            { -1, +1, 3, 0, 0 },
            { 0, +1, 2, 0, 0 },
            { +1, +1, 9, 0, 0 }
        };
        core::sort::quick(&features[0], 9, [](const feature::point& lhs, const feature::point& rhs) {
            return lhs.response > rhs.response;
        });
        for (unsigned int i = 0; i < 9; ++i) {
            REQUIRE(is_value_approx(features[i].response, static_cast<float>(9 - i)));
        }
    }
    {
        constexpr static const auto by_response_descending = [](const feature::point& lhs, const feature::point& rhs) {
            return lhs.response > rhs.response;
        };

        feature::point features_empty[1] = {};
        core::sort::quick(&features_empty[0], 0, by_response_descending);

        feature::point features_single[1] = { { 0, 0, 42, 0, 0 } };
        core::sort::quick(&features_single[0], 1, by_response_descending);
        REQUIRE(is_value_approx(features_single[0].response, 42));

        feature::point features_pair_sorted[2] = { { 0, 0, 2, 0, 0 }, { 0, 0, 1, 0, 0 } };
        core::sort::quick(&features_pair_sorted[0], 2, by_response_descending);
        REQUIRE(is_value_approx(features_pair_sorted[0].response, 2));
        REQUIRE(is_value_approx(features_pair_sorted[1].response, 1));

        feature::point features_pair_unsorted[2] = { { 0, 0, 1, 0, 0 }, { 0, 0, 2, 0, 0 } };
        core::sort::quick(&features_pair_unsorted[0], 2, by_response_descending);
        REQUIRE(is_value_approx(features_pair_unsorted[0].response, 2));
        REQUIRE(is_value_approx(features_pair_unsorted[1].response, 1));
    }
    {
        constexpr static const size_t count = 8192;
        constexpr static const auto log2_ceil = [](size_t n) -> double {
            double result = 1;
            while (n > 1) {
                n >>= 1;
                ++result;
            }
            return result;
        };
        const double comparison_budget = static_cast<double>(count) * log2_ceil(count) * 20.0;

        std::vector<feature::point> features_reverse_sorted(count);
        for (size_t i = 0; i < count; ++i) {
            features_reverse_sorted[i] = { 0, 0, static_cast<float>(i), 0, 0 };
        }
        size_t comparisons_reverse_sorted = 0;
        core::sort::quick(features_reverse_sorted.data(), features_reverse_sorted.size(), [&comparisons_reverse_sorted](const feature::point& lhs, const feature::point& rhs) {
            ++comparisons_reverse_sorted;
            return lhs.response > rhs.response;
        });
        for (size_t i = 0; i < count; ++i) {
            REQUIRE(is_value_approx(features_reverse_sorted[i].response, static_cast<float>(count - 1 - i)));
        }
        REQUIRE(static_cast<double>(comparisons_reverse_sorted) < comparison_budget);

        std::vector<feature::point> features_already_sorted(count);
        for (size_t i = 0; i < count; ++i) {
            features_already_sorted[i] = { 0, 0, static_cast<float>(count - 1 - i), 0, 0 };
        }
        size_t comparisons_already_sorted = 0;
        core::sort::quick(features_already_sorted.data(), features_already_sorted.size(), [&comparisons_already_sorted](const feature::point& lhs, const feature::point& rhs) {
            ++comparisons_already_sorted;
            return lhs.response > rhs.response;
        });
        for (size_t i = 0; i < count; ++i) {
            REQUIRE(is_value_approx(features_already_sorted[i].response, static_cast<float>(count - 1 - i)));
        }
        REQUIRE(static_cast<double>(comparisons_already_sorted) < comparison_budget);
    }
    {
        constexpr static const size_t count = 8192;
        constexpr static const auto log2_ceil = [](size_t n) -> double {
            double result = 1;
            while (n > 1) {
                n >>= 1;
                ++result;
            }
            return result;
        };
        const double comparison_budget = static_cast<double>(count) * log2_ceil(count) * 20.0;

        std::vector<feature::point> features_few_distinct(count);
        for (size_t i = 0; i < count; ++i) {
            features_few_distinct[i] = { 0, 0, static_cast<float>(i % 5), 0, 0 };
        }
        size_t comparisons_few_distinct = 0;
        core::sort::quick(features_few_distinct.data(), features_few_distinct.size(), [&comparisons_few_distinct](const feature::point& lhs, const feature::point& rhs) {
            ++comparisons_few_distinct;
            return lhs.response > rhs.response;
        });
        for (size_t i = 1; i < count; ++i) {
            REQUIRE(features_few_distinct[i - 1].response >= features_few_distinct[i].response);
        }
        REQUIRE(static_cast<double>(comparisons_few_distinct) < comparison_budget);

        std::vector<feature::point> features_all_equal(count);
        for (size_t i = 0; i < count; ++i) {
            features_all_equal[i] = { static_cast<float>(i), 0, 7, 0, 0 };
        }
        size_t comparisons_all_equal = 0;
        core::sort::quick(features_all_equal.data(), features_all_equal.size(), [&comparisons_all_equal](const feature::point& lhs, const feature::point& rhs) {
            ++comparisons_all_equal;
            return lhs.response > rhs.response;
        });
        for (size_t i = 1; i < count; ++i) {
            REQUIRE(features_all_equal[i - 1].response >= features_all_equal[i].response);
        }
        REQUIRE(static_cast<double>(comparisons_all_equal) < comparison_budget);
    }

    return EXIT_SUCCESS;
}
