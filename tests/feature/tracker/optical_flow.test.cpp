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

#include "feature/tracker/optical_flow.hpp"

#include "image/pyramid.hpp"

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

static inline double texture(double x, double y) {
    return 128.0 +
           55.0 * std::sin(0.21 * x) * std::sin(0.19 * y) +
           35.0 * std::cos(0.15 * x + 0.10 * y) +
           25.0 * std::sin(0.30 * x - 0.12 * y);
}

static inline double texture_other(double x, double y) {
    return 128.0 +
           90.0 * std::sin(0.52 * x + 1.3) * std::sin(0.47 * y - 0.7) +
           55.0 * std::cos(0.33 * x - 0.40 * y + 2.1);
}

template <typename function_type>
static inline image::image make_image(size_t rows, size_t cols, function_type function) {
    image::image result(rows, cols);
    for (size_t y = 0; y < rows; ++y) {
        for (size_t x = 0; x < cols; ++x) {
            double value = function(static_cast<double>(x), static_cast<double>(y));
            if (value < 0.0) {
                value = 0.0;
            }
            if (value > 255.0) {
                value = 255.0;
            }
            result.get_data()[y * cols + x] = static_cast<unsigned char>(value + 0.5);
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    constexpr static const size_t dimension = 160;
    {
        const image::image base = make_image(dimension, dimension, texture);
        const image::pyramid pyramid(base);
        REQUIRE(pyramid.size() == 3);
        for (size_t level = 0; level < pyramid.size(); ++level) {
            const size_t expected = dimension >> level;
            REQUIRE(pyramid[level].get_rows() == expected);
            REQUIRE(pyramid[level].get_cols() == expected);
        }
        const image::pyramid degenerate(make_image(8, 8, texture));
        REQUIRE(degenerate.size() == 1);
    }

    {
        const float shift_x = 3.0f;
        const float shift_y = 2.0f;
        const image::image previous = make_image(dimension, dimension, texture);
        const image::image next = make_image(dimension, dimension, [=](double x, double y) {
            return texture(x - static_cast<double>(shift_x), y - static_cast<double>(shift_y));
        });
        const image::pyramid pyramid_previous(previous);
        const image::pyramid pyramid_next(next);

        constexpr static const size_t count = 5;
        const float points_x[count] = { 50.0f, 70.0f, 90.0f, 60.0f, 100.0f };
        const float points_y[count] = { 50.0f, 60.0f, 80.0f, 100.0f, 55.0f };
        feature::tracker::optical_flow::result results[count];
        feature::tracker::optical_flow::track(pyramid_previous, pyramid_next, points_x, points_y, count, results);

        for (size_t i = 0; i < count; ++i) {
            REQUIRE(results[i].tracked);
            REQUIRE(std::abs(static_cast<double>(results[i].x - points_x[i]) - static_cast<double>(shift_x)) < 0.3);
            REQUIRE(std::abs(static_cast<double>(results[i].y - points_y[i]) - static_cast<double>(shift_y)) < 0.3);
        }
    }

    {
        const float shift_x = 1.5f;
        const float shift_y = -0.75f;
        const image::image previous = make_image(dimension, dimension, texture);
        const image::image next = make_image(dimension, dimension, [=](double x, double y) {
            return texture(x - static_cast<double>(shift_x), y - static_cast<double>(shift_y));
        });
        const image::pyramid pyramid_previous(previous);
        const image::pyramid pyramid_next(next);

        constexpr static const size_t count = 5;
        const float points_x[count] = { 50.0f, 70.0f, 90.0f, 60.0f, 100.0f };
        const float points_y[count] = { 50.0f, 60.0f, 80.0f, 100.0f, 55.0f };
        feature::tracker::optical_flow::result results[count];
        feature::tracker::optical_flow::track(pyramid_previous, pyramid_next, points_x, points_y, count, results);

        for (size_t i = 0; i < count; ++i) {
            REQUIRE(results[i].tracked);
            REQUIRE(std::abs(static_cast<double>(results[i].x - points_x[i]) - static_cast<double>(shift_x)) < 0.3);
            REQUIRE(std::abs(static_cast<double>(results[i].y - points_y[i]) - static_cast<double>(shift_y)) < 0.3);
        }
    }

    {
        image::image previous = make_image(dimension, dimension, texture);
        for (size_t y = 55; y < 105; ++y) {
            for (size_t x = 55; x < 105; ++x) {
                previous.get_data()[y * dimension + x] = 128;
            }
        }
        const float shift_x = 6.0f;
        image::image next = make_image(dimension, dimension, [=](double x, double y) {
            return texture(x - static_cast<double>(shift_x), y);
        });
        for (size_t y = 55; y < 105; ++y) {
            for (size_t x = static_cast<size_t>(55 + 6); x < static_cast<size_t>(105 + 6); ++x) {
                next.get_data()[y * dimension + x] = 128;
            }
        }
        const image::pyramid pyramid_previous(previous);
        const image::pyramid pyramid_next(next);

        constexpr static const size_t count = 3;
        const float points_x[count] = { 80.0f, 150.0f, 40.0f };
        const float points_y[count] = { 80.0f, 80.0f, 40.0f };
        feature::tracker::optical_flow::result results[count];
        feature::tracker::optical_flow::track(pyramid_previous, pyramid_next, points_x, points_y, count, results);

        REQUIRE(results[0].tracked == false);
        REQUIRE(results[1].tracked == false);
        REQUIRE(results[2].tracked == true);
        REQUIRE(std::abs(static_cast<double>(results[2].x - points_x[2]) - static_cast<double>(shift_x)) < 0.3);
        REQUIRE(std::abs(static_cast<double>(results[2].y - points_y[2])) < 0.3);
    }

    {
        const float shift_x = 2.0f;
        const float shift_y = 1.0f;
        const image::image previous = make_image(dimension, dimension, texture);
        image::image next = make_image(dimension, dimension, [=](double x, double y) {
            return texture(x - static_cast<double>(shift_x), y - static_cast<double>(shift_y));
        });
        for (size_t y = 40; y < 95; ++y) {
            for (size_t x = 30; x < 85; ++x) {
                double value = texture_other(static_cast<double>(x), static_cast<double>(y));
                if (value < 0.0) {
                    value = 0.0;
                }
                if (value > 255.0) {
                    value = 255.0;
                }
                next.get_data()[y * dimension + x] = static_cast<unsigned char>(value + 0.5);
            }
        }
        const image::pyramid pyramid_previous(previous);
        const image::pyramid pyramid_next(next);

        constexpr static const size_t count = 2;
        const float points_x[count] = { 57.0f, 125.0f };
        const float points_y[count] = { 67.0f, 125.0f };

        feature::tracker::optical_flow::result naive[count];
        feature::tracker::optical_flow::track(pyramid_previous, pyramid_next, points_x, points_y, count, naive, 7, 30, 1e-3f, 10000.0f, false);
        REQUIRE(naive[0].tracked == true);
        REQUIRE(naive[1].tracked == true);

        feature::tracker::optical_flow::result gated[count];
        feature::tracker::optical_flow::track(pyramid_previous, pyramid_next, points_x, points_y, count, gated, 7, 30, 1e-3f, 10000.0f, true, 1.0f);
        REQUIRE(gated[0].tracked == false);
        REQUIRE(gated[1].tracked == true);
        REQUIRE(std::abs(static_cast<double>(gated[1].x - points_x[1]) - static_cast<double>(shift_x)) < 0.3);
        REQUIRE(std::abs(static_cast<double>(gated[1].y - points_y[1]) - static_cast<double>(shift_y)) < 0.3);
    }

    {
        const float shift_x = 2.0f;
        const float shift_y = 1.0f;
        const image::image previous = make_image(dimension, dimension, texture);
        image::image next = make_image(dimension, dimension, [=](double x, double y) {
            return texture(x - static_cast<double>(shift_x), y - static_cast<double>(shift_y));
        });
        for (size_t y = 40; y < 95; ++y) {
            for (size_t x = 30; x < 85; ++x) {
                double value = texture_other(static_cast<double>(x), static_cast<double>(y));
                if (value < 0.0) {
                    value = 0.0;
                }
                if (value > 255.0) {
                    value = 255.0;
                }
                next.get_data()[y * dimension + x] = static_cast<unsigned char>(value + 0.5);
            }
        }
        const image::pyramid pyramid_previous(previous);
        const image::pyramid pyramid_next(next);

        constexpr static const size_t count = 2;
        const float points_x[count] = { 57.0f, 125.0f };
        const float points_y[count] = { 67.0f, 125.0f };

        feature::tracker::optical_flow::result ungated[count];
        feature::tracker::optical_flow::track(pyramid_previous, pyramid_next, points_x, points_y, count, ungated, 7, 30, 1e-3f, 10000.0f, false);
        REQUIRE(ungated[0].tracked == true);
        REQUIRE(ungated[1].tracked == true);
        REQUIRE(ungated[1].error < 10.0f);
        REQUIRE(ungated[0].error > 25.0f);

        const float error_threshold = 15.0f;
        feature::tracker::optical_flow::result gated[count];
        feature::tracker::optical_flow::track(pyramid_previous, pyramid_next, points_x, points_y, count, gated, 7, 30, 1e-3f, error_threshold, false);
        REQUIRE(gated[0].tracked == false);
        REQUIRE(gated[1].tracked == true);
    }

    return EXIT_SUCCESS;
}
