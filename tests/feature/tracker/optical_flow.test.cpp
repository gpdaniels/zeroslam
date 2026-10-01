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

static inline double next_random_unit(unsigned long long& seed) {
    seed += 0x9E3779B97F4A7C15ull;
    unsigned long long z = seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z = z ^ (z >> 31);
    return static_cast<double>(z >> 11) / 9007199254740992.0;
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
        const image::image base = make_image(dimension, dimension, texture);
        constexpr static const int half_window = 7;
        constexpr static const int width = (2 * half_window) + 1;
        float values[width * width];
        float gradients_x[width * width];
        float gradients_y[width * width];
        const float centre_x = 71.3f;
        const float centre_y = 64.8f;
        REQUIRE(feature::tracker::optical_flow::window_in_bounds(base, centre_x, centre_y, half_window));
        feature::tracker::optical_flow::sample_gradients(base, centre_x, centre_y, half_window, values, gradients_x, gradients_y);
        for (int j = 0; j < width; ++j) {
            for (int i = 0; i < width; ++i) {
                const float x = centre_x + static_cast<float>(i - half_window);
                const float y = centre_y + static_cast<float>(j - half_window);
                const float expected_x = 0.5f * (feature::tracker::optical_flow::sample_bilinear(base, x + 1.0f, y) - feature::tracker::optical_flow::sample_bilinear(base, x - 1.0f, y));
                const float expected_y = 0.5f * (feature::tracker::optical_flow::sample_bilinear(base, x, y + 1.0f) - feature::tracker::optical_flow::sample_bilinear(base, x, y - 1.0f));
                REQUIRE(values[(j * width) + i] == feature::tracker::optical_flow::sample_bilinear(base, x, y));
                REQUIRE(std::abs(static_cast<double>(gradients_x[(j * width) + i] - expected_x)) < 1e-3);
                REQUIRE(std::abs(static_cast<double>(gradients_y[(j * width) + i] - expected_y)) < 1e-3);
            }
        }
    }

    {
        // Coarse levels that cannot contribute are skipped rather than failing the track. Levels 2 and 3 of a period 8
        // lattice are degenerate (the pyramid's binomial blur removes its alias exactly) while levels 0 and 1 are not.
        constexpr static const size_t cols = 640;
        constexpr static const size_t rows = 480;
        const double shift_x = 1.7;
        const double shift_y = -1.2;
        const auto lattice = [](const double x, const double y) {
            const double k = 2.0 * 3.14159265358979323846 / 8.0;
            return 128.0 + 30.0 * (std::cos((k * x) + 0.4) + std::cos((k * y) + 1.1) + (0.7 * std::cos((k * (x + y)) + 2.3)) + (0.5 * std::cos((k * (x - y)) + 0.2)));
        };
        const image::pyramid pyramid_previous(make_image(rows, cols, lattice));
        const image::pyramid pyramid_next(make_image(rows, cols, [&](double x, double y) {
            return lattice(x - shift_x, y - shift_y);
        }));
        REQUIRE(pyramid_previous.size() == 4);
        std::vector<float> points_x;
        std::vector<float> points_y;
        for (int j = 0; j < 16; ++j) {
            for (int i = 0; i < 24; ++i) {
                points_x.push_back(80.5f + (20.0f * static_cast<float>(i)));
                points_y.push_back(80.5f + (20.0f * static_cast<float>(j)));
            }
        }
        std::vector<feature::tracker::optical_flow::result> results(points_x.size());
        feature::tracker::optical_flow::track(pyramid_previous, pyramid_next, points_x.data(), points_y.data(), points_x.size(), results.data());
        double sum = 0.0;
        for (size_t i = 0; i < points_x.size(); ++i) {
            REQUIRE(results[i].tracked);
            const double error_x = static_cast<double>(results[i].x - points_x[i]) - shift_x;
            const double error_y = static_cast<double>(results[i].y - points_y[i]) - shift_y;
            sum += (error_x * error_x) + (error_y * error_y);
        }
        REQUIRE(std::sqrt(sum / static_cast<double>(points_x.size())) < 0.02);
    }

    {
        // A texture of 0.9-1.8 rad/px is flat at level 3 and leaves only aliased residue at levels 1 and 2, which drives
        // the flow to wrong minima once the flat level is skipped. Those tracks are rejected, none is accepted wrong.
        constexpr static const size_t cols = 640;
        constexpr static const size_t rows = 480;
        constexpr static const int waves = 12;
        double frequency_x[waves];
        double frequency_y[waves];
        double phase[waves];
        unsigned long long seed = 5;
        for (int wave = 0; wave < waves; ++wave) {
            const double angle = 2.0 * 3.14159265358979323846 * next_random_unit(seed);
            const double frequency = 0.9 + (0.9 * next_random_unit(seed));
            frequency_x[wave] = frequency * std::cos(angle);
            frequency_y[wave] = frequency * std::sin(angle);
            phase[wave] = 2.0 * 3.14159265358979323846 * next_random_unit(seed);
        }
        const auto fine = [&](const double x, const double y) {
            double value = 128.0;
            for (int wave = 0; wave < waves; ++wave) {
                value += 12.0 * std::sin((frequency_x[wave] * x) + (frequency_y[wave] * y) + phase[wave]);
            }
            return value;
        };
        const double shift_x = 1.7;
        const double shift_y = -1.2;
        const image::pyramid pyramid_previous(make_image(rows, cols, fine));
        const image::pyramid pyramid_next(make_image(rows, cols, [&](double x, double y) {
            return fine(x - shift_x, y - shift_y);
        }));
        std::vector<float> points_x;
        std::vector<float> points_y;
        for (int j = 0; j < 16; ++j) {
            for (int i = 0; i < 24; ++i) {
                points_x.push_back(80.5f + (20.0f * static_cast<float>(i)));
                points_y.push_back(80.5f + (20.0f * static_cast<float>(j)));
            }
        }
        std::vector<feature::tracker::optical_flow::result> results(points_x.size());
        feature::tracker::optical_flow::track(pyramid_previous, pyramid_next, points_x.data(), points_y.data(), points_x.size(), results.data());
        for (size_t i = 0; i < points_x.size(); ++i) {
            if (results[i].tracked) {
                REQUIRE(std::hypot(static_cast<double>(results[i].x - points_x[i]) - shift_x, static_cast<double>(results[i].y - points_y[i]) - shift_y) < 0.5);
            }
        }
    }

    {
        // Points near the border moving toward it: with continue_outside a coarse iteration that leaves the image returns
        // to the level's entry flow and the finer levels still track them, by default such a track is rejected.
        constexpr static const size_t cols = 640;
        constexpr static const size_t rows = 480;
        const double shift = 8.0;
        const image::pyramid pyramid_previous(make_image(rows, cols, texture));
        const image::pyramid pyramid_left(make_image(rows, cols, [&](double x, double y) {
            return texture(x + shift, y + 0.4);
        }));
        const image::pyramid pyramid_up(make_image(rows, cols, [&](double x, double y) {
            return texture(x - 0.3, y + shift);
        }));
        std::vector<float> inset(24);
        std::vector<float> spread(24);
        for (size_t k = 0; k < inset.size(); ++k) {
            inset[k] = static_cast<float>(18 + k) + 0.5f;
            spread[k] = static_cast<float>(100 + (12 * k)) + 0.5f;
        }
        for (const bool continue_outside : { true, false }) {
            std::vector<feature::tracker::optical_flow::result> left(inset.size());
            feature::tracker::optical_flow::track(pyramid_previous, pyramid_left, inset.data(), spread.data(), inset.size(), left.data(), 7, 30, 1e-3f, 40.0f, true, 1.0f, nullptr, nullptr, false, continue_outside);
            std::vector<feature::tracker::optical_flow::result> up(inset.size());
            feature::tracker::optical_flow::track(pyramid_previous, pyramid_up, spread.data(), inset.data(), inset.size(), up.data(), 7, 30, 1e-3f, 40.0f, true, 1.0f, nullptr, nullptr, false, continue_outside);
            size_t tracked = 0;
            for (size_t k = 0; k < inset.size(); ++k) {
                if (left[k].tracked) {
                    ++tracked;
                    REQUIRE(std::hypot(static_cast<double>(left[k].x - inset[k]) + shift, static_cast<double>(left[k].y - spread[k]) + 0.4) < 0.1);
                }
                if (up[k].tracked) {
                    ++tracked;
                    REQUIRE(std::hypot(static_cast<double>(up[k].x - spread[k]) - 0.3, static_cast<double>(up[k].y - inset[k]) + shift) < 0.1);
                }
            }
            if (continue_outside) {
                REQUIRE(tracked == 2 * inset.size());
            }
            else {
                REQUIRE(tracked < inset.size());
            }
        }
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

        feature::tracker::optical_flow::result plain[count];
        feature::tracker::optical_flow::track(pyramid_previous, pyramid_next, points_x, points_y, count, plain, 7, 30, 1e-3f, 10000.0f, true, 1.0f, nullptr, nullptr, false);
        feature::tracker::optical_flow::result damped[count];
        feature::tracker::optical_flow::track(pyramid_previous, pyramid_next, points_x, points_y, count, damped, 7, 30, 1e-3f, 10000.0f, true, 1.0f, nullptr, nullptr, true);

        for (size_t i = 0; i < count; ++i) {
            REQUIRE(damped[i].tracked);
            REQUIRE(plain[i].tracked);
            REQUIRE(std::abs(static_cast<double>(damped[i].x - points_x[i]) - static_cast<double>(shift_x)) < 0.3);
            REQUIRE(std::abs(static_cast<double>(damped[i].y - points_y[i]) - static_cast<double>(shift_y)) < 0.3);
            REQUIRE(std::abs(static_cast<double>(damped[i].x - plain[i].x)) < 0.3);
            REQUIRE(std::abs(static_cast<double>(damped[i].y - plain[i].y)) < 0.3);
        }

        feature::tracker::optical_flow::result repeated[count];
        feature::tracker::optical_flow::track(pyramid_previous, pyramid_next, points_x, points_y, count, repeated, 7, 30, 1e-3f, 10000.0f, true, 1.0f, nullptr, nullptr, true);
        for (size_t i = 0; i < count; ++i) {
            REQUIRE(repeated[i].x == damped[i].x);
            REQUIRE(repeated[i].y == damped[i].y);
        }
    }

    return EXIT_SUCCESS;
}
