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

#include "feature/tracker/wavelet_flow.hpp"

#include "image/image.hpp"
#include "image/quaternion_wavelet.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
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

namespace {
    constexpr static const size_t dimension = 512;
    constexpr static const int wave_count = 56;
    constexpr static const double pi = 3.14159265358979323846;

    double next_random_unit(unsigned long long& seed) {
        seed += 0x9E3779B97F4A7C15ull;
        unsigned long long z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z = z ^ (z >> 31);
        return static_cast<double>(z >> 11) / 9007199254740992.0;
    }

    struct texture_field final {
        double frequency_x[wave_count];
        double frequency_y[wave_count];
        double amplitude[wave_count];
        double phase[wave_count];

        explicit texture_field(unsigned long long seed) {
            constexpr static const double lowest = 0.03;
            constexpr static const double highest = 2.40;
            for (int wave = 0; wave < wave_count; ++wave) {
                const double angle = 2.0 * pi * next_random_unit(seed);
                const double frequency = lowest * std::exp(next_random_unit(seed) * std::log(highest / lowest));
                this->frequency_x[wave] = frequency * std::cos(angle);
                this->frequency_y[wave] = frequency * std::sin(angle);
                this->amplitude[wave] = lowest / frequency;
                this->phase[wave] = 2.0 * pi * next_random_unit(seed);
            }
            double energy = 0.0;
            for (int wave = 0; wave < wave_count; ++wave) {
                energy += this->amplitude[wave] * this->amplitude[wave];
            }
            const double normalisation = 45.0 / std::sqrt(2.0 * energy);
            for (int wave = 0; wave < wave_count; ++wave) {
                this->amplitude[wave] *= normalisation;
            }
        }

        double evaluate(const double x, const double y) const {
            double value = 0.0;
            for (int wave = 0; wave < wave_count; ++wave) {
                value += this->amplitude[wave] * std::sin(this->frequency_x[wave] * x + this->frequency_y[wave] * y + this->phase[wave]);
            }
            return 128.0 + value;
        }
    };

    image::image render(const texture_field& field, const double shift_x, const double shift_y, const double gain = 1.0, const double bias = 0.0, const double noise = 0.0, unsigned long long seed = 3) {
        image::image result(dimension, dimension);
        for (size_t y = 0; y < dimension; ++y) {
            for (size_t x = 0; x < dimension; ++x) {
                double value = field.evaluate(static_cast<double>(x) - shift_x, static_cast<double>(y) - shift_y);
                value = (value * gain) + bias;
                if (noise > 0.0) {
                    value += noise * ((next_random_unit(seed) + next_random_unit(seed) + next_random_unit(seed)) - 1.5);
                }
                if (value < 0.0) {
                    value = 0.0;
                }
                if (value > 255.0) {
                    value = 255.0;
                }
                result.get_data()[(y * dimension) + x] = static_cast<unsigned char>(value + 0.5);
            }
        }
        return result;
    }

    void probe_points(std::vector<float>& points_x, std::vector<float>& points_y) {
        for (size_t y = 96; y + 96 < dimension; y += 24) {
            for (size_t x = 96; x + 96 < dimension; x += 24) {
                points_x.push_back(static_cast<float>(x) + 0.5f);
                points_y.push_back(static_cast<float>(y) + 0.5f);
            }
        }
    }

    double median_error(const texture_field& field, const image::quaternion_wavelet& reference, const image::image& moved, const double shift_x, const double shift_y, const feature::tracker::wavelet_flow::options& settings, size_t& kept) {
        static_cast<void>(field);
        std::vector<float> points_x;
        std::vector<float> points_y;
        probe_points(points_x, points_y);
        const image::quaternion_wavelet current(moved);
        std::vector<feature::tracker::wavelet_flow::result> results(points_x.size());
        feature::tracker::wavelet_flow::track(reference, current, points_x.data(), points_y.data(), points_x.size(), results.data(), settings);
        std::vector<double> errors;
        for (size_t index = 0; index < points_x.size(); ++index) {
            if (!results[index].tracked) {
                continue;
            }
            const double error_x = static_cast<double>(results[index].x - points_x[index]) - shift_x;
            const double error_y = static_cast<double>(results[index].y - points_y[index]) - shift_y;
            errors.push_back(std::sqrt((error_x * error_x) + (error_y * error_y)));
        }
        kept = errors.size();
        if (errors.empty()) {
            return -1.0;
        }
        std::sort(errors.begin(), errors.end());
        return errors[errors.size() / 2];
    }
}

int main() {
    const texture_field field(11);
    const image::image base = render(field, 0.0, 0.0);
    const image::quaternion_wavelet reference(base);
    REQUIRE(reference.size() >= 6);

    feature::tracker::wavelet_flow::options settings;

    {
        const double shifts_x[8] = { 0.05, 0.25, 0.0, 1.3, 3.0, -5.0, 9.0, -15.0 };
        const double shifts_y[8] = { 0.05, 0.0, 0.4, -0.7, 2.0, 4.0, -6.0, 9.0 };
        for (int test = 0; test < 8; ++test) {
            size_t kept = 0;
            const double error = median_error(field, reference, render(field, shifts_x[test], shifts_y[test]), shifts_x[test], shifts_y[test], settings, kept);
            REQUIRE(kept > 20);
            REQUIRE(error >= 0.0);
            REQUIRE(error < 0.30);
        }
    }

    {
        size_t kept = 0;
        const double error = median_error(field, reference, render(field, 80.0, 60.0), 80.0, 60.0, settings, kept);
        REQUIRE((kept == 0) || (error > 1.0));
    }

    {
        size_t kept_plain = 0;
        const double plain = median_error(field, reference, render(field, 2.3, -1.7), 2.3, -1.7, settings, kept_plain);
        size_t kept_bright = 0;
        const double bright = median_error(field, reference, render(field, 2.3, -1.7, 1.25, 0.0), 2.3, -1.7, settings, kept_bright);
        size_t kept_dim = 0;
        const double dim = median_error(field, reference, render(field, 2.3, -1.7, 0.7, 40.0), 2.3, -1.7, settings, kept_dim);
        REQUIRE(kept_plain > 20);
        REQUIRE(kept_bright > 20);
        REQUIRE(kept_dim > 20);
        REQUIRE(std::abs(bright - plain) < 0.05);
        REQUIRE(std::abs(dim - plain) < 0.05);
    }

    {
        size_t kept_plain = 0;
        const double plain = median_error(field, reference, render(field, 1.4, 0.9), 1.4, 0.9, settings, kept_plain);
        size_t kept_noisy = 0;
        const double noisy = median_error(field, reference, render(field, 1.4, 0.9, 1.0, 0.0, 4.0), 1.4, 0.9, settings, kept_noisy);
        REQUIRE(kept_plain > 100);
        REQUIRE(kept_noisy > (kept_plain * 9) / 10);
        REQUIRE(noisy < plain + 0.1);
        size_t kept_loud = 0;
        const double loud = median_error(field, reference, render(field, 1.4, 0.9, 1.0, 0.0, 16.0), 1.4, 0.9, settings, kept_loud);
        REQUIRE(kept_loud < kept_plain / 4);
        REQUIRE((kept_loud == 0) || (loud < 1.0));
    }

    {
        feature::tracker::wavelet_flow::options analytic = settings;
        analytic.measured_frequency = false;
        size_t kept_measured = 0;
        size_t kept_analytic = 0;
        const double measured = median_error(field, reference, render(field, 3.0, 2.0), 3.0, 2.0, settings, kept_measured);
        const double fixed = median_error(field, reference, render(field, 3.0, 2.0), 3.0, 2.0, analytic, kept_analytic);
        REQUIRE(kept_measured > 20);
        REQUIRE(kept_analytic > 20);
        REQUIRE(measured < fixed);
    }

    {
        const image::image flat(dimension, dimension);
        for (size_t index = 0; index < dimension * dimension; ++index) {
            flat.get_data()[index] = static_cast<unsigned char>(128);
        }
        const image::quaternion_wavelet blank(flat);
        std::vector<float> points_x;
        std::vector<float> points_y;
        probe_points(points_x, points_y);
        std::vector<feature::tracker::wavelet_flow::result> results(points_x.size());
        feature::tracker::wavelet_flow::track(blank, blank, points_x.data(), points_y.data(), points_x.size(), results.data(), settings);
        for (size_t index = 0; index < points_x.size(); ++index) {
            REQUIRE(!results[index].tracked);
        }
    }

    {
        const image::quaternion_wavelet empty;
        std::vector<float> points_x;
        std::vector<float> points_y;
        probe_points(points_x, points_y);
        std::vector<feature::tracker::wavelet_flow::result> results(points_x.size());
        feature::tracker::wavelet_flow::track(empty, reference, points_x.data(), points_y.data(), points_x.size(), results.data(), settings);
        for (size_t index = 0; index < points_x.size(); ++index) {
            REQUIRE(!results[index].tracked);
        }
        feature::tracker::wavelet_flow::track(reference, empty, points_x.data(), points_y.data(), points_x.size(), results.data(), settings);
        for (size_t index = 0; index < points_x.size(); ++index) {
            REQUIRE(!results[index].tracked);
        }
    }

    {
        std::vector<float> points_x;
        std::vector<float> points_y;
        probe_points(points_x, points_y);
        const image::image moved = render(field, 60.0, 0.0);
        const image::quaternion_wavelet current(moved);
        std::vector<float> guess_x(points_x.size(), 58.0f);
        std::vector<float> guess_y(points_x.size(), 0.0f);
        std::vector<feature::tracker::wavelet_flow::result> results(points_x.size());
        feature::tracker::wavelet_flow::track(reference, current, points_x.data(), points_y.data(), points_x.size(), results.data(), settings, guess_x.data(), guess_y.data());
        size_t good = 0;
        for (size_t index = 0; index < points_x.size(); ++index) {
            if (results[index].tracked && (std::abs(static_cast<double>(results[index].x - points_x[index]) - 60.0) < 0.5)) {
                ++good;
            }
        }
        REQUIRE(good > points_x.size() / 3);
    }

    {
        feature::tracker::wavelet_flow::options robust = settings;
        robust.robust = true;
        size_t kept_plain = 0;
        size_t kept_robust = 0;
        const double plain = median_error(field, reference, render(field, 1.7, -2.2), 1.7, -2.2, settings, kept_plain);
        const double reweighted = median_error(field, reference, render(field, 1.7, -2.2), 1.7, -2.2, robust, kept_robust);
        REQUIRE(kept_robust >= (kept_plain * 9) / 10);
        REQUIRE(std::abs(reweighted - plain) < 0.05);

        image::image split = render(field, 1.0, 0.0);
        const image::image other = render(field, -6.0, 3.0);
        for (size_t y = 0; y < dimension; ++y) {
            for (size_t x = dimension / 2; x < dimension; ++x) {
                split.get_data()[(y * dimension) + x] = other.get_data()[(y * dimension) + x];
            }
        }
        const image::quaternion_wavelet contaminated(split);
        std::vector<float> points_x;
        std::vector<float> points_y;
        for (size_t y = 120; y + 120 < dimension; y += 16) {
            points_x.push_back(static_cast<float>(dimension / 2) - 8.0f + 0.5f);
            points_y.push_back(static_cast<float>(y) + 0.5f);
        }
        std::vector<feature::tracker::wavelet_flow::result> plain_results(points_x.size());
        std::vector<feature::tracker::wavelet_flow::result> robust_results(points_x.size());
        feature::tracker::wavelet_flow::options loose = settings;
        loose.forward_backward = false;
        loose.max_error = 100.0f;
        feature::tracker::wavelet_flow::options loose_robust = loose;
        loose_robust.robust = true;
        feature::tracker::wavelet_flow::track(reference, contaminated, points_x.data(), points_y.data(), points_x.size(), plain_results.data(), loose);
        feature::tracker::wavelet_flow::track(reference, contaminated, points_x.data(), points_y.data(), points_x.size(), robust_results.data(), loose_robust);
        double plain_sum = 0.0;
        double robust_sum = 0.0;
        for (size_t index = 0; index < points_x.size(); ++index) {
            plain_sum += std::hypot(static_cast<double>(plain_results[index].x - points_x[index]) - 1.0, static_cast<double>(plain_results[index].y - points_y[index]));
            robust_sum += std::hypot(static_cast<double>(robust_results[index].x - points_x[index]) - 1.0, static_cast<double>(robust_results[index].y - points_y[index]));
        }
        REQUIRE(robust_sum < plain_sum);
    }

    {
        std::vector<float> points_x;
        std::vector<float> points_y;
        probe_points(points_x, points_y);
        std::vector<feature::tracker::wavelet_flow::anchor> anchors(points_x.size());
        size_t built = 0;
        for (size_t index = 0; index < points_x.size(); ++index) {
            if (feature::tracker::wavelet_flow::build_anchor(reference, points_x[index], points_y[index], settings, anchors[index])) {
                ++built;
                REQUIRE(!anchors[index].levels.empty());
                REQUIRE(anchors[index].levels.front().index >= anchors[index].levels.back().index);
                REQUIRE(anchors[index].levels.back().index == 2);
                REQUIRE(!anchors[index].levels.back().frequencies.empty());
            }
        }
        REQUIRE(built == points_x.size());
        const double run_x[4] = { 0.8, 2.1, 3.9, 6.5 };
        const double run_y[4] = { -0.4, -1.3, -2.0, -1.1 };
        std::vector<float> displacement_x(points_x.size(), 0.0f);
        std::vector<float> displacement_y(points_x.size(), 0.0f);
        for (int frame = 0; frame < 4; ++frame) {
            const image::quaternion_wavelet current(render(field, run_x[frame], run_y[frame], (frame == 3) ? 1.3 : 1.0));
            std::vector<double> errors;
            for (size_t index = 0; index < points_x.size(); ++index) {
                feature::tracker::wavelet_flow::result outcome;
                if (!feature::tracker::wavelet_flow::align(current, anchors[index], settings, displacement_x[index], displacement_y[index], outcome)) {
                    continue;
                }
                REQUIRE(std::abs(static_cast<double>(outcome.x - points_x[index]) - static_cast<double>(displacement_x[index])) < 1e-4);
                errors.push_back(std::hypot(static_cast<double>(displacement_x[index]) - run_x[frame], static_cast<double>(displacement_y[index]) - run_y[frame]));
            }
            REQUIRE(errors.size() > (points_x.size() * 9) / 10);
            std::sort(errors.begin(), errors.end());
            REQUIRE(errors[errors.size() / 2] < 0.3);
        }
        const image::image flat(dimension, dimension);
        for (size_t index = 0; index < dimension * dimension; ++index) {
            flat.get_data()[index] = static_cast<unsigned char>(128);
        }
        feature::tracker::wavelet_flow::anchor empty;
        feature::tracker::wavelet_flow::result outcome;
        float zero_x = 0.0f;
        float zero_y = 0.0f;
        REQUIRE(feature::tracker::wavelet_flow::build_anchor(image::quaternion_wavelet(flat), points_x[0], points_y[0], settings, empty));
        REQUIRE(!feature::tracker::wavelet_flow::align(reference, empty, settings, zero_x, zero_y, outcome));
        REQUIRE(!outcome.tracked);
    }

    {
        const image::quaternion_wavelet dense_reference(base, 0, 2, true);
        REQUIRE(dense_reference[2].spacing == 2.0);
        REQUIRE(dense_reference[2].cols == dimension / 2);
        REQUIRE(std::abs(dense_reference[2].wavelet_centre - (0.5 * reference[2].wavelet_centre)) < 1e-9);
        std::vector<float> points_x;
        std::vector<float> points_y;
        probe_points(points_x, points_y);
        const double shifts_x[3] = { 0.25, 3.0, 9.0 };
        const double shifts_y[3] = { 0.0, 2.0, -6.0 };
        for (int test = 0; test < 3; ++test) {
            const image::image moved = render(field, shifts_x[test], shifts_y[test]);
            const image::quaternion_wavelet dense_current(moved, 0, 2, true);
            const image::quaternion_wavelet current(moved);
            std::vector<feature::tracker::wavelet_flow::result> dense_results(points_x.size());
            std::vector<feature::tracker::wavelet_flow::result> results(points_x.size());
            feature::tracker::wavelet_flow::track(dense_reference, dense_current, points_x.data(), points_y.data(), points_x.size(), dense_results.data(), settings);
            feature::tracker::wavelet_flow::track(reference, current, points_x.data(), points_y.data(), points_x.size(), results.data(), settings);
            std::vector<double> dense_errors;
            std::vector<double> errors;
            for (size_t index = 0; index < points_x.size(); ++index) {
                if (dense_results[index].tracked) {
                    dense_errors.push_back(std::hypot(static_cast<double>(dense_results[index].x - points_x[index]) - shifts_x[test], static_cast<double>(dense_results[index].y - points_y[index]) - shifts_y[test]));
                }
                if (results[index].tracked) {
                    errors.push_back(std::hypot(static_cast<double>(results[index].x - points_x[index]) - shifts_x[test], static_cast<double>(results[index].y - points_y[index]) - shifts_y[test]));
                }
            }
            REQUIRE(dense_errors.size() >= (errors.size() * 9) / 10);
            std::sort(dense_errors.begin(), dense_errors.end());
            std::sort(errors.begin(), errors.end());
            REQUIRE(dense_errors[dense_errors.size() / 2] < errors[errors.size() / 2] + 0.02);
        }
    }

    {
        std::vector<float> points_x;
        std::vector<float> points_y;
        probe_points(points_x, points_y);
        feature::tracker::wavelet_flow::options seeded = settings;
        seeded.guess_first = true;
        {
            const image::quaternion_wavelet current(render(field, 60.0, 0.0));
            std::vector<float> guess_x(points_x.size(), 59.0f);
            std::vector<float> guess_y(points_x.size(), 0.0f);
            std::vector<feature::tracker::wavelet_flow::result> results(points_x.size());
            feature::tracker::wavelet_flow::track(reference, current, points_x.data(), points_y.data(), points_x.size(), results.data(), seeded, guess_x.data(), guess_y.data());
            size_t good = 0;
            for (size_t index = 0; index < points_x.size(); ++index) {
                if (results[index].tracked && (std::abs(static_cast<double>(results[index].x - points_x[index]) - 60.0) < 0.5)) {
                    ++good;
                }
            }
            REQUIRE(good > points_x.size() / 2);
        }
        {
            const image::quaternion_wavelet current(render(field, 2.0, 1.0));
            std::vector<float> guess_x(points_x.size(), 41.0f);
            std::vector<float> guess_y(points_x.size(), -37.0f);
            std::vector<feature::tracker::wavelet_flow::result> results(points_x.size());
            feature::tracker::wavelet_flow::track(reference, current, points_x.data(), points_y.data(), points_x.size(), results.data(), seeded, guess_x.data(), guess_y.data());
            size_t good = 0;
            for (size_t index = 0; index < points_x.size(); ++index) {
                if (results[index].tracked && (std::hypot(static_cast<double>(results[index].x - points_x[index]) - 2.0, static_cast<double>(results[index].y - points_y[index]) - 1.0) < 0.5)) {
                    ++good;
                }
            }
            REQUIRE(good > (points_x.size() * 3) / 4);
        }
    }

    {
        std::vector<float> points_x;
        std::vector<float> points_y;
        probe_points(points_x, points_y);
        const image::quaternion_wavelet current(render(field, 2.0, -1.0));
        std::vector<feature::tracker::wavelet_flow::result> first(points_x.size());
        std::vector<feature::tracker::wavelet_flow::result> second(points_x.size());
        feature::tracker::wavelet_flow::track(reference, current, points_x.data(), points_y.data(), points_x.size(), first.data(), settings);
        feature::tracker::wavelet_flow::track(reference, current, points_x.data(), points_y.data(), points_x.size(), second.data(), settings);
        for (size_t index = 0; index < points_x.size(); ++index) {
            REQUIRE(first[index].tracked == second[index].tracked);
            REQUIRE(first[index].x == second[index].x);
            REQUIRE(first[index].y == second[index].y);
            REQUIRE(first[index].error == second[index].error);
        }
    }

    return EXIT_SUCCESS;
}
