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

#include "image/quaternion_wavelet.hpp"

#include "image/image.hpp"

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
    constexpr static const size_t dimension = 384;
    constexpr static const int wave_count = 48;
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

    image::image render(const texture_field& field, const size_t size, const double shift_x, const double shift_y, const double gain = 1.0, const double bias = 0.0) {
        image::image result(size, size);
        for (size_t y = 0; y < size; ++y) {
            for (size_t x = 0; x < size; ++x) {
                double value = field.evaluate(static_cast<double>(x) - shift_x, static_cast<double>(y) - shift_y);
                value = (value * gain) + bias;
                if (value < 0.0) {
                    value = 0.0;
                }
                if (value > 255.0) {
                    value = 255.0;
                }
                result.get_data()[(y * size) + x] = static_cast<unsigned char>(value + 0.5);
            }
        }
        return result;
    }

    double wrap(double angle) {
        while (angle >= pi) {
            angle -= 2.0 * pi;
        }
        while (angle < -pi) {
            angle += 2.0 * pi;
        }
        return angle;
    }
}

int main() {
    {
        const double* const tables[4] = {
            image::quaternion_wavelet::first_lowpass_a(),
            image::quaternion_wavelet::first_lowpass_b(),
            image::quaternion_wavelet::later_lowpass_a(),
            image::quaternion_wavelet::later_lowpass_b()
        };
        for (int table = 0; table < 4; ++table) {
            double sum = 0.0;
            double energy = 0.0;
            for (int tap = 0; tap < image::quaternion_wavelet::filter_length; ++tap) {
                sum += tables[table][tap];
                energy += tables[table][tap] * tables[table][tap];
            }
            REQUIRE(std::abs(sum - std::sqrt(2.0)) < 1e-7);
            REQUIRE(std::abs(energy - 1.0) < 1e-7);
            for (int shift = 1; shift * 2 < image::quaternion_wavelet::filter_length; ++shift) {
                double correlation = 0.0;
                for (int tap = 0; tap + (2 * shift) < image::quaternion_wavelet::filter_length; ++tap) {
                    correlation += tables[table][tap] * tables[table][tap + (2 * shift)];
                }
                REQUIRE(std::abs(correlation) < 1e-7);
            }
        }
        for (int tap = 0; tap < image::quaternion_wavelet::filter_length; ++tap) {
            REQUIRE(image::quaternion_wavelet::later_lowpass_a()[tap] == image::quaternion_wavelet::later_lowpass_b()[image::quaternion_wavelet::filter_length - 1 - tap]);
        }
        double highpass[image::quaternion_wavelet::filter_length];
        image::quaternion_wavelet::quadrature_mirror(image::quaternion_wavelet::first_lowpass_a(), &highpass[0]);
        double highpass_sum = 0.0;
        double cross = 0.0;
        for (int tap = 0; tap < image::quaternion_wavelet::filter_length; ++tap) {
            highpass_sum += highpass[tap];
            cross += highpass[tap] * image::quaternion_wavelet::first_lowpass_a()[tap];
        }
        REQUIRE(std::abs(highpass_sum) < 1e-9);
        REQUIRE(std::abs(cross) < 1e-9);
    }

    {
        double wavelet[6];
        double scaling[6];
        image::quaternion_wavelet::spectral_centres(6, &wavelet[0], &scaling[0]);
        for (int level = 2; level <= 6; ++level) {
            REQUIRE(wavelet[level - 1] > 0.70);
            REQUIRE(wavelet[level - 1] < 0.85);
            REQUIRE(scaling[level - 1] > 0.10);
            REQUIRE(scaling[level - 1] < 0.35);
        }
        REQUIRE(wavelet[0] < 0.55);
    }

    const texture_field field(7);
    const image::image base = render(field, dimension, 0.0, 0.0);

    {
        const image::quaternion_wavelet transform(base);
        REQUIRE(transform.size() >= 4);
        REQUIRE(transform[1].rows == 0);
        REQUIRE(transform[1].cols == 0);
        for (size_t level = 2; level <= transform.size(); ++level) {
            REQUIRE(transform[level].rows == dimension >> level);
            REQUIRE(transform[level].cols == dimension >> level);
            REQUIRE(transform[level].bands[0].size() == transform[level].rows * transform[level].cols * image::quaternion_wavelet::component_count);
            REQUIRE(transform.at(level, image::quaternion_wavelet::band::diagonal, 0, 0) != nullptr);
            REQUIRE(transform.at(level, image::quaternion_wavelet::band::diagonal, transform[level].cols, 0) == nullptr);
        }
        REQUIRE(transform.at(transform.size() + 1, image::quaternion_wavelet::band::diagonal, 0, 0) == nullptr);
        const image::quaternion_wavelet complete(base, 0, 1);
        REQUIRE(complete[1].rows == dimension / 2);
    }

    {
        const float components[4] = { 0.7f, -0.3f, 0.45f, 0.11f };
        float plus_real = 0.0f;
        float plus_imaginary = 0.0f;
        float minus_real = 0.0f;
        float minus_imaginary = 0.0f;
        image::quaternion_wavelet::complex_pair(&components[0], plus_real, plus_imaginary, minus_real, minus_imaginary);
        REQUIRE(std::abs(static_cast<double>(plus_real) - (0.7 - 0.11)) < 1e-6);
        REQUIRE(std::abs(static_cast<double>(plus_imaginary) - (-0.3 + 0.45)) < 1e-6);
        REQUIRE(std::abs(static_cast<double>(minus_real) - (0.7 + 0.11)) < 1e-6);
        REQUIRE(std::abs(static_cast<double>(minus_imaginary) - (-0.3 - 0.45)) < 1e-6);
        const double plus_angle = std::atan2(static_cast<double>(plus_imaginary), static_cast<double>(plus_real));
        const double minus_angle = std::atan2(static_cast<double>(minus_imaginary), static_cast<double>(minus_real));
        double first = 0.0;
        double second = 0.0;
        double third = 0.0;
        REQUIRE(image::quaternion_wavelet::phases(&components[0], first, second, third));
        REQUIRE(std::abs(first - (0.5 * (plus_angle + minus_angle))) < 1e-9);
        REQUIRE(std::abs(second - (0.5 * (plus_angle - minus_angle))) < 1e-9);
        REQUIRE(third >= -0.25 * pi);
        REQUIRE(third <= 0.25 * pi);
        const double modulus = static_cast<double>(image::quaternion_wavelet::modulus(&components[0]));
        const double plus_power = (static_cast<double>(plus_real) * static_cast<double>(plus_real)) + (static_cast<double>(plus_imaginary) * static_cast<double>(plus_imaginary));
        const double minus_power = (static_cast<double>(minus_real) * static_cast<double>(minus_real)) + (static_cast<double>(minus_imaginary) * static_cast<double>(minus_imaginary));
        REQUIRE(std::abs((plus_power + minus_power) - (2.0 * modulus * modulus)) < 1e-6);
        const float zero[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        REQUIRE(!image::quaternion_wavelet::phases(&zero[0], first, second, third));
    }

    {
        const image::quaternion_wavelet reference(base);
        const image::quaternion_wavelet moved(render(field, dimension, 4.0, 0.0));
        double worst = 0.0;
        double scale = 0.0;
        for (size_t y = 8; y + 8 < reference[2].rows; ++y) {
            for (size_t x = 8; x + 8 < reference[2].cols; ++x) {
                const float* const from = reference.at(2, image::quaternion_wavelet::band::diagonal, x, y);
                const float* const to = moved.at(2, image::quaternion_wavelet::band::diagonal, x + 1, y);
                for (int component = 0; component < image::quaternion_wavelet::component_count; ++component) {
                    const double difference = std::abs(static_cast<double>(from[component] - to[component]));
                    if (difference > worst) {
                        worst = difference;
                    }
                }
                const double modulus = static_cast<double>(image::quaternion_wavelet::modulus(from));
                if (modulus > scale) {
                    scale = modulus;
                }
            }
        }
        REQUIRE(scale > 1.0);
        REQUIRE(worst < 0.02 * scale);
    }

    {
        const image::quaternion_wavelet reference(base);
        const double shift = 1.0;
        const image::quaternion_wavelet moved(render(field, dimension, shift, 0.0));
        const image::quaternion_wavelet::level& level = reference[3];
        const double spacing = 8.0;
        std::vector<double> ratios;
        std::vector<double> slopes;
        for (size_t y = 4; y + 4 < level.rows; ++y) {
            for (size_t x = 4; x + 4 < level.cols; ++x) {
                const float* const from = reference.at(3, image::quaternion_wavelet::band::vertical, x, y);
                const float* const to = moved.at(3, image::quaternion_wavelet::band::vertical, x, y);
                const double modulus_from = static_cast<double>(image::quaternion_wavelet::modulus(from));
                const double modulus_to = static_cast<double>(image::quaternion_wavelet::modulus(to));
                if (modulus_from < 6.0) {
                    continue;
                }
                ratios.push_back(modulus_to / modulus_from);
                float from_plus_real = 0.0f;
                float from_plus_imaginary = 0.0f;
                float from_minus_real = 0.0f;
                float from_minus_imaginary = 0.0f;
                image::quaternion_wavelet::complex_pair(from, from_plus_real, from_plus_imaginary, from_minus_real, from_minus_imaginary);
                float to_plus_real = 0.0f;
                float to_plus_imaginary = 0.0f;
                float to_minus_real = 0.0f;
                float to_minus_imaginary = 0.0f;
                image::quaternion_wavelet::complex_pair(to, to_plus_real, to_plus_imaginary, to_minus_real, to_minus_imaginary);
                const double real = (static_cast<double>(to_plus_real) * static_cast<double>(from_plus_real)) + (static_cast<double>(to_plus_imaginary) * static_cast<double>(from_plus_imaginary));
                const double imaginary = (static_cast<double>(to_plus_imaginary) * static_cast<double>(from_plus_real)) - (static_cast<double>(to_plus_real) * static_cast<double>(from_plus_imaginary));
                slopes.push_back(wrap(std::atan2(imaginary, real)) / (-2.0 * pi * (shift / spacing)));
            }
        }
        REQUIRE(ratios.size() > 50);
        double ratio_sum = 0.0;
        for (const double ratio : ratios) {
            ratio_sum += ratio;
        }
        REQUIRE(std::abs((ratio_sum / static_cast<double>(ratios.size())) - 1.0) < 0.1);
        double slope_sum = 0.0;
        for (const double slope : slopes) {
            slope_sum += slope;
        }
        const double measured = slope_sum / static_cast<double>(slopes.size());
        REQUIRE(measured > 0.5 * reference[3].wavelet_centre);
        REQUIRE(measured < 1.5 * reference[3].wavelet_centre);
    }

    {
        const image::quaternion_wavelet reference(base);
        const image::quaternion_wavelet brighter(render(field, dimension, 0.0, 0.0, 1.2, 6.0));
        std::vector<double> phase_moves;
        std::vector<double> ratios;
        for (size_t y = 4; y + 4 < reference[4].rows; ++y) {
            for (size_t x = 4; x + 4 < reference[4].cols; ++x) {
                const float* const from = reference.at(4, image::quaternion_wavelet::band::diagonal, x, y);
                const float* const to = brighter.at(4, image::quaternion_wavelet::band::diagonal, x, y);
                const double modulus_from = static_cast<double>(image::quaternion_wavelet::modulus(from));
                if (modulus_from < 20.0) {
                    continue;
                }
                float from_pair[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                float to_pair[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                image::quaternion_wavelet::complex_pair(from, from_pair[0], from_pair[1], from_pair[2], from_pair[3]);
                image::quaternion_wavelet::complex_pair(to, to_pair[0], to_pair[1], to_pair[2], to_pair[3]);
                for (int half = 0; half < 2; ++half) {
                    const double from_real = static_cast<double>(from_pair[half * 2]);
                    const double from_imaginary = static_cast<double>(from_pair[(half * 2) + 1]);
                    const double to_real = static_cast<double>(to_pair[half * 2]);
                    const double to_imaginary = static_cast<double>(to_pair[(half * 2) + 1]);
                    if (((from_real * from_real) + (from_imaginary * from_imaginary)) < (20.0 * 20.0)) {
                        continue;
                    }
                    const double real = (to_real * from_real) + (to_imaginary * from_imaginary);
                    const double imaginary = (to_imaginary * from_real) - (to_real * from_imaginary);
                    phase_moves.push_back(std::abs(wrap(std::atan2(imaginary, real))));
                }
                ratios.push_back(static_cast<double>(image::quaternion_wavelet::modulus(to)) / modulus_from);
            }
        }
        REQUIRE(phase_moves.size() > 100);
        std::sort(phase_moves.begin(), phase_moves.end());
        std::sort(ratios.begin(), ratios.end());
        REQUIRE(phase_moves[phase_moves.size() / 2] < 0.02);
        REQUIRE(phase_moves[(phase_moves.size() * 9) / 10] < 0.05);
        REQUIRE(std::abs(ratios[ratios.size() / 2] - 1.2) < 0.02);
    }

    {
        const image::quaternion_wavelet decimated(base);
        const image::quaternion_wavelet dense(base, 0, 2, true);
        REQUIRE(dense.size() == decimated.size());
        REQUIRE(dense[2].spacing == 2.0);
        REQUIRE(decimated[2].spacing == 4.0);
        REQUIRE(dense[2].rows == 2 * decimated[2].rows);
        REQUIRE(dense[2].cols == 2 * decimated[2].cols);
        REQUIRE(std::abs(dense[2].wavelet_centre - (0.5 * decimated[2].wavelet_centre)) < 1e-12);
        REQUIRE(std::abs(dense[2].scaling_centre - (0.5 * decimated[2].scaling_centre)) < 1e-12);
        for (size_t y = 0; y < decimated[2].rows; ++y) {
            for (size_t x = 0; x < decimated[2].cols; ++x) {
                for (int which = 0; which < image::quaternion_wavelet::band_count; ++which) {
                    const float* const from = decimated.at(2, static_cast<image::quaternion_wavelet::band>(which), x, y);
                    const float* const to = dense.at(2, static_cast<image::quaternion_wavelet::band>(which), 2 * x, 2 * y);
                    for (int component = 0; component < image::quaternion_wavelet::component_count; ++component) {
                        REQUIRE(from[component] == to[component]);
                    }
                }
            }
        }
        for (size_t level = 3; level <= dense.size(); ++level) {
            REQUIRE(dense[level].spacing == decimated[level].spacing);
            REQUIRE(dense[level].bands[0] == decimated[level].bands[0]);
        }
        const image::quaternion_wavelet shifted(render(field, dimension, 2.0, 0.0), 0, 2, true);
        double worst = 0.0;
        double scale = 0.0;
        for (size_t y = 8; y + 8 < decimated[2].rows; ++y) {
            for (size_t x = 8; x + 8 < decimated[2].cols; ++x) {
                const float* const from = dense.at(2, image::quaternion_wavelet::band::diagonal, (2 * x) + 1, 2 * y);
                const float* const to = shifted.at(2, image::quaternion_wavelet::band::diagonal, (2 * x) + 2, 2 * y);
                for (int component = 0; component < image::quaternion_wavelet::component_count; ++component) {
                    worst = std::max(worst, std::abs(static_cast<double>(from[component] - to[component])));
                }
                scale = std::max(scale, static_cast<double>(image::quaternion_wavelet::modulus(from)));
            }
        }
        REQUIRE(scale > 1.0);
        REQUIRE(worst < 0.02 * scale);
    }

    {
        const image::quaternion_wavelet empty;
        REQUIRE(empty.size() == 0);
        REQUIRE(empty.at(1, image::quaternion_wavelet::band::diagonal, 0, 0) == nullptr);
        const image::image small(8, 8);
        const image::quaternion_wavelet tiny(small);
        REQUIRE(tiny.size() == 0);
    }

    {
        const image::quaternion_wavelet first(base);
        const image::quaternion_wavelet second(base);
        REQUIRE(first.size() == second.size());
        for (size_t level = 2; level <= first.size(); ++level) {
            for (int which = 0; which < image::quaternion_wavelet::band_count; ++which) {
                REQUIRE(first[level].bands[which].size() == second[level].bands[which].size());
                for (size_t index = 0; index < first[level].bands[which].size(); ++index) {
                    REQUIRE(first[level].bands[which][index] == second[level].bands[which][index]);
                }
            }
        }
    }

    {
        constexpr static const size_t side = 96;
        image::image original(side, side);
        image::image transposed(side, side);
        unsigned long long seed = 0x5eed0101ull;
        for (size_t y = 0; y < side; ++y) {
            for (size_t x = 0; x < side; ++x) {
                const unsigned char value = static_cast<unsigned char>(255.0 * next_random_unit(seed));
                original.get_data()[(y * side) + x] = value;
                transposed.get_data()[(x * side) + y] = value;
            }
        }
        const image::quaternion_wavelet forward(original, 0, 1);
        const image::quaternion_wavelet flipped(transposed, 0, 1);
        REQUIRE(forward.size() == flipped.size());
        for (size_t level = 1; level <= forward.size(); ++level) {
            double worst = 0.0;
            double scale = 0.0;
            for (size_t y = 0; y < forward[level].rows; ++y) {
                for (size_t x = 0; x < forward[level].cols; ++x) {
                    const float* const lhs = forward.at(level, image::quaternion_wavelet::band::diagonal, x, y);
                    const float* const rhs = flipped.at(level, image::quaternion_wavelet::band::diagonal, y, x);
                    const int swapped[image::quaternion_wavelet::component_count] = { 0, 2, 1, 3 };
                    for (int component = 0; component < image::quaternion_wavelet::component_count; ++component) {
                        worst = std::max(worst, static_cast<double>(std::abs(lhs[component] - rhs[swapped[component]])));
                        scale = std::max(scale, static_cast<double>(std::abs(lhs[component])));
                    }
                }
            }
            REQUIRE(scale > 1.0);
            REQUIRE(worst < 1e-5 * scale);
        }
    }

    return EXIT_SUCCESS;
}
