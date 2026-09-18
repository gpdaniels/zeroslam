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

#include "feature/tracker/patch_flow.hpp"

#include "image/image.hpp"
#include "image/pyramid.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

namespace {
    constexpr static const size_t dimension = 192;
    constexpr static const int wave_count = 8;
    constexpr static const double pi = 3.14159265358979323846;

    double next_random_unit(unsigned long long& seed) {
        seed += 0x9E3779B97F4A7C15ull;
        unsigned long long z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z = z ^ (z >> 31);
        return static_cast<double>(z >> 11) / 9007199254740992.0;
    }

    struct texture_field {
        double frequency_x[wave_count];
        double frequency_y[wave_count];
        double phase[wave_count];
        double amplitude[wave_count];

        double evaluate(const double x, const double y) const {
            double value = 100.0;
            for (int wave = 0; wave < wave_count; ++wave) {
                value += this->amplitude[wave] * std::sin(this->frequency_x[wave] * x + this->frequency_y[wave] * y + this->phase[wave]);
            }
            return value;
        }
    };

    texture_field make_texture(unsigned long long& seed) {
        texture_field field;
        for (int wave = 0; wave < wave_count; ++wave) {
            const double angle = 2.0 * pi * next_random_unit(seed);
            const double frequency = 0.10 + 0.25 * next_random_unit(seed);
            field.frequency_x[wave] = frequency * std::cos(angle);
            field.frequency_y[wave] = frequency * std::sin(angle);
            field.phase[wave] = 2.0 * pi * next_random_unit(seed);
            field.amplitude[wave] = 60.0 / static_cast<double>(wave_count);
        }
        return field;
    }

    template <typename function_type>
    image::image make_image(const size_t size, function_type function) {
        image::image result(size, size);
        for (size_t y = 0; y < size; ++y) {
            for (size_t x = 0; x < size; ++x) {
                double value = function(static_cast<double>(x), static_cast<double>(y));
                if (value < 0.0) {
                    value = 0.0;
                }
                if (value > 255.0) {
                    value = 255.0;
                }
                result.get_data()[y * size + x] = static_cast<unsigned char>(value + 0.5);
            }
        }
        return result;
    }

    bool track_point(const image::pyramid& target, const image::pyramid& source, const float x, const float y, const feature::tracker::patch_flow::options& settings, feature::tracker::patch_flow::state& warp, feature::tracker::patch_flow::result& out) {
        feature::tracker::patch_flow::anchor anchored;
        if (!feature::tracker::patch_flow::build_anchor(source, x, y, settings, anchored)) {
            return false;
        }
        return feature::tracker::patch_flow::align(target, anchored, settings, warp, out);
    }

    feature::tracker::patch_flow::options make_options(const feature::tracker::patch_flow::model_kind model) {
        feature::tracker::patch_flow::options settings;
        settings.model = model;
        settings.half_window = 9;
        return settings;
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    unsigned long long seed = 0x5EED5EED5EED5EEDull;
    const texture_field field = make_texture(seed);
    const image::image base = make_image(dimension, [&](const double x, const double y) {
        return field.evaluate(x, y);
    });
    const image::pyramid pyramid_base(base);
    REQUIRE(pyramid_base.size() == 3);

    constexpr static const int probe_count = 4;
    const float probe_x[probe_count] = { 60.5f, 90.5f, 120.5f, 75.5f };
    const float probe_y[probe_count] = { 60.5f, 70.5f, 100.5f, 115.5f };

    {
        REQUIRE(feature::tracker::patch_flow::parameter_count(feature::tracker::patch_flow::model_kind::translation) == 2);
        REQUIRE(feature::tracker::patch_flow::parameter_count(feature::tracker::patch_flow::model_kind::affine) == 6);
        REQUIRE(feature::tracker::patch_flow::parameter_count(feature::tracker::patch_flow::model_kind::translation_illumination) == 4);
        REQUIRE(feature::tracker::patch_flow::parameter_count(feature::tracker::patch_flow::model_kind::affine_illumination) == 8);
        REQUIRE(!feature::tracker::patch_flow::has_illumination(feature::tracker::patch_flow::model_kind::translation));
        REQUIRE(!feature::tracker::patch_flow::has_illumination(feature::tracker::patch_flow::model_kind::affine));
        REQUIRE(feature::tracker::patch_flow::has_illumination(feature::tracker::patch_flow::model_kind::translation_illumination));
        REQUIRE(feature::tracker::patch_flow::has_illumination(feature::tracker::patch_flow::model_kind::affine_illumination));
    }

    {
        const feature::tracker::patch_flow::options settings = make_options(feature::tracker::patch_flow::model_kind::affine_illumination);
        feature::tracker::patch_flow::anchor anchored;
        REQUIRE(feature::tracker::patch_flow::build_anchor(pyramid_base, probe_x[0], probe_y[0], settings, anchored));
        REQUIRE(anchored.levels == 3);
        REQUIRE(anchored.parameters == 8);
        REQUIRE(anchored.half_window == 9);
        REQUIRE(anchored.centre_x == probe_x[0] - 0.5f);
        REQUIRE(anchored.centre_y == probe_y[0] - 0.5f);
        REQUIRE(anchored.values.size() == static_cast<size_t>(3 * 19 * 19));
        REQUIRE(anchored.gradients_x.size() == anchored.values.size());
        REQUIRE(anchored.gradients_y.size() == anchored.values.size());
        REQUIRE(anchored.inverse_hessian.size() == static_cast<size_t>(3 * 8 * 8));
    }

    {
        const double shift_x = 0.37;
        const double shift_y = -0.62;
        const image::image shifted = make_image(dimension, [&](const double x, const double y) {
            return field.evaluate(x - shift_x, y - shift_y);
        });
        const image::pyramid pyramid_shifted(shifted);
        for (int probe = 0; probe < probe_count; ++probe) {
            const feature::tracker::patch_flow::options settings = make_options(feature::tracker::patch_flow::model_kind::translation);
            feature::tracker::patch_flow::state warp;
            feature::tracker::patch_flow::result out;
            REQUIRE(track_point(pyramid_shifted, pyramid_base, probe_x[probe], probe_y[probe], settings, warp, out));
            REQUIRE(std::abs(static_cast<double>(out.x - probe_x[probe]) - shift_x) < 0.05);
            REQUIRE(std::abs(static_cast<double>(out.y - probe_y[probe]) - shift_y) < 0.05);
            REQUIRE(out.error < 1.0f);
            REQUIRE(warp.linear_xx == 1.0f);
            REQUIRE(warp.linear_xy == 0.0f);
            REQUIRE(warp.linear_yx == 0.0f);
            REQUIRE(warp.linear_yy == 1.0f);
            REQUIRE(warp.gain == 0.0f);
            REQUIRE(warp.bias == 0.0f);
        }
    }

    {
        const double shift_x = -6.0;
        const double shift_y = 4.0;
        const image::image shifted = make_image(dimension, [&](const double x, const double y) {
            return field.evaluate(x - shift_x, y - shift_y);
        });
        const image::pyramid pyramid_shifted(shifted);
        for (int probe = 0; probe < probe_count; ++probe) {
            const feature::tracker::patch_flow::options settings = make_options(feature::tracker::patch_flow::model_kind::translation);
            feature::tracker::patch_flow::state warp;
            feature::tracker::patch_flow::result out;
            REQUIRE(track_point(pyramid_shifted, pyramid_base, probe_x[probe], probe_y[probe], settings, warp, out));
            REQUIRE(std::abs(static_cast<double>(out.x - probe_x[probe]) - shift_x) < 0.05);
            REQUIRE(std::abs(static_cast<double>(out.y - probe_y[probe]) - shift_y) < 0.05);
        }
    }

    {
        const double centre = 95.0;
        const double angle = 3.0 * pi / 180.0;
        const double magnification = 1.03;
        const double linear_xx = magnification * std::cos(angle);
        const double linear_xy = -magnification * std::sin(angle);
        const double linear_yx = magnification * std::sin(angle);
        const double linear_yy = magnification * std::cos(angle);
        const double determinant = linear_xx * linear_yy - linear_xy * linear_yx;
        const double inverse_xx = linear_yy / determinant;
        const double inverse_xy = -linear_xy / determinant;
        const double inverse_yx = -linear_yx / determinant;
        const double inverse_yy = linear_xx / determinant;
        const image::image warped = make_image(dimension, [&](const double x, const double y) {
            const double offset_x = x - centre;
            const double offset_y = y - centre;
            return field.evaluate(centre + inverse_xx * offset_x + inverse_xy * offset_y, centre + inverse_yx * offset_x + inverse_yy * offset_y);
        });
        const image::pyramid pyramid_warped(warped);
        for (int probe = 0; probe < probe_count; ++probe) {
            const double offset_x = static_cast<double>(probe_x[probe]) - 0.5 - centre;
            const double offset_y = static_cast<double>(probe_y[probe]) - 0.5 - centre;
            const double expected_x = centre + linear_xx * offset_x + linear_xy * offset_y + 0.5;
            const double expected_y = centre + linear_yx * offset_x + linear_yy * offset_y + 0.5;

            const feature::tracker::patch_flow::options affine_settings = make_options(feature::tracker::patch_flow::model_kind::affine);
            feature::tracker::patch_flow::state affine_warp;
            feature::tracker::patch_flow::result affine_out;
            REQUIRE(track_point(pyramid_warped, pyramid_base, probe_x[probe], probe_y[probe], affine_settings, affine_warp, affine_out));
            REQUIRE(std::abs(static_cast<double>(affine_out.x) - expected_x) < 0.15);
            REQUIRE(std::abs(static_cast<double>(affine_out.y) - expected_y) < 0.15);
            REQUIRE(std::abs(static_cast<double>(affine_warp.linear_xx) - linear_xx) < 0.025);
            REQUIRE(std::abs(static_cast<double>(affine_warp.linear_xy) - linear_xy) < 0.025);
            REQUIRE(std::abs(static_cast<double>(affine_warp.linear_yx) - linear_yx) < 0.025);
            REQUIRE(std::abs(static_cast<double>(affine_warp.linear_yy) - linear_yy) < 0.025);

            const feature::tracker::patch_flow::options translation_settings = make_options(feature::tracker::patch_flow::model_kind::translation);
            feature::tracker::patch_flow::state translation_warp;
            feature::tracker::patch_flow::result translation_out;
            REQUIRE(track_point(pyramid_warped, pyramid_base, probe_x[probe], probe_y[probe], translation_settings, translation_warp, translation_out));
            REQUIRE(affine_out.error < translation_out.error);
            const double affine_distance = std::abs(static_cast<double>(affine_out.x) - expected_x) + std::abs(static_cast<double>(affine_out.y) - expected_y);
            const double translation_distance = std::abs(static_cast<double>(translation_out.x) - expected_x) + std::abs(static_cast<double>(translation_out.y) - expected_y);
            REQUIRE(affine_distance < translation_distance);
        }
    }

    {
        const double shift_x = 0.45;
        const double shift_y = 0.8;
        const double gain = 0.25;
        const double bias = 25.0;
        const image::image exposed = make_image(dimension, [&](const double x, const double y) {
            return (1.0 + gain) * field.evaluate(x - shift_x, y - shift_y) + bias;
        });
        const image::pyramid pyramid_exposed(exposed);
        for (int probe = 0; probe < probe_count; ++probe) {
            const feature::tracker::patch_flow::options plain_settings = make_options(feature::tracker::patch_flow::model_kind::translation);
            feature::tracker::patch_flow::state plain_warp;
            feature::tracker::patch_flow::result plain_out;
            REQUIRE(!track_point(pyramid_exposed, pyramid_base, probe_x[probe], probe_y[probe], plain_settings, plain_warp, plain_out));

            const feature::tracker::patch_flow::options affine_settings = make_options(feature::tracker::patch_flow::model_kind::affine);
            feature::tracker::patch_flow::state affine_warp;
            feature::tracker::patch_flow::result affine_out;
            REQUIRE(!track_point(pyramid_exposed, pyramid_base, probe_x[probe], probe_y[probe], affine_settings, affine_warp, affine_out));

            const feature::tracker::patch_flow::options illuminated_settings = make_options(feature::tracker::patch_flow::model_kind::translation_illumination);
            feature::tracker::patch_flow::state illuminated_warp;
            feature::tracker::patch_flow::result illuminated_out;
            REQUIRE(track_point(pyramid_exposed, pyramid_base, probe_x[probe], probe_y[probe], illuminated_settings, illuminated_warp, illuminated_out));
            REQUIRE(std::abs(static_cast<double>(illuminated_out.x - probe_x[probe]) - shift_x) < 0.05);
            REQUIRE(std::abs(static_cast<double>(illuminated_out.y - probe_y[probe]) - shift_y) < 0.05);
            REQUIRE(illuminated_out.error < 1.0f);
            REQUIRE(std::abs(static_cast<double>(illuminated_warp.gain) - gain) < 0.05);
            REQUIRE(std::abs(static_cast<double>(illuminated_warp.bias) - bias) < 5.0);
            const double illuminated_prediction = (1.0 + static_cast<double>(illuminated_warp.gain)) * 100.0 + static_cast<double>(illuminated_warp.bias);
            REQUIRE(std::abs(illuminated_prediction - ((1.0 + gain) * 100.0 + bias)) < 1.0);

            const feature::tracker::patch_flow::options full_settings = make_options(feature::tracker::patch_flow::model_kind::affine_illumination);
            feature::tracker::patch_flow::state full_warp;
            feature::tracker::patch_flow::result full_out;
            REQUIRE(track_point(pyramid_exposed, pyramid_base, probe_x[probe], probe_y[probe], full_settings, full_warp, full_out));
            REQUIRE(std::abs(static_cast<double>(full_out.x - probe_x[probe]) - shift_x) < 0.06);
            REQUIRE(std::abs(static_cast<double>(full_out.y - probe_y[probe]) - shift_y) < 0.06);
            const double full_prediction = (1.0 + static_cast<double>(full_warp.gain)) * 100.0 + static_cast<double>(full_warp.bias);
            REQUIRE(std::abs(full_prediction - ((1.0 + gain) * 100.0 + bias)) < 1.0);
        }
    }

    {
        const image::image flat = make_image(dimension, [](const double x, const double y) {
            static_cast<void>(x);
            static_cast<void>(y);
            return 128.0;
        });
        const image::pyramid pyramid_flat(flat);
        feature::tracker::patch_flow::anchor anchored;
        REQUIRE(!feature::tracker::patch_flow::build_anchor(pyramid_flat, 96.5f, 96.5f, feature::tracker::patch_flow::options(), anchored));
        REQUIRE(anchored.levels == 0);
        REQUIRE(anchored.values.empty());
    }

    {
        feature::tracker::patch_flow::anchor anchored;
        REQUIRE(!feature::tracker::patch_flow::build_anchor(pyramid_base, 1.5f, 1.5f, feature::tracker::patch_flow::options(), anchored));
        REQUIRE(!feature::tracker::patch_flow::build_anchor(pyramid_base, 96.5f, 3.5f, feature::tracker::patch_flow::options(), anchored));
        REQUIRE(!feature::tracker::patch_flow::build_anchor(pyramid_base, static_cast<float>(dimension) - 2.5f, 96.5f, feature::tracker::patch_flow::options(), anchored));
        REQUIRE(anchored.levels == 0);
        feature::tracker::patch_flow::anchor shallow;
        REQUIRE(feature::tracker::patch_flow::build_anchor(pyramid_base, 20.5f, 20.5f, feature::tracker::patch_flow::options(), shallow));
        REQUIRE(shallow.levels > 0);
        REQUIRE(shallow.levels < 3);
    }

    {
        const image::image shifted = make_image(dimension, [&](const double x, const double y) {
            return field.evaluate(x - 1.25, y + 0.5);
        });
        const image::pyramid pyramid_shifted(shifted);
        const feature::tracker::patch_flow::options settings = make_options(feature::tracker::patch_flow::model_kind::affine_illumination);
        feature::tracker::patch_flow::anchor anchored;
        REQUIRE(feature::tracker::patch_flow::build_anchor(pyramid_base, probe_x[0], probe_y[0], settings, anchored));
        feature::tracker::patch_flow::state first_warp;
        feature::tracker::patch_flow::result first_out;
        REQUIRE(feature::tracker::patch_flow::align(pyramid_shifted, anchored, settings, first_warp, first_out));
        feature::tracker::patch_flow::state second_warp;
        feature::tracker::patch_flow::result second_out;
        REQUIRE(feature::tracker::patch_flow::align(pyramid_shifted, anchored, settings, second_warp, second_out));
        REQUIRE(first_out.x == second_out.x);
        REQUIRE(first_out.y == second_out.y);
        REQUIRE(first_out.error == second_out.error);
        REQUIRE(first_out.tracked == second_out.tracked);
        REQUIRE(first_warp.linear_xx == second_warp.linear_xx);
        REQUIRE(first_warp.linear_xy == second_warp.linear_xy);
        REQUIRE(first_warp.linear_yx == second_warp.linear_yx);
        REQUIRE(first_warp.linear_yy == second_warp.linear_yy);
        REQUIRE(first_warp.translation_x == second_warp.translation_x);
        REQUIRE(first_warp.translation_y == second_warp.translation_y);
        REQUIRE(first_warp.gain == second_warp.gain);
        REQUIRE(first_warp.bias == second_warp.bias);
        float position_x = 0.0f;
        float position_y = 0.0f;
        feature::tracker::patch_flow::position(anchored, first_warp, position_x, position_y);
        REQUIRE(position_x == first_out.x);
        REQUIRE(position_y == first_out.y);
    }

    {
        const image::image shifted = make_image(dimension, [&](const double x, const double y) {
            return field.evaluate(x - 2.5, y - 1.5);
        });
        const image::pyramid pyramid_shifted(shifted);
        const feature::tracker::patch_flow::options settings = make_options(feature::tracker::patch_flow::model_kind::translation);
        feature::tracker::patch_flow::anchor anchored;
        REQUIRE(feature::tracker::patch_flow::build_anchor(pyramid_base, probe_x[1], probe_y[1], settings, anchored));
        feature::tracker::patch_flow::state cold_warp;
        feature::tracker::patch_flow::result cold_out;
        REQUIRE(feature::tracker::patch_flow::align(pyramid_shifted, anchored, settings, cold_warp, cold_out));
        feature::tracker::patch_flow::state warm_warp;
        warm_warp.translation_x = 2.0f;
        warm_warp.translation_y = 2.0f;
        feature::tracker::patch_flow::result warm_out;
        REQUIRE(feature::tracker::patch_flow::align(pyramid_shifted, anchored, settings, warm_warp, warm_out));
        REQUIRE(std::abs(static_cast<double>(warm_out.x - cold_out.x)) < 0.02);
        REQUIRE(std::abs(static_cast<double>(warm_out.y - cold_out.y)) < 0.02);
    }

    std::printf("All patch flow tests passed.\n");
    return 0;
}
