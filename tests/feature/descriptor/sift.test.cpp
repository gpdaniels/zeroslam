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

#include "feature/descriptor/sift.hpp"

#include "feature/angle/orb.hpp"
#include "match/distance/hamming.hpp"

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

static double texture(double x, double y) {
    return 128.0 + 55.0 * std::sin(0.21 * x) * std::sin(0.19 * y) + 35.0 * std::cos(0.15 * x + 0.10 * y) + 25.0 * std::sin(0.30 * x - 0.12 * y);
}

static std::vector<unsigned char> render(const int size, const double angle, const double shift_x, const double shift_y) {
    std::vector<unsigned char> image(static_cast<size_t>(size * size));
    const double centre = 0.5 * static_cast<double>(size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const double dx = static_cast<double>(x) - centre;
            const double dy = static_cast<double>(y) - centre;
            const double sx = (std::cos(angle) * dx) + (std::sin(angle) * dy) + centre + shift_x;
            const double sy = (-std::sin(angle) * dx) + (std::cos(angle) * dy) + centre + shift_y;
            image[static_cast<size_t>((y * size) + x)] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, texture(sx, sy) + 0.5)));
        }
    }
    return image;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);
    constexpr static const int size = 96;
    const int centre = size / 2;
    const std::vector<unsigned char> plain = render(size, 0.0, 0.0, 0.0);
    const unsigned char* const at_centre = plain.data() + (centre * size) + centre;
    const float angle = feature::angle::orb::dominant_angle(at_centre, size);

    feature::descriptor::binary<256> first;
    feature::descriptor::binary<256> again;
    feature::descriptor::sift::describe(at_centre, size, angle, first);
    feature::descriptor::sift::describe(at_centre, size, angle, again);
    REQUIRE(match::distance::hamming::distance(first, again) == 0);

    unsigned int set_bits = 0;
    for (int byte = 0; byte < 32; ++byte) {
        for (int bit = 0; bit < 8; ++bit) {
            set_bits += (static_cast<unsigned int>(first.data[byte]) >> bit) & 1u;
        }
    }
    REQUIRE((set_bits > 40) && (set_bits < 216));

    const std::vector<unsigned char> shifted = render(size, 0.0, 0.4, -0.3);
    feature::descriptor::binary<256> descriptor_shifted;
    feature::descriptor::sift::describe(shifted.data() + (centre * size) + centre, size, feature::angle::orb::dominant_angle(shifted.data() + (centre * size) + centre, size), descriptor_shifted);
    REQUIRE(match::distance::hamming::distance(first, descriptor_shifted) < 40);

    const std::vector<unsigned char> rotated = render(size, 0.6, 0.0, 0.0);
    const unsigned char* const rotated_centre = rotated.data() + (centre * size) + centre;
    feature::descriptor::binary<256> descriptor_rotated;
    feature::descriptor::sift::describe(rotated_centre, size, feature::angle::orb::dominant_angle(rotated_centre, size), descriptor_rotated);
    REQUIRE(match::distance::hamming::distance(first, descriptor_rotated) < 70);

    const unsigned char* const elsewhere = plain.data() + ((centre + 17) * size) + centre - 23;
    feature::descriptor::binary<256> descriptor_elsewhere;
    feature::descriptor::sift::describe(elsewhere, size, feature::angle::orb::dominant_angle(elsewhere, size), descriptor_elsewhere);
    REQUIRE(match::distance::hamming::distance(first, descriptor_elsewhere) > 70);
    float vector[feature::descriptor::sift::dimensions];
    feature::descriptor::sift::describe_float(at_centre, size, angle, nullptr, vector);
    double squares = 0.0;
    for (int d = 0; d < feature::descriptor::sift::dimensions; ++d) {
        REQUIRE(vector[d] >= 0.0f);
        squares += static_cast<double>(vector[d]) * static_cast<double>(vector[d]);
    }
    REQUIRE(std::abs(squares - 1.0) < 1e-3);
    const float tilt[4] = { 1.0f, 0.0f, 0.0f, 1.0f / 1.2f };
    float tilted[feature::descriptor::sift::dimensions];
    feature::descriptor::sift::describe_float(at_centre, size, angle, &tilt[0], tilted);
    feature::descriptor::binary<256> descriptor_tilted;
    feature::descriptor::sift::binarise(tilted, descriptor_tilted);
    REQUIRE(match::distance::hamming::distance(first, descriptor_tilted) < 70);

    {
        // A zero offset matches the overloads without one bit for bit.
        float offset_vector[feature::descriptor::sift::dimensions];
        feature::descriptor::sift::describe_float(at_centre, size, 0.0f, 0.0f, angle, nullptr, offset_vector);
        for (int d = 0; d < feature::descriptor::sift::dimensions; ++d) {
            REQUIRE(offset_vector[d] == vector[d]);
        }
        feature::descriptor::sift::describe_float(at_centre, size, 0.0f, 0.0f, angle, &tilt[0], offset_vector);
        for (int d = 0; d < feature::descriptor::sift::dimensions; ++d) {
            REQUIRE(offset_vector[d] == tilted[d]);
        }
        feature::descriptor::binary<256> offset_descriptor;
        feature::descriptor::sift::describe(at_centre, size, 0.0f, 0.0f, angle, offset_descriptor);
        REQUIRE(match::distance::hamming::distance(first, offset_descriptor) == 0);
    }

    {
        // Content shifted by a fraction of a pixel is found again by describing at the fractional offset, not at the nearest pixel.
        double fractional_total = 0.0;
        double nearest_total = 0.0;
        for (int shift = 0; shift < 4; ++shift) {
            const double shift_x = -0.45 + 0.3 * static_cast<double>(shift);
            const double shift_y = 0.4 - 0.25 * static_cast<double>(shift);
            const std::vector<unsigned char> moved = render(size, 0.0, shift_x, shift_y);
            float fractional[feature::descriptor::sift::dimensions];
            float nearest[feature::descriptor::sift::dimensions];
            feature::descriptor::sift::describe_float(moved.data() + (centre * size) + centre, size, static_cast<float>(-shift_x), static_cast<float>(-shift_y), angle, nullptr, fractional);
            feature::descriptor::sift::describe_float(moved.data() + (centre * size) + centre, size, std::round(static_cast<float>(-shift_x)), std::round(static_cast<float>(-shift_y)), angle, nullptr, nearest);
            double fractional_squares = 0.0;
            double nearest_squares = 0.0;
            for (int d = 0; d < feature::descriptor::sift::dimensions; ++d) {
                fractional_squares += static_cast<double>(fractional[d] - vector[d]) * static_cast<double>(fractional[d] - vector[d]);
                nearest_squares += static_cast<double>(nearest[d] - vector[d]) * static_cast<double>(nearest[d] - vector[d]);
            }
            REQUIRE(std::sqrt(fractional_squares) < 0.03);
            fractional_total += std::sqrt(fractional_squares);
            nearest_total += std::sqrt(nearest_squares);
        }
        REQUIRE(2.0 * fractional_total < nearest_total);
    }

    {
        // The affine path equals the plain description of the patch the affine warps into view; the samples sit at R A u, so the image is warped by R A R^T.
        const int patch_size = 101;
        const int patch_centre = patch_size / 2;
        const double patch_angle = 0.3;
        for (int direction = 0; direction < 4; ++direction) {
            const double phi = static_cast<double>(direction) * 0.25 * 3.14159265358979323846;
            const double stretch = 0.5;
            const double a[4] = { std::cos(phi) * std::cos(phi) + stretch * std::sin(phi) * std::sin(phi), (stretch - 1.0) * std::cos(phi) * std::sin(phi), (stretch - 1.0) * std::cos(phi) * std::sin(phi), std::sin(phi) * std::sin(phi) + stretch * std::cos(phi) * std::cos(phi) };
            const double c = std::cos(patch_angle);
            const double s = std::sin(patch_angle);
            const double ra[4] = { c * a[0] - s * a[2], c * a[1] - s * a[3], s * a[0] + c * a[2], s * a[1] + c * a[3] };
            const double warp[4] = { ra[0] * c - ra[1] * s, ra[0] * s + ra[1] * c, ra[2] * c - ra[3] * s, ra[2] * s + ra[3] * c };
            std::vector<unsigned char> patch(static_cast<size_t>(patch_size * patch_size));
            for (int y = 0; y < patch_size; ++y) {
                for (int x = 0; x < patch_size; ++x) {
                    const double u = static_cast<double>(x - patch_centre);
                    const double v = static_cast<double>(y - patch_centre);
                    const double value = texture(static_cast<double>(centre) + warp[0] * u + warp[1] * v, static_cast<double>(centre) + warp[2] * u + warp[3] * v);
                    patch[static_cast<size_t>((y * patch_size) + x)] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, value + 0.5)));
                }
            }
            const float affine[4] = { static_cast<float>(a[0]), static_cast<float>(a[1]), static_cast<float>(a[2]), static_cast<float>(a[3]) };
            float expected[feature::descriptor::sift::dimensions];
            float warped[feature::descriptor::sift::dimensions];
            feature::descriptor::sift::describe_float(patch.data() + (patch_centre * patch_size) + patch_centre, patch_size, static_cast<float>(patch_angle), nullptr, expected);
            feature::descriptor::sift::describe_float(at_centre, size, static_cast<float>(patch_angle), &affine[0], warped);
            double warp_squares = 0.0;
            for (int d = 0; d < feature::descriptor::sift::dimensions; ++d) {
                warp_squares += static_cast<double>(expected[d] - warped[d]) * static_cast<double>(expected[d] - warped[d]);
            }
            REQUIRE(std::sqrt(warp_squares) < 0.05);
        }
    }

    {
        // The first half of the bits compares each dimension with the median and the second half with a partner, where no pair is compared twice.
        float values[feature::descriptor::sift::dimensions];
        for (int d = 0; d < feature::descriptor::sift::dimensions; ++d) {
            values[d] = static_cast<float>((d * 37) % 128) / 128.0f;
        }
        feature::descriptor::binary<256> descriptor;
        feature::descriptor::sift::binarise(values, descriptor);
        std::vector<float> sorted(&values[0], &values[0] + feature::descriptor::sift::dimensions);
        std::sort(sorted.begin(), sorted.end());
        const float median = sorted[feature::descriptor::sift::dimensions / 2];
        int complements = 0;
        for (int d = 0; d < feature::descriptor::sift::dimensions; ++d) {
            const int partner = (d < 64) ? (d + 64) : ((d + 8) % 128);
            const bool above_median = ((descriptor.data[d >> 3] >> (d & 7)) & 1) != 0;
            const bool above_partner = ((descriptor.data[(128 + d) >> 3] >> ((128 + d) & 7)) & 1) != 0;
            REQUIRE(above_median == (values[d] > median));
            REQUIRE(above_partner == (values[d] > values[partner]));
            if (d >= 64) {
                const bool first_half = ((descriptor.data[(64 + d) >> 3] >> ((64 + d) & 7)) & 1) != 0;
                complements += (first_half != above_partner) ? 1 : 0;
            }
        }
        REQUIRE(complements < 64);
    }

    {
        // The 512-bit code is a thermometer: a dimension sets one bit per threshold it exceeds, so the Hamming distance between
        // two codes is the sum over dimensions of how many thresholds lie between their values.
        float lower[feature::descriptor::sift::dimensions];
        float upper[feature::descriptor::sift::dimensions];
        int expected_distance = 0;
        for (int d = 0; d < feature::descriptor::sift::dimensions; ++d) {
            lower[d] = 0.15f * static_cast<float>((d * 37) % 128) / 128.0f;
            upper[d] = lower[d] + ((d % 3 == 0) ? 0.04f : 0.0f);
            for (int level = 0; level < feature::descriptor::sift::thermometer_levels; ++level) {
                const float threshold = feature::descriptor::sift::thermometer_thresholds[level];
                expected_distance += ((lower[d] > threshold) != (upper[d] > threshold)) ? 1 : 0;
            }
        }
        feature::descriptor::binary<512> code_lower;
        feature::descriptor::binary<512> code_upper;
        feature::descriptor::sift::binarise(lower, code_lower);
        feature::descriptor::sift::binarise(upper, code_upper);
        for (int d = 0; d < feature::descriptor::sift::dimensions; ++d) {
            for (int level = 0; level < feature::descriptor::sift::thermometer_levels; ++level) {
                const int bit = (level * feature::descriptor::sift::dimensions) + d;
                const bool set = ((code_lower.data[bit >> 3] >> (bit & 7)) & 1) != 0;
                REQUIRE(set == (lower[d] > feature::descriptor::sift::thermometer_thresholds[level]));
            }
        }
        REQUIRE(expected_distance > 0);
        REQUIRE(static_cast<int>(match::distance::hamming::distance(code_lower, code_upper)) == expected_distance);
    }
    return EXIT_SUCCESS;
}
