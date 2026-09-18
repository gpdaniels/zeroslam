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

#pragma once
#ifndef ZEROSLAM_IMAGE_QUATERNION_WAVELET_HPP
#define ZEROSLAM_IMAGE_QUATERNION_WAVELET_HPP

#include "image/image.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace image {
    class quaternion_wavelet final {
    public:
        enum class band {
            horizontal,
            vertical,
            diagonal
        };

        constexpr static const int band_count = 3;
        constexpr static const int component_count = 4;
        constexpr static const size_t minimum_dimension = 6;
        constexpr static const int filter_length = 10;

        struct level final {
            size_t rows = 0;
            size_t cols = 0;
            double spacing = 1.0;
            std::vector<float> bands[band_count];
            double wavelet_centre = 0.0;
            double scaling_centre = 0.0;
        };

    private:
        std::vector<level> levels;

    public:
        quaternion_wavelet();

        explicit quaternion_wavelet(const image& base, size_t level_count = 0, size_t finest_band = 2, bool undecimated_finest = false);

        size_t size() const;

        const level& operator[](const size_t index) const;

        const float* at(const size_t index, const band which, const size_t x, const size_t y) const;

        static double centre_x(const level& data, const band which);
        static double centre_y(const level& data, const band which);

        static void complex_pair(const float* const components, float& plus_real, float& plus_imaginary, float& minus_real, float& minus_imaginary);

        static float modulus(const float* const components);

        static bool phases(const float* const components, double& first, double& second, double& third);

        static const double* first_lowpass_a();
        static const double* first_lowpass_b();
        static const double* later_lowpass_a();
        static const double* later_lowpass_b();
        static void quadrature_mirror(const double* const lowpass, double* const highpass);

        constexpr static const size_t maximum_levels = 12;

        static void spectral_centres(const size_t level_count, double* const wavelet_out, double* const scaling_out);
    };
}

#endif // ZEROSLAM_IMAGE_QUATERNION_WAVELET_HPP
