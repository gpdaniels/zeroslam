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
#ifndef ZEROSLAM_FEATURE_TRACKER_OPTICAL_FLOW_HPP
#define ZEROSLAM_FEATURE_TRACKER_OPTICAL_FLOW_HPP

#include "image/image.hpp"
#include "image/pyramid.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::tracker {
    class optical_flow final {
    public:
        constexpr static const int maximum_half_window = 15;
        constexpr static const int maximum_window_width = 2 * maximum_half_window + 1;
        constexpr static const int maximum_window_area = maximum_window_width * maximum_window_width;
        constexpr static const int maximum_padded_width = maximum_window_width + 2;
        constexpr static const int maximum_padded_area = maximum_padded_width * maximum_padded_width;
        // A weak coarse level is skipped with its flow propagated. The finer levels may then hold only aliased residue of a
        // fine texture, whose estimate lands on a wrong minimum with an error comparable to the template's own contrast,
        // so such a track must end with a mean absolute error below this fraction of the template's mean absolute
        // deviation.
        constexpr static const double skipped_level_relative_error = 0.5;

        // With continue_outside a coarse level whose iteration leaves the image restarts the next level from its entry
        // flow instead of rejecting the track. It is off by default: the tracks it keeps lie in a band about 64-130 px
        // from the image border, and on undistorted fisheye frames (LaMAria R_02_easy) they raised the median
        // trajectory error over five runs from 0.46 m to 2.57 m.

        struct result final {
            float x;
            float y;
            bool tracked;
            float error;
        };

    public:
        static float sample_bilinear(const image::image& source, float x, float y);

        static bool window_in_bounds(const image::image& source, float x, float y, int half_window);
        static void sample_table(float centre, float flow, float shift, int half_window, int* __restrict const base, float* __restrict const weight);
        static void sample_window(const unsigned char* __restrict const data, int width, const int* __restrict const base_x, const float* __restrict const weight_x, const int* __restrict const base_y, const float* __restrict const weight_y, int half_window, float* __restrict const window);

        // Samples the window and its central difference gradients from one window padded by a pixel on each side.
        static void sample_gradients(const image::image& source, float centre_x, float centre_y, int half_window, float* __restrict const values, float* __restrict const gradients_x, float* __restrict const gradients_y);

        static bool track_single(
            const image::pyramid& pyramid_previous,
            const image::pyramid& pyramid_next,
            float seed_x,
            float seed_y,
            int half_window,
            int max_iterations,
            double min_eigenvalue,
            float max_error,
            float& result_x,
            float& result_y,
            float& result_error,
            float guess_x = 0.0f,
            float guess_y = 0.0f,
            bool damped_steps = false,
            bool continue_outside = false
        );

        static void track(
            const image::pyramid& pyramid_previous,
            const image::pyramid& pyramid_next,
            const float* points_x,
            const float* points_y,
            size_t count,
            result* results_out,
            int half_window = 7,
            int max_iterations = 30,
            float min_eigenvalue = 1e-3f,
            float max_error = 10000.0f,
            bool forward_backward = true,
            float fb_threshold = 1.0f,
            const float* guess_x = nullptr,
            const float* guess_y = nullptr,
            bool damped_steps = false,
            bool continue_outside = false
        );
    };
}

#endif // ZEROSLAM_FEATURE_TRACKER_OPTICAL_FLOW_HPP
