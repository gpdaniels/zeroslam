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
#ifndef ZEROSLAM_FEATURE_TRACKER_WAVELET_FLOW_HPP
#define ZEROSLAM_FEATURE_TRACKER_WAVELET_FLOW_HPP

#include "feature/tracker/patch_flow.hpp"
#include "image/pyramid.hpp"
#include "image/quaternion_wavelet.hpp"

namespace feature::tracker {
    class wavelet_flow final {
    public:
        constexpr static const int maximum_half_window = 6;
        constexpr static const int maximum_window_width = (2 * maximum_half_window) + 1;
        constexpr static const int maximum_window_area = maximum_window_width * maximum_window_width;
        constexpr static const int maximum_measurements = image::quaternion_wavelet::band_count * 2 * maximum_window_area;

        struct result final {
            float x;
            float y;
            bool tracked;
            float error;
        };

        struct options final {
            int half_window = 2;
            int coarsest_level = 6;
            int finest_level = 2;
            int iterations = 3;
            float min_modulus = 1.0f;
            double min_eigenvalue = 1.0;
            float max_error = 1.0f;
            bool forward_backward = true;
            float fb_threshold = 1.0f;
            bool measured_frequency = true;
            bool robust = false;
            bool guess_first = false;
            // With pyramids, each estimate is refined and verified in the intensity domain by a gain and bias
            // compensated patch alignment, which must converge within refine_bound pixels of the phase estimate.
            int refine_levels = 1;
            int refine_half_window = 7;
            int refine_iterations = 30;
            double refine_min_eigenvalue = 1e-3;
            float refine_max_error = 40.0f;
            float refine_bound = 4.0f;
        };

        struct anchor final {
            struct level final {
                int index = 0;
                double spacing = 1.0;
                int stride = 1;
                int node_x = 0;
                int node_y = 0;
                int radius = 0;
                double centre_x[image::quaternion_wavelet::band_count] = { 0.0, 0.0, 0.0 };
                double centre_y[image::quaternion_wavelet::band_count] = { 0.0, 0.0, 0.0 };
                std::vector<float> pairs;
                std::vector<float> frequencies;
            };

            float centre_x = 0.0f;
            float centre_y = 0.0f;
            std::vector<level> levels;
            patch_flow::anchor refine;
        };

    public:
        static bool track_single(
            const image::quaternion_wavelet& previous,
            const image::quaternion_wavelet& next,
            float seed_x,
            float seed_y,
            const options& settings,
            float guess_x,
            float guess_y,
            float& result_x,
            float& result_y,
            float& result_error
        );

        static void track(
            const image::quaternion_wavelet& previous,
            const image::quaternion_wavelet& next,
            const float* points_x,
            const float* points_y,
            size_t count,
            result* results_out,
            const options& settings,
            const float* guess_x = nullptr,
            const float* guess_y = nullptr,
            const image::pyramid* pyramid_previous = nullptr,
            const image::pyramid* pyramid_next = nullptr
        );

        static bool build_anchor(const image::quaternion_wavelet& transform, float x, float y, const options& settings, anchor& out, const image::pyramid* pyramid = nullptr);

        static bool align(const image::quaternion_wavelet& transform, const anchor& anchored, const options& settings, float& displacement_x, float& displacement_y, result& out, const image::pyramid* pyramid = nullptr);

    private:
        struct window final {
            double spacing = 1.0;
            double gate_spacing = 1.0;
            int stride = 1;
            int node_x = 0;
            int node_y = 0;
            int radius = 0;
            const double* centre_x = nullptr;
            const double* centre_y = nullptr;
            const float* pairs = nullptr;
            const float* frequencies = nullptr;
        };

        static double wrap(double angle);

        static double unwrap_towards(double angle, double target);

        static void measure(const float* const template_node, const float* const current_node, bool minus, double& difference, double& weight);

        static int fitted_radius(const image::quaternion_wavelet::level& data, int node_x, int node_y, int half_window, int margin, int stride);

        static void sample_window(const image::quaternion_wavelet::level& data, const window& shape, float* pairs, float* frequencies);

        static int level_strides(const image::quaternion_wavelet::level& data, int level, int* strides);

        static bool solve_level(const window& shape, const image::quaternion_wavelet::level& current, const options& settings, double& displacement_x, double& displacement_y, double& error, bool evaluate_only = false);

        static bool bounded_level(const window& shape, const image::quaternion_wavelet::level& current, const options& settings, bool bounded, double bound, double& displacement_x, double& displacement_y, double& error);

        static patch_flow::options refine_options(const options& settings);

        static bool refine(const patch_flow::anchor& template_patch, const image::pyramid& target, const options& settings, float estimate_x, float estimate_y, float& refined_x, float& refined_y);
    };
}

#endif // ZEROSLAM_FEATURE_TRACKER_WAVELET_FLOW_HPP
