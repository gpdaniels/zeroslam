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

#include "math/math.hpp"

namespace feature::tracker {
    float optical_flow::sample_bilinear(const image::image& source, float x, float y) {
        const int width = static_cast<int>(source.get_cols());
        const int height = static_cast<int>(source.get_rows());
        const int floor_x = static_cast<int>(math::floor(static_cast<double>(x)));
        const int floor_y = static_cast<int>(math::floor(static_cast<double>(y)));
        const int low_x = math::max(0, math::min(floor_x, width - 1));
        const int low_y = math::max(0, math::min(floor_y, height - 1));
        const int high_x = math::max(0, math::min(floor_x + 1, width - 1));
        const int high_y = math::max(0, math::min(floor_y + 1, height - 1));
        const float weight_x = math::max(0.0f, math::min(x - static_cast<float>(floor_x), 1.0f));
        const float weight_y = math::max(0.0f, math::min(y - static_cast<float>(floor_y), 1.0f));
        const unsigned char* const data = source.get_data();
        const float value_00 = static_cast<float>(data[static_cast<size_t>(low_y) * static_cast<size_t>(width) + static_cast<size_t>(low_x)]);
        const float value_01 = static_cast<float>(data[static_cast<size_t>(low_y) * static_cast<size_t>(width) + static_cast<size_t>(high_x)]);
        const float value_10 = static_cast<float>(data[static_cast<size_t>(high_y) * static_cast<size_t>(width) + static_cast<size_t>(low_x)]);
        const float value_11 = static_cast<float>(data[static_cast<size_t>(high_y) * static_cast<size_t>(width) + static_cast<size_t>(high_x)]);
        const float top = value_00 * (1.0f - weight_x) + value_01 * weight_x;
        const float bottom = value_10 * (1.0f - weight_x) + value_11 * weight_x;
        return top * (1.0f - weight_y) + bottom * weight_y;
    }

    bool optical_flow::window_in_bounds(const image::image& source, float x, float y, int half_window) {
        const float width = static_cast<float>(source.get_cols());
        const float height = static_cast<float>(source.get_rows());
        const float margin = static_cast<float>(half_window) + 1.0f;
        return (x - margin >= 0.0f) &&
               (y - margin >= 0.0f) &&
               (x + margin + 1.0f <= width - 1.0f) &&
               (y + margin + 1.0f <= height - 1.0f);
    }

    bool optical_flow::track_single(
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
        float& result_error
    ) {
        half_window = math::max(1, math::min(half_window, maximum_half_window));

        result_x = seed_x;
        result_y = seed_y;
        result_error = max_error + 1.0f;

        const int levels = static_cast<int>(math::min(pyramid_previous.size(), pyramid_next.size()));
        if (levels < 1) {
            return false;
        }

        int start_level = -1;
        for (int level = levels - 1; level >= 0; --level) {
            const float scale = 1.0f / static_cast<float>(1 << level);
            if (optical_flow::window_in_bounds(pyramid_previous[static_cast<size_t>(level)], seed_x * scale, seed_y * scale, half_window)) {
                start_level = level;
                break;
            }
        }
        if (start_level < 0) {
            return false;
        }

        const int window_width = 2 * half_window + 1;
        const double window_area = static_cast<double>(window_width) * static_cast<double>(window_width);

        float gradient_x[maximum_window_area];
        float gradient_y[maximum_window_area];
        float previous_value[maximum_window_area];

        float flow_x = 0.0f;
        float flow_y = 0.0f;

        for (int level = start_level; level >= 0; --level) {
            const image::image& previous = pyramid_previous[static_cast<size_t>(level)];
            const image::image& next = pyramid_next[static_cast<size_t>(level)];
            const float scale = 1.0f / static_cast<float>(1 << level);
            const float center_x = seed_x * scale;
            const float center_y = seed_y * scale;

            if (!optical_flow::window_in_bounds(previous, center_x, center_y, half_window)) {
                return false;
            }

            double structure_xx = 0.0;
            double structure_xy = 0.0;
            double structure_yy = 0.0;
            int index = 0;
            for (int offset_y = -half_window; offset_y <= half_window; ++offset_y) {
                for (int offset_x = -half_window; offset_x <= half_window; ++offset_x) {
                    const float sample_x = center_x + static_cast<float>(offset_x);
                    const float sample_y = center_y + static_cast<float>(offset_y);
                    const float derivative_x = 0.5f * (optical_flow::sample_bilinear(previous, sample_x + 1.0f, sample_y) - optical_flow::sample_bilinear(previous, sample_x - 1.0f, sample_y));
                    const float derivative_y = 0.5f * (optical_flow::sample_bilinear(previous, sample_x, sample_y + 1.0f) - optical_flow::sample_bilinear(previous, sample_x, sample_y - 1.0f));
                    gradient_x[index] = derivative_x;
                    gradient_y[index] = derivative_y;
                    previous_value[index] = optical_flow::sample_bilinear(previous, sample_x, sample_y);
                    structure_xx += static_cast<double>(derivative_x) * static_cast<double>(derivative_x);
                    structure_xy += static_cast<double>(derivative_x) * static_cast<double>(derivative_y);
                    structure_yy += static_cast<double>(derivative_y) * static_cast<double>(derivative_y);
                    ++index;
                }
            }

            const double determinant = structure_xx * structure_yy - structure_xy * structure_xy;
            const double trace = structure_xx + structure_yy;
            const double eigen_gap = math::sqrt(math::max(0.0, trace * trace - 4.0 * determinant));
            const double smaller_eigenvalue = 0.5 * (trace - eigen_gap);
            if (!(determinant > 0.0) || (smaller_eigenvalue < min_eigenvalue * window_area)) {
                return false;
            }

            for (int iteration = 0; iteration < max_iterations; ++iteration) {
                if (!optical_flow::window_in_bounds(next, center_x + flow_x, center_y + flow_y, half_window)) {
                    return false;
                }
                double mismatch_x = 0.0;
                double mismatch_y = 0.0;
                index = 0;
                for (int offset_y = -half_window; offset_y <= half_window; ++offset_y) {
                    for (int offset_x = -half_window; offset_x <= half_window; ++offset_x) {
                        const float sample_x = center_x + static_cast<float>(offset_x);
                        const float sample_y = center_y + static_cast<float>(offset_y);
                        const float difference = previous_value[index] - optical_flow::sample_bilinear(next, sample_x + flow_x, sample_y + flow_y);
                        mismatch_x += static_cast<double>(difference) * static_cast<double>(gradient_x[index]);
                        mismatch_y += static_cast<double>(difference) * static_cast<double>(gradient_y[index]);
                        ++index;
                    }
                }
                const double step_x = (structure_yy * mismatch_x - structure_xy * mismatch_y) / determinant;
                const double step_y = (structure_xx * mismatch_y - structure_xy * mismatch_x) / determinant;
                flow_x += static_cast<float>(step_x);
                flow_y += static_cast<float>(step_y);
                if ((step_x * step_x + step_y * step_y) < (0.03 * 0.03)) {
                    break;
                }
            }

            if (level > 0) {
                flow_x *= 2.0f;
                flow_y *= 2.0f;
            }
        }

        result_x = seed_x + flow_x;
        result_y = seed_y + flow_y;

        const image::image& previous_full = pyramid_previous[0];
        const image::image& next_full = pyramid_next[0];
        if (!optical_flow::window_in_bounds(next_full, result_x, result_y, half_window)) {
            return false;
        }
        double error_sum = 0.0;
        int error_count = 0;
        for (int offset_y = -half_window; offset_y <= half_window; ++offset_y) {
            for (int offset_x = -half_window; offset_x <= half_window; ++offset_x) {
                const float previous_sample = optical_flow::sample_bilinear(previous_full, seed_x + static_cast<float>(offset_x), seed_y + static_cast<float>(offset_y));
                const float next_sample = optical_flow::sample_bilinear(next_full, result_x + static_cast<float>(offset_x), result_y + static_cast<float>(offset_y));
                error_sum += math::abs(static_cast<double>(previous_sample) - static_cast<double>(next_sample));
                ++error_count;
            }
        }
        result_error = static_cast<float>(error_sum / static_cast<double>(error_count));
        return result_error <= max_error;
    }

    void optical_flow::track(
        const image::pyramid& pyramid_previous,
        const image::pyramid& pyramid_next,
        const float* points_x,
        const float* points_y,
        size_t count,
        result* results_out,
        int half_window,
        int max_iterations,
        float min_eigenvalue,
        float max_error,
        bool forward_backward,
        float fb_threshold
    ) {
        half_window = math::max(1, math::min(half_window, maximum_half_window));
        if (max_iterations < 1) {
            max_iterations = 1;
        }
        const double min_eigenvalue_as_double = static_cast<double>(min_eigenvalue);

        for (size_t point = 0; point < count; ++point) {
            const float seed_x = points_x[point];
            const float seed_y = points_y[point];

            float forward_x = seed_x;
            float forward_y = seed_y;
            float forward_error = max_error + 1.0f;
            const bool forward_ok = optical_flow::track_single(pyramid_previous, pyramid_next, seed_x, seed_y, half_window, max_iterations, min_eigenvalue_as_double, max_error, forward_x, forward_y, forward_error);

            results_out[point].x = forward_x;
            results_out[point].y = forward_y;
            results_out[point].error = forward_error;
            results_out[point].tracked = false;

            if (!forward_ok) {
                continue;
            }

            if (forward_backward) {
                float backward_x = forward_x;
                float backward_y = forward_y;
                float backward_error = max_error + 1.0f;
                const bool backward_ok = optical_flow::track_single(pyramid_next, pyramid_previous, forward_x, forward_y, half_window, max_iterations, min_eigenvalue_as_double, max_error, backward_x, backward_y, backward_error);
                if (!backward_ok) {
                    continue;
                }
                const float delta_x = backward_x - seed_x;
                const float delta_y = backward_y - seed_y;
                if ((delta_x * delta_x + delta_y * delta_y) > (fb_threshold * fb_threshold)) {
                    continue;
                }
            }

            results_out[point].tracked = true;
        }
    }
}
