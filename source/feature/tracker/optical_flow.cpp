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

#include "core/coordinates.hpp"
#include "core/thread_pool.hpp"
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

    void optical_flow::sample_table(const float centre, const float flow, const float shift, const int half_window, int* __restrict const base, float* __restrict const weight) {
        const int window_width = 2 * half_window + 1;
        for (int i = 0; i < window_width; ++i) {
            const float position = ((centre + static_cast<float>(i - half_window)) + flow) + shift;
            base[i] = static_cast<int>(math::floor(static_cast<double>(position)));
            weight[i] = position - static_cast<float>(base[i]);
        }
    }

    void optical_flow::sample_window(const unsigned char* __restrict const data, const int width, const int* __restrict const base_x, const float* __restrict const weight_x, const int* __restrict const base_y, const float* __restrict const weight_y, const int half_window, float* __restrict const window) {
        const int window_width = 2 * half_window + 1;
        float* __restrict out = window;
        for (int j = 0; j < window_width; ++j) {
            const unsigned char* __restrict const row = data + static_cast<size_t>(base_y[j]) * static_cast<size_t>(width);
            const unsigned char* __restrict const row_below = row + width;
            const float row_weight = weight_y[j];
            for (int i = 0; i < window_width; ++i) {
                const int x = base_x[i];
                const float top = static_cast<float>(row[x]) * (1.0f - weight_x[i]) + static_cast<float>(row[x + 1]) * weight_x[i];
                const float bottom = static_cast<float>(row_below[x]) * (1.0f - weight_x[i]) + static_cast<float>(row_below[x + 1]) * weight_x[i];
                out[i] = top * (1.0f - row_weight) + bottom * row_weight;
            }
            out += window_width;
        }
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
        float& result_error,
        float guess_x,
        float guess_y,
        bool damped_steps
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
        const int window_area = window_width * window_width;
        const double window_area_as_double = static_cast<double>(window_area);
        float gradient_x[maximum_window_area];
        float gradient_y[maximum_window_area];
        float previous_value[maximum_window_area];
        float next_value[maximum_window_area];
        float flow_x = guess_x / static_cast<float>(1 << start_level);
        float flow_y = guess_y / static_cast<float>(1 << start_level);
        for (int level = start_level; level >= 0; --level) {
            const image::image& previous = pyramid_previous[static_cast<size_t>(level)];
            const image::image& next = pyramid_next[static_cast<size_t>(level)];
            const float scale = 1.0f / static_cast<float>(1 << level);
            const float center_x = seed_x * scale;
            const float center_y = seed_y * scale;
            if (!optical_flow::window_in_bounds(previous, center_x, center_y, half_window)) {
                return false;
            }
            int base_x[3][maximum_window_width];
            float weight_x[3][maximum_window_width];
            int base_y[3][maximum_window_width];
            float weight_y[3][maximum_window_width];
            optical_flow::sample_table(center_x, 0.0f, 0.0f, half_window, base_x[0], weight_x[0]);
            optical_flow::sample_table(center_x, 0.0f, 1.0f, half_window, base_x[1], weight_x[1]);
            optical_flow::sample_table(center_x, 0.0f, -1.0f, half_window, base_x[2], weight_x[2]);
            optical_flow::sample_table(center_y, 0.0f, 0.0f, half_window, base_y[0], weight_y[0]);
            optical_flow::sample_table(center_y, 0.0f, 1.0f, half_window, base_y[1], weight_y[1]);
            optical_flow::sample_table(center_y, 0.0f, -1.0f, half_window, base_y[2], weight_y[2]);
            const unsigned char* __restrict const previous_data = previous.get_data();
            const int previous_width = static_cast<int>(previous.get_cols());
            optical_flow::sample_window(previous_data, previous_width, base_x[0], weight_x[0], base_y[0], weight_y[0], half_window, previous_value);
            optical_flow::sample_window(previous_data, previous_width, base_x[1], weight_x[1], base_y[0], weight_y[0], half_window, gradient_x);
            optical_flow::sample_window(previous_data, previous_width, base_x[2], weight_x[2], base_y[0], weight_y[0], half_window, next_value);
            for (int index = 0; index < window_area; ++index) {
                gradient_x[index] = 0.5f * (gradient_x[index] - next_value[index]);
            }
            optical_flow::sample_window(previous_data, previous_width, base_x[0], weight_x[0], base_y[1], weight_y[1], half_window, gradient_y);
            optical_flow::sample_window(previous_data, previous_width, base_x[0], weight_x[0], base_y[2], weight_y[2], half_window, next_value);
            double structure_xx = 0.0;
            double structure_xy = 0.0;
            double structure_yy = 0.0;
            for (int index = 0; index < window_area; ++index) {
                const float derivative_x = gradient_x[index];
                const float derivative_y = 0.5f * (gradient_y[index] - next_value[index]);
                gradient_y[index] = derivative_y;
                structure_xx += static_cast<double>(derivative_x) * static_cast<double>(derivative_x);
                structure_xy += static_cast<double>(derivative_x) * static_cast<double>(derivative_y);
                structure_yy += static_cast<double>(derivative_y) * static_cast<double>(derivative_y);
            }
            const double determinant = structure_xx * structure_yy - structure_xy * structure_xy;
            const double trace = structure_xx + structure_yy;
            const double eigen_gap = math::sqrt(math::max(0.0, trace * trace - 4.0 * determinant));
            const double smaller_eigenvalue = 0.5 * (trace - eigen_gap);
            if (!(determinant > 0.0) || (smaller_eigenvalue < min_eigenvalue * window_area_as_double)) {
                return false;
            }
            double previous_direction_x = 0.0;
            double previous_direction_y = 0.0;
            for (int iteration = 0; iteration < max_iterations; ++iteration) {
                if (!optical_flow::window_in_bounds(next, center_x + flow_x, center_y + flow_y, half_window)) {
                    return false;
                }
                optical_flow::sample_table(center_x, flow_x, 0.0f, half_window, base_x[0], weight_x[0]);
                optical_flow::sample_table(center_y, flow_y, 0.0f, half_window, base_y[0], weight_y[0]);
                optical_flow::sample_window(next.get_data(), static_cast<int>(next.get_cols()), base_x[0], weight_x[0], base_y[0], weight_y[0], half_window, next_value);
                double mismatch_x = 0.0;
                double mismatch_y = 0.0;
                for (int index = 0; index < window_area; ++index) {
                    const float difference = previous_value[index] - next_value[index];
                    mismatch_x += static_cast<double>(difference) * static_cast<double>(gradient_x[index]);
                    mismatch_y += static_cast<double>(difference) * static_cast<double>(gradient_y[index]);
                }
                const double step_x = (structure_yy * mismatch_x - structure_xy * mismatch_y) / determinant;
                const double step_y = (structure_xx * mismatch_y - structure_xy * mismatch_x) / determinant;
                double applied_x = step_x;
                double applied_y = step_y;
                if (damped_steps) {
                    const double step_length = math::sqrt((step_x * step_x) + (step_y * step_y));
                    if (step_length > 0.03) {
                        const double direction_x = step_x / step_length;
                        const double direction_y = step_y / step_length;
                        if (((direction_x * previous_direction_x) + (direction_y * previous_direction_y)) < -0.5) {
                            applied_x = 0.5 * step_x;
                            applied_y = 0.5 * step_y;
                        }
                        previous_direction_x = direction_x;
                        previous_direction_y = direction_y;
                    }
                }
                flow_x += static_cast<float>(applied_x);
                flow_y += static_cast<float>(applied_y);
                if (((applied_x * applied_x) + (applied_y * applied_y)) < (0.03 * 0.03)) {
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
        int base_x[maximum_window_width];
        float weight_x[maximum_window_width];
        int base_y[maximum_window_width];
        float weight_y[maximum_window_width];
        optical_flow::sample_table(seed_x, 0.0f, 0.0f, half_window, base_x, weight_x);
        optical_flow::sample_table(seed_y, 0.0f, 0.0f, half_window, base_y, weight_y);
        optical_flow::sample_window(previous_full.get_data(), static_cast<int>(previous_full.get_cols()), base_x, weight_x, base_y, weight_y, half_window, previous_value);
        optical_flow::sample_table(result_x, 0.0f, 0.0f, half_window, base_x, weight_x);
        optical_flow::sample_table(result_y, 0.0f, 0.0f, half_window, base_y, weight_y);
        optical_flow::sample_window(next_full.get_data(), static_cast<int>(next_full.get_cols()), base_x, weight_x, base_y, weight_y, half_window, next_value);
        double error_sum = 0.0;
        for (int index = 0; index < window_area; ++index) {
            error_sum += math::abs(static_cast<double>(previous_value[index]) - static_cast<double>(next_value[index]));
        }
        result_error = static_cast<float>(error_sum / window_area_as_double);
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
        float fb_threshold,
        const float* guess_x,
        const float* guess_y,
        bool damped_steps
    ) {
        half_window = math::max(1, math::min(half_window, maximum_half_window));
        if (max_iterations < 1) {
            max_iterations = 1;
        }
        const double min_eigenvalue_as_double = static_cast<double>(min_eigenvalue);

        core::thread_pool::instance().parallel_for(count, 16, [&](const size_t point) {
            const float seed_x = core::to_pixel_index_position(points_x[point]);
            const float seed_y = core::to_pixel_index_position(points_y[point]);
            float forward_x = seed_x;
            float forward_y = seed_y;
            float forward_error = max_error + 1.0f;
            const bool forward_ok = optical_flow::track_single(pyramid_previous, pyramid_next, seed_x, seed_y, half_window, max_iterations, min_eigenvalue_as_double, max_error, forward_x, forward_y, forward_error, (guess_x != nullptr) ? guess_x[point] : 0.0f, (guess_y != nullptr) ? guess_y[point] : 0.0f, damped_steps);
            results_out[point].x = core::to_pixel_centre(forward_x);
            results_out[point].y = core::to_pixel_centre(forward_y);
            results_out[point].error = forward_error;
            results_out[point].tracked = false;
            if (!forward_ok) {
                return;
            }
            if (forward_backward) {
                float backward_x = forward_x;
                float backward_y = forward_y;
                float backward_error = max_error + 1.0f;
                const bool backward_ok = optical_flow::track_single(pyramid_next, pyramid_previous, forward_x, forward_y, half_window, max_iterations, min_eigenvalue_as_double, max_error, backward_x, backward_y, backward_error, seed_x - forward_x, seed_y - forward_y, damped_steps);
                if (!backward_ok) {
                    return;
                }
                const float delta_x = backward_x - seed_x;
                const float delta_y = backward_y - seed_y;
                if ((delta_x * delta_x + delta_y * delta_y) > (fb_threshold * fb_threshold)) {
                    return;
                }
            }
            results_out[point].tracked = true;
        });
    }
}
