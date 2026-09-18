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

#include "core/coordinates.hpp"
#include "core/sort.hpp"
#include "core/thread_pool.hpp"
#include "math/math.hpp"

namespace feature::tracker {
    double wavelet_flow::wrap(double angle) {
        const double turn = 2.0 * math::pi<double>();
        while (angle >= math::pi<double>()) {
            angle -= turn;
        }
        while (angle < -math::pi<double>()) {
            angle += turn;
        }
        return angle;
    }

    double wavelet_flow::unwrap_towards(const double angle, const double target) {
        const double turn = 2.0 * math::pi<double>();
        return angle + (turn * static_cast<double>(math::round((target - angle) / turn)));
    }

    void wavelet_flow::measure(const float* const template_node, const float* const current_node, const bool minus, double& difference, double& weight) {
        float template_plus_real = 0.0f;
        float template_plus_imaginary = 0.0f;
        float template_minus_real = 0.0f;
        float template_minus_imaginary = 0.0f;
        image::quaternion_wavelet::complex_pair(template_node, template_plus_real, template_plus_imaginary, template_minus_real, template_minus_imaginary);
        float current_plus_real = 0.0f;
        float current_plus_imaginary = 0.0f;
        float current_minus_real = 0.0f;
        float current_minus_imaginary = 0.0f;
        image::quaternion_wavelet::complex_pair(current_node, current_plus_real, current_plus_imaginary, current_minus_real, current_minus_imaginary);
        const double template_real = static_cast<double>(minus ? template_minus_real : template_plus_real);
        const double template_imaginary = static_cast<double>(minus ? template_minus_imaginary : template_plus_imaginary);
        const double current_real = static_cast<double>(minus ? current_minus_real : current_plus_real);
        const double current_imaginary = static_cast<double>(minus ? current_minus_imaginary : current_plus_imaginary);
        const double real = (current_real * template_real) + (current_imaginary * template_imaginary);
        const double imaginary = (current_imaginary * template_real) - (current_real * template_imaginary);
        difference = ((real == 0.0) && (imaginary == 0.0)) ? 0.0 : math::atan2(imaginary, real);
        const double template_power = (template_real * template_real) + (template_imaginary * template_imaginary);
        const double current_power = (current_real * current_real) + (current_imaginary * current_imaginary);
        weight = math::sqrt(template_power * current_power);
    }

    int wavelet_flow::fitted_radius(const image::quaternion_wavelet::level& data, const int node_x, const int node_y, const int half_window, const int margin, const int stride) {
        const int columns = static_cast<int>(data.cols);
        const int rows = static_cast<int>(data.rows);
        int radius = half_window;
        radius = math::min(radius, (node_x - margin) / stride);
        radius = math::min(radius, (node_y - margin) / stride);
        radius = math::min(radius, ((columns - 1 - node_x) - margin) / stride);
        radius = math::min(radius, ((rows - 1 - node_y) - margin) / stride);
        return radius;
    }

    int wavelet_flow::level_strides(const image::quaternion_wavelet::level& data, const int level, int* const strides) {
        const int dense = static_cast<int>(math::round(static_cast<double>(1 << level) / data.spacing));
        int count = 0;
        if (dense > 1) {
            strides[count++] = dense;
        }
        strides[count++] = 1;
        return count;
    }

    void wavelet_flow::sample_window(const image::quaternion_wavelet::level& data, const window& shape, float* const pairs, float* const frequencies) {
        const double turn = 2.0 * math::pi<double>();
        const int columns = static_cast<int>(data.cols);
        const int width = (2 * shape.radius) + 1;
        const int area = width * width;
        for (int which = 0; which < image::quaternion_wavelet::band_count; ++which) {
            const std::vector<float>& band_values = data.bands[which];
            const double centre_x = shape.centre_x[which];
            const double centre_y = shape.centre_y[which];
            float* const band_pairs = pairs + (which * area * 4);
            float* const band_frequencies = (frequencies != nullptr) ? (frequencies + (which * area * 4)) : nullptr;
            for (int down = -shape.radius; down <= shape.radius; ++down) {
                for (int across = -shape.radius; across <= shape.radius; ++across) {
                    const int slot = (((down + shape.radius) * width) + (across + shape.radius)) * 4;
                    const size_t node = static_cast<size_t>(((shape.node_y + (shape.stride * down)) * columns) + (shape.node_x + (shape.stride * across))) * image::quaternion_wavelet::component_count;
                    image::quaternion_wavelet::complex_pair(&band_values[node], band_pairs[slot + 0], band_pairs[slot + 1], band_pairs[slot + 2], band_pairs[slot + 3]);
                    if ((band_frequencies == nullptr) || (shape.stride != 1)) {
                        continue;
                    }
                    const size_t step_x = image::quaternion_wavelet::component_count;
                    const size_t step_y = static_cast<size_t>(columns) * image::quaternion_wavelet::component_count;
                    for (int sign = 0; sign < 2; ++sign) {
                        const bool minus = (sign == 1);
                        const double signed_centre_y = minus ? -centre_y : centre_y;
                        double forward = 0.0;
                        double backward = 0.0;
                        double ignored = 0.0;
                        wavelet_flow::measure(&band_values[node], &band_values[node + step_x], minus, forward, ignored);
                        wavelet_flow::measure(&band_values[node - step_x], &band_values[node], minus, backward, ignored);
                        band_frequencies[slot + (sign * 2) + 0] = static_cast<float>(0.5 * (wavelet_flow::unwrap_towards(forward, turn * centre_x) + wavelet_flow::unwrap_towards(backward, turn * centre_x)) / turn);
                        wavelet_flow::measure(&band_values[node], &band_values[node + step_y], minus, forward, ignored);
                        wavelet_flow::measure(&band_values[node - step_y], &band_values[node], minus, backward, ignored);
                        band_frequencies[slot + (sign * 2) + 1] = static_cast<float>(0.5 * (wavelet_flow::unwrap_towards(forward, turn * signed_centre_y) + wavelet_flow::unwrap_towards(backward, turn * signed_centre_y)) / turn);
                    }
                }
            }
        }
    }

    bool wavelet_flow::solve_level(const window& shape, const image::quaternion_wavelet::level& current, const options& settings, double& displacement_x, double& displacement_y, double& error) {
        const double turn = 2.0 * math::pi<double>();
        const double spacing = shape.spacing;
        const int radius = shape.radius;
        const int width = (2 * radius) + 1;
        const int area = width * width;
        const int columns = static_cast<int>(current.cols);
        const int rows = static_cast<int>(current.rows);
        const double floor_power = math::max(0.0, static_cast<double>(settings.min_modulus)) * math::max(0.0, static_cast<double>(settings.min_modulus));
        float delta[wavelet_flow::maximum_measurements];
        float jacobian_x[wavelet_flow::maximum_measurements];
        float jacobian_y[wavelet_flow::maximum_measurements];
        float weight[wavelet_flow::maximum_measurements];
        int count = 0;
        int measured_x = 0;
        int measured_y = 0;
        const auto measure_window = [&](const int offset_x, const int offset_y) {
            count = 0;
            measured_x = offset_x;
            measured_y = offset_y;
            const int current_x = shape.node_x + (shape.stride * offset_x);
            const int current_y = shape.node_y + (shape.stride * offset_y);
            const int reach = shape.stride * radius;
            if ((current_x - reach < 0) || (current_y - reach < 0) || (current_x + reach >= columns) || (current_y + reach >= rows)) {
                return false;
            }
            for (int which = 0; which < image::quaternion_wavelet::band_count; ++which) {
                const double centre_x = shape.centre_x[which];
                const double centre_y = shape.centre_y[which];
                const std::vector<float>& current_band = current.bands[which];
                const float* const pairs = shape.pairs + (which * area * 4);
                const float* const frequencies = (shape.frequencies != nullptr) ? (shape.frequencies + (which * area * 4)) : nullptr;
                for (int down = -radius; down <= radius; ++down) {
                    for (int across = -radius; across <= radius; ++across) {
                        const int slot = (((down + radius) * width) + (across + radius)) * 4;
                        const size_t node = static_cast<size_t>(((current_y + (shape.stride * down)) * columns) + (current_x + (shape.stride * across))) * image::quaternion_wavelet::component_count;
                        float current_pair[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                        image::quaternion_wavelet::complex_pair(&current_band[node], current_pair[0], current_pair[1], current_pair[2], current_pair[3]);
                        for (int sign = 0; sign < 2; ++sign) {
                            const bool minus = (sign == 1);
                            const double template_real = static_cast<double>(pairs[slot + (sign * 2) + 0]);
                            const double template_imaginary = static_cast<double>(pairs[slot + (sign * 2) + 1]);
                            const double current_real = static_cast<double>(current_pair[sign * 2]);
                            const double current_imaginary = static_cast<double>(current_pair[(sign * 2) + 1]);
                            const double template_power = (template_real * template_real) + (template_imaginary * template_imaginary);
                            const double current_power = (current_real * current_real) + (current_imaginary * current_imaginary);
                            if ((template_power < floor_power) || (current_power < floor_power)) {
                                continue;
                            }
                            const double real = (current_real * template_real) + (current_imaginary * template_imaginary);
                            const double imaginary = (current_imaginary * template_real) - (current_real * template_imaginary);
                            const double frequency_x = (frequencies != nullptr) ? static_cast<double>(frequencies[slot + (sign * 2) + 0]) : centre_x;
                            const double frequency_y = (frequencies != nullptr) ? static_cast<double>(frequencies[slot + (sign * 2) + 1]) : (minus ? -centre_y : centre_y);
                            delta[count] = static_cast<float>(math::atan2(imaginary, real));
                            jacobian_x[count] = static_cast<float>(-turn * frequency_x / spacing);
                            jacobian_y[count] = static_cast<float>(-turn * frequency_y / spacing);
                            weight[count] = static_cast<float>(math::sqrt(template_power * current_power));
                            ++count;
                        }
                    }
                }
            }
            return count > 0;
        };
        const auto solve = [&](const int offset_x, const int offset_y, const double knee, double& step_x, double& step_y, double& rms) {
            const double remainder_x = displacement_x - (spacing * static_cast<double>(offset_x));
            const double remainder_y = displacement_y - (spacing * static_cast<double>(offset_y));
            double normal_xx = 0.0;
            double normal_xy = 0.0;
            double normal_yy = 0.0;
            double gradient_x = 0.0;
            double gradient_y = 0.0;
            double weight_sum = 0.0;
            double plain_sum = 0.0;
            double residual_sum = 0.0;
            for (int index = 0; index < count; ++index) {
                const double jx = static_cast<double>(jacobian_x[index]);
                const double jy = static_cast<double>(jacobian_y[index]);
                const double residual = wavelet_flow::wrap(static_cast<double>(delta[index]) - ((jx * remainder_x) + (jy * remainder_y)));
                const double plain = static_cast<double>(weight[index]);
                const double magnitude = math::abs(residual);
                const double robust = ((knee > 0.0) && (magnitude > knee)) ? (plain * knee / magnitude) : plain;
                normal_xx += robust * jx * jx;
                normal_xy += robust * jx * jy;
                normal_yy += robust * jy * jy;
                gradient_x += robust * jx * residual;
                gradient_y += robust * jy * residual;
                weight_sum += robust;
                plain_sum += plain;
                residual_sum += plain * residual * residual;
            }
            if (!(plain_sum > 0.0) || !(weight_sum > 0.0)) {
                return false;
            }
            rms = math::sqrt(residual_sum / plain_sum);
            const double determinant = (normal_xx * normal_yy) - (normal_xy * normal_xy);
            const double trace = normal_xx + normal_yy;
            if (!(determinant > 0.0)) {
                return false;
            }
            const double eigen_gap = math::sqrt(math::max(0.0, (trace * trace) - (4.0 * determinant)));
            const double smaller_eigenvalue = 0.5 * (trace - eigen_gap);
            if ((smaller_eigenvalue * shape.gate_spacing * shape.gate_spacing) < (settings.min_eigenvalue * weight_sum)) {
                return false;
            }
            step_x = ((normal_yy * gradient_x) - (normal_xy * gradient_y)) / determinant;
            step_y = ((normal_xx * gradient_y) - (normal_xy * gradient_x)) / determinant;
            return math::isfinite(step_x) && math::isfinite(step_y);
        };
        int settled_x = 0;
        int settled_y = 0;
        bool settled = false;
        double step_x = 0.0;
        double step_y = 0.0;
        double rms = 0.0;
        for (int iteration = 0; iteration < math::max(1, settings.iterations); ++iteration) {
            const int offset_x = static_cast<int>(math::round(displacement_x / spacing));
            const int offset_y = static_cast<int>(math::round(displacement_y / spacing));
            if (!measure_window(offset_x, offset_y) || !solve(offset_x, offset_y, 0.0, step_x, step_y, rms)) {
                break;
            }
            displacement_x += step_x;
            displacement_y += step_y;
            settled_x = offset_x;
            settled_y = offset_y;
            settled = true;
            if ((static_cast<int>(math::round(displacement_x / spacing)) == offset_x) && (static_cast<int>(math::round(displacement_y / spacing)) == offset_y)) {
                break;
            }
        }
        if (!settled) {
            return false;
        }
        if ((measured_x != settled_x) || (measured_y != settled_y) || (count == 0)) {
            if (!measure_window(settled_x, settled_y)) {
                return false;
            }
        }
        if (settings.robust) {
            for (int pass = 0; pass < 2; ++pass) {
                const double remainder_x = displacement_x - (spacing * static_cast<double>(settled_x));
                const double remainder_y = displacement_y - (spacing * static_cast<double>(settled_y));
                float magnitude[wavelet_flow::maximum_measurements] = {};
                int order[wavelet_flow::maximum_measurements] = {};
                double total = 0.0;
                for (int index = 0; index < count; ++index) {
                    magnitude[index] = static_cast<float>(math::abs(wavelet_flow::wrap(static_cast<double>(delta[index]) - ((static_cast<double>(jacobian_x[index]) * remainder_x) + (static_cast<double>(jacobian_y[index]) * remainder_y)))));
                    order[index] = index;
                    total += static_cast<double>(weight[index]);
                }
                core::sort::heap(&order[0], static_cast<size_t>(count), [&magnitude](const int lhs, const int rhs) {
                    return magnitude[lhs] < magnitude[rhs];
                });
                double running = 0.0;
                double median = static_cast<double>(magnitude[order[count - 1]]);
                for (int index = 0; index < count; ++index) {
                    running += static_cast<double>(weight[order[index]]);
                    if (running >= 0.5 * total) {
                        median = static_cast<double>(magnitude[order[index]]);
                        break;
                    }
                }
                const double knee = math::max(2.5 * median, 0.05);
                if (!solve(settled_x, settled_y, knee, step_x, step_y, rms)) {
                    break;
                }
                displacement_x += step_x;
                displacement_y += step_y;
            }
        }
        if (solve(settled_x, settled_y, 0.0, step_x, step_y, rms)) {
            error = rms;
        }
        return true;
    }

    bool wavelet_flow::track_single(
        const image::quaternion_wavelet& previous,
        const image::quaternion_wavelet& next,
        const float seed_x,
        const float seed_y,
        const options& settings,
        const float guess_x,
        const float guess_y,
        float& result_x,
        float& result_y,
        float& result_error
    ) {
        result_x = seed_x + guess_x;
        result_y = seed_y + guess_y;
        result_error = settings.max_error + 1.0f;
        const int half_window = math::max(1, math::min(settings.half_window, wavelet_flow::maximum_half_window));
        const int depth = static_cast<int>(math::min(previous.size(), next.size()));
        const int finest = math::max(2, settings.finest_level);
        const int coarsest = math::min(depth, settings.coarsest_level);
        if (coarsest < finest) {
            return false;
        }
        double displacement_x = static_cast<double>(guess_x);
        double displacement_y = static_cast<double>(guess_y);
        bool solved = false;
        double error = static_cast<double>(settings.max_error) + 1.0;
        float pairs[image::quaternion_wavelet::band_count * wavelet_flow::maximum_window_area * 4];
        float frequencies[image::quaternion_wavelet::band_count * wavelet_flow::maximum_window_area * 4];
        double centres_x[image::quaternion_wavelet::band_count];
        double centres_y[image::quaternion_wavelet::band_count];
        for (int level = coarsest; level >= finest; --level) {
            const image::quaternion_wavelet::level& template_level = previous[static_cast<size_t>(level)];
            const image::quaternion_wavelet::level& current_level = next[static_cast<size_t>(level)];
            if ((template_level.rows == 0) || (template_level.rows != current_level.rows) || (template_level.cols != current_level.cols) || (template_level.spacing != current_level.spacing)) {
                continue;
            }
            int strides[2] = { 1, 1 };
            const int passes = wavelet_flow::level_strides(template_level, level, &strides[0]);
            for (int pass = 0; pass < passes; ++pass) {
                window shape;
                shape.stride = strides[pass];
                shape.spacing = template_level.spacing * static_cast<double>(shape.stride);
                shape.gate_spacing = static_cast<double>(1 << level);
                shape.node_x = static_cast<int>(math::floor(static_cast<double>(seed_x) / template_level.spacing));
                shape.node_y = static_cast<int>(math::floor(static_cast<double>(seed_y) / template_level.spacing));
                const bool last_pass = (level == finest) && (pass == passes - 1);
                const bool measure_frequency = settings.measured_frequency && last_pass && (shape.stride == 1);
                const int dense_factor = static_cast<int>(math::round(static_cast<double>(1 << level) / shape.spacing));
                shape.radius = wavelet_flow::fitted_radius(template_level, shape.node_x, shape.node_y, math::min(half_window * dense_factor, wavelet_flow::maximum_half_window), measure_frequency ? 1 : 0, shape.stride);
                if (shape.radius < 1) {
                    continue;
                }
                for (int which = 0; which < image::quaternion_wavelet::band_count; ++which) {
                    centres_x[which] = image::quaternion_wavelet::centre_x(template_level, static_cast<image::quaternion_wavelet::band>(which)) * static_cast<double>(shape.stride);
                    centres_y[which] = image::quaternion_wavelet::centre_y(template_level, static_cast<image::quaternion_wavelet::band>(which)) * static_cast<double>(shape.stride);
                }
                shape.centre_x = &centres_x[0];
                shape.centre_y = &centres_y[0];
                shape.pairs = &pairs[0];
                shape.frequencies = measure_frequency ? &frequencies[0] : nullptr;
                wavelet_flow::sample_window(template_level, shape, &pairs[0], measure_frequency ? &frequencies[0] : nullptr);
                if (wavelet_flow::solve_level(shape, current_level, settings, displacement_x, displacement_y, error)) {
                    solved = true;
                }
            }
        }
        result_x = seed_x + static_cast<float>(displacement_x);
        result_y = seed_y + static_cast<float>(displacement_y);
        if (!solved) {
            return false;
        }
        result_error = static_cast<float>(error);
        return result_error <= settings.max_error;
    }

    void wavelet_flow::track(
        const image::quaternion_wavelet& previous,
        const image::quaternion_wavelet& next,
        const float* const points_x,
        const float* const points_y,
        const size_t count,
        result* const results_out,
        const options& settings,
        const float* const guess_x,
        const float* const guess_y
    ) {
        core::thread_pool::instance().parallel_for(count, 16, [&](const size_t point) {
            const float seed_x = core::to_pixel_index_position(points_x[point]);
            const float seed_y = core::to_pixel_index_position(points_y[point]);
            const bool has_guess = (guess_x != nullptr) && (guess_y != nullptr) && ((guess_x[point] != 0.0f) || (guess_y[point] != 0.0f));
            const float first_x = (settings.guess_first && has_guess) ? guess_x[point] : 0.0f;
            const float first_y = (settings.guess_first && has_guess) ? guess_y[point] : 0.0f;
            float forward_x = seed_x;
            float forward_y = seed_y;
            float forward_error = settings.max_error + 1.0f;
            bool forward_ok = wavelet_flow::track_single(previous, next, seed_x, seed_y, settings, first_x, first_y, forward_x, forward_y, forward_error);
            if (!forward_ok && has_guess) {
                const float second_x = settings.guess_first ? 0.0f : guess_x[point];
                const float second_y = settings.guess_first ? 0.0f : guess_y[point];
                float retry_x = seed_x;
                float retry_y = seed_y;
                float retry_error = settings.max_error + 1.0f;
                if (wavelet_flow::track_single(previous, next, seed_x, seed_y, settings, second_x, second_y, retry_x, retry_y, retry_error)) {
                    forward_x = retry_x;
                    forward_y = retry_y;
                    forward_error = retry_error;
                    forward_ok = true;
                }
            }
            results_out[point].x = core::to_pixel_centre(forward_x);
            results_out[point].y = core::to_pixel_centre(forward_y);
            results_out[point].error = forward_error;
            results_out[point].tracked = false;
            if (!forward_ok) {
                return;
            }
            if (settings.forward_backward) {
                float backward_x = forward_x;
                float backward_y = forward_y;
                float backward_error = settings.max_error + 1.0f;
                const bool backward_ok = wavelet_flow::track_single(next, previous, forward_x, forward_y, settings, seed_x - forward_x, seed_y - forward_y, backward_x, backward_y, backward_error);
                if (!backward_ok) {
                    return;
                }
                const float drift_x = backward_x - seed_x;
                const float drift_y = backward_y - seed_y;
                if (((drift_x * drift_x) + (drift_y * drift_y)) > (settings.fb_threshold * settings.fb_threshold)) {
                    return;
                }
            }
            results_out[point].tracked = true;
        });
    }

    bool wavelet_flow::build_anchor(const image::quaternion_wavelet& transform, const float x, const float y, const options& settings, anchor& out) {
        out.levels.clear();
        out.centre_x = core::to_pixel_index_position(x);
        out.centre_y = core::to_pixel_index_position(y);
        const int half_window = math::max(1, math::min(settings.half_window, wavelet_flow::maximum_half_window));
        const int finest = math::max(2, settings.finest_level);
        const int coarsest = math::min(static_cast<int>(transform.size()), settings.coarsest_level);
        for (int level = coarsest; level >= finest; --level) {
            const image::quaternion_wavelet::level& data = transform[static_cast<size_t>(level)];
            if (data.rows == 0) {
                continue;
            }
            int strides[2] = { 1, 1 };
            const int passes = wavelet_flow::level_strides(data, level, &strides[0]);
            for (int pass = 0; pass < passes; ++pass) {
                anchor::level stored;
                stored.index = level;
                stored.stride = strides[pass];
                stored.spacing = data.spacing * static_cast<double>(stored.stride);
                stored.node_x = static_cast<int>(math::floor(static_cast<double>(out.centre_x) / data.spacing));
                stored.node_y = static_cast<int>(math::floor(static_cast<double>(out.centre_y) / data.spacing));
                const bool last_pass = (level == finest) && (pass == passes - 1);
                const bool measure_frequency = settings.measured_frequency && last_pass && (stored.stride == 1);
                const int dense_factor = static_cast<int>(math::round(static_cast<double>(1 << level) / stored.spacing));
                stored.radius = wavelet_flow::fitted_radius(data, stored.node_x, stored.node_y, math::min(half_window * dense_factor, wavelet_flow::maximum_half_window), measure_frequency ? 1 : 0, stored.stride);
                if (stored.radius < 1) {
                    continue;
                }
                for (int which = 0; which < image::quaternion_wavelet::band_count; ++which) {
                    stored.centre_x[which] = image::quaternion_wavelet::centre_x(data, static_cast<image::quaternion_wavelet::band>(which)) * static_cast<double>(stored.stride);
                    stored.centre_y[which] = image::quaternion_wavelet::centre_y(data, static_cast<image::quaternion_wavelet::band>(which)) * static_cast<double>(stored.stride);
                }
                const int width = (2 * stored.radius) + 1;
                const size_t floats = static_cast<size_t>(image::quaternion_wavelet::band_count) * static_cast<size_t>(width * width) * 4u;
                stored.pairs.assign(floats, 0.0f);
                if (measure_frequency) {
                    stored.frequencies.assign(floats, 0.0f);
                }
                window shape;
                shape.stride = stored.stride;
                shape.spacing = stored.spacing;
                shape.gate_spacing = static_cast<double>(1 << level);
                shape.node_x = stored.node_x;
                shape.node_y = stored.node_y;
                shape.radius = stored.radius;
                shape.centre_x = &stored.centre_x[0];
                shape.centre_y = &stored.centre_y[0];
                shape.pairs = stored.pairs.data();
                shape.frequencies = measure_frequency ? stored.frequencies.data() : nullptr;
                wavelet_flow::sample_window(data, shape, stored.pairs.data(), measure_frequency ? stored.frequencies.data() : nullptr);
                out.levels.push_back(static_cast<anchor::level&&>(stored));
            }
        }
        return !out.levels.empty();
    }

    bool wavelet_flow::align(const image::quaternion_wavelet& transform, const anchor& anchored, const options& settings, float& displacement_x, float& displacement_y, result& out) {
        double running_x = static_cast<double>(displacement_x);
        double running_y = static_cast<double>(displacement_y);
        double error = static_cast<double>(settings.max_error) + 1.0;
        bool solved = false;
        for (size_t entry = 0; entry < anchored.levels.size(); ++entry) {
            const anchor::level& stored = anchored.levels[entry];
            if ((stored.index < 1) || (static_cast<size_t>(stored.index) > transform.size())) {
                continue;
            }
            const image::quaternion_wavelet::level& current = transform[static_cast<size_t>(stored.index)];
            if ((current.rows == 0) || (current.spacing * static_cast<double>(stored.stride) != stored.spacing)) {
                continue;
            }
            window shape;
            shape.stride = stored.stride;
            shape.spacing = stored.spacing;
            shape.gate_spacing = static_cast<double>(1 << stored.index);
            shape.node_x = stored.node_x;
            shape.node_y = stored.node_y;
            shape.radius = stored.radius;
            shape.centre_x = &stored.centre_x[0];
            shape.centre_y = &stored.centre_y[0];
            shape.pairs = stored.pairs.data();
            shape.frequencies = stored.frequencies.empty() ? nullptr : stored.frequencies.data();
            if (wavelet_flow::solve_level(shape, current, settings, running_x, running_y, error)) {
                solved = true;
            }
        }
        out.x = core::to_pixel_centre(anchored.centre_x + static_cast<float>(running_x));
        out.y = core::to_pixel_centre(anchored.centre_y + static_cast<float>(running_y));
        out.error = static_cast<float>(error);
        out.tracked = false;
        if (!solved) {
            return false;
        }
        displacement_x = static_cast<float>(running_x);
        displacement_y = static_cast<float>(running_y);
        out.tracked = out.error <= settings.max_error;
        return out.tracked;
    }
}
