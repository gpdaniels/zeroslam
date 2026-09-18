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

#include "core/coordinates.hpp"
#include "feature/tracker/optical_flow.hpp"
#include "math/math.hpp"

namespace feature::tracker {
    int patch_flow::parameter_count(const model_kind model) {
        switch (model) {
            case model_kind::translation:
                return 2;
            case model_kind::affine:
                return 6;
            case model_kind::translation_illumination:
                return 4;
            case model_kind::affine_illumination:
                return 8;
        }
        return 2;
    }

    bool patch_flow::has_illumination(const model_kind model) {
        switch (model) {
            case model_kind::translation:
                return false;
            case model_kind::affine:
                return false;
            case model_kind::translation_illumination:
                return true;
            case model_kind::affine_illumination:
                return true;
        }
        return false;
    }

    void patch_flow::position(const anchor& anchored, const state& warp, float& x, float& y) {
        x = core::to_pixel_centre(anchored.centre_x + warp.translation_x);
        y = core::to_pixel_centre(anchored.centre_y + warp.translation_y);
    }

    void patch_flow::model_columns(const model_kind model, const double gradient_x, const double gradient_y, const double offset_x, const double offset_y, const double value, double* const columns) {
        switch (model) {
            case model_kind::translation:
                columns[0] = gradient_x;
                columns[1] = gradient_y;
                break;
            case model_kind::affine:
                columns[0] = gradient_x * offset_x;
                columns[1] = gradient_x * offset_y;
                columns[2] = gradient_y * offset_x;
                columns[3] = gradient_y * offset_y;
                columns[4] = gradient_x;
                columns[5] = gradient_y;
                break;
            case model_kind::translation_illumination:
                columns[0] = gradient_x;
                columns[1] = gradient_y;
                columns[2] = value;
                columns[3] = 1.0;
                break;
            case model_kind::affine_illumination:
                columns[0] = gradient_x * offset_x;
                columns[1] = gradient_x * offset_y;
                columns[2] = gradient_y * offset_x;
                columns[3] = gradient_y * offset_y;
                columns[4] = gradient_x;
                columns[5] = gradient_y;
                columns[6] = value;
                columns[7] = 1.0;
                break;
        }
    }

    void patch_flow::unpack(const model_kind model, const double* const parameters, double* const linear, double* const translation, double& gain, double& bias) {
        linear[0] = 0.0;
        linear[1] = 0.0;
        linear[2] = 0.0;
        linear[3] = 0.0;
        translation[0] = 0.0;
        translation[1] = 0.0;
        gain = 0.0;
        bias = 0.0;
        switch (model) {
            case model_kind::translation:
                translation[0] = parameters[0];
                translation[1] = parameters[1];
                break;
            case model_kind::affine:
                linear[0] = parameters[0];
                linear[1] = parameters[1];
                linear[2] = parameters[2];
                linear[3] = parameters[3];
                translation[0] = parameters[4];
                translation[1] = parameters[5];
                break;
            case model_kind::translation_illumination:
                translation[0] = parameters[0];
                translation[1] = parameters[1];
                gain = parameters[2];
                bias = parameters[3];
                break;
            case model_kind::affine_illumination:
                linear[0] = parameters[0];
                linear[1] = parameters[1];
                linear[2] = parameters[2];
                linear[3] = parameters[3];
                translation[0] = parameters[4];
                translation[1] = parameters[5];
                gain = parameters[6];
                bias = parameters[7];
                break;
        }
    }

    double patch_flow::step_motion(const model_kind model, const double* const step, const int half_window) {
        double linear[4];
        double translation[2];
        double gain = 0.0;
        double bias = 0.0;
        patch_flow::unpack(model, step, linear, translation, gain, bias);
        static_cast<void>(gain);
        static_cast<void>(bias);
        const double extent = static_cast<double>(half_window);
        double motion = 0.0;
        for (int corner = 0; corner < 4; ++corner) {
            const double x = ((corner & 1) != 0) ? extent : -extent;
            const double y = ((corner & 2) != 0) ? extent : -extent;
            const double moved_x = linear[0] * x + linear[1] * y + translation[0];
            const double moved_y = linear[2] * x + linear[3] * y + translation[1];
            motion = math::max(motion, math::sqrt(moved_x * moved_x + moved_y * moved_y));
        }
        return motion;
    }

    bool patch_flow::compose_inverse(const model_kind model, const double* const step, state& warp) {
        double linear[4];
        double translation[2];
        double gain = 0.0;
        double bias = 0.0;
        patch_flow::unpack(model, step, linear, translation, gain, bias);
        for (int index = 0; index < 4; ++index) {
            if (!math::isfinite(linear[index])) {
                return false;
            }
        }
        if (!math::isfinite(translation[0]) || !math::isfinite(translation[1]) || !math::isfinite(gain) || !math::isfinite(bias)) {
            return false;
        }
        const double increment_xx = 1.0 + linear[0];
        const double increment_xy = linear[1];
        const double increment_yx = linear[2];
        const double increment_yy = 1.0 + linear[3];
        const double determinant = increment_xx * increment_yy - increment_xy * increment_yx;
        if (!(math::abs(determinant) > 1.0e-6)) {
            return false;
        }
        const double inverse_xx = increment_yy / determinant;
        const double inverse_xy = -increment_xy / determinant;
        const double inverse_yx = -increment_yx / determinant;
        const double inverse_yy = increment_xx / determinant;
        const double inverse_tx = -(inverse_xx * translation[0] + inverse_xy * translation[1]);
        const double inverse_ty = -(inverse_yx * translation[0] + inverse_yy * translation[1]);
        const double current_xx = static_cast<double>(warp.linear_xx);
        const double current_xy = static_cast<double>(warp.linear_xy);
        const double current_yx = static_cast<double>(warp.linear_yx);
        const double current_yy = static_cast<double>(warp.linear_yy);
        const double composed_xx = current_xx * inverse_xx + current_xy * inverse_yx;
        const double composed_xy = current_xx * inverse_xy + current_xy * inverse_yy;
        const double composed_yx = current_yx * inverse_xx + current_yy * inverse_yx;
        const double composed_yy = current_yx * inverse_xy + current_yy * inverse_yy;
        const double composed_determinant = composed_xx * composed_yy - composed_xy * composed_yx;
        if (!(composed_determinant > 0.0625) || !(composed_determinant < 16.0)) {
            return false;
        }
        warp.translation_x = static_cast<float>(current_xx * inverse_tx + current_xy * inverse_ty + static_cast<double>(warp.translation_x));
        warp.translation_y = static_cast<float>(current_yx * inverse_tx + current_yy * inverse_ty + static_cast<double>(warp.translation_y));
        warp.linear_xx = static_cast<float>(composed_xx);
        warp.linear_xy = static_cast<float>(composed_xy);
        warp.linear_yx = static_cast<float>(composed_yx);
        warp.linear_yy = static_cast<float>(composed_yy);
        warp.gain = static_cast<float>(static_cast<double>(warp.gain) + gain);
        warp.bias = static_cast<float>(static_cast<double>(warp.bias) + bias);
        return true;
    }

    bool patch_flow::warp_in_bounds(const image::image& source, const float centre_x, const float centre_y, const state& warp, const int half_window) {
        const float width = static_cast<float>(source.get_cols());
        const float height = static_cast<float>(source.get_rows());
        const float extent = static_cast<float>(half_window);
        for (int corner = 0; corner < 4; ++corner) {
            const float x = ((corner & 1) != 0) ? extent : -extent;
            const float y = ((corner & 2) != 0) ? extent : -extent;
            const float mapped_x = centre_x + (warp.linear_xx * x + warp.linear_xy * y + warp.translation_x);
            const float mapped_y = centre_y + (warp.linear_yx * x + warp.linear_yy * y + warp.translation_y);
            if (!math::isfinite(mapped_x) || !math::isfinite(mapped_y)) {
                return false;
            }
            if ((mapped_x < 1.0f) || (mapped_y < 1.0f) || (mapped_x > width - 2.0f) || (mapped_y > height - 2.0f)) {
                return false;
            }
        }
        return true;
    }

    bool patch_flow::invert(const double* const values, const int size, double* const inverse) {
        double working[patch_flow::maximum_parameters * patch_flow::maximum_parameters] = {};
        double scale = 0.0;
        for (int row = 0; row < size; ++row) {
            for (int column = 0; column < size; ++column) {
                working[row * size + column] = values[row * size + column];
                inverse[row * size + column] = (row == column) ? 1.0 : 0.0;
                scale = math::max(scale, math::abs(values[row * size + column]));
            }
        }
        const double epsilon = scale * 1.0e-12;
        for (int column = 0; column < size; ++column) {
            int pivot_row = column;
            double pivot_scale = math::abs(working[column * size + column]);
            for (int row = column + 1; row < size; ++row) {
                const double candidate = math::abs(working[row * size + column]);
                if (candidate > pivot_scale) {
                    pivot_row = row;
                    pivot_scale = candidate;
                }
            }
            if (!(pivot_scale > epsilon)) {
                return false;
            }
            if (pivot_row != column) {
                for (int index = 0; index < size; ++index) {
                    const double swap_working = working[column * size + index];
                    working[column * size + index] = working[pivot_row * size + index];
                    working[pivot_row * size + index] = swap_working;
                    const double swap_inverse = inverse[column * size + index];
                    inverse[column * size + index] = inverse[pivot_row * size + index];
                    inverse[pivot_row * size + index] = swap_inverse;
                }
            }
            const double divisor = working[column * size + column];
            for (int index = 0; index < size; ++index) {
                working[column * size + index] /= divisor;
                inverse[column * size + index] /= divisor;
            }
            working[column * size + column] = 1.0;
            for (int row = 0; row < size; ++row) {
                if (row == column) {
                    continue;
                }
                const double factor = working[row * size + column];
                if (factor == 0.0) {
                    continue;
                }
                for (int index = 0; index < size; ++index) {
                    working[row * size + index] -= factor * working[column * size + index];
                    inverse[row * size + index] -= factor * inverse[column * size + index];
                }
                working[row * size + column] = 0.0;
            }
        }
        for (int index = 0; index < size * size; ++index) {
            if (!math::isfinite(inverse[index])) {
                return false;
            }
        }
        return true;
    }

    bool patch_flow::build_anchor(const image::pyramid& pyramid, const float x, const float y, const options& settings, anchor& out) {
        out = anchor();
        const int available = static_cast<int>(pyramid.size());
        if (available < 1) {
            return false;
        }
        const int half_window = math::max(1, math::min(settings.half_window, patch_flow::maximum_half_window));
        const int window_width = 2 * half_window + 1;
        const int window_area = window_width * window_width;
        const double window_area_as_double = static_cast<double>(window_area);
        const int parameters = patch_flow::parameter_count(settings.model);
        const float centre_x = core::to_pixel_index_position(x);
        const float centre_y = core::to_pixel_index_position(y);
        int levels = math::max(1, math::min(settings.levels, available));
        while (levels > 1) {
            const float scale = 1.0f / static_cast<float>(1 << (levels - 1));
            if (optical_flow::window_in_bounds(pyramid[static_cast<size_t>(levels - 1)], centre_x * scale, centre_y * scale, half_window)) {
                break;
            }
            --levels;
        }
        if (!optical_flow::window_in_bounds(pyramid[0], centre_x, centre_y, half_window)) {
            return false;
        }
        out.model = settings.model;
        out.parameters = parameters;
        out.half_window = half_window;
        out.levels = levels;
        out.centre_x = centre_x;
        out.centre_y = centre_y;
        out.values.resize(static_cast<size_t>(levels * window_area));
        out.gradients_x.resize(static_cast<size_t>(levels * window_area));
        out.gradients_y.resize(static_cast<size_t>(levels * window_area));
        out.inverse_hessian.resize(static_cast<size_t>(levels * parameters * parameters));
        for (int level = 0; level < levels; ++level) {
            const image::image& source = pyramid[static_cast<size_t>(level)];
            const float scale = 1.0f / static_cast<float>(1 << level);
            const float level_x = centre_x * scale;
            const float level_y = centre_y * scale;
            if (!optical_flow::window_in_bounds(source, level_x, level_y, half_window)) {
                out = anchor();
                return false;
            }
            float* const values = out.values.data() + static_cast<size_t>(level * window_area);
            float* const gradients_x = out.gradients_x.data() + static_cast<size_t>(level * window_area);
            float* const gradients_y = out.gradients_y.data() + static_cast<size_t>(level * window_area);
            double hessian[patch_flow::maximum_parameters * patch_flow::maximum_parameters] = {};
            double structure_xx = 0.0;
            double structure_xy = 0.0;
            double structure_yy = 0.0;
            int index = 0;
            for (int j = -half_window; j <= half_window; ++j) {
                for (int i = -half_window; i <= half_window; ++i) {
                    const float base_x = level_x + static_cast<float>(i);
                    const float base_y = level_y + static_cast<float>(j);
                    const float value = optical_flow::sample_bilinear(source, base_x, base_y);
                    const float gradient_x = 0.5f * (optical_flow::sample_bilinear(source, base_x + 1.0f, base_y) - optical_flow::sample_bilinear(source, base_x - 1.0f, base_y));
                    const float gradient_y = 0.5f * (optical_flow::sample_bilinear(source, base_x, base_y + 1.0f) - optical_flow::sample_bilinear(source, base_x, base_y - 1.0f));
                    values[index] = value;
                    gradients_x[index] = gradient_x;
                    gradients_y[index] = gradient_y;
                    structure_xx += static_cast<double>(gradient_x) * static_cast<double>(gradient_x);
                    structure_xy += static_cast<double>(gradient_x) * static_cast<double>(gradient_y);
                    structure_yy += static_cast<double>(gradient_y) * static_cast<double>(gradient_y);
                    double columns[patch_flow::maximum_parameters] = {};
                    patch_flow::model_columns(settings.model, static_cast<double>(gradient_x), static_cast<double>(gradient_y), static_cast<double>(i), static_cast<double>(j), static_cast<double>(value), columns);
                    for (int row = 0; row < parameters; ++row) {
                        for (int column = row; column < parameters; ++column) {
                            hessian[row * parameters + column] += columns[row] * columns[column];
                        }
                    }
                    ++index;
                }
            }
            for (int row = 0; row < parameters; ++row) {
                for (int column = 0; column < row; ++column) {
                    hessian[row * parameters + column] = hessian[column * parameters + row];
                }
            }
            const double determinant = structure_xx * structure_yy - structure_xy * structure_xy;
            const double trace = structure_xx + structure_yy;
            const double eigen_gap = math::sqrt(math::max(0.0, trace * trace - 4.0 * determinant));
            const double smaller_eigenvalue = 0.5 * (trace - eigen_gap);
            if (!(determinant > 0.0) || (smaller_eigenvalue < settings.min_eigenvalue * window_area_as_double)) {
                out = anchor();
                return false;
            }
            double inverse[patch_flow::maximum_parameters * patch_flow::maximum_parameters] = {};
            if (!patch_flow::invert(hessian, parameters, inverse)) {
                out = anchor();
                return false;
            }
            double* const stored = out.inverse_hessian.data() + static_cast<size_t>(level * parameters * parameters);
            for (int entry = 0; entry < parameters * parameters; ++entry) {
                stored[entry] = inverse[entry];
            }
        }
        return true;
    }

    bool patch_flow::align(const image::pyramid& pyramid, const anchor& anchored, const options& settings, state& warp, result& out) {
        out.x = core::to_pixel_centre(anchored.centre_x + warp.translation_x);
        out.y = core::to_pixel_centre(anchored.centre_y + warp.translation_y);
        out.tracked = false;
        out.error = settings.max_error + 1.0f;
        if ((anchored.levels < 1) || (anchored.levels > static_cast<int>(pyramid.size()))) {
            return false;
        }
        const int half_window = anchored.half_window;
        const int window_width = 2 * half_window + 1;
        const int window_area = window_width * window_width;
        const int parameters = anchored.parameters;
        if ((parameters < 1) || (parameters > patch_flow::maximum_parameters)) {
            return false;
        }
        const int iterations = math::max(1, settings.iterations);
        const double convergence = static_cast<double>(settings.convergence);
        state current = warp;
        const float start_scale = 1.0f / static_cast<float>(1 << (anchored.levels - 1));
        current.translation_x = warp.translation_x * start_scale;
        current.translation_y = warp.translation_y * start_scale;
        for (int level = anchored.levels - 1; level >= 0; --level) {
            const image::image& source = pyramid[static_cast<size_t>(level)];
            const float scale = 1.0f / static_cast<float>(1 << level);
            const float level_x = anchored.centre_x * scale;
            const float level_y = anchored.centre_y * scale;
            const float* const values = anchored.values.data() + static_cast<size_t>(level * window_area);
            const float* const gradients_x = anchored.gradients_x.data() + static_cast<size_t>(level * window_area);
            const float* const gradients_y = anchored.gradients_y.data() + static_cast<size_t>(level * window_area);
            const double* const inverse = anchored.inverse_hessian.data() + static_cast<size_t>(level * parameters * parameters);
            double previous_direction[patch_flow::maximum_parameters] = {};
            bool have_direction = false;
            for (int iteration = 0; iteration < iterations; ++iteration) {
                if (!patch_flow::warp_in_bounds(source, level_x, level_y, current, half_window)) {
                    return false;
                }
                const double gain = static_cast<double>(current.gain);
                const double bias = static_cast<double>(current.bias);
                double projection[patch_flow::maximum_parameters] = {};
                int index = 0;
                for (int j = -half_window; j <= half_window; ++j) {
                    for (int i = -half_window; i <= half_window; ++i) {
                        const float offset_x = static_cast<float>(i);
                        const float offset_y = static_cast<float>(j);
                        const float sample_x = level_x + (current.linear_xx * offset_x + current.linear_xy * offset_y + current.translation_x);
                        const float sample_y = level_y + (current.linear_yx * offset_x + current.linear_yy * offset_y + current.translation_y);
                        const float sample = optical_flow::sample_bilinear(source, sample_x, sample_y);
                        const double value = static_cast<double>(values[index]);
                        const double residual = static_cast<double>(sample) - (1.0 + gain) * value - bias;
                        double columns[patch_flow::maximum_parameters] = {};
                        patch_flow::model_columns(anchored.model, static_cast<double>(gradients_x[index]), static_cast<double>(gradients_y[index]), static_cast<double>(i), static_cast<double>(j), value, columns);
                        for (int parameter = 0; parameter < parameters; ++parameter) {
                            projection[parameter] += columns[parameter] * residual;
                        }
                        ++index;
                    }
                }
                double step[patch_flow::maximum_parameters] = {};
                for (int row = 0; row < parameters; ++row) {
                    double total = 0.0;
                    for (int column = 0; column < parameters; ++column) {
                        total += inverse[row * parameters + column] * projection[column];
                    }
                    step[row] = total;
                }
                if (settings.damped_steps) {
                    double norm = 0.0;
                    for (int parameter = 0; parameter < parameters; ++parameter) {
                        norm += step[parameter] * step[parameter];
                    }
                    norm = math::sqrt(norm);
                    if ((norm > 0.0) && (patch_flow::step_motion(anchored.model, step, half_window) > convergence)) {
                        double direction[patch_flow::maximum_parameters] = {};
                        for (int parameter = 0; parameter < parameters; ++parameter) {
                            direction[parameter] = step[parameter] / norm;
                        }
                        if (have_direction) {
                            double dot = 0.0;
                            for (int parameter = 0; parameter < parameters; ++parameter) {
                                dot += direction[parameter] * previous_direction[parameter];
                            }
                            if (dot < -0.5) {
                                for (int parameter = 0; parameter < parameters; ++parameter) {
                                    step[parameter] *= 0.5;
                                }
                            }
                        }
                        for (int parameter = 0; parameter < parameters; ++parameter) {
                            previous_direction[parameter] = direction[parameter];
                        }
                        have_direction = true;
                    }
                }
                const double motion = patch_flow::step_motion(anchored.model, step, half_window);
                if (!patch_flow::compose_inverse(anchored.model, step, current)) {
                    return false;
                }
                if (motion < convergence) {
                    break;
                }
            }
            if (level > 0) {
                current.translation_x *= 2.0f;
                current.translation_y *= 2.0f;
            }
        }
        const image::image& full = pyramid[0];
        if (!patch_flow::warp_in_bounds(full, anchored.centre_x, anchored.centre_y, current, half_window)) {
            return false;
        }
        const float* const values = anchored.values.data();
        const double gain = static_cast<double>(current.gain);
        const double bias = static_cast<double>(current.bias);
        double error_sum = 0.0;
        int index = 0;
        for (int j = -half_window; j <= half_window; ++j) {
            for (int i = -half_window; i <= half_window; ++i) {
                const float offset_x = static_cast<float>(i);
                const float offset_y = static_cast<float>(j);
                const float sample_x = anchored.centre_x + (current.linear_xx * offset_x + current.linear_xy * offset_y + current.translation_x);
                const float sample_y = anchored.centre_y + (current.linear_yx * offset_x + current.linear_yy * offset_y + current.translation_y);
                const float sample = optical_flow::sample_bilinear(full, sample_x, sample_y);
                const double residual = static_cast<double>(sample) - (1.0 + gain) * static_cast<double>(values[index]) - bias;
                error_sum += math::abs(residual);
                ++index;
            }
        }
        warp = current;
        out.x = core::to_pixel_centre(anchored.centre_x + current.translation_x);
        out.y = core::to_pixel_centre(anchored.centre_y + current.translation_y);
        out.error = static_cast<float>(error_sum / static_cast<double>(window_area));
        out.tracked = (out.error <= settings.max_error);
        return out.tracked;
    }
}
