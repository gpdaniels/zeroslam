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
#ifndef ZEROSLAM_FEATURE_TRACKER_PATCH_FLOW_HPP
#define ZEROSLAM_FEATURE_TRACKER_PATCH_FLOW_HPP

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
    class patch_flow final {
    public:
        constexpr static const int maximum_half_window = 15;
        constexpr static const int maximum_window_width = 2 * maximum_half_window + 1;
        constexpr static const int maximum_window_area = maximum_window_width * maximum_window_width;
        constexpr static const int maximum_parameters = 8;

        enum class model_kind {
            translation,
            affine,
            translation_illumination,
            affine_illumination
        };

        struct options final {
            model_kind model = model_kind::translation;
            int half_window = 7;
            int levels = 4;
            int iterations = 30;
            float convergence = 0.03f;
            double min_eigenvalue = 1e-3;
            float max_error = 40.0f;
            bool damped_steps = false;
        };

        struct anchor final {
            model_kind model = model_kind::translation;
            int parameters = 2;
            int half_window = 7;
            int levels = 0;
            float centre_x = 0.0f;
            float centre_y = 0.0f;
            std::vector<float> values;
            std::vector<float> gradients_x;
            std::vector<float> gradients_y;
            std::vector<double> inverse_hessian;
        };

        struct state final {
            float linear_xx = 1.0f;
            float linear_xy = 0.0f;
            float linear_yx = 0.0f;
            float linear_yy = 1.0f;
            float translation_x = 0.0f;
            float translation_y = 0.0f;
            float gain = 0.0f;
            float bias = 0.0f;
        };

        struct result final {
            float x;
            float y;
            bool tracked;
            float error;
        };

    public:
        static int parameter_count(model_kind model);

        static bool has_illumination(model_kind model);

        static void position(const anchor& anchored, const state& warp, float& x, float& y);

        static bool build_anchor(const image::pyramid& pyramid, float x, float y, const options& settings, anchor& out);

        static bool align(const image::pyramid& pyramid, const anchor& anchored, const options& settings, state& warp, result& out);

    private:
        static void model_columns(model_kind model, double gradient_x, double gradient_y, double offset_x, double offset_y, double value, double* columns);

        static void unpack(model_kind model, const double* parameters, double* linear, double* translation, double& gain, double& bias);

        static double step_motion(model_kind model, const double* step, int half_window);

        static bool compose_inverse(model_kind model, const double* step, state& warp);

        static bool warp_in_bounds(const image::image& source, float centre_x, float centre_y, const state& warp, int half_window);

        static bool invert(const double* values, int size, double* inverse);
    };
}

#endif // ZEROSLAM_FEATURE_TRACKER_PATCH_FLOW_HPP
