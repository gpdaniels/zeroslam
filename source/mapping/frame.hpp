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
#ifndef ZEROSLAM_MAPPING_FRAME_HPP
#define ZEROSLAM_MAPPING_FRAME_HPP

#include "feature/descriptor/binary.hpp"
#include "feature/point.hpp"
#include "feature/score/structure_tensor.hpp"
#include "feature/tracker/patch_flow.hpp"
#include "feature/tracker/tracker.hpp"
#include "image/image.hpp"
#include "image/pyramid.hpp"
#include "math/matrix.hpp"
#include "optimisation/factor_graph.hpp"
#include "sensor/camera.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mapping {
    class frame final {
    public:
        struct settings final {
            enum class detector_kind {
                fast,
                structure_tensor,
                mser
            };
            enum class refiner_kind {
                none,
                subpixel,
                structure_tensor
            };
            enum class tracker_kind {
                klt,
                extrema
            };
            detector_kind detector = detector_kind::fast;
            feature::score::structure_tensor::measure detector_measure = feature::score::structure_tensor::measure::harris;
            float detector_sigma = 2.5f;
            refiner_kind refiner = refiner_kind::subpixel;
            feature::score::structure_tensor::measure refiner_measure = feature::score::structure_tensor::measure::rohr;
            float refiner_sigma = 1.5f;
            enum class association_kind {
                klt,
                match,
                both
            };
            enum class descriptor_kind {
                orb,
                teblid,
                bsift
            };
            tracker_kind tracker = tracker_kind::klt;
            association_kind association = association_kind::both;
            descriptor_kind descriptor = descriptor_kind::orb;
            bool affine = false;
            bool blur_weighting = true;

            float descriptor_distance_scale() const {
                return (this->descriptor == descriptor_kind::orb) ? 1.0f : 0.75f;
            }

            bool lines = false;
            bool line_pose = false;
            double line_angle = 0.0;
            optimisation::factor_graph::strategy solver = optimisation::factor_graph::strategy::dense_schur;
            optimisation::factor_graph::precision solver_precision = optimisation::factor_graph::precision::double_precision;
            bool cull_keyframes = true;
            int global_adjustment_keyframes = 10;
            bool inverse_depth = true;
            bool fixed_budget = false;
            int pose_outlier_limit = 0;
            bool klt_damped_steps = false;
            float track_collision_distance = 2.0f;
            feature::tracker::tracker::flow_kind flow = feature::tracker::tracker::flow_kind::intensity;
            int wavelet_half_window = 2;
            int wavelet_levels = 6;
            bool wavelet_robust = false;
            bool wavelet_undecimated = false;
            feature::tracker::tracker::wavelet_seed_kind wavelet_seed = feature::tracker::tracker::wavelet_seed_kind::rest;
            bool anchored_patches = false;
            feature::tracker::patch_flow::model_kind anchor_model = feature::tracker::patch_flow::model_kind::translation;
            float anchor_refresh_error = 0.0f;
        };

    public:
        int id;
        double laplacian_variance;
        double measurement_sigma;
        math::matrix<double, 3, 3> rotation;
        math::matrix<double, 3, 1> translation;
        sensor::model camera;
        image::pyramid image_pyramid;
        std::vector<std::vector<feature::point>> keypoint_pyramid;
        std::vector<std::vector<feature::descriptor::binary<256>>> descriptor_pyramid;

        std::vector<feature::point> keypoints;
        std::vector<feature::descriptor::binary<256>> descriptors;

    public:
        float to_level0_x(const float level_x, const size_t level) const {
            return level_x * this->image_pyramid.scale_x(level);
        }

        float to_level0_y(const float level_y, const size_t level) const {
            return level_y * this->image_pyramid.scale_y(level);
        }

    public:
        math::matrix<double, 3, 4> get_pose() const {
            return { { { rotation[0][0], rotation[0][1], rotation[0][2], translation[0] },
                       { rotation[1][0], rotation[1][1], rotation[1][2], translation[1] },
                       { rotation[2][0], rotation[2][1], rotation[2][2], translation[2] } } };
        }

    public:
        constexpr static const int level0_features = 500;

        constexpr static const float detector_threshold = 2.0f;

        frame();

        frame(int input_id, const sensor::model& camera_intrinsics, const image::image& input_image_grey, const settings& frontend);

        frame(int input_id, const sensor::model& camera_intrinsics, const image::image& input_image_grey);
    };
}

#endif // ZEROSLAM_MAPPING_FRAME_HPP
