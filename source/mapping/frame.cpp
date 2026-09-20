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

#include "mapping/frame.hpp"

#include "core/assert.hpp"
#include "core/coordinates.hpp"
#include "core/filter.hpp"
#include "core/sort.hpp"
#include "core/thread_pool.hpp"
#include "feature/angle/orb.hpp"
#include "feature/descriptor/orb.hpp"
#include "feature/descriptor/sift.hpp"
#include "feature/descriptor/teblid.hpp"
#include "feature/detector/fast.hpp"
#include "feature/detector/mser.hpp"
#include "feature/detector/structure_tensor.hpp"
#include "feature/distributor/square_covering.hpp"
#include "feature/refiner/structure_tensor.hpp"
#include "feature/refiner/subpixel.hpp"
#include "feature/score/fast.hpp"
#include "feature/suppressor/fast.hpp"
#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <utility>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mapping {
    frame::frame() {
        this->id = -1;
        this->rotation = math::matrix<double, 3, 3>::identity();
        this->translation = math::matrix<double, 3, 1>::zero();
        this->laplacian_variance = 0.0;
        this->measurement_sigma = 1.0;
    }

    frame::frame(int input_id, const sensor::model& camera_intrinsics, const image::image& input_image_grey)
        : frame(input_id, camera_intrinsics, input_image_grey, settings()) {
    }

    frame::frame(int input_id, const sensor::model& camera_intrinsics, const image::image& input_image_grey, const settings& frontend) {
        this->id = input_id;
        this->rotation = math::matrix<double, 3, 3>::identity();
        this->translation = math::matrix<double, 3, 1>::zero();
        this->camera = camera_intrinsics;

        this->image_pyramid = image::pyramid(input_image_grey);
        this->laplacian_variance = 0.0;
        this->measurement_sigma = 1.0;
        {
            const image::image& base = this->image_pyramid[0];
            const int cols = static_cast<int>(base.get_cols());
            const int rows = static_cast<int>(base.get_rows());
            const unsigned char* const data = base.get_data();
            double sum = 0.0;
            double sum_squared = 0.0;
            size_t count = 0;
            for (int y = 1; y + 1 < rows; y += 2) {
                for (int x = 1; x + 1 < cols; x += 2) {
                    const int index = (y * cols) + x;
                    const double laplacian = (4.0 * data[index]) - data[index - 1] - data[index + 1] - data[index - cols] - data[index + cols];
                    sum += laplacian;
                    sum_squared += laplacian * laplacian;
                    ++count;
                }
            }
            if (count > 0) {
                const double mean = sum / static_cast<double>(count);
                this->laplacian_variance = math::max(0.0, (sum_squared / static_cast<double>(count)) - (mean * mean));
            }
        }
        const size_t octaves = this->image_pyramid.size();
        this->keypoint_pyramid.reserve(octaves);
        this->descriptor_pyramid.reserve(octaves);
        for (size_t o = 0; o < octaves; ++o) {
            const image::image& image_grey = this->image_pyramid[o];
            const int width = static_cast<int>(image_grey.get_cols());
            const int height = static_cast<int>(image_grey.get_rows());

            std::vector<feature::point> kps(50000);
            size_t feature_count = 0;
            if (frontend.detector == settings::detector_kind::mser) {
                feature_count = feature::detector::mser::detect(image_grey.get_data(), width, height, width, feature::detector::mser::options(), kps.size(), kps.data());
            }
            else if (frontend.detector == settings::detector_kind::fast) {
                feature_count = feature::detector::fast::detect(image_grey.get_data(), width, height, width, 7, kps.size(), kps.data());
            }
            else {
                feature_count = feature::detector::structure_tensor::detect(image_grey.get_data(), width, height, width, frontend.detector_measure, frontend.detector_sigma, frame::detector_threshold, kps.size(), kps.data());
            }
            kps.resize(feature_count);

            constexpr static const int border = 25;
            ASSERT((frontend.refiner != settings::refiner_kind::structure_tensor) || (feature::refiner::structure_tensor::footprint(frontend.refiner_sigma) <= border), "The structure tensor refiner reads outside the feature border.");
            size_t prune_edge_count = kps.size();
            core::filter::remove_if(kps.data(), prune_edge_count, [width, height](const feature::point& feature) {
                return (feature.x < static_cast<float>(border)) || (feature.x >= static_cast<float>(width - border)) || (feature.y < static_cast<float>(border)) || (feature.y >= static_cast<float>(height - border));
            });
            kps.resize(static_cast<size_t>(prune_edge_count));

            if (frontend.detector == settings::detector_kind::fast) {
                for (size_t i = 0; i < kps.size(); ++i) {
                    const unsigned char* feature = image_grey.get_data() + static_cast<size_t>(kps[i].y) * image_grey.get_cols() + static_cast<size_t>(kps[i].x);
                    const float response = feature::score::fast::score(feature, width);
                    kps[i].response = response;
                }
            }

            size_t prune_score_count = kps.size();
            core::filter::remove_if(kps.data(), prune_score_count, [](const feature::point& feature) {
                return feature.response < 0;
            });
            kps.resize(static_cast<size_t>(prune_score_count));

            core::sort::quick(kps.data(), kps.size(), [](const feature::point& lhs, const feature::point& rhs) {
                return lhs.y == rhs.y ? lhs.x < rhs.x : lhs.y < rhs.y;
            });
            std::vector<feature::point> features_suppressed(kps.size());
            const size_t suppressed_count = feature::suppressor::fast::suppress(kps.data(), kps.size(), image_grey.get_rows(), features_suppressed.data());
            features_suppressed.resize(suppressed_count);
            kps = std::move(features_suppressed);

            core::sort::quick(kps.data(), kps.size(), [](const feature::point& lhs, const feature::point& rhs) {
                if (lhs.response != rhs.response) {
                    return lhs.response > rhs.response;
                }
                if (lhs.y != rhs.y) {
                    return lhs.y > rhs.y;
                }
                return lhs.x > rhs.x;
            });
            const double area_ratio = static_cast<double>(this->image_pyramid.scale_x(o)) * static_cast<double>(this->image_pyramid.scale_y(o));
            const int distribute_minimum = math::max(1, static_cast<int>(static_cast<double>(frame::level0_features) / area_ratio));
            std::vector<feature::point> features_distributed(static_cast<size_t>(4 * distribute_minimum));
            const int distributed_count = feature::distributor::square_covering::distribute(kps.data(), static_cast<int>(kps.size()), width, height, distribute_minimum, static_cast<int>(features_distributed.size()), features_distributed.data());
            features_distributed.resize(static_cast<size_t>(distributed_count));
            kps = std::move(features_distributed);
            if (frontend.fixed_budget && (kps.size() > static_cast<size_t>(distribute_minimum))) {
                kps.resize(static_cast<size_t>(distribute_minimum));
            }

            std::vector<feature::descriptor::binary<256>> des;
            des.resize(kps.size());
            core::thread_pool::instance().parallel_for(kps.size(), 32, [&](const size_t i) {
                const unsigned char* feature = image_grey.get_data() + static_cast<size_t>(kps[i].y) * image_grey.get_cols() + static_cast<size_t>(kps[i].x);
                float offset_x = 0;
                float offset_y = 0;
                bool refined = false;
                switch (frontend.refiner) {
                    case settings::refiner_kind::none: {
                    } break;
                    case settings::refiner_kind::subpixel: {
                        refined = feature::refiner::subpixel::refine(feature, width, offset_x, offset_y);
                    } break;
                    case settings::refiner_kind::structure_tensor: {
                        refined = feature::refiner::structure_tensor::refine(feature, width, frontend.refiner_measure, frontend.refiner_sigma, offset_x, offset_y);
                    } break;
                }
                if (frontend.descriptor != settings::descriptor_kind::orb) {
                    const float angle = feature::angle::orb::dominant_angle(feature, width);
                    if (frontend.descriptor == settings::descriptor_kind::teblid) {
                        feature::descriptor::teblid::describe(feature, width, angle, des[i]);
                    }
                    else {
                        feature::descriptor::sift::describe(feature, width, angle, des[i]);
                    }
                    if (refined) {
                        kps[i].x += offset_x;
                        kps[i].y += offset_y;
                    }
                    return;
                }
                if (!refined) {
                    const float angle = feature::angle::orb::dominant_angle(feature, width);
                    feature::descriptor::orb::describe(feature, width, angle, des[i]);
                    return;
                }

                unsigned char patch[41][41];
                feature::refiner::subpixel::patch_41x41_bilinear(feature, width, offset_x, offset_y, &patch[0][0]);
                const float angle = feature::angle::orb::dominant_angle(&patch[20][20], 41);
                feature::descriptor::orb::describe(&patch[20][20], 41, angle, des[i]);
                kps[i].x += offset_x;
                kps[i].y += offset_y;
            });
            for (feature::point& kp : kps) {
                kp.octave = static_cast<int>(o);
            }
            this->keypoint_pyramid.push_back(std::move(kps));
            this->descriptor_pyramid.push_back(std::move(des));
        }

        const size_t exported_levels = this->keypoint_pyramid.size();
        size_t exported_count = 0;
        for (size_t level = 0; level < exported_levels; ++level) {
            exported_count += this->keypoint_pyramid[level].size();
        }
        this->keypoints.reserve(exported_count);
        this->descriptors.reserve(exported_count);
        for (size_t level = 0; level < exported_levels; ++level) {
            const std::vector<feature::point>& level_keypoints = this->keypoint_pyramid[level];
            const std::vector<feature::descriptor::binary<256>>& level_descriptors = this->descriptor_pyramid[level];
            for (size_t i = 0; i < level_keypoints.size(); ++i) {
                feature::point exported = level_keypoints[i];
                exported.x = core::to_pixel_centre(this->to_level0_x(exported.x, level));
                exported.y = core::to_pixel_centre(this->to_level0_y(exported.y, level));
                this->keypoints.push_back(exported);
                this->descriptors.push_back(level_descriptors[i]);
            }
        }
    }
}
