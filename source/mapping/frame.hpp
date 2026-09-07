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

#include "core/filter.hpp"
#include "core/sort.hpp"
#include "feature/angle/orb.hpp"
#include "feature/descriptor/binary.hpp"
#include "feature/descriptor/orb.hpp"
#include "feature/detector/fast.hpp"
#include "feature/distributor/square_covering.hpp"
#include "feature/point.hpp"
#include "feature/refiner/subpixel.hpp"
#include "feature/score/fast.hpp"
#include "feature/suppressor/fast.hpp"
#include "image/image.hpp"
#include "image/pyramid.hpp"
#include "math/matrix.hpp"
#include "sensor/camera.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mapping {
    class frame {
    public:
        static inline int id_generator = 0;
        int id;
        math::matrix<double, 3, 3> rotation;
        math::matrix<double, 3, 1> translation;
        sensor::pinhole camera;
        image::pyramid image_pyramid;
        std::vector<std::vector<feature::point>> keypoint_pyramid;
        std::vector<std::vector<feature::descriptor::binary<256>>> descriptor_pyramid;

    public:
        math::matrix<double, 3, 4> get_pose() const {
            return { { { rotation[0][0], rotation[0][1], rotation[0][2], translation[0] },
                       { rotation[1][0], rotation[1][1], rotation[1][2], translation[1] },
                       { rotation[2][0], rotation[2][1], rotation[2][2], translation[2] } } };
        }

    public:
        frame() {
            this->id = -1;
            this->rotation = math::matrix<double, 3, 3>::zero();
            this->translation = math::matrix<double, 3, 1>::zero();
        }

        frame(const sensor::pinhole camera_intrinsics, const image::image& input_image_grey) {
            this->id = frame::id_generator++;
            this->rotation = math::matrix<double, 3, 3>::identity();
            this->translation = math::matrix<double, 3, 1>::zero();
            this->camera = camera_intrinsics;

            this->image_pyramid = image::pyramid(input_image_grey);
            this->keypoint_pyramid.reserve(this->image_pyramid.size());
            this->descriptor_pyramid.reserve(this->image_pyramid.size());
            for (size_t o = 0; o < this->image_pyramid.size(); ++o) {
                const image::image& image_grey = this->image_pyramid[o];
                const int image_cols = static_cast<int>(image_grey.get_cols());
                const int image_rows = static_cast<int>(image_grey.get_rows());

                // Detect features.
                std::vector<feature::point> kps;
                kps.resize(50000);
                const size_t feature_count = feature::detector::fast::detect(image_grey.get_data(), image_cols, image_rows, image_cols, 7, kps.size(), kps.data());
                kps.resize(feature_count);

                // Prune edge features.
                // Note: The border must cover all sampling performed around a keypoint:
                // - The rotated descriptor pattern reads up to round(13 * sqrt(2)) = 18 pixels from the feature.
                // - The subpixel patch samples up to 20 (patch radius) + 4 (maximum accepted refinement offset) + 1 (bilinear interpolation) = 25 pixels from the feature.
                constexpr static const int border = 25;
                size_t prune_edge_count = kps.size();
                core::filter::remove_if(kps.data(), prune_edge_count, [image_cols, image_rows](const feature::point& feature) {
                    return (feature.x < border) || (feature.x >= static_cast<float>(image_cols - border)) || (feature.y < border) || (feature.y >= static_cast<float>(image_rows - border));
                });
                kps.resize(static_cast<size_t>(prune_edge_count));

                // Score features.
                for (size_t i = 0; i < kps.size(); ++i) {
                    const unsigned char* feature = image_grey.get_data() + static_cast<size_t>(kps[i].y) * image_grey.get_cols() + static_cast<size_t>(kps[i].x);
                    const float response = feature::score::fast::score(feature, image_cols);
                    kps[i].response = response;
                }

                // Prune low score features.
                size_t prune_score_count = kps.size();
                core::filter::remove_if(kps.data(), prune_score_count, [](const feature::point& feature) {
                    return feature.response < 0;
                });
                kps.resize(static_cast<size_t>(prune_score_count));

                // Non maximally suppress features.
                // Note: This suppression function assumes points are sorted in row major.
                core::sort::quick(kps.data(), kps.size(), [](const feature::point& lhs, const feature::point& rhs) {
                    return lhs.y == rhs.y ? lhs.x < rhs.x : lhs.y < rhs.y;
                });
                std::vector<feature::point> features_suppressed(kps.size());
                const size_t suppressed_count = feature::suppressor::fast::suppress(kps.data(), kps.size(), image_grey.get_rows(), features_suppressed.data());
                features_suppressed.resize(suppressed_count);
                kps = std::move(features_suppressed);

                // Distribute features.
                // Note: The distribution function assumes points are sorted by response, so sort first.
                core::sort::quick(kps.data(), kps.size(), [](const feature::point& lhs, const feature::point& rhs) {
                    if (lhs.response != rhs.response) {
                        return lhs.response > rhs.response;
                    }
                    if (lhs.y != rhs.y) {
                        return lhs.y > rhs.y;
                    }
                    return lhs.x > rhs.x;
                });
                std::vector<feature::point> features_distributed(2000);
                const int distributed_count = feature::distributor::square_covering::distribute(kps.data(), static_cast<int>(kps.size()), image_cols, image_rows, 500, static_cast<int>(features_distributed.size()), features_distributed.data());
                features_distributed.resize(static_cast<size_t>(distributed_count));
                kps = std::move(features_distributed);

                // Describe features.
                std::vector<feature::descriptor::binary<256>> des;
                des.resize(static_cast<size_t>(distributed_count));
                for (size_t i = 0; i < static_cast<size_t>(distributed_count); ++i) {
                    const unsigned char* feature = image_grey.get_data() + static_cast<size_t>(kps[i].y) * image_grey.get_cols() + static_cast<size_t>(kps[i].x);
                    // Attempt subpixel refinement of feature.
                    float offset_x = 0;
                    float offset_y = 0;
                    if (!feature::refiner::subpixel::refine(feature, image_cols, offset_x, offset_y)) {
                        // If unsuccessful, proceed using unrefined feature.
                        const float angle = feature::angle::orb::dominant_angle(feature, image_cols);
                        feature::descriptor::orb::describe(feature, image_cols, angle, des[i]);
                        continue;
                    }

                    // If successful, calculate descriptors from a subpixel patch.
                    unsigned char patch[41][41];
                    feature::refiner::subpixel::patch_41x41_bilinear(feature, image_cols, offset_x, offset_y, &patch[0][0]);
                    const float angle = feature::angle::orb::dominant_angle(&patch[20][20], 41);
                    feature::descriptor::orb::describe(&patch[20][20], 41, angle, des[i]);
                    kps[i].x += offset_x;
                    kps[i].y += offset_y;
                }
                this->keypoint_pyramid.push_back(std::move(kps));
                this->descriptor_pyramid.push_back(std::move(des));
            }
        }
    };
}

#endif // ZEROSLAM_MAPPING_FRAME_HPP
