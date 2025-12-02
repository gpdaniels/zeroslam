/*
Copyright (C) 2025 Geoffrey Daniels. https://gpdaniels.com/

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
#ifndef FRAME_HPP
#define FRAME_HPP

#include "camera.hpp"
#include "feature.hpp"
#include "image.hpp"
#include "matrix.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace frame {
    class frame {
    public:
        static inline int id_generator = 0;
        int id;
        matrix::matrix<double, 3, 3> rotation;
        matrix::matrix<double, 3, 1> translation;
        camera::pinhole camera;
        std::vector<image::image> image_pyramid;
        std::vector<std::vector<feature::point>> keypoint_pyramid;
        std::vector<std::vector<feature::descriptor>> descriptor_pyramid;

    public:
        matrix::matrix<double, 3, 4> get_pose() const {
            return { { { rotation[0][0], rotation[0][1], rotation[0][2], translation[0] },
                       { rotation[1][0], rotation[1][1], rotation[1][2], translation[1] },
                       { rotation[2][0], rotation[2][1], rotation[2][2], translation[2] } } };
        }

    public:
        frame() {
            this->id = -1;
            this->rotation = matrix::matrix<double, 3, 3>::zero();
            this->translation = matrix::matrix<double, 3, 1>::zero();
        }

        frame(const camera::pinhole camera_intrinsics, const image::image& input_image_grey) {
            this->id = frame::id_generator++;
            this->rotation = matrix::matrix<double, 3, 3>::identity();
            this->translation = matrix::matrix<double, 3, 1>::zero();
            this->camera = camera_intrinsics;

            // Pyramid.
            // Note: The number of octaves is calculated from floored powers of two required to represent the smallest image dimension.
            // Note: We subtract five levels to ensure the smalles image size has a minimum dimension of at least 32 pixels.
            const int octaves = static_cast<int>(math::floor(math::log(static_cast<double>(math::min(input_image_grey.get_cols(), input_image_grey.get_rows()))) / math::log(2.0))) - 5;
            this->image_pyramid.reserve(octaves);
            this->keypoint_pyramid.reserve(octaves);
            this->descriptor_pyramid.reserve(octaves);
            this->image_pyramid.push_back(input_image_grey);
            for (int o = 0; o < octaves; ++o) {
                if (o > 0) {
                    // Copy, blur, resize.
                    const image::image& previous = this->image_pyramid.back();
                    image::image blurred(previous.get_rows(), previous.get_cols());
                    image::blur(previous.get_data(), previous.get_cols(), previous.get_rows(), previous.get_cols(), blurred.get_data());
                    image::image next(previous.get_rows() / 2, previous.get_cols() / 2);
                    image::resize(blurred.get_data(), blurred.get_cols(), blurred.get_rows(), next.get_cols(), next.get_rows(), next.get_data());
                    this->image_pyramid.push_back(next);
                }

                const image::image& image_grey = this->image_pyramid.back();

                // Detect features.
                std::vector<feature::point> kps;
                kps.resize(50000);
                const int feature_count = feature::detect(image_grey.get_data(), image_grey.get_cols(), image_grey.get_rows(), image_grey.get_cols(), 7, static_cast<int>(kps.size()), kps.data());
                kps.resize(static_cast<size_t>(feature_count));

                // Prune edge features.
                size_t prune_edge_count = kps.size();
                prune(kps.data(), prune_edge_count, [&image_grey](const feature::point& feature) {
                    return (feature.x < 16) || (feature.x > static_cast<int>(image_grey.get_cols()) - 16) || (feature.y < 16) || (feature.y > static_cast<int>(image_grey.get_rows()) - 16);
                });
                kps.resize(static_cast<size_t>(prune_edge_count));

                // Score features.
                for (size_t i = 0; i < static_cast<size_t>(feature_count); ++i) {
                    const unsigned char* feature = image_grey.get_data() + static_cast<size_t>(kps[i].y) * image_grey.get_cols() + static_cast<size_t>(kps[i].x);
                    const float response = feature::score(feature, image_grey.get_cols());
                    kps[i].response = response;
                }

                // Prune low score features.
                size_t prune_score_count = kps.size();
                prune(kps.data(), prune_score_count, [](const feature::point& feature) {
                    return feature.response < 0;
                });
                kps.resize(static_cast<size_t>(prune_score_count));

                // Non maximally suppress features.
                // Note: This suppression function assumes points are sorted in row major.
                feature::sort(kps.data(), kps.size(), [](const feature::point& lhs, const feature::point& rhs) {
                    return lhs.y == rhs.y ? lhs.x < rhs.x : lhs.y < rhs.y;
                });
                std::vector<feature::point> features_suppressed(kps.size());
                const int suppressed_count = feature::suppress(kps.data(), kps.size(), image_grey.get_rows(), features_suppressed.data());
                features_suppressed.resize(suppressed_count);
                kps = std::move(features_suppressed);

                // Distribute features.
                // Note: This distribution function assumes points are sorted by response.
                feature::sort(kps.data(), kps.size(), [](const feature::point& lhs, const feature::point& rhs) {
                    return lhs.response > rhs.response;
                });
                std::vector<feature::point> features_distributed(2000);
                const int distributed_count = distribute(kps.data(), static_cast<int>(kps.size()), image_grey.get_cols(), image_grey.get_rows(), 500, static_cast<int>(features_distributed.size()), features_distributed.data());
                features_distributed.resize(static_cast<size_t>(distributed_count));
                kps = std::move(features_distributed);

                // Describe features.
                std::vector<feature::descriptor> des;
                des.resize(static_cast<size_t>(distributed_count));
                for (size_t i = 0; i < static_cast<size_t>(distributed_count); ++i) {
                    const unsigned char* feature = image_grey.get_data() + static_cast<size_t>(kps[i].y) * image_grey.get_cols() + static_cast<size_t>(kps[i].x);
                    // Attempt subpixel refinement of feature.
                    float offset_x = 0;
                    float offset_y = 0;
                    if (!feature::refine(feature, image_grey.get_cols(), offset_x, offset_y)) {
                        // If unsuccessful, proceed using unrefined feature.
                        const float angle = feature::dominant_angle(feature, image_grey.get_cols());
                        feature::describe(feature, image_grey.get_cols(), angle, des[i]);
                        continue;
                    }

                    // If successful, calculate descriptors from a subpixel patch.
                    unsigned char patch[31][31];
                    feature::patch_bilinear(feature, image_grey.get_cols(), offset_x, offset_y, &patch[0][0]);
                    const float angle = feature::dominant_angle(&patch[15][15], 31);
                    feature::describe(&patch[15][15], 31, angle, des[i]);
                    kps[i].x += offset_x;
                    kps[i].y += offset_y;
                }
                this->keypoint_pyramid.push_back(std::move(kps));
                this->descriptor_pyramid.push_back(std::move(des));
            }
        }
    };
}

#endif // FRAME_HPP