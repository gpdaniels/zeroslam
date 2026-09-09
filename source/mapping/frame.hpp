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
#include "image/image.hpp"
#include "image/pyramid.hpp"
#include "math/matrix.hpp"
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

    public:
        int id;
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

        frame();

        frame(int input_id, const sensor::model& camera_intrinsics, const image::image& input_image_grey);
    };
}

#endif // ZEROSLAM_MAPPING_FRAME_HPP
