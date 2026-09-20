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

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        mapping::frame frame;
    }
    {
        const math::matrix<double, 3, 3> intrinsics = { { { 1.0, 0.0, 0.5 }, { 0.0, 1.0, 0.5 }, { 0.0, 0.0, 1.0 } } };
        sensor::model camera(std::vector<double>{ intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }.data(), 12);
        image::image image(256, 256);
        for (size_t i = 0; i < image.get_rows(); ++i) {
            for (size_t j = 0; j < image.get_cols(); ++j) {
                if ((i < 64) || (i >= image.get_rows() - 64) || (j < 64) || (j >= image.get_cols() - 64)) {
                    image.get_data()[i * image.get_cols() + j] = 0;
                }
                else {
                    image.get_data()[i * image.get_cols() + j] = 255;
                }
            }
        }
        mapping::frame frame_0(0, camera, image);
        REQUIRE(frame_0.id == 0);

        mapping::frame::settings frontend;
        frontend.detector = mapping::frame::settings::detector_kind::structure_tensor;
        frontend.refiner = mapping::frame::settings::refiner_kind::structure_tensor;
        for (feature::score::structure_tensor::measure kind : { feature::score::structure_tensor::measure::klt, feature::score::structure_tensor::measure::harris, feature::score::structure_tensor::measure::rohr }) {
            frontend.detector_measure = kind;
            frontend.refiner_measure = kind;
            mapping::frame frame_tensor(0, camera, image, frontend);
            REQUIRE(!frame_tensor.keypoints.empty());
            REQUIRE(frame_tensor.keypoints.size() == frame_tensor.descriptors.size());
        }
        frontend.refiner = mapping::frame::settings::refiner_kind::none;
        mapping::frame frame_unrefined(0, camera, image, frontend);
        REQUIRE(!frame_unrefined.keypoints.empty());
        frontend.fixed_budget = true;
        mapping::frame frame_budgeted(0, camera, image, frontend);
        REQUIRE(!frame_budgeted.keypoints.empty());
        REQUIRE(frame_budgeted.keypoints.size() <= frame_unrefined.keypoints.size());

        image::image textured(480, 640);
        unsigned int state = 2026u;
        for (size_t i = 0; i < textured.get_rows() * textured.get_cols(); ++i) {
            state = (state * 1664525u) + 1013904223u;
            textured.get_data()[i] = static_cast<unsigned char>(state >> 24);
        }
        mapping::frame::settings budget;
        budget.fixed_budget = true;
        mapping::frame frame_textured(0, camera, textured, budget);
        REQUIRE(frame_textured.keypoint_pyramid.size() == frame_textured.descriptor_pyramid.size());
        for (size_t level = 0; level < frame_textured.keypoint_pyramid.size(); ++level) {
            REQUIRE(frame_textured.keypoint_pyramid[level].size() == frame_textured.descriptor_pyramid[level].size());
        }
    }

    {
        const math::matrix<double, 3, 3> intrinsics = { { { 1.0, 0.0, 0.5 }, { 0.0, 1.0, 0.5 }, { 0.0, 0.0, 1.0 } } };
        sensor::model camera(std::vector<double>{ intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }.data(), 12);

        image::image tiny(16, 16);
        for (size_t i = 0; i < tiny.get_rows(); ++i) {
            for (size_t j = 0; j < tiny.get_cols(); ++j) {
                tiny.get_data()[i * tiny.get_cols() + j] = static_cast<unsigned char>((i * 16 + j) % 256);
            }
        }
        mapping::frame frame_tiny(0, camera, tiny);
        REQUIRE(frame_tiny.image_pyramid.size() >= 1);
        REQUIRE(frame_tiny.keypoint_pyramid.size() >= 1);
        REQUIRE(frame_tiny.descriptor_pyramid.size() >= 1);

        image::image small(48, 48);
        for (size_t i = 0; i < small.get_rows(); ++i) {
            for (size_t j = 0; j < small.get_cols(); ++j) {
                small.get_data()[i * small.get_cols() + j] = static_cast<unsigned char>((i + j) % 256);
            }
        }
        mapping::frame frame_small(1, camera, small);
        REQUIRE(frame_small.image_pyramid.size() >= 1);
        REQUIRE(frame_small.keypoint_pyramid.size() >= 1);
        REQUIRE(frame_small.descriptor_pyramid.size() >= 1);
    }

    {
        const math::matrix<double, 3, 3> intrinsics = { { { 320.0, 0.0, 320.0 }, { 0.0, 320.0, 240.0 }, { 0.0, 0.0, 1.0 } } };
        sensor::model camera(std::vector<double>{ intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }.data(), 12);
        image::image textured(480, 640);
        unsigned int state = 12345u;
        for (size_t i = 0; i < textured.get_rows(); ++i) {
            for (size_t j = 0; j < textured.get_cols(); ++j) {
                state = (state * 1664525u) + 1013904223u;
                const unsigned int checker = (((i / 8) + (j / 8)) % 2) ? 200u : 40u;
                textured.get_data()[i * textured.get_cols() + j] = static_cast<unsigned char>((checker + ((state >> 16) % 32u)) % 256u);
            }
        }
        mapping::frame::settings frontend;
        frontend.refiner = mapping::frame::settings::refiner_kind::none;
        const mapping::frame unrefined(0, camera, textured, frontend);
        REQUIRE(!unrefined.keypoints.empty());
        for (size_t level = 0; level < unrefined.keypoint_pyramid.size(); ++level) {
            for (const feature::point& local : unrefined.keypoint_pyramid[level]) {
                REQUIRE(local.x == static_cast<float>(static_cast<int>(local.x)));
                REQUIRE(local.y == static_cast<float>(static_cast<int>(local.y)));
            }
        }
        bool coarse_seen = false;
        for (const feature::point& exported : unrefined.keypoints) {
            coarse_seen = coarse_seen || (exported.octave > 0);
            REQUIRE(exported.x - static_cast<float>(static_cast<int>(exported.x)) == 0.5f);
            REQUIRE(exported.y - static_cast<float>(static_cast<int>(exported.y)) == 0.5f);
        }
        REQUIRE(coarse_seen);
    }

    {
        const math::matrix<double, 3, 3> intrinsics = { { { 320.0, 0.0, 320.0 }, { 0.0, 320.0, 240.0 }, { 0.0, 0.0, 1.0 } } };
        sensor::model camera(std::vector<double>{ intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }.data(), 12);

        image::image textured(480, 640);
        unsigned int state = 12345u;
        for (size_t i = 0; i < textured.get_rows(); ++i) {
            for (size_t j = 0; j < textured.get_cols(); ++j) {
                state = (state * 1664525u) + 1013904223u;
                const unsigned int checker = (((i / 8) + (j / 8)) % 2) ? 200u : 40u;
                textured.get_data()[i * textured.get_cols() + j] = static_cast<unsigned char>((checker + ((state >> 16) % 32u)) % 256u);
            }
        }

        const mapping::frame built(0, camera, textured);
        REQUIRE(built.image_pyramid.size() == 4);
        REQUIRE(built.image_pyramid[1].get_cols() == 320);
        REQUIRE(built.image_pyramid[1].get_rows() == 240);
        REQUIRE(built.image_pyramid[2].get_cols() == 160);
        REQUIRE(built.image_pyramid[2].get_rows() == 120);
        REQUIRE(built.image_pyramid[3].get_cols() == 80);
        REQUIRE(built.image_pyramid[3].get_rows() == 60);
        REQUIRE(built.image_pyramid.scale_x(0) == 1.0f);
        REQUIRE(built.image_pyramid.scale_y(0) == 1.0f);
        REQUIRE(built.image_pyramid.scale_x(1) == 2.0f);
        REQUIRE(built.image_pyramid.scale_x(2) == 4.0f);
        REQUIRE(built.image_pyramid.back().get_rows() >= image::pyramid::minimum_dimension);

        size_t expected_total = 0;
        for (size_t level = 0; level < built.keypoint_pyramid.size(); ++level) {
            expected_total += built.keypoint_pyramid[level].size();
        }
        REQUIRE(built.keypoints.size() == expected_total);
        REQUIRE(built.descriptors.size() == expected_total);
        REQUIRE(!built.keypoint_pyramid[0].empty());
        REQUIRE(built.keypoints.size() > built.keypoint_pyramid[0].size());
        for (size_t i = 1; i < built.keypoints.size(); ++i) {
            REQUIRE(built.keypoints[i].octave >= built.keypoints[i - 1].octave);
        }
        size_t flat_index = 0;
        for (size_t level = 0; level < built.keypoint_pyramid.size(); ++level) {
            for (size_t i = 0; i < built.keypoint_pyramid[level].size(); ++i) {
                const feature::point& exported = built.keypoints[flat_index];
                const feature::point& local = built.keypoint_pyramid[level][i];
                REQUIRE(exported.octave == static_cast<int>(level));
                REQUIRE(local.octave == static_cast<int>(level));
                REQUIRE(exported.x == (local.x * built.image_pyramid.scale_x(level)) + 0.5f);
                REQUIRE(exported.y == (local.y * built.image_pyramid.scale_y(level)) + 0.5f);
                REQUIRE(local.x >= 0.0f);
                REQUIRE(local.y >= 0.0f);
                REQUIRE(local.x < static_cast<float>(built.image_pyramid[level].get_cols()));
                REQUIRE(local.y < static_cast<float>(built.image_pyramid[level].get_rows()));
                REQUIRE(exported.response == local.response);
                REQUIRE(exported.x >= 0.0f);
                REQUIRE(exported.y >= 0.0f);
                REQUIRE(exported.x < static_cast<float>(textured.get_cols()));
                REQUIRE(exported.y < static_cast<float>(textured.get_rows()));
                ++flat_index;
            }
        }

        REQUIRE(built.keypoint_pyramid[0].size() <= static_cast<size_t>(4 * mapping::frame::level0_features));
        for (size_t level = 1; level < built.keypoint_pyramid.size(); ++level) {
            REQUIRE(built.keypoint_pyramid[level].size() <= built.keypoint_pyramid[level - 1].size());
        }
    }

    return EXIT_SUCCESS;
}
