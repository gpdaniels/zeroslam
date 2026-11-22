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

#include "sensor/camera.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

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
        static_assert(sensor::pinhole::parameter_count == 4, "The pinhole alias must name the pinhole camera.");
        static_assert(sensor::camera::pinhole_opencv<double>::parameter_count <= sensor::model::maximum_parameter_count, "The largest model must fit a buffer of the maximum parameter count.");
        static_assert(sensor::camera::fisheye_624<double>::parameter_count <= sensor::model::maximum_parameter_count, "The largest model must fit a buffer of the maximum parameter count.");
        static_assert(sizeof(sensor::camera::pinhole_opencv<double>) <= sensor::model::maximum_size, "The largest model must fit the erased model.");
        const double parameters[12] = { 0.8, 0.8, 0.5, 0.375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        const sensor::model model(sensor::camera::pinhole_radial_tangential<double>(parameters, 12));
        REQUIRE(model.get_parameter_count() == 12);
    }

    {
        // Every model is made by its name, with its parameters, and every one fits the erased model.
        struct named_model {
            const char* name;
            size_t count;
            double parameters[18];
        };

        const named_model models[] = {
            { "pinhole", 4, { 400.0, 410.0, 320.5, 240.5 } },
            { "pinhole_radial_tangential", 12, { 400.0, 410.0, 320.5, 240.5, -0.2, 0.05, 0.001, -0.001, 0.0, 0.0, 0.0, 0.0 } },
            { "pinhole_opencv", 18, { 400.0, 410.0, 320.5, 240.5, -0.2, 0.05, 0.001, -0.001, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, -0.001, 0.0, 0.01, 0.0 } },
            { "kannala_brandt", 8, { 300.0, 300.0, 320.5, 240.5, 0.01, -0.002, 0.0, 0.0 } },
            { "fisheye_624", 16, { 241.0, 241.0, 316.9, 237.3, -0.02, 0.1, -0.07, 0.01, 0.003, -0.001, 0.0003, -0.0002, -0.0005, 0.0, 0.0003, 0.0 } },
            { "field_of_view", 5, { 300.0, 310.0, 320.5, 240.5, 0.9 } },
            { "unified", 5, { 350.0, 360.0, 320.5, 240.5, 0.8 } },
            { "extended_unified", 6, { 300.0, 310.0, 320.5, 240.5, 0.6, 1.1 } },
            { "double_sphere", 6, { 349.7, 349.6, 365.6, 249.9, -0.2, 0.56 } }
        };
        for (const named_model& entry : models) {
            sensor::model model;
            REQUIRE(sensor::camera::make_model(entry.name, &entry.parameters[0], entry.count, model));
            REQUIRE(std::strcmp(model.name(), entry.name) == 0);
            REQUIRE(model.get_parameter_count() == entry.count);
            double recovered[18];
            REQUIRE(model.get_parameters(&recovered[0], entry.count));
            for (size_t index = 0; index < entry.count; ++index) {
                REQUIRE(recovered[index] == entry.parameters[index]);
            }
            const double point[3] = { 0.1, -0.2, 2.0 };
            double pixel[2];
            REQUIRE(model.project(&point[0], &pixel[0]));
            double ray[3];
            REQUIRE(model.unproject(&pixel[0], &ray[0]));
            REQUIRE(std::abs((ray[0] / ray[2]) - 0.05) < 1e-9);
            REQUIRE(std::abs((ray[1] / ray[2]) + 0.1) < 1e-9);
            // A wrong count leaves the model as it was.
            REQUIRE(!sensor::camera::make_model(entry.name, &entry.parameters[0], entry.count - 1, model));
            REQUIRE(std::strcmp(model.name(), entry.name) == 0);
            // Every model says what its parameters are: the focal lengths and principal point first, then distortion.
            sensor::camera::parameter_role roles[18];
            REQUIRE(!model.get_parameter_roles(&roles[0], entry.count + 1));
            REQUIRE(model.get_parameter_roles(&roles[0], entry.count));
            REQUIRE(roles[0] == sensor::camera::parameter_role::focal_length);
            REQUIRE(roles[1] == sensor::camera::parameter_role::focal_length);
            REQUIRE(roles[2] == sensor::camera::parameter_role::principal_point);
            REQUIRE(roles[3] == sensor::camera::parameter_role::principal_point);
            for (size_t index = 4; index < entry.count; ++index) {
                REQUIRE((roles[index] == sensor::camera::parameter_role::distortion) || (roles[index] == sensor::camera::parameter_role::distortion_higher));
            }
            REQUIRE((entry.count == 4) || (roles[4] == sensor::camera::parameter_role::distortion));
            double intrinsics[4];
            sensor::camera::pinhole_of(model, intrinsics);
            REQUIRE(intrinsics[2] == entry.parameters[2]);
            REQUIRE(intrinsics[3] == entry.parameters[3]);
        }
        sensor::model model;
        REQUIRE(!sensor::camera::make_model("pinhole_radial", &models[0].parameters[0], 4, model));
        REQUIRE(!sensor::camera::make_model("pinholes", &models[0].parameters[0], 4, model));
        REQUIRE(!sensor::camera::make_model(nullptr, &models[0].parameters[0], 4, model));
        REQUIRE(!sensor::camera::make_model("pinhole", nullptr, 4, model));
        REQUIRE(!model.is_valid());
        const double negative[4] = { -400.0, 410.0, 320.5, 240.5 };
        REQUIRE(!sensor::camera::make_model("pinhole", &negative[0], 4, model));
        double intrinsics[4];
        sensor::camera::pinhole_of(model, intrinsics);
        REQUIRE(intrinsics[0] == 1.0);
        REQUIRE(intrinsics[1] == 1.0);
        REQUIRE(intrinsics[2] == 0.0);
        REQUIRE(intrinsics[3] == 0.0);
    }

    {
        // The image plane takes rays within 80 degrees of the optical axis, whatever their length.
        const double straight[3] = { 0.0, 0.0, 1.0 };
        const double unit_depth[3] = { 5.6, 0.0, 1.0 };
        const double too_far[3] = { 5.8, 0.0, 1.0 };
        const double sideways[3] = { 1.0, 0.0, 0.0 };
        const double behind[3] = { 0.0, 0.0, -1.0 };
        REQUIRE(sensor::camera::on_image_plane(&straight[0]));
        REQUIRE(sensor::camera::on_image_plane(&unit_depth[0]));
        REQUIRE(!sensor::camera::on_image_plane(&too_far[0]));
        REQUIRE(!sensor::camera::on_image_plane(&sideways[0]));
        REQUIRE(!sensor::camera::on_image_plane(&behind[0]));
        for (const double degrees : { 79.9, 80.1 }) {
            const double angle = degrees * 3.141592653589793 / 180.0;
            const double ray[3] = { 3.0 * std::sin(angle), 0.0, 3.0 * std::cos(angle) };
            REQUIRE(sensor::camera::on_image_plane(&ray[0]) == (degrees < 80.0));
        }
    }

    return EXIT_SUCCESS;
}
