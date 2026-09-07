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

#include "sensor/camera/model.hpp"

#include "sensor/camera/pinhole.hpp"
#include "sensor/camera/pinhole_radial_tangential.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

template <typename type>
static void exercise() {
    using model = sensor::camera::model<type>;
    const type pinhole_parameters[4] = { type(0.8), type(0.8), type(0.5), type(0.375) };
    const type radial_parameters[12] = { type(0.8), type(0.8), type(0.5), type(0.375), type(-0.2), type(0.05), type(0.001), type(-0.001), type(0), type(0), type(0), type(0) };
    const type point[3] = { type(0.1), type(-0.2), type(2) };

    {
        REQUIRE(std::strcmp(sensor::camera::pinhole<type>::name, "pinhole") == 0);
        REQUIRE(std::strcmp(sensor::camera::pinhole_radial_tangential<type>::name, "pinhole_radial_tangential") == 0);
        REQUIRE(std::strcmp(model(sensor::camera::pinhole<type>()).name(), "pinhole") == 0);
        REQUIRE(std::strcmp(model(sensor::camera::pinhole_radial_tangential<type>()).name(), "pinhole_radial_tangential") == 0);
    }

    {
        model empty;
        REQUIRE(!empty.is_valid());
        REQUIRE(empty.get_parameter_count() == 0);
        REQUIRE(empty.name()[0] == 0);
        type pixel[2] = { type(7), type(7) };
        REQUIRE(!empty.project(point, pixel));
        type ray[3] = {};
        REQUIRE(!empty.unproject(pixel, ray));
        REQUIRE(!empty.set_parameters(pinhole_parameters, 4));
        type parameters[4] = { type(3), type(3), type(3), type(3) };
        REQUIRE(!empty.get_parameters(parameters, 4));
        REQUIRE(parameters[0] == type(3));
        const model copy(empty);
        REQUIRE(!copy.is_valid());
    }

    {
        const sensor::camera::pinhole<type> concrete(pinhole_parameters, 4);
        const model erased(concrete);
        REQUIRE(erased.is_valid());
        REQUIRE(erased.get_parameter_count() == 4);
        REQUIRE(std::strcmp(erased.name(), "pinhole") == 0);
        REQUIRE(std::strcmp(erased.name(), sensor::camera::pinhole<type>::name) == 0);
        type parameters[4] = {};
        REQUIRE(!erased.get_parameters(parameters, 3));
        REQUIRE(!erased.get_parameters(nullptr, 4));
        REQUIRE(erased.get_parameters(parameters, 4));
        for (size_t i = 0; i < 4; ++i) {
            REQUIRE(parameters[i] == pinhole_parameters[i]);
        }
        type pixel_direct[2] = {};
        type pixel_erased[2] = {};
        type jacobian_direct[6] = {};
        type jacobian_erased[6] = {};
        REQUIRE(concrete.project(point, pixel_direct, jacobian_direct));
        REQUIRE(erased.project(point, pixel_erased, jacobian_erased));
        REQUIRE(pixel_direct[0] == pixel_erased[0]);
        REQUIRE(pixel_direct[1] == pixel_erased[1]);
        for (size_t k = 0; k < 6; ++k) {
            REQUIRE(jacobian_direct[k] == jacobian_erased[k]);
        }
        type ray_direct[3] = {};
        type ray_erased[3] = {};
        REQUIRE(concrete.unproject(pixel_direct, ray_direct));
        REQUIRE(erased.unproject(pixel_erased, ray_erased));
        for (size_t k = 0; k < 3; ++k) {
            REQUIRE(ray_direct[k] == ray_erased[k]);
        }
        const type behind[3] = { type(0.1), type(0.1), type(-1) };
        REQUIRE(!erased.project(behind, pixel_erased));
    }

    {
        model erased = sensor::camera::pinhole_radial_tangential<type>(radial_parameters, 12);
        REQUIRE(erased.get_parameter_count() == 12);
        REQUIRE(std::strcmp(erased.name(), "pinhole_radial_tangential") == 0);
        model moved(static_cast<model&&>(model(erased)));
        REQUIRE(moved.is_valid());
        REQUIRE(std::strcmp(moved.name(), "pinhole_radial_tangential") == 0);
        model move_assigned;
        move_assigned = static_cast<model&&>(moved);
        REQUIRE(move_assigned.get_parameter_count() == 12);
        REQUIRE(!erased.set_parameters(pinhole_parameters, 4));
        REQUIRE(erased.get_parameter_count() == 12);
        model copy(erased);
        model assigned;
        assigned = erased;
        type pixel[3][2] = {};
        REQUIRE(erased.project(point, pixel[0]));
        REQUIRE(copy.project(point, pixel[1]));
        REQUIRE(assigned.project(point, pixel[2]));
        REQUIRE(pixel[0][0] == pixel[1][0]);
        REQUIRE(pixel[0][0] == pixel[2][0]);
        REQUIRE(pixel[0][1] == pixel[1][1]);
        REQUIRE(pixel[0][1] == pixel[2][1]);
        const type changed[12] = { type(0.9), type(0.9), type(0.5), type(0.375), type(0), type(0), type(0), type(0), type(0), type(0), type(0), type(0) };
        REQUIRE(copy.set_parameters(changed, 12));
        REQUIRE(copy.project(point, pixel[1]));
        REQUIRE(pixel[0][0] != pixel[1][0]);
        REQUIRE(erased.project(point, pixel[2]));
        REQUIRE(pixel[0][0] == pixel[2][0]);
        assigned = model(sensor::camera::pinhole<type>(pinhole_parameters, 4));
        REQUIRE(assigned.get_parameter_count() == 4);
        assigned.clear();
        REQUIRE(!assigned.is_valid());
    }

    {
        const type zero_distortion[12] = { type(0.8), type(0.8), type(0.5), type(0.375), type(0), type(0), type(0), type(0), type(0), type(0), type(0), type(0) };
        const model models[2] = { model(sensor::camera::pinhole<type>(pinhole_parameters, 4)), model(sensor::camera::pinhole_radial_tangential<type>(zero_distortion, 12)) };
        for (int i = -5; i <= 5; ++i) {
            for (int j = -5; j <= 5; ++j) {
                const type sample[3] = { type(i) * type(0.07), type(j) * type(0.05), type(1.5) };
                type pixel[2][2] = {};
                type jacobian[2][6] = {};
                type ray[2][3] = {};
                for (int m = 0; m < 2; ++m) {
                    REQUIRE(models[m].project(sample, pixel[m], jacobian[m]));
                    REQUIRE(models[m].unproject(pixel[m], ray[m]));
                }
                REQUIRE(pixel[0][0] == pixel[1][0]);
                REQUIRE(pixel[0][1] == pixel[1][1]);
                for (size_t k = 0; k < 6; ++k) {
                    REQUIRE(jacobian[0][k] == jacobian[1][k]);
                }
                REQUIRE(std::abs(static_cast<double>(ray[0][0] - ray[1][0])) < 1e-6);
                REQUIRE(std::abs(static_cast<double>(ray[0][1] - ray[1][1])) < 1e-6);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(sensor::camera::pinhole<float>) <= sensor::camera::model<float>::maximum_size, "The pinhole model must fit.");
    static_assert(sizeof(sensor::camera::pinhole<double>) <= sensor::camera::model<double>::maximum_size, "The pinhole model must fit.");
    static_assert(sizeof(sensor::camera::pinhole_radial_tangential<float>) <= sensor::camera::model<float>::maximum_size, "The pinhole radial tangential model must fit.");
    static_assert(sizeof(sensor::camera::pinhole_radial_tangential<double>) <= sensor::camera::model<double>::maximum_size, "The pinhole radial tangential model must fit.");
    exercise<double>();
    exercise<float>();

    return EXIT_SUCCESS;
}
