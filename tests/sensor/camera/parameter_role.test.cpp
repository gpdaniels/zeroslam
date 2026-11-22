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

#include "sensor/camera/parameter_role.hpp"

#include "sensor/camera/model.hpp"
#include "sensor/camera/pinhole.hpp"
#include "sensor/camera/pinhole_radial_tangential.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

// A model that does not say what its parameters are.
class anonymous_model final {
public:
    constexpr static const char* name = "anonymous";

    size_t get_parameter_count() const {
        return 1;
    }

    bool set_parameters(const double* const parameters, const size_t parameters_length) {
        return (parameters != nullptr) && (parameters_length == 1);
    }

    bool get_parameters(double* const parameters, const size_t parameters_length) const {
        if ((parameters == nullptr) || (parameters_length != 1)) {
            return false;
        }
        parameters[0] = 1.0;
        return true;
    }

    bool project(const double* const point_xyz, double* const point_xy, double* const jacobian_projection, double* const jacobian_parameters) const {
        static_cast<void>(jacobian_projection);
        static_cast<void>(jacobian_parameters);
        point_xy[0] = point_xyz[0] / point_xyz[2];
        point_xy[1] = point_xyz[1] / point_xyz[2];
        return true;
    }

    bool unproject(const double* const point_xy, double* const ray_xyz, double* const jacobian_unprojection) const {
        static_cast<void>(jacobian_unprojection);
        ray_xyz[0] = point_xy[0];
        ray_xyz[1] = point_xy[1];
        ray_xyz[2] = 1.0;
        return true;
    }
};

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    using sensor::camera::parameter_role;

    {
        // The roles are distinct values, unknown being the default of a value initialised role.
        const parameter_role defaulted{};
        REQUIRE(defaulted == parameter_role::unknown);
        REQUIRE(parameter_role::focal_length != parameter_role::principal_point);
        REQUIRE(parameter_role::distortion != parameter_role::distortion_higher);
    }

    {
        const sensor::camera::model<double> camera{ sensor::camera::pinhole<double>() };
        parameter_role roles[4] = {};
        REQUIRE(camera.get_parameter_roles(&roles[0], 4));
        REQUIRE(roles[0] == parameter_role::focal_length);
        REQUIRE(roles[1] == parameter_role::focal_length);
        REQUIRE(roles[2] == parameter_role::principal_point);
        REQUIRE(roles[3] == parameter_role::principal_point);
        REQUIRE(!camera.get_parameter_roles(&roles[0], 3));
        REQUIRE(!camera.get_parameter_roles(nullptr, 4));
    }

    {
        const sensor::camera::model<float> camera{ sensor::camera::pinhole_radial_tangential<float>() };
        parameter_role roles[12] = {};
        REQUIRE(camera.get_parameter_roles(&roles[0], 12));
        const parameter_role expected[12] = {
            parameter_role::focal_length,
            parameter_role::focal_length,
            parameter_role::principal_point,
            parameter_role::principal_point,
            parameter_role::distortion,
            parameter_role::distortion,
            parameter_role::distortion,
            parameter_role::distortion,
            parameter_role::distortion_higher,
            parameter_role::distortion_higher,
            parameter_role::distortion_higher,
            parameter_role::distortion_higher
        };
        for (size_t index = 0; index < 12; ++index) {
            REQUIRE(roles[index] == expected[index]);
        }
        REQUIRE(!camera.get_parameter_roles(&roles[0], 4));
    }

    {
        // A model may leave its roles unsaid, and an empty model has none.
        const sensor::camera::model<double> camera{ anonymous_model() };
        REQUIRE(camera.is_valid());
        parameter_role roles[1] = { parameter_role::focal_length };
        REQUIRE(!camera.get_parameter_roles(&roles[0], 1));
        REQUIRE(roles[0] == parameter_role::focal_length);
        const sensor::camera::model<double> empty;
        REQUIRE(!empty.get_parameter_roles(&roles[0], 1));
    }

    return EXIT_SUCCESS;
}
