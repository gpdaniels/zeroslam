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

#include "mapping/rig.hpp"

#include "math/lie.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/extrinsic.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-9) {
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

static bool is_transform_approx(const math::se3<double>& lhs, const math::se3<double>& rhs) {
    const math::matrix<double, 6, 1> error = (lhs * rhs.inverse()).log();
    for (size_t i = 0; i < 6; ++i) {
        if (!(std::abs(error[i]) < 1e-9)) {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        // A monocular rig: one camera, the reference, at the identity, with no metric scale.
        mapping::rig mono;
        REQUIRE(mono.reference() == -1);
        REQUIRE(mono.camera_count() == 0);
        const size_t camera = mono.add(1, mapping::rig::sensor_kind::camera, math::se3<double>::identity(), true);
        REQUIRE(camera == 0);
        REQUIRE(mono.reference() == 0);
        REQUIRE(mono.camera_count() == 1);
        REQUIRE(mono.camera(0) == 0);
        REQUIRE(!mono.metric());
        REQUIRE(!mono.refines(0));
        const optimisation::vertex node = mono.extrinsic_vertex(0);
        REQUIRE(node.is_fixed());
        REQUIRE(node.get_parameters()[6] == 1.0);
    }

    {
        // An imu body with two cameras on it, as EuRoC: the extrinsics are given in the body's frame and read in the first
        // camera's, which is the reference whatever its own extrinsic.
        const math::se3<double> body_from_left(math::so3<double>::exp({ { 0.01, -0.02, 1.56 } }), { { -0.02, -0.06, 0.01 } });
        const math::se3<double> body_from_right(math::so3<double>::exp({ { 0.012, -0.018, 1.57 } }), { { -0.02, 0.05, 0.01 } });
        mapping::rig euroc;
        const size_t imu = euroc.add(7, mapping::rig::sensor_kind::imu, math::se3<double>::identity(), false);
        REQUIRE(euroc.reference() == -1);
        const size_t left = euroc.add(1, mapping::rig::sensor_kind::camera, body_from_left, false);
        const size_t right = euroc.add(2, mapping::rig::sensor_kind::camera, body_from_right, true);
        REQUIRE(euroc.reference() == static_cast<int>(left));
        REQUIRE(euroc.camera_count() == 2);
        REQUIRE(euroc.camera(0) == left);
        REQUIRE(euroc.camera(1) == right);
        size_t found = 99;
        REQUIRE(euroc.find(2, found) && (found == right));
        REQUIRE(!euroc.find(3, found));
        REQUIRE(is_transform_approx(euroc.reference_from_sensor(left), math::se3<double>::identity()));
        REQUIRE(is_transform_approx(euroc.reference_from_sensor(right), body_from_left.inverse() * body_from_right));
        REQUIRE(is_transform_approx(euroc.reference_from_sensor(imu), body_from_left.inverse()));
        // The baseline, about 11 cm, gives the map a metric scale, and each distance is kept as it was declared.
        REQUIRE(euroc.metric());
        REQUIRE(is_value_approx(euroc.at(right).declared_distance, std::sqrt((body_from_left.inverse() * body_from_right).translation().get_length_squared())));
        REQUIRE(is_value_approx(euroc.at(imu).declared_distance, std::sqrt(body_from_left.translation().get_length_squared())));

        // Only the refined sensor's vertex is free, and what an adjustment leaves in it moves the sensor in the reference
        // frame while the reference stays where it is on the body.
        REQUIRE(euroc.refines(right));
        REQUIRE(!euroc.refines(left));
        REQUIRE(!euroc.refines(imu));
        optimisation::vertex node = euroc.extrinsic_vertex(right);
        REQUIRE(!node.is_fixed());
        REQUIRE(euroc.extrinsic_vertex(left).is_fixed());
        const double step[6] = { 0.01, -0.02, 0.005, 0.001, 0.002, -0.003 };
        node.plus(&step[0]);
        euroc.read_extrinsic_vertex(right, node);
        REQUIRE(is_transform_approx(euroc.reference_from_sensor(right), optimisation::vertices::extrinsic::from_parameters(node.get_parameters())));
        REQUIRE(is_transform_approx(euroc.at(left).body_from_sensor, body_from_left));
        // A held sensor's vertex is not read back.
        optimisation::vertex held = euroc.extrinsic_vertex(imu);
        held.set_fixed(false);
        held.plus(&step[0]);
        euroc.read_extrinsic_vertex(imu, held);
        REQUIRE(is_transform_approx(euroc.at(imu).body_from_sensor, math::se3<double>::identity()));
    }

    {
        // Two cameras at the same place, a depth sensor tied to the first: no metric baseline between cameras.
        mapping::rig coincident;
        coincident.add(1, mapping::rig::sensor_kind::camera, math::se3<double>::identity(), false);
        const size_t depth = coincident.add(5, mapping::rig::sensor_kind::depth, math::se3<double>(math::so3<double>::identity(), { { 0.025, 0.0, 0.0 } }), false, 1);
        coincident.add(2, mapping::rig::sensor_kind::camera, math::se3<double>(math::so3<double>::exp({ { 0.0, 1.5, 0.0 } }), { { 0.0, 0.0, 0.0 } }), false);
        REQUIRE(coincident.at(depth).attached_to == 1);
        REQUIRE(!coincident.metric());
        coincident.clear();
        REQUIRE(coincident.size() == 0);
        REQUIRE(coincident.reference() == -1);
    }

    return EXIT_SUCCESS;
}
