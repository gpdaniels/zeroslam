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

#include "optimisation/vertices/intrinsics.hpp"

#include "optimisation/vertex.hpp"
#include "optimisation/vertices/pose.hpp"
#include "sensor/camera/model.hpp"
#include "sensor/camera/pinhole.hpp"
#include "sensor/camera/pinhole_radial_tangential.hpp"

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

// Whether the camera projects a point where a camera made afresh from the parameters does.
static bool projects_as(const sensor::camera::model<double>& camera, const double* const parameters) {
    const sensor::camera::pinhole_radial_tangential<double> fresh(parameters, 12);
    const double point[3] = { 0.3, -0.2, 2.0 };
    double expected[2] = {};
    double projected[2] = {};
    return fresh.project(&point[0], &expected[0]) && camera.project(&point[0], &projected[0]) && (expected[0] == projected[0]) && (expected[1] == projected[1]);
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(optimisation::vertices::intrinsics) <= optimisation::vertex::maximum_size, "The intrinsics vertex must fit.");

    const double initial[12] = { 500.0, 510.0, 320.0, 240.0, -0.2, 0.05, 0.001, -0.002, 0.0, 0.0, 0.0, 0.0 };
    const sensor::camera::model<double> camera{ sensor::camera::pinhole_radial_tangential<double>(&initial[0], 12) };
    // The focal lengths and the principal point free, the distortion held.
    const bool focal_centre[12] = { true, true, true, true, false, false, false, false, false, false, false, false };

    {
        optimisation::vertex vertex = optimisation::vertices::intrinsics::make(camera, &focal_centre[0], 12);
        REQUIRE(vertex.is_valid());
        REQUIRE(vertex.get_parameter_dimensions() == 12);
        REQUIRE(vertex.get_local_dimensions() == 4);
        for (size_t index = 0; index < 12; ++index) {
            REQUIRE(vertex.get_parameters()[index] == initial[index]);
        }
        const optimisation::vertices::intrinsics* const policy = vertex.get_policy<optimisation::vertices::intrinsics>();
        REQUIRE(policy != nullptr);
        REQUIRE(vertex.get_policy<optimisation::vertices::pose>() == nullptr);
        REQUIRE(policy->get_parameter_count() == 12);
        REQUIRE(policy->get_local_count() == 4);
        for (int local = 0; local < 4; ++local) {
            REQUIRE(policy->get_parameter_index(local) == local);
            REQUIRE(policy->get_local_index(local) == local);
        }
        REQUIRE(policy->get_parameter_index(4) == -1);
        REQUIRE(policy->get_local_index(4) == -1);
        REQUIRE(policy->get_local_index(11) == -1);
        REQUIRE(projects_as(policy->get_camera(), vertex.get_parameters()));

        // A step moves the free parameters only, additively, and the model with them.
        const double delta[4] = { 5.0, -2.0, 1.5, -0.5 };
        vertex.backup();
        vertex.plus(&delta[0]);
        REQUIRE(vertex.get_parameters()[0] == 505.0);
        REQUIRE(vertex.get_parameters()[1] == 508.0);
        REQUIRE(vertex.get_parameters()[2] == 321.5);
        REQUIRE(vertex.get_parameters()[3] == 239.5);
        for (size_t index = 4; index < 12; ++index) {
            REQUIRE(vertex.get_parameters()[index] == initial[index]);
        }
        REQUIRE(projects_as(policy->get_camera(), vertex.get_parameters()));

        // Restoring the backup, and setting the parameters, take the model along.
        vertex.restore();
        REQUIRE(vertex.get_parameters()[0] == 500.0);
        REQUIRE(projects_as(policy->get_camera(), &initial[0]));
        const double moved[12] = { 480.0, 490.0, 300.0, 250.0, -0.1, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        REQUIRE(vertex.set_parameters(&moved[0], 12));
        REQUIRE(projects_as(policy->get_camera(), &moved[0]));

        // A step the model refuses, to a focal length that is not positive, is not taken.
        const double refused[4] = { -600.0, 0.0, 10.0, 10.0 };
        vertex.plus(&refused[0]);
        for (size_t index = 0; index < 12; ++index) {
            REQUIRE(vertex.get_parameters()[index] == moved[index]);
        }
        REQUIRE(projects_as(policy->get_camera(), &moved[0]));

        // A copy holds a model of its own.
        optimisation::vertex copy(vertex);
        const double copy_delta[4] = { 1.0, 1.0, 1.0, 1.0 };
        copy.plus(&copy_delta[0]);
        REQUIRE(projects_as(copy.get_policy<optimisation::vertices::intrinsics>()->get_camera(), copy.get_parameters()));
        REQUIRE(projects_as(policy->get_camera(), &moved[0]));
        REQUIRE(copy.get_parameters()[0] == 481.0);
        optimisation::vertex moved_vertex(static_cast<optimisation::vertex&&>(copy));
        REQUIRE(moved_vertex.get_local_dimensions() == 4);
        REQUIRE(projects_as(moved_vertex.get_policy<optimisation::vertices::intrinsics>()->get_camera(), moved_vertex.get_parameters()));
        optimisation::vertex assigned;
        assigned = moved_vertex;
        REQUIRE(assigned.get_parameter_dimensions() == 12);
        REQUIRE(assigned.get_local_dimensions() == 4);
        REQUIRE(projects_as(assigned.get_policy<optimisation::vertices::intrinsics>()->get_camera(), assigned.get_parameters()));
    }

    {
        // The free columns of a jacobian by every parameter, in the order of the parameters.
        const bool scattered[12] = { true, false, false, true, true, false, false, false, false, false, false, true };
        const optimisation::vertices::intrinsics policy(camera, &scattered[0], 12);
        REQUIRE(policy.get_local_count() == 4);
        REQUIRE(policy.get_parameter_index(0) == 0);
        REQUIRE(policy.get_parameter_index(1) == 3);
        REQUIRE(policy.get_parameter_index(2) == 4);
        REQUIRE(policy.get_parameter_index(3) == 11);
        REQUIRE(policy.get_local_index(11) == 3);
        double full[2 * 12] = {};
        for (size_t index = 0; index < 24; ++index) {
            full[index] = static_cast<double>(index);
        }
        double selected[2 * 4] = {};
        policy.select_free_columns(&full[0], 2, -2.0, &selected[0]);
        const double expected[8] = { 0.0, -6.0, -8.0, -22.0, -24.0, -30.0, -32.0, -46.0 };
        for (size_t index = 0; index < 8; ++index) {
            REQUIRE(selected[index] == expected[index]);
        }
    }

    {
        // The focal lengths moving together keep their ratio, the step going to the first in full and to the other in
        // proportion, and the jacobian by the shared dimension sums theirs in that proportion.
        const int tied[12] = { 0, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
        optimisation::vertex vertex = optimisation::vertices::intrinsics::make(camera, &tied[0], 12);
        REQUIRE(vertex.is_valid());
        REQUIRE(vertex.get_local_dimensions() == 2);
        const optimisation::vertices::intrinsics* const policy = vertex.get_policy<optimisation::vertices::intrinsics>();
        REQUIRE(policy->get_local_index(0) == 0);
        REQUIRE(policy->get_local_index(1) == 0);
        REQUIRE(policy->get_local_index(2) == 1);
        REQUIRE(policy->get_local_index(3) == -1);
        REQUIRE(policy->get_parameter_index(0) == 0);
        REQUIRE(policy->get_parameter_index(1) == 2);
        REQUIRE(policy->get_weight(0) == 1.0);
        REQUIRE(policy->get_weight(1) == 510.0 / 500.0);
        REQUIRE(policy->get_weight(2) == 1.0);
        REQUIRE(policy->get_weight(3) == 0.0);
        const double delta[2] = { 25.0, -3.0 };
        vertex.plus(&delta[0]);
        REQUIRE(vertex.get_parameters()[0] == 525.0);
        REQUIRE(std::abs(vertex.get_parameters()[1] - (510.0 * 1.05)) < 1e-9);
        REQUIRE(std::abs((vertex.get_parameters()[1] / vertex.get_parameters()[0]) - (510.0 / 500.0)) < 1e-12);
        REQUIRE(vertex.get_parameters()[2] == 317.0);
        REQUIRE(vertex.get_parameters()[3] == 240.0);
        REQUIRE(projects_as(policy->get_camera(), vertex.get_parameters()));
        double full[2 * 12] = {};
        for (size_t index = 0; index < 24; ++index) {
            full[index] = static_cast<double>(index + 1);
        }
        double selected[2 * 2] = {};
        policy->select_free_columns(&full[0], 2, -1.0, &selected[0]);
        REQUIRE(std::abs(selected[0] - (-(1.0 + (2.0 * 510.0 / 500.0)))) < 1e-12);
        REQUIRE(selected[1] == -3.0);
        REQUIRE(std::abs(selected[2] - (-(13.0 + (14.0 * 510.0 / 500.0)))) < 1e-12);
        REQUIRE(selected[3] == -15.0);

        // Local dimensions that skip one, or a local dimension out of range, make an invalid vertex.
        const int skipping[12] = { 0, 0, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
        REQUIRE(!optimisation::vertices::intrinsics::make(camera, &skipping[0], 12).is_valid());
        const int below[12] = { 0, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
        REQUIRE(!optimisation::vertices::intrinsics::make(camera, &below[0], 12).is_valid());
    }

    {
        // A vertex with nothing free, a mask of the wrong length, or a camera without parameters is invalid.
        const bool none[12] = {};
        REQUIRE(!optimisation::vertices::intrinsics::make(camera, &none[0], 12).is_valid());
        REQUIRE(!optimisation::vertices::intrinsics::make(camera, &focal_centre[0], 4).is_valid());
        REQUIRE(!optimisation::vertices::intrinsics::make(camera, static_cast<const bool*>(nullptr), 12).is_valid());
        REQUIRE(!optimisation::vertices::intrinsics::make(camera, static_cast<const int*>(nullptr), 12).is_valid());
        REQUIRE(!optimisation::vertices::intrinsics::make(sensor::camera::model<double>(), &focal_centre[0], 0).is_valid());
        const optimisation::vertex invalid{ optimisation::vertices::intrinsics(camera, &none[0], 12) };
        REQUIRE(!invalid.is_valid());
        REQUIRE(invalid.get_parameter_dimensions() == 0);
        REQUIRE(invalid.get_local_dimensions() == 0);
        REQUIRE(invalid.get_policy<optimisation::vertices::intrinsics>() == nullptr);
    }

    {
        // Any model, here a pinhole with its four parameters, all free.
        const double pinhole_parameters[4] = { 400.0, 400.0, 200.0, 150.0 };
        const sensor::camera::model<double> pinhole{ sensor::camera::pinhole<double>(&pinhole_parameters[0], 4) };
        const bool all[4] = { true, true, true, true };
        optimisation::vertex vertex = optimisation::vertices::intrinsics::make(pinhole, &all[0], 4);
        REQUIRE(vertex.is_valid());
        REQUIRE(vertex.get_parameter_dimensions() == 4);
        REQUIRE(vertex.get_local_dimensions() == 4);
        const double delta[4] = { 10.0, -10.0, 1.0, 2.0 };
        vertex.plus(&delta[0]);
        double parameters[4] = {};
        REQUIRE(vertex.get_policy<optimisation::vertices::intrinsics>()->get_camera().get_parameters(&parameters[0], 4));
        REQUIRE(parameters[0] == 410.0);
        REQUIRE(parameters[1] == 390.0);
        REQUIRE(parameters[2] == 201.0);
        REQUIRE(parameters[3] == 152.0);
    }

    return EXIT_SUCCESS;
}
