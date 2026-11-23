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

#include "optimisation/edges/gravity_alignment.hpp"

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/relative_similarity.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/similarity.hpp"

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

static optimisation::vertex make_similarity(const math::so3<double>& rotation, const math::matrix<double, 3, 1>& translation) {
    optimisation::vertex vertex{ optimisation::vertices::similarity() };
    const double parameters[8] = { translation[0], translation[1], translation[2], rotation.get_quaternion()[1], rotation.get_quaternion()[2], rotation.get_quaternion()[3], rotation.get_quaternion()[0], 1.0 };
    REQUIRE(vertex.set_parameters(&parameters[0], 8));
    return vertex;
}

static math::so3<double> rotation_of(const optimisation::vertex& vertex) {
    const double* const p = vertex.get_parameters();
    return math::so3<double>(p[6], p[3], p[4], p[5]);
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(sizeof(optimisation::edges::gravity_alignment) <= optimisation::edge::maximum_size, "The gravity alignment edge must fit.");

    const math::matrix<double, 3, 1> down = { { 0.0, 0.0, -9.81 } };
    const math::so3<double> rotation = math::so3<double>::exp({ { 0.3, -0.2, 1.0 } });

    // The residual is zero where the rotation turns gravity onto the observation, and is the difference of the directions
    // otherwise; a turn about gravity changes nothing.
    {
        optimisation::vertex vertex = make_similarity(rotation, { { 1.0, 2.0, 3.0 } });
        const optimisation::edges::gravity_alignment policy(down, true);
        optimisation::edge edge{ policy };
        const math::matrix<double, 3, 1> observed = policy.turned(vertex.get_parameters());
        const math::matrix<double, 3, 1> expected = rotation.inverse() * math::matrix<double, 3, 1>({ { 0.0, 0.0, -1.0 } });
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(std::abs(observed[i] - expected[i]) < 1e-12);
        }
        edge.set_observation(math::matrix<double, 0, 0>(3, 1, observed.data()));
        REQUIRE(edge.add_vertex(&vertex));
        edge.compute_residual();
        REQUIRE(edge.chi2() < 1e-24);
        optimisation::vertex turned = make_similarity(math::so3<double>::exp({ { 0.0, 0.0, 0.8 } }) * rotation, { { 1.0, 2.0, 3.0 } });
        optimisation::edge about_gravity{ policy };
        about_gravity.set_observation(edge.get_observation());
        REQUIRE(about_gravity.add_vertex(&turned));
        about_gravity.compute_residual();
        REQUIRE(about_gravity.chi2() < 1e-24);
        optimisation::vertex tilted = make_similarity(math::so3<double>::exp({ { 0.1, 0.0, 0.0 } }) * rotation, { { 1.0, 2.0, 3.0 } });
        optimisation::edge tilt{ policy };
        tilt.set_observation(edge.get_observation());
        REQUIRE(tilt.add_vertex(&tilted));
        tilt.compute_residual();
        REQUIRE(std::abs(std::sqrt(tilt.chi2()) - 2.0 * std::sin(0.05)) < 1e-9);
        // Without the inverse the rotation itself turns gravity.
        const optimisation::edges::gravity_alignment forward(down, false);
        const math::matrix<double, 3, 1> forward_turned = forward.turned(vertex.get_parameters());
        const math::matrix<double, 3, 1> forward_expected = rotation * math::matrix<double, 3, 1>({ { 0.0, 0.0, -1.0 } });
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(std::abs(forward_turned[i] - forward_expected[i]) < 1e-12);
        }
    }

    // A pose graph of two similarities joined by a measurement that tilts the second: with the second's gravity held it takes
    // the measurement's turn about gravity and its position but keeps its tilt.
    {
        optimisation::factor_graph graph;
        optimisation::vertex first = make_similarity(math::so3<double>::identity(), math::matrix<double, 3, 1>::zero());
        first.set_fixed(true);
        optimisation::vertex* const first_vertex = graph.add_vertex(static_cast<optimisation::vertex&&>(first));
        optimisation::vertex* const second_vertex = graph.add_vertex(make_similarity(rotation, { { 1.0, 0.0, 0.0 } }));
        const optimisation::edges::gravity_alignment policy(down, true);
        const math::matrix<double, 3, 1> held = policy.turned(second_vertex->get_parameters());
        optimisation::edge hold{ policy };
        hold.set_observation(math::matrix<double, 0, 0>(3, 1, held.data()));
        hold.set_information(math::matrix<double, 0, 0>::identity(3, 3) * 1e8);
        REQUIRE(hold.add_vertex(second_vertex));
        REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(hold)) != nullptr);
        const math::so3<double> measured = math::so3<double>::exp({ { 0.05, -0.04, 0.3 } }) * rotation;
        const double measurement[8] = { 1.5, 0.2, -0.1, measured.get_quaternion()[1], measured.get_quaternion()[2], measured.get_quaternion()[3], measured.get_quaternion()[0], 1.0 };
        optimisation::edge relative{ optimisation::edges::relative_similarity() };
        relative.set_observation(math::matrix<double, 0, 0>(8, 1, &measurement[0]));
        REQUIRE(relative.add_vertex(first_vertex));
        REQUIRE(relative.add_vertex(second_vertex));
        REQUIRE(graph.add_edge(static_cast<optimisation::edge&&>(relative)) != nullptr);
        REQUIRE(graph.solve(50, true) > 0);
        const math::matrix<double, 3, 1> after = policy.turned(second_vertex->get_parameters());
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(std::abs(after[i] - held[i]) < 1e-4);
            REQUIRE(std::abs(second_vertex->get_parameters()[i] - measurement[i]) < 1e-3);
        }
        // The turn about gravity is the measurement's.
        const math::so3<double> result = rotation_of(*second_vertex);
        const math::matrix<double, 3, 1> forward_axis = { { 1.0, 0.0, 0.0 } };
        const math::matrix<double, 3, 1> result_axis = result * forward_axis;
        const math::matrix<double, 3, 1> measured_axis = measured * forward_axis;
        const double result_heading = std::atan2(result_axis[1], result_axis[0]);
        const double measured_heading = std::atan2(measured_axis[1], measured_axis[0]);
        REQUIRE(std::abs(result_heading - measured_heading) < 0.06);
    }

    return EXIT_SUCCESS;
}
