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

#include "optimisation/vertices/calibration.hpp"

#include "optimisation/vertex.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const double values[4] = { 0.3, 2.7, 1.0, 0.02 };

    // A vertex has a local dimension for each free value, in the order of the values, and moves only those.
    {
        optimisation::vertex vertex = optimisation::vertices::calibration::make(&values[0], 4, 0x9u);
        REQUIRE(vertex.is_valid());
        REQUIRE(std::strcmp(vertex.name(), "calibration") == 0);
        REQUIRE(!vertex.is_fixed());
        REQUIRE(vertex.get_parameter_dimensions() == 9);
        REQUIRE(vertex.get_local_dimensions() == 2);
        const double* const parameters = vertex.get_parameters();
        REQUIRE(optimisation::vertices::calibration::mask(parameters) == 0x9u);
        REQUIRE(optimisation::vertices::calibration::local_index(parameters, 0) == 0);
        REQUIRE(optimisation::vertices::calibration::local_index(parameters, 1) == -1);
        REQUIRE(optimisation::vertices::calibration::local_index(parameters, 2) == -1);
        REQUIRE(optimisation::vertices::calibration::local_index(parameters, 3) == 1);
        REQUIRE(optimisation::vertices::calibration::local_index(parameters, 8) == -1);
        for (size_t i = 4; i < 8; ++i) {
            REQUIRE(parameters[i] == 0.0);
        }

        const double delta[2] = { 0.01, -0.005 };
        vertex.plus(&delta[0]);
        REQUIRE(vertex.get_parameters()[0] == 0.3 + 0.01);
        REQUIRE(vertex.get_parameters()[1] == 2.7);
        REQUIRE(vertex.get_parameters()[2] == 1.0);
        REQUIRE(vertex.get_parameters()[3] == 0.02 - 0.005);
        REQUIRE(optimisation::vertices::calibration::mask(vertex.get_parameters()) == 0x9u);

        // The erased vertex copies its policy and parameters, and backs them up and restores them.
        optimisation::vertex copy = vertex;
        REQUIRE(copy.get_local_dimensions() == 2);
        copy.backup();
        copy.plus(&delta[0]);
        REQUIRE(copy.get_parameters()[3] != vertex.get_parameters()[3]);
        copy.restore();
        REQUIRE(copy.get_parameters()[3] == vertex.get_parameters()[3]);
    }

    // Every count of free values from one to eight makes a vertex of that many local dimensions.
    {
        const double eight[8] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        for (unsigned int count = 1; count <= 8; ++count) {
            const unsigned int mask = (1u << count) - 1u;
            optimisation::vertex vertex = optimisation::vertices::calibration::make(&eight[0], 8, mask << (8u - count));
            REQUIRE(vertex.get_local_dimensions() == static_cast<int>(count));
            double delta[8] = {};
            for (size_t i = 0; i < 8; ++i) {
                delta[i] = 0.5;
            }
            vertex.plus(&delta[0]);
            for (size_t i = 0; i < 8; ++i) {
                const bool moved = i >= (8u - count);
                REQUIRE(vertex.get_parameters()[i] == (eight[i] + (moved ? 0.5 : 0.0)));
            }
        }
    }

    // No free value makes a fixed vertex; mask bits past the values given, and values past the eight, are dropped.
    {
        optimisation::vertex none = optimisation::vertices::calibration::make(&values[0], 4, 0u);
        REQUIRE(none.is_fixed());
        REQUIRE(optimisation::vertices::calibration::mask(none.get_parameters()) == 0u);
        optimisation::vertex beyond = optimisation::vertices::calibration::make(&values[0], 2, 0xFCu);
        REQUIRE(beyond.is_fixed());
        optimisation::vertex partial = optimisation::vertices::calibration::make(&values[0], 2, 0xFFu);
        REQUIRE(partial.get_local_dimensions() == 2);
        REQUIRE(optimisation::vertices::calibration::mask(partial.get_parameters()) == 0x3u);
        optimisation::vertex zeros = optimisation::vertices::calibration::make(nullptr, 3, 0x1u);
        REQUIRE(zeros.get_parameters()[0] == 0.0);

        // A mask that is not one is read as none.
        double corrupt[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0 };
        REQUIRE(optimisation::vertices::calibration::mask(&corrupt[0]) == 0u);
        corrupt[8] = 300.0;
        REQUIRE(optimisation::vertices::calibration::mask(&corrupt[0]) == 0u);
    }

    return EXIT_SUCCESS;
}
