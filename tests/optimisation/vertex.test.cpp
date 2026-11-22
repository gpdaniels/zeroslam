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

#include "optimisation/vertex.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

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

class scalar_vertex final {
public:
    constexpr static const char* name = "scalar";
    constexpr static const int parameter_count = 2;
    constexpr static const int local_count = 2;

    void plus(double* const parameters, const double* const delta) const {
        for (int i = 0; i < scalar_vertex::local_count; ++i) {
            parameters[i] += delta[i];
        }
    }
};

// A vertex whose sizes are known when it is made, which keeps the sum of its parameters wherever they are written.
class sized_vertex final {
public:
    constexpr static const char* name = "sized";
    constexpr static const int parameter_count = 1;
    constexpr static const int local_count = 1;

    int parameters_held;
    int locals_held;
    double* sum;

    int get_parameter_count() const {
        return this->parameters_held;
    }

    int get_local_count() const {
        return this->locals_held;
    }

    void plus(double* const parameters, const double* const delta) const {
        for (int i = 0; i < this->locals_held; ++i) {
            parameters[i] += delta[i];
        }
        this->synchronise(parameters);
    }

    void synchronise(const double* const parameters) const {
        *this->sum = 0.0;
        for (int i = 0; i < this->parameters_held; ++i) {
            *this->sum += parameters[i];
        }
    }
};

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    optimisation::vertex vertex{ scalar_vertex() };
    REQUIRE(vertex.is_valid());
    REQUIRE(vertex.get_parameter_dimensions() == 2);
    REQUIRE(vertex.get_local_dimensions() == 2);
    REQUIRE(vertex.get_parameters()[0] == 0.0);
    REQUIRE(vertex.get_parameters()[1] == 0.0);
    REQUIRE(!vertex.is_fixed());
    REQUIRE(!vertex.is_marginalised());
    REQUIRE(vertex.get_ordering_id() == 0);

    vertex.set_fixed();
    vertex.set_marginalised();
    vertex.set_ordering_id(7);
    REQUIRE(vertex.is_fixed());
    REQUIRE(vertex.is_marginalised());
    REQUIRE(vertex.get_ordering_id() == 7);
    vertex.set_fixed(false);
    REQUIRE(!vertex.is_fixed());

    optimisation::vertex empty;
    REQUIRE(!empty.is_valid());
    REQUIRE(empty.get_local_dimensions() == 0);
    const double empty_delta[2] = { 1.0, 1.0 };
    empty.plus(&empty_delta[0]);
    REQUIRE(!empty.set_parameters(&empty_delta[0], 2));

    const double initial[2] = { 1.0, -2.0 };
    REQUIRE(vertex.set_parameters(&initial[0], 2));
    REQUIRE(!vertex.set_parameters(&initial[0], 3));
    REQUIRE(!vertex.set_parameters(nullptr, 2));
    vertex.backup();
    const double delta[2] = { 0.5, 0.25 };
    vertex.plus(&delta[0]);
    REQUIRE(is_value_approx(vertex.get_parameters()[0], 1.5));
    REQUIRE(is_value_approx(vertex.get_parameters()[1], -1.75));
    vertex.restore();
    REQUIRE(vertex.get_parameters()[0] == 1.0);
    REQUIRE(vertex.get_parameters()[1] == -2.0);

    double copied[2] = {};
    REQUIRE(vertex.get_parameters(&copied[0], 2));
    REQUIRE(copied[0] == 1.0);
    REQUIRE(copied[1] == -2.0);
    REQUIRE(!vertex.get_parameters(&copied[0], 1));
    REQUIRE(!vertex.get_parameters(nullptr, 2));

    {
        optimisation::vertex original{ scalar_vertex() };
        const double values[2] = { 4.0, 5.0 };
        REQUIRE(original.set_parameters(&values[0], 2));
        optimisation::vertex copy = original;
        const double changed[2] = { 9.0, 9.0 };
        REQUIRE(original.set_parameters(&changed[0], 2));
        REQUIRE(copy.get_parameters()[0] == 4.0);
        REQUIRE(copy.get_parameters()[1] == 5.0);
        REQUIRE(original.get_parameters()[0] == 9.0);

        // Assignment takes the type, the flags and the parameters of the other vertex, and a self assignment leaves it alone.
        optimisation::vertex assigned;
        REQUIRE(!assigned.is_valid());
        assigned = original;
        REQUIRE(assigned.is_valid());
        REQUIRE(std::strcmp(assigned.name(), "scalar") == 0);
        REQUIRE(assigned.get_parameters()[0] == 9.0);
        REQUIRE(assigned.get_local_dimensions() == 2);
        optimisation::vertex& assigned_alias = assigned;
        assigned = assigned_alias;
        REQUIRE(assigned.is_valid());
        REQUIRE(assigned.get_parameters()[0] == 9.0);
        assigned.set_fixed();
        assigned.set_ordering_id(4);
        optimisation::vertex move_assigned{ scalar_vertex() };
        move_assigned = std::move(assigned);
        REQUIRE(move_assigned.is_valid());
        REQUIRE(move_assigned.is_fixed());
        REQUIRE(move_assigned.get_ordering_id() == 4);
        REQUIRE(move_assigned.get_parameters()[0] == 9.0);
        move_assigned.plus(&delta[0]);
        REQUIRE(is_value_approx(move_assigned.get_parameters()[0], 9.5));
    }

    // An erased vertex has no type to name.
    REQUIRE(std::strcmp(empty.name(), "") == 0);

    {
        // The sizes a vertex is made with override its constants, and are checked then: a vertex that cannot hold them is
        // invalid. Its own policy is reachable by its type, and writing its parameters from outside synchronises it.
        double sum = 0.0;
        optimisation::vertex sized{ sized_vertex{ 20, 3, &sum } };
        REQUIRE(sized.is_valid());
        REQUIRE(sized.get_parameter_dimensions() == 20);
        REQUIRE(sized.get_local_dimensions() == 3);
        REQUIRE(sized.get_policy<sized_vertex>() != nullptr);
        REQUIRE(sized.get_policy<sized_vertex>()->parameters_held == 20);
        REQUIRE(sized.get_policy<scalar_vertex>() == nullptr);
        double values[20] = {};
        for (size_t i = 0; i < 20; ++i) {
            values[i] = 1.0;
        }
        REQUIRE(!sized.set_parameters(&values[0], 7));
        REQUIRE(sized.set_parameters(&values[0], 20));
        REQUIRE(sum == 20.0);
        sized.backup();
        const double step[3] = { 1.0, 2.0, 3.0 };
        sized.plus(&step[0]);
        REQUIRE(sum == 26.0);
        sized.restore();
        REQUIRE(sum == 20.0);
        const optimisation::vertex sized_copy(sized);
        REQUIRE(sized_copy.get_parameter_dimensions() == 20);
        REQUIRE(sized_copy.get_local_dimensions() == 3);
        optimisation::vertex sized_assigned;
        sized_assigned = sized_copy;
        REQUIRE(sized_assigned.get_local_dimensions() == 3);
        sized_assigned.clear();
        REQUIRE(sized_assigned.get_parameter_dimensions() == 0);
        REQUIRE(sized_assigned.get_local_dimensions() == 0);
        REQUIRE(!optimisation::vertex(sized_vertex{ static_cast<int>(optimisation::vertex::maximum_parameters) + 1, 1, &sum }).is_valid());
        REQUIRE(!optimisation::vertex(sized_vertex{ 4, 5, &sum }).is_valid());
        REQUIRE(!optimisation::vertex(sized_vertex{ 4, 0, &sum }).is_valid());
        REQUIRE(!optimisation::vertex(sized_vertex{ 0, 0, &sum }).is_valid());
        REQUIRE(optimisation::vertex(sized_vertex{ 4, 0, &sum }).get_policy<sized_vertex>() == nullptr);
    }

    return EXIT_SUCCESS;
}
