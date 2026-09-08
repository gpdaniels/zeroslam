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

#include "optimisation/loss.hpp"

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

class squared final {
public:
    constexpr static const char* name = "squared";

    void compute(const double error_squared, math::matrix<double, 3, 1>& rho) const {
        rho[0] = error_squared * error_squared;
        rho[1] = 2.0 * error_squared;
        rho[2] = 2.0;
    }
};

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        const optimisation::loss erased{ squared() };
        REQUIRE(erased.is_valid());
        REQUIRE(erased.name()[0] == 's');
        math::matrix<double, 3, 1> rho;
        erased.compute(3.0, rho);
        REQUIRE(rho[0] == 9.0);
        REQUIRE(rho[1] == 6.0);
        REQUIRE(rho[2] == 2.0);
    }

    {
        optimisation::loss empty;
        REQUIRE(!empty.is_valid());
        math::matrix<double, 3, 1> rho;
        empty.compute(3.0, rho);
        REQUIRE(rho[0] == 3.0);
        REQUIRE(rho[1] == 1.0);
        REQUIRE(rho[2] == 0.0);
        empty = optimisation::loss(squared());
        REQUIRE(empty.is_valid());
        const optimisation::loss copy = empty;
        copy.compute(2.0, rho);
        REQUIRE(rho[0] == 4.0);
        empty.clear();
        REQUIRE(!empty.is_valid());
        REQUIRE(copy.is_valid());
    }

    return EXIT_SUCCESS;
}
