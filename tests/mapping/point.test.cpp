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

#include "mapping/point.hpp"

#include "math/lie.hpp"

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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    {
        mapping::point p;
    }
    {
        mapping::point p0(0, { { 1, 2, 3 } }, { { 4, 5, 6 } });
        REQUIRE(p0.id == 0);
        mapping::point p1(1, { { 4, 5, 6 } }, { { 7, 8, 9 } });
        REQUIRE(p1.id == 1);
    }

    {
        mapping::point p(2, { { 1.0, -0.5, 4.0 } }, { { 0, 0, 0 } });
        const math::matrix<double, 3, 3> rotation = math::so3<double>::exp({ { 0.1, -0.2, 0.05 } }).get_matrix();
        const math::matrix<double, 3, 1> translation({ 0.3, 0.1, -0.4 });
        REQUIRE(p.anchor(rotation, translation));
        REQUIRE(p.inverse_depth);
        REQUIRE(!p.at_infinity());
        const math::matrix<double, 3, 1> before = p.location;
        p.update_location_from_inverse_depth();
        REQUIRE(std::abs((p.location - before).get_length_squared()) < 1.0e-18);
        mapping::point behind(3, { { 0.0, 0.0, -1.0 } }, { { 0, 0, 0 } });
        REQUIRE(!behind.anchor(math::matrix<double, 3, 3>::identity(), math::matrix<double, 3, 1>::zero()));
        REQUIRE(!behind.inverse_depth);
        mapping::point distant(4, math::matrix<double, 3, 1>::zero(), { { 0, 0, 0 } });
        distant.anchor_at_infinity(math::matrix<double, 3, 3>::identity(), math::matrix<double, 3, 1>({ 1.0, 2.0, 3.0 }), { { 0.5, -0.25, 1.0 } });
        REQUIRE(distant.at_infinity());
        const math::matrix<double, 3, 1> centre = -(math::transpose(math::matrix<double, 3, 3>::identity()) * math::matrix<double, 3, 1>({ 1.0, 2.0, 3.0 }));
        const math::matrix<double, 3, 1> offset = distant.location - centre;
        REQUIRE(std::abs(offset[0] / offset[2] - 0.5) < 1.0e-9);
        REQUIRE(std::abs(offset[1] / offset[2] + 0.25) < 1.0e-9);
        REQUIRE(offset[2] > 1.0e3);
    }
    {
        mapping::point p(7, { { 0.0, 0.0, 4.0 } }, { { 1.0, 1.0, 1.0 } });
        math::matrix<double, 0, 0> jacobian(2, 3, math::matrix<double, 2, 3>{ { { 100.0, 0.0, 10.0 }, { 0.0, 100.0, 0.0 } } }.data());
        REQUIRE(p.uncertainty == mapping::point::uncertainty_kind::unknown);
        math::matrix<double, 2, 2> w = p.observation_information(jacobian, 2.0);
        REQUIRE(std::abs(w[0][0] - 0.25) < 1.0e-12);
        REQUIRE(std::abs(w[1][1] - 0.25) < 1.0e-12);
        REQUIRE(std::abs(w[0][1]) < 1.0e-12);

        math::matrix<double, 3, 3> information = math::matrix<double, 3, 3>::zero();
        information[0][0] = 1.0e4;
        information[1][1] = 1.0e4;
        information[2][2] = 1.0;
        p.set_information(information, { { 0.0, 0.0, 1.0 } });
        REQUIRE(p.uncertainty == mapping::point::uncertainty_kind::estimated);
        REQUIRE(std::abs(p.covariance[2][2] - 1.0) < 1.0e-12);
        w = p.observation_information(jacobian, 1.0);
        REQUIRE(std::abs(w[0][0] - 1.0) < 1.0e-12);
        REQUIRE(std::abs(w[1][1] - 1.0) < 1.0e-12);
        REQUIRE(std::abs(w[0][1]) < 1.0e-12);

        p.set_information(math::matrix<double, 3, 3>::zero(), { { 0.0, 0.0, 1.0 } });
        REQUIRE(p.uncertainty == mapping::point::uncertainty_kind::estimated);
        REQUIRE(p.anchor(math::matrix<double, 3, 3>::identity(), math::matrix<double, 3, 1>::zero()));
        p.set_information(math::matrix<double, 3, 3>::zero(), { { 0.0, 0.0, 1.0 } });
        REQUIRE(p.uncertainty == mapping::point::uncertainty_kind::unbounded);
        w = p.observation_information(jacobian, 1.0);
        REQUIRE(std::abs(w[0][0]) < 1.0e-12);
        REQUIRE(std::abs(w[1][1] - 1.0) < 1.0e-12);
        REQUIRE(std::abs(w[0][1]) < 1.0e-12);

        mapping::point distant(8, { { 0.0, 0.0, 1.0 } }, { { 1.0, 1.0, 1.0 } });
        distant.anchor_at_infinity(math::matrix<double, 3, 3>::identity(), math::matrix<double, 3, 1>::zero(), { { 0.5, -0.25, 1.0 } });
        REQUIRE(distant.uncertainty == mapping::point::uncertainty_kind::unbounded);
        REQUIRE(std::abs(distant.depth_direction[2] - 1.0) < 1.0e-12);
    }

    return EXIT_SUCCESS;
}
