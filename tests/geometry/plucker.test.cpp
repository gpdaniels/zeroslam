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

#include "geometry/plucker.hpp"

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
        const math::matrix<double, 3, 1> a({ 1.0, 2.0, 3.0 });
        const math::matrix<double, 3, 1> b({ 4.0, -1.0, 5.0 });
        geometry::plucker line;
        REQUIRE(geometry::plucker::from_points(a, b, line));
        REQUIRE(std::abs(line.direction.get_length_squared() - 1.0) < 1.0e-12);
        REQUIRE(std::abs(geometry::plucker::dot(line.moment, line.direction)) < 1.0e-12);
        REQUIRE(std::abs((line.project_point(a) - a).get_length_squared()) < 1.0e-18);
        REQUIRE(std::abs((line.project_point(b) - b).get_length_squared()) < 1.0e-18);
        geometry::plucker degenerate;
        REQUIRE(!geometry::plucker::from_points(a, a, degenerate));
    }

    {
        geometry::plucker line;
        REQUIRE(geometry::plucker::from_points({ { 1.0, 2.0, 0.0 } }, { { 1.0, 2.0, 1.0 } }, line));
        REQUIRE(std::abs(line.distance_to_origin() - std::sqrt(5.0)) < 1.0e-12);
        const math::matrix<double, 3, 1> closest = line.closest_point_to_origin();
        REQUIRE(std::abs(closest[0] - 1.0) < 1.0e-12);
        REQUIRE(std::abs(closest[1] - 2.0) < 1.0e-12);
        REQUIRE(std::abs(closest[2] - 0.0) < 1.0e-12);
    }

    {
        geometry::plucker line;
        REQUIRE(geometry::plucker::from_planes({ { 1.0, 0.0, 0.0 } }, -1.0, { { 0.0, 1.0, 0.0 } }, -2.0, 0.01, line));
        REQUIRE(std::abs(std::abs(line.direction[2]) - 1.0) < 1.0e-12);
        const math::matrix<double, 3, 1> closest = line.closest_point_to_origin();
        REQUIRE(std::abs(closest[0] - 1.0) < 1.0e-12);
        REQUIRE(std::abs(closest[1] - 2.0) < 1.0e-12);
        geometry::plucker degenerate;
        REQUIRE(!geometry::plucker::from_planes({ { 1.0, 0.0, 0.0 } }, -1.0, { { 1.0, 0.001, 0.0 } }, -2.0, 0.01, degenerate));
    }

    {
        const math::matrix<double, 3, 1> a({ 0.5, -1.0, 2.0 });
        const math::matrix<double, 3, 1> b({ 2.0, 1.0, 4.0 });
        geometry::plucker line;
        REQUIRE(geometry::plucker::from_points(a, b, line));
        const math::se3<double> transform = math::se3<double>::exp({ { 0.2, -0.1, 0.3, 0.5, 1.0, -0.7 } });
        const math::matrix<double, 3, 3> rotation = transform.rotation().get_matrix();
        const math::matrix<double, 3, 1> translation = transform.translation();
        const geometry::plucker transformed = line.transformed(rotation, translation);
        geometry::plucker expected;
        REQUIRE(geometry::plucker::from_points((rotation * a) + translation, (rotation * b) + translation, expected));
        REQUIRE(std::abs((transformed.direction - expected.direction).get_length_squared()) < 1.0e-18);
        REQUIRE(std::abs((transformed.moment - expected.moment).get_length_squared()) < 1.0e-18);
        REQUIRE(std::abs(transformed.direction.get_length_squared() - 1.0) < 1.0e-12);
        REQUIRE(std::abs(geometry::plucker::dot(transformed.moment, transformed.direction)) < 1.0e-12);
    }

    {
        geometry::plucker line;
        REQUIRE(geometry::plucker::from_points({ { 1.0, 2.0, 0.0 } }, { { 1.0, 2.0, 1.0 } }, line));
        math::matrix<double, 3, 1> closest;
        REQUIRE(line.closest_point_to_ray({ { 0.0, 0.0, 0.0 } }, { { 1.0, 0.0, 0.0 } }, closest));
        REQUIRE(std::abs(closest[0] - 1.0) < 1.0e-12);
        REQUIRE(std::abs(closest[1] - 2.0) < 1.0e-12);
        REQUIRE(std::abs(closest[2] - 0.0) < 1.0e-12);
        REQUIRE(!line.closest_point_to_ray({ { 0.0, 0.0, 0.0 } }, { { 0.0, 0.0, 1.0 } }, closest));
    }

    {
        geometry::plucker line;
        REQUIRE(geometry::plucker::from_points({ { 1.0, 2.0, 3.0 } }, { { -2.0, 0.5, 4.0 } }, line));
        math::matrix<double, 3, 3> u;
        double w1 = 0.0;
        double w2 = 0.0;
        double scale = 0.0;
        line.to_orthonormal(u, w1, w2, scale);
        REQUIRE(std::abs((w1 * w1) + (w2 * w2) - 1.0) < 1.0e-12);
        const math::matrix<double, 3, 3> identity_check = math::transpose(u) * u;
        for (size_t r = 0; r < 3; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                REQUIRE(std::abs(identity_check[r][c] - ((r == c) ? 1.0 : 0.0)) < 1.0e-12);
            }
        }
        const geometry::plucker recomposed = geometry::plucker::from_orthonormal(u, w1, w2, scale);
        REQUIRE(std::abs((recomposed.moment - line.moment).get_length_squared()) < 1.0e-18);
        REQUIRE(std::abs((recomposed.direction - line.direction).get_length_squared()) < 1.0e-18);

        geometry::plucker updated = line;
        REQUIRE(updated.oplus(0.0, 0.0, 0.0, 0.0));
        REQUIRE(std::abs((updated.moment - line.moment).get_length_squared()) < 1.0e-18);
        REQUIRE(std::abs((updated.direction - line.direction).get_length_squared()) < 1.0e-18);
        REQUIRE(updated.oplus(0.01, -0.02, 0.03, -0.01));
        REQUIRE(std::abs(updated.direction.get_length_squared() - 1.0) < 1.0e-12);
        REQUIRE(std::abs(geometry::plucker::dot(updated.moment, updated.direction)) < 1.0e-12);
        REQUIRE((updated.direction - line.direction).get_length_squared() > 1.0e-8);
    }

    {
        geometry::plucker line(math::matrix<double, 3, 1>::zero(), math::matrix<double, 3, 1>({ 0.0, 1.0, 0.0 }));
        math::matrix<double, 3, 3> u;
        double w1 = 0.0;
        double w2 = 0.0;
        double scale = 0.0;
        line.to_orthonormal(u, w1, w2, scale);
        REQUIRE(std::abs(w1) < 1.0e-12);
        REQUIRE(std::abs(w2 - 1.0) < 1.0e-12);
        geometry::plucker updated = line;
        REQUIRE(updated.oplus(0.0, 0.0, 0.0, 0.1));
        REQUIRE(updated.moment.get_length_squared() > 1.0e-6);
        REQUIRE(std::abs(geometry::plucker::dot(updated.moment, updated.direction)) < 1.0e-12);
    }

    return EXIT_SUCCESS;
}
