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

#include "estimation/robust/solver/similarity.hpp"

#include "core/random_pcg.hpp"
#include "math/lie.hpp"
#include "math/matrix.hpp"

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

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-8) {
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    core::random_pcg random(0x5eed0044ull);
    const math::matrix<double, 3, 3> rotation = math::so3<double>::exp({ { 0.3, -0.2, 0.5 } }).get_matrix();
    const math::matrix<double, 3, 1> translation{ { 0.5, -0.3, 1.0 } };
    const double scale = 1.7;

    {
        constexpr static const size_t inlier_count = 30;
        constexpr static const size_t outlier_count = 10;
        constexpr static const size_t correspondence_count = inlier_count + outlier_count;
        estimation::correspondence_3d_3d<double> data[correspondence_count];
        for (size_t i = 0; i < correspondence_count; ++i) {
            data[i].lhs = math::matrix<double, 3, 1>{ { random.get_random(-2.0, 2.0), random.get_random(-2.0, 2.0), random.get_random(-2.0, 2.0) } };
            data[i].rhs = (scale * (rotation * data[i].lhs)) + translation;
            if (i >= inlier_count) {
                data[i].rhs = data[i].rhs + math::matrix<double, 3, 1>{ { random.get_random(0.5, 2.0), random.get_random(-2.0, -0.5), random.get_random(0.5, 2.0) } };
            }
        }

        float residuals[correspondence_count];
        size_t inliers[correspondence_count];
        size_t inliers_size = 0;
        estimation::robust::solver::similarity<double>::model_type model{};
        REQUIRE(estimation::robust::solver::similarity<double>::solve(data, correspondence_count, 0.01f, residuals, inliers, inliers_size, model));
        REQUIRE(inliers_size == inlier_count);
        for (size_t i = 0; i < inliers_size; ++i) {
            REQUIRE(inliers[i] < inlier_count);
        }
        for (size_t i = 0; i < correspondence_count; ++i) {
            REQUIRE(std::isfinite(residuals[i]));
            REQUIRE((residuals[i] < 0.01f) == (i < inlier_count));
        }
        REQUIRE(is_value_approx(model.scale, scale, 1e-6));
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                REQUIRE(is_value_approx(model.rotation[row][col], rotation[row][col], 1e-6));
            }
            REQUIRE(is_value_approx(model.translation[row], translation[row], 1e-6));
        }
    }

    {
        constexpr static const size_t correspondence_count = 40;
        estimation::correspondence_3d_3d<double> data[correspondence_count];
        for (size_t i = 0; i < correspondence_count; ++i) {
            data[i].lhs = math::matrix<double, 3, 1>{ { random.get_random(-2.0, 2.0), random.get_random(-2.0, 2.0), random.get_random(-2.0, 2.0) } };
            data[i].rhs = (scale * (rotation * data[i].lhs)) + translation + math::matrix<double, 3, 1>{ { random.get_random(-0.01, 0.01), random.get_random(-0.01, 0.01), random.get_random(-0.01, 0.01) } };
        }

        float residuals[correspondence_count];
        size_t inliers[correspondence_count];
        size_t inliers_size = 0;
        estimation::robust::solver::similarity<double>::model_type model{};
        REQUIRE(estimation::robust::solver::similarity<double>::solve(data, correspondence_count, 0.05f, residuals, inliers, inliers_size, model));
        REQUIRE(inliers_size == correspondence_count);
        REQUIRE(is_value_approx(model.scale, scale, 1e-2));
        for (size_t row = 0; row < 3; ++row) {
            REQUIRE(is_value_approx(model.translation[row], translation[row], 2e-2));
        }
    }

    {
        estimation::correspondence_3d_3d<double> data[2] = {};
        float residuals[2];
        size_t inliers[2];
        size_t inliers_size = 0;
        estimation::robust::solver::similarity<double>::model_type model{};
        REQUIRE(!estimation::robust::solver::similarity<double>::solve(data, 2, 0.01f, residuals, inliers, inliers_size, model));
    }

    return EXIT_SUCCESS;
}
