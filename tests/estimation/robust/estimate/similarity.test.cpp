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

#include "estimation/robust/estimate/similarity.hpp"

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

    static_assert(estimation::robust::estimate::similarity<double>::sample_size == 3);
    static_assert(estimation::robust::estimate::similarity<double>::models_size == 1);

    const estimation::robust::estimate::similarity<double> estimator;
    core::random_pcg random(0x5eed0033ull);
    for (int trial = 0; trial < 16; ++trial) {
        const math::matrix<double, 3, 3> rotation = math::so3<double>::exp({ { random.get_random(-1.0, 1.0), random.get_random(-1.0, 1.0), random.get_random(-1.0, 1.0) } }).get_matrix();
        const math::matrix<double, 3, 1> translation{ { random.get_random(-2.0, 2.0), random.get_random(-2.0, 2.0), random.get_random(-2.0, 2.0) } };
        const double scale = random.get_random(0.5, 2.0);
        estimation::correspondence_3d_3d<double> data[6];
        for (size_t i = 0; i < 6; ++i) {
            data[i].lhs = math::matrix<double, 3, 1>{ { random.get_random(-1.5, 1.5), random.get_random(-1.5, 1.5), random.get_random(-1.5, 1.5) } };
            data[i].rhs = (scale * (rotation * data[i].lhs)) + translation;
        }

        estimation::robust::estimate::similarity<double>::model models[1];
        REQUIRE(estimator.generate_models(data, 3, models) == 1);
        REQUIRE(is_value_approx(models[0].scale, scale, 1e-6));
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                REQUIRE(is_value_approx(models[0].rotation[row][col], rotation[row][col], 1e-6));
            }
            REQUIRE(is_value_approx(models[0].translation[row], translation[row], 1e-6));
        }
        REQUIRE(estimator.generate_models(data, 2, models) == 0);

        float residuals[6];
        estimator.compute_residuals(data, 6, models[0], residuals);
        for (size_t i = 0; i < 6; ++i) {
            REQUIRE(residuals[i] < 1e-5f);
        }
        data[4].rhs[0] += 0.05;
        estimator.compute_residuals(data, 6, models[0], residuals);
        REQUIRE(is_value_approx(static_cast<double>(residuals[4]), 0.05, 1e-4));
        REQUIRE(residuals[3] < 1e-5f);
    }

    {
        estimation::correspondence_3d_3d<double> data[3];
        for (size_t i = 0; i < 3; ++i) {
            data[i].lhs = math::matrix<double, 3, 1>{ { 1.0, 2.0, 3.0 } };
            data[i].rhs = math::matrix<double, 3, 1>{ { static_cast<double>(i), 0.0, 0.0 } };
        }
        estimation::robust::estimate::similarity<double>::model models[1];
        REQUIRE(estimator.generate_models(data, 3, models) == 0);
    }

    return EXIT_SUCCESS;
}
