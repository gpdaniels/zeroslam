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

#include "estimation/robust/estimate/p3p.hpp"

#include "core/random_pcg.hpp"
#include "math/lie.hpp"
#include "math/math.hpp"

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

static inline math::matrix<double, 3, 3> make_rotation(core::random_pcg& random, double maximum_angle) {
    return math::so3<double>::rotation(random.get_random(-maximum_angle, maximum_angle), random.get_random(-maximum_angle, maximum_angle), random.get_random(-maximum_angle, maximum_angle)).get_matrix();
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(estimation::robust::estimate::p3p<double>::sample_size == 3);
    static_assert(estimation::robust::estimate::p3p<double>::models_size == 4);

    const estimation::robust::estimate::p3p<double> estimator;
    core::random_pcg random(0x5eed0022ull);

    for (int trial = 0; trial < 16; ++trial) {
        const math::matrix<double, 3, 3> rotation = make_rotation(random, 0.5);
        const math::matrix<double, 3, 1> translation{ { random.get_random(-1.0, 1.0), random.get_random(-1.0, 1.0), random.get_random(-0.5, 0.5) } };

        estimation::correspondence_2d_3d<double> data[6];
        for (size_t i = 0; i < 6; ++i) {
            const math::matrix<double, 3, 1> world{ { random.get_random(-1.5, 1.5), random.get_random(-1.5, 1.5), random.get_random(3.0, 8.0) } };
            const math::matrix<double, 3, 1> camera = (rotation * world) + translation;
            data[i].lhs = math::matrix<double, 2, 1>{ { camera[0] / camera[2], camera[1] / camera[2] } };
            data[i].rhs = world;
        }

        // One of the generated poses is the true pose.
        estimation::robust::estimate::p3p<double>::model models[4];
        const size_t count = estimator.generate_models(data, 3, models);
        REQUIRE(count >= 1);
        REQUIRE(count <= 4);
        size_t matching = 0;
        for (size_t m = 0; m < count; ++m) {
            bool matches = true;
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    matches = matches && is_value_approx(models[m].rotation[row][col], rotation[row][col], 1e-6);
                }
                matches = matches && is_value_approx(models[m].translation[row], translation[row], 1e-6);
            }
            if (matches) {
                ++matching;
            }
        }
        REQUIRE(matching == 1);
        REQUIRE(estimator.generate_models(data, 2, models) == 0);

        // Residuals vanish on true correspondences and grow with a displaced observation.
        estimation::robust::estimate::p3p<double>::model truth;
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                truth.rotation[row][col] = rotation[row][col];
            }
            truth.translation[row] = translation[row];
        }
        float residuals[6];
        estimator.compute_residuals(data, 6, truth, residuals);
        for (size_t i = 0; i < 6; ++i) {
            REQUIRE(residuals[i] < 1e-6f);
        }
        data[4].lhs[0] += 0.05;
        estimator.compute_residuals(data, 6, truth, residuals);
        REQUIRE(residuals[4] > 1e-4f);
        REQUIRE(residuals[3] < 1e-6f);
    }

    return EXIT_SUCCESS;
}
