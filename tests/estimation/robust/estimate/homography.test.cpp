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

#include "estimation/robust/estimate/homography.hpp"

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

    static_assert(estimation::robust::estimate::homography<double>::sample_size == 4);
    static_assert(estimation::robust::estimate::homography<double>::models_size == 1);

    const estimation::robust::estimate::homography<double> estimator;
    core::random_pcg random(0x5eed0021ull);

    for (int trial = 0; trial < 16; ++trial) {
        const math::matrix<double, 3, 3> rotation = make_rotation(random, 0.4);
        const math::matrix<double, 3, 1> translation{ { random.get_random(-0.5, 0.5), random.get_random(-0.5, 0.5), random.get_random(-0.3, 0.3) } };
        const double plane_a = random.get_random(-0.3, 0.3);
        const double plane_b = random.get_random(-0.3, 0.3);

        estimation::correspondence_2d_2d<double> data[8];
        for (size_t i = 0; i < 8; ++i) {
            const double x = random.get_random(-1.0, 1.0);
            const double y = random.get_random(-1.0, 1.0);
            const math::matrix<double, 3, 1> world{ { x, y, 4.0 + (plane_a * x) + (plane_b * y) } };
            const math::matrix<double, 3, 1> camera = (rotation * world) + translation;
            data[i].lhs = math::matrix<double, 2, 1>{ { world[0] / world[2], world[1] / world[2] } };
            data[i].rhs = math::matrix<double, 2, 1>{ { camera[0] / camera[2], camera[1] / camera[2] } };
        }

        // The model from four points transfers the other four, rhs to lhs.
        estimation::robust::estimate::homography<double>::model model;
        REQUIRE(estimator.generate_models(data, 4, &model) == 1);
        REQUIRE(estimator.generate_models(data, 3, &model) == 0);
        for (size_t i = 4; i < 8; ++i) {
            const double px = (model.homography[0][0] * data[i].rhs[0]) + (model.homography[0][1] * data[i].rhs[1]) + model.homography[0][2];
            const double py = (model.homography[1][0] * data[i].rhs[0]) + (model.homography[1][1] * data[i].rhs[1]) + model.homography[1][2];
            const double pw = (model.homography[2][0] * data[i].rhs[0]) + (model.homography[2][1] * data[i].rhs[1]) + model.homography[2][2];
            REQUIRE(is_value_approx(px / pw, data[i].lhs[0], 1e-7));
            REQUIRE(is_value_approx(py / pw, data[i].lhs[1], 1e-7));
        }

        // Residuals vanish on the plane and grow off it.
        float residuals[8];
        estimator.compute_residuals(data, 8, model, residuals);
        for (size_t i = 0; i < 8; ++i) {
            REQUIRE(residuals[i] < 1e-10f);
        }
        data[5].lhs[1] += 0.02;
        estimator.compute_residuals(data, 8, model, residuals);
        REQUIRE(residuals[5] > 1e-5f);
        REQUIRE(residuals[6] < 1e-10f);
    }

    // A singular homography marks everything an outlier.
    {
        estimation::robust::estimate::homography<double>::model singular;
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                singular.homography[row][col] = 0.0;
            }
        }
        singular.homography[0][0] = 1.0;
        estimation::correspondence_2d_2d<double> data[2];
        data[0].lhs = math::matrix<double, 2, 1>{ { 0.1, 0.2 } };
        data[0].rhs = math::matrix<double, 2, 1>{ { 0.3, 0.4 } };
        data[1] = data[0];
        float residuals[2];
        estimator.compute_residuals(data, 2, singular, residuals);
        REQUIRE(std::isinf(residuals[0]));
        REQUIRE(std::isinf(residuals[1]));
    }

    return EXIT_SUCCESS;
}
