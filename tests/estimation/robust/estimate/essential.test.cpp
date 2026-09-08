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

#include "estimation/robust/estimate/essential.hpp"

#include "core/random_pcg.hpp"
#include "geometry/essential.hpp"
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

    static_assert(estimation::robust::estimate::essential<double>::sample_size == 5);
    static_assert(estimation::robust::estimate::essential<double>::models_size == 10);

    const estimation::robust::estimate::essential<double> estimator;
    core::random_pcg random(0x5eed0020ull);

    for (int trial = 0; trial < 16; ++trial) {
        const math::matrix<double, 3, 3> rotation = make_rotation(random, 0.5);
        const math::matrix<double, 3, 1> translation{ { random.get_random(-1.0, 1.0), random.get_random(-1.0, 1.0), random.get_random(0.5, 1.5) } };
        math::matrix<double, 3, 3> essential;
        geometry::essential<double>::from_poses(rotation, translation, essential);

        estimation::correspondence_2d_2d<double> data[8];
        for (size_t i = 0; i < 8; ++i) {
            const math::matrix<double, 3, 1> world{ { random.get_random(-1.0, 1.0), random.get_random(-1.0, 1.0), random.get_random(3.0, 8.0) } };
            const math::matrix<double, 3, 1> camera = (rotation * world) + translation;
            data[i].lhs = math::matrix<double, 2, 1>{ { world[0] / world[2], world[1] / world[2] } };
            data[i].rhs = math::matrix<double, 2, 1>{ { camera[0] / camera[2], camera[1] / camera[2] } };
        }

        // One of the generated models is the true essential matrix up to scale and sign.
        estimation::robust::estimate::essential<double>::model models[10];
        const size_t count = estimator.generate_models(data, 5, models);
        REQUIRE(count >= 1);
        REQUIRE(count <= 10);
        size_t matching = 0;
        for (size_t m = 0; m < count; ++m) {
            double dot = 0.0;
            double norm_model = 0.0;
            double norm_true = 0.0;
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    dot += models[m].essential[row][col] * essential[row][col];
                    norm_model += models[m].essential[row][col] * models[m].essential[row][col];
                    norm_true += essential[row][col] * essential[row][col];
                }
            }
            if (is_value_approx(std::abs(dot) / std::sqrt(norm_model * norm_true), 1.0, 1e-6)) {
                ++matching;
            }
        }
        REQUIRE(matching >= 1);
        REQUIRE(estimator.generate_models(data, 4, models) == 0);

        // Residuals vanish on true correspondences and grow with a displaced observation.
        estimation::robust::estimate::essential<double>::model truth;
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                truth.essential[row][col] = essential[row][col];
            }
        }
        float residuals[8];
        estimator.compute_residuals(data, 8, truth, residuals);
        for (size_t i = 0; i < 8; ++i) {
            REQUIRE(residuals[i] < 1e-12f);
        }
        const math::matrix<double, 3, 1> line = essential * math::matrix<double, 3, 1>{ { data[3].lhs[0], data[3].lhs[1], 1.0 } };
        const double line_normal = std::sqrt((line[0] * line[0]) + (line[1] * line[1]));
        data[3].rhs[0] += 0.05 * line[0] / line_normal;
        data[3].rhs[1] += 0.05 * line[1] / line_normal;
        estimator.compute_residuals(data, 8, truth, residuals);
        REQUIRE(residuals[3] > 1e-5f);
        REQUIRE(residuals[2] < 1e-12f);
    }

    return EXIT_SUCCESS;
}
