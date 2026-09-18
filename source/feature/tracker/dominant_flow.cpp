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

#include "feature/tracker/dominant_flow.hpp"

#include "math/math.hpp"
#include "math/matrix.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::tracker {
    void dominant_flow::affine::apply(const float x, const float y, float& out_x, float& out_y) const {
        out_x = this->a00 * x + this->a01 * y + this->bx;
        out_y = this->a10 * x + this->a11 * y + this->by;
    }

    bool dominant_flow::affine::invert(affine& inverse) const {
        const float determinant = this->a00 * this->a11 - this->a01 * this->a10;
        if (math::abs(determinant) < 1.0e-6f) {
            return false;
        }
        affine result;
        result.a00 = this->a11 / determinant;
        result.a01 = -this->a01 / determinant;
        result.a10 = -this->a10 / determinant;
        result.a11 = this->a00 / determinant;
        result.bx = -(result.a00 * this->bx + result.a01 * this->by);
        result.by = -(result.a10 * this->bx + result.a11 * this->by);
        inverse = result;
        return true;
    }

    bool dominant_flow::fit(
        const float* __restrict const from_x,
        const float* __restrict const from_y,
        const float* __restrict const to_x,
        const float* __restrict const to_y,
        const size_t count,
        const options& settings,
        affine& flow
    ) {
        if (count < 3) {
            return false;
        }

        std::vector<float> differences(count);
        for (size_t i = 0; i < count; ++i) {
            differences[i] = to_x[i] - from_x[i];
        }
        std::sort(differences.begin(), differences.end());
        const float median_x = differences[count / 2];
        for (size_t i = 0; i < count; ++i) {
            differences[i] = to_y[i] - from_y[i];
        }
        std::sort(differences.begin(), differences.end());
        const float median_y = differences[count / 2];

        affine current;
        current.bx = median_x;
        current.by = median_y;

        const double sigma_squared = static_cast<double>(settings.sigma) * static_cast<double>(settings.sigma);
        for (int iteration = 0; iteration < settings.iterations; ++iteration) {
            math::matrix<double, 3, 3> normal = math::matrix<double, 3, 3>::zero();
            math::matrix<double, 3, 1> rhs_x = math::matrix<double, 3, 1>::zero();
            math::matrix<double, 3, 1> rhs_y = math::matrix<double, 3, 1>::zero();
            for (size_t i = 0; i < count; ++i) {
                float predicted_x;
                float predicted_y;
                current.apply(from_x[i], from_y[i], predicted_x, predicted_y);
                const double residual_x = static_cast<double>(to_x[i]) - static_cast<double>(predicted_x);
                const double residual_y = static_cast<double>(to_y[i]) - static_cast<double>(predicted_y);
                const double residual_squared = residual_x * residual_x + residual_y * residual_y;
                const double denominator = residual_squared + sigma_squared;
                const double weight = (sigma_squared * sigma_squared) / (denominator * denominator);
                const double phi[3] = { static_cast<double>(from_x[i]), static_cast<double>(from_y[i]), 1.0 };
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t column = 0; column < 3; ++column) {
                        normal[row][column] += weight * phi[row] * phi[column];
                    }
                    rhs_x[row] += weight * phi[row] * static_cast<double>(to_x[i]);
                    rhs_y[row] += weight * phi[row] * static_cast<double>(to_y[i]);
                }
            }
            math::matrix<double, 3, 3> normal_inverse;
            if (!math::invert(normal, normal_inverse)) {
                return false;
            }
            const math::matrix<double, 3, 1> row_x = normal_inverse * rhs_x;
            const math::matrix<double, 3, 1> row_y = normal_inverse * rhs_y;
            affine next;
            next.a00 = static_cast<float>(row_x[0]);
            next.a01 = static_cast<float>(row_x[1]);
            next.bx = static_cast<float>(row_x[2]);
            next.a10 = static_cast<float>(row_y[0]);
            next.a11 = static_cast<float>(row_y[1]);
            next.by = static_cast<float>(row_y[2]);
            const float change = math::max(
                math::max(
                    math::max(math::abs(next.a00 - current.a00), math::abs(next.a01 - current.a01)),
                    math::max(math::abs(next.a10 - current.a10), math::abs(next.a11 - current.a11))
                ),
                math::max(math::abs(next.bx - current.bx), math::abs(next.by - current.by))
            );
            current = next;
            if (change < settings.convergence) {
                break;
            }
        }
        flow = current;
        return true;
    }
}
