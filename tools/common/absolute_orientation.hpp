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

#pragma once
#ifndef ZEROSLAM_TOOLS_COMMON_ABSOLUTE_ORIENTATION_HPP
#define ZEROSLAM_TOOLS_COMMON_ABSOLUTE_ORIENTATION_HPP

#include "decomposition.hpp"

namespace alignment {
    using size_t = decltype(sizeof(0));

    constexpr static const double degenerate_conditioning = 1e-4;

    inline bool absolute_orientation(
        const double* const ground_truth_x,
        const double* const ground_truth_y,
        const double* const ground_truth_z,
        const size_t ground_truth_size,
        const double* const estimated_x,
        const double* const estimated_y,
        const double* const estimated_z,
        const size_t estimated_size,
        double& optimal_scale,
        double optimal_rotation[3][3],
        double optimal_translation[3],
        const bool overlap_first = false,
        const double* const prior_rotation = nullptr,
        double* const conditioning = nullptr
    ) {
        if ((ground_truth_size != estimated_size) || (ground_truth_size < 3)) {
            return false;
        }
        const size_t count = ground_truth_size;
        const double* const ground_truth[3] = { ground_truth_x, ground_truth_y, ground_truth_z };
        const double* const estimated[3] = { estimated_x, estimated_y, estimated_z };

        double ground_truth_anchor[3] = {};
        double estimated_anchor[3] = {};
        for (size_t axis = 0; axis < 3; ++axis) {
            if (overlap_first) {
                ground_truth_anchor[axis] = ground_truth[axis][0];
                estimated_anchor[axis] = estimated[axis][0];
            }
            else {
                for (size_t index = 0; index < count; ++index) {
                    ground_truth_anchor[axis] += ground_truth[axis][index];
                    estimated_anchor[axis] += estimated[axis][index];
                }
                ground_truth_anchor[axis] /= static_cast<double>(count);
                estimated_anchor[axis] /= static_cast<double>(count);
            }
        }

        double covariance[3][3] = {};
        double estimated_variance = 0.0;
        for (size_t index = 0; index < count; ++index) {
            double delta_estimated[3];
            double delta_ground_truth[3];
            for (size_t axis = 0; axis < 3; ++axis) {
                delta_estimated[axis] = estimated[axis][index] - estimated_anchor[axis];
                delta_ground_truth[axis] = ground_truth[axis][index] - ground_truth_anchor[axis];
                estimated_variance += delta_estimated[axis] * delta_estimated[axis];
            }
            for (size_t row = 0; row < 3; ++row) {
                for (size_t column = 0; column < 3; ++column) {
                    covariance[row][column] += delta_estimated[row] * delta_ground_truth[column];
                }
            }
        }

        double u[3][3];
        double s[3][3];
        double vt[3][3];
        if (!decomposition::singular_value_3x3(covariance, u, s, vt)) {
            return false;
        }

        const double stiffest = s[0][0] + s[1][1];
        const double slackest = s[1][1] + s[2][2];
        const double fit_conditioning = (stiffest > 0.0) ? (slackest / stiffest) : 0.0;
        if (conditioning != nullptr) {
            *conditioning = fit_conditioning;
        }

        if (prior_rotation != nullptr) {
            double regularised[3][3];
            for (size_t row = 0; row < 3; ++row) {
                for (size_t column = 0; column < 3; ++column) {
                    regularised[row][column] = covariance[row][column] + (degenerate_conditioning * s[0][0] * prior_rotation[(3 * column) + row]);
                }
            }
            if (!decomposition::singular_value_3x3(regularised, u, s, vt)) {
                return false;
            }
        }

        double v_ut[3][3];
        for (size_t row = 0; row < 3; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                v_ut[row][column] = 0.0;
                for (size_t k = 0; k < 3; ++k) {
                    v_ut[row][column] += vt[k][row] * u[column][k];
                }
            }
        }
        const double determinant =
            v_ut[0][0] * (v_ut[1][1] * v_ut[2][2] - v_ut[2][1] * v_ut[1][2]) -
            v_ut[0][1] * (v_ut[1][0] * v_ut[2][2] - v_ut[1][2] * v_ut[2][0]) +
            v_ut[0][2] * (v_ut[1][0] * v_ut[2][1] - v_ut[1][1] * v_ut[2][0]);
        const double reflection = (determinant > 0.0) ? 1.0 : -1.0;
        for (size_t row = 0; row < 3; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                optimal_rotation[row][column] = 0.0;
                for (size_t k = 0; k < 3; ++k) {
                    const double d = (k == 2) ? reflection : 1.0;
                    optimal_rotation[row][column] += vt[k][row] * d * u[column][k];
                }
            }
        }
        double correlation = 0.0;
        for (size_t row = 0; row < 3; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                correlation += optimal_rotation[column][row] * covariance[row][column];
            }
        }
        optimal_scale = (estimated_variance > 0.0) ? (correlation / estimated_variance) : 1.0;
        for (size_t row = 0; row < 3; ++row) {
            optimal_translation[row] = ground_truth_anchor[row];
            for (size_t column = 0; column < 3; ++column) {
                optimal_translation[row] -= optimal_scale * optimal_rotation[row][column] * estimated_anchor[column];
            }
        }
        return true;
    }
}

#endif // ZEROSLAM_TOOLS_COMMON_ABSOLUTE_ORIENTATION_HPP
