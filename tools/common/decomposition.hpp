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
#ifndef ZEROSLAM_TOOLS_COMMON_DECOMPOSITION_HPP
#define ZEROSLAM_TOOLS_COMMON_DECOMPOSITION_HPP

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace decomposition {
    inline bool singular_value_3x3(const double a[3][3], double u[3][3], double s[3][3], double vt[3][3]) {
        double work[3][3];
        double v[3][3];
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                if (!std::isfinite(a[row][column])) {
                    return false;
                }
                work[row][column] = a[row][column];
                v[row][column] = (row == column) ? 1.0 : 0.0;
            }
        }
        for (int sweep = 0; sweep < 60; ++sweep) {
            double off_diagonal = 0.0;
            for (int p = 0; p < 2; ++p) {
                for (int q = p + 1; q < 3; ++q) {
                    double alpha = 0.0;
                    double beta = 0.0;
                    double gamma = 0.0;
                    for (int row = 0; row < 3; ++row) {
                        alpha += work[row][p] * work[row][p];
                        beta += work[row][q] * work[row][q];
                        gamma += work[row][p] * work[row][q];
                    }
                    if ((gamma == 0.0) || (std::fabs(gamma) <= 1e-300)) {
                        continue;
                    }
                    off_diagonal = std::fmax(off_diagonal, std::fabs(gamma) / std::sqrt(alpha * beta));
                    const double zeta = (beta - alpha) / (2.0 * gamma);
                    const double t = (zeta >= 0.0 ? 1.0 : -1.0) / (std::fabs(zeta) + std::sqrt(1.0 + (zeta * zeta)));
                    const double c = 1.0 / std::sqrt(1.0 + (t * t));
                    const double sn = c * t;
                    for (int row = 0; row < 3; ++row) {
                        const double work_p = work[row][p];
                        const double work_q = work[row][q];
                        work[row][p] = (c * work_p) - (sn * work_q);
                        work[row][q] = (sn * work_p) + (c * work_q);
                        const double v_p = v[row][p];
                        const double v_q = v[row][q];
                        v[row][p] = (c * v_p) - (sn * v_q);
                        v[row][q] = (sn * v_p) + (c * v_q);
                    }
                }
            }
            if (off_diagonal < 1e-15) {
                break;
            }
        }
        double sigma[3];
        int order[3] = { 0, 1, 2 };
        for (int column = 0; column < 3; ++column) {
            sigma[column] = std::sqrt((work[0][column] * work[0][column]) + (work[1][column] * work[1][column]) + (work[2][column] * work[2][column]));
        }
        for (int i = 0; i < 2; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                if (sigma[order[j]] > sigma[order[i]]) {
                    const int swap = order[i];
                    order[i] = order[j];
                    order[j] = swap;
                }
            }
        }
        const double largest = sigma[order[0]];
        const double tolerance = largest * 1e-13;
        int rank = 0;
        for (int index = 0; index < 3; ++index) {
            const int column = order[index];
            for (int row = 0; row < 3; ++row) {
                s[row][index] = 0.0;
                vt[index][row] = v[row][column];
            }
            s[index][index] = sigma[column];
            if (sigma[column] > tolerance) {
                for (int row = 0; row < 3; ++row) {
                    u[row][index] = work[row][column] / sigma[column];
                }
                ++rank;
            }
        }
        if (rank < 3) {
            if (rank == 0) {
                for (int row = 0; row < 3; ++row) {
                    u[row][0] = (row == 0) ? 1.0 : 0.0;
                }
                rank = 1;
            }
            if (rank == 1) {
                const int axis = (std::fabs(u[0][0]) < 0.9) ? 0 : 1;
                double candidate[3] = { 0.0, 0.0, 0.0 };
                candidate[axis] = 1.0;
                const double dot = candidate[0] * u[0][0] + candidate[1] * u[1][0] + candidate[2] * u[2][0];
                double length = 0.0;
                for (int row = 0; row < 3; ++row) {
                    candidate[row] -= dot * u[row][0];
                    length += candidate[row] * candidate[row];
                }
                length = std::sqrt(length);
                for (int row = 0; row < 3; ++row) {
                    u[row][1] = candidate[row] / length;
                }
            }
            u[0][2] = (u[1][0] * u[2][1]) - (u[2][0] * u[1][1]);
            u[1][2] = (u[2][0] * u[0][1]) - (u[0][0] * u[2][1]);
            u[2][2] = (u[0][0] * u[1][1]) - (u[1][0] * u[0][1]);
        }
        return true;
    }
}

#endif // ZEROSLAM_TOOLS_COMMON_DECOMPOSITION_HPP
