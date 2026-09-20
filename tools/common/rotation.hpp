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
#ifndef ZEROSLAM_TOOLS_COMMON_ROTATION_HPP
#define ZEROSLAM_TOOLS_COMMON_ROTATION_HPP

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace rotation {
    inline void quaternion_to_matrix(const double* const quaternion_xyzw, double* const matrix) {
        const double two_x = 2.0 * quaternion_xyzw[0];
        const double two_y = 2.0 * quaternion_xyzw[1];
        const double two_z = 2.0 * quaternion_xyzw[2];
        const double two_x_x = two_x * quaternion_xyzw[0];
        const double two_x_y = two_x * quaternion_xyzw[1];
        const double two_x_z = two_x * quaternion_xyzw[2];
        const double two_x_w = two_x * quaternion_xyzw[3];
        const double two_y_y = two_y * quaternion_xyzw[1];
        const double two_y_z = two_y * quaternion_xyzw[2];
        const double two_y_w = two_y * quaternion_xyzw[3];
        const double two_z_z = two_z * quaternion_xyzw[2];
        const double two_z_w = two_z * quaternion_xyzw[3];
        matrix[0] = 1.0 - (two_y_y + two_z_z);
        matrix[1] = two_x_y - two_z_w;
        matrix[2] = two_x_z + two_y_w;
        matrix[3] = two_x_y + two_z_w;
        matrix[4] = 1.0 - (two_x_x + two_z_z);
        matrix[5] = two_y_z - two_x_w;
        matrix[6] = two_x_z - two_y_w;
        matrix[7] = two_y_z + two_x_w;
        matrix[8] = 1.0 - (two_x_x + two_y_y);
    }

    inline void matrix_to_quaternion(const double* const matrix, double* const quaternion_xyzw) {
        const auto at = [&](const int row, const int column) {
            return matrix[(3 * row) + column];
        };
        const double trace = at(0, 0) + at(1, 1) + at(2, 2);
        if (trace > 0) {
            const double square_root = std::sqrt(trace + 1);
            const double inverse_two_square_root = 0.5 / square_root;
            quaternion_xyzw[3] = 0.5 * square_root;
            quaternion_xyzw[0] = (at(2, 1) - at(1, 2)) * inverse_two_square_root;
            quaternion_xyzw[1] = (at(0, 2) - at(2, 0)) * inverse_two_square_root;
            quaternion_xyzw[2] = (at(1, 0) - at(0, 1)) * inverse_two_square_root;
            return;
        }
        const int i = (at(0, 0) < at(1, 1)) ? (1 + (at(1, 1) < at(2, 2))) : (2 * (at(0, 0) < at(2, 2)));
        const int j = (i + 1) % 3;
        const int k = (j + 1) % 3;
        const double square_root = std::sqrt(at(i, i) - at(j, j) - at(k, k) + 1.0);
        const double inverse_two_square_root = 0.5 / square_root;
        quaternion_xyzw[3] = (at(k, j) - at(j, k)) * inverse_two_square_root;
        quaternion_xyzw[i] = 0.5 * square_root;
        quaternion_xyzw[j] = (at(i, j) + at(j, i)) * inverse_two_square_root;
        quaternion_xyzw[k] = (at(i, k) + at(k, i)) * inverse_two_square_root;
    }

    inline void slerp(const double* const from_xyzw, const double* const to_xyzw, const double fraction, double* const result_xyzw) {
        double to[4] = { to_xyzw[0], to_xyzw[1], to_xyzw[2], to_xyzw[3] };
        double cosine = (from_xyzw[0] * to[0]) + (from_xyzw[1] * to[1]) + (from_xyzw[2] * to[2]) + (from_xyzw[3] * to[3]);
        if (cosine < 0.0) {
            cosine = -cosine;
            for (int i = 0; i < 4; ++i) {
                to[i] = -to[i];
            }
        }
        double weight_from = 1.0 - fraction;
        double weight_to = fraction;
        if (cosine < 0.9995) {
            const double angle = std::acos(cosine > 1.0 ? 1.0 : cosine);
            const double sine = std::sin(angle);
            weight_from = std::sin((1.0 - fraction) * angle) / sine;
            weight_to = std::sin(fraction * angle) / sine;
        }
        double length = 0.0;
        for (int i = 0; i < 4; ++i) {
            result_xyzw[i] = (weight_from * from_xyzw[i]) + (weight_to * to[i]);
            length += result_xyzw[i] * result_xyzw[i];
        }
        length = std::sqrt(length);
        if (length > 0.0) {
            for (int i = 0; i < 4; ++i) {
                result_xyzw[i] /= length;
            }
        }
    }

    struct rigid {
        double rotation[9] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
        double translation[3] = { 0.0, 0.0, 0.0 };

        static rigid from_matrix(const double* const matrix_4x4) {
            rigid result;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    result.rotation[(3 * row) + column] = matrix_4x4[(row * 4) + column];
                }
                result.translation[row] = matrix_4x4[(row * 4) + 3];
            }
            return result;
        }

        static rigid from_pose(const double* const position, const double* const quaternion_xyzw) {
            rigid result;
            rotation::quaternion_to_matrix(quaternion_xyzw, &result.rotation[0]);
            for (int row = 0; row < 3; ++row) {
                result.translation[row] = position[row];
            }
            return result;
        }

        rigid inverse() const {
            rigid result;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    result.rotation[(3 * row) + column] = this->rotation[(3 * column) + row];
                }
            }
            for (int row = 0; row < 3; ++row) {
                double sum = 0.0;
                for (int column = 0; column < 3; ++column) {
                    sum += result.rotation[(3 * row) + column] * this->translation[column];
                }
                result.translation[row] = -sum;
            }
            return result;
        }

        rigid operator*(const rigid& other) const {
            rigid result;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    double sum = 0.0;
                    for (int k = 0; k < 3; ++k) {
                        sum += this->rotation[(3 * row) + k] * other.rotation[(3 * k) + column];
                    }
                    result.rotation[(3 * row) + column] = sum;
                }
                double sum = 0.0;
                for (int k = 0; k < 3; ++k) {
                    sum += this->rotation[(3 * row) + k] * other.translation[k];
                }
                result.translation[row] = sum + this->translation[row];
            }
            return result;
        }

        void to_pose(double* const position, double* const quaternion_xyzw) const {
            for (int row = 0; row < 3; ++row) {
                position[row] = this->translation[row];
            }
            rotation::matrix_to_quaternion(&this->rotation[0], quaternion_xyzw);
        }
    };
}

#endif // ZEROSLAM_TOOLS_COMMON_ROTATION_HPP
