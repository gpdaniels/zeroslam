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
#ifndef TRAJECTORY_ALIGNMENT_ABSOLUTE_ORIENTATION_HPP
#define TRAJECTORY_ALIGNMENT_ABSOLUTE_ORIENTATION_HPP

#include "matrix_decomposition_singular_value.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

void compute_centroids(
    const double* const ground_truth_x,
    const double* const ground_truth_y,
    const double* const ground_truth_z,
    const double* const estimated_x,
    const double* const estimated_y,
    const double* const estimated_z,
    const size_t point_count,
    const bool use_first_pose_for_overlap,
    double& ground_truth_centroid_x,
    double& ground_truth_centroid_y,
    double& ground_truth_centroid_z,
    double& estimated_centroid_x,
    double& estimated_centroid_y,
    double& estimated_centroid_z
) {
    if (use_first_pose_for_overlap) {
        ground_truth_centroid_x = ground_truth_x[0];
        ground_truth_centroid_y = ground_truth_y[0];
        ground_truth_centroid_z = ground_truth_z[0];
        estimated_centroid_x = estimated_x[0];
        estimated_centroid_y = estimated_y[0];
        estimated_centroid_z = estimated_z[0];
    } else {
        double ground_truth_sum_x = 0;
        double ground_truth_sum_y = 0;
        double ground_truth_sum_z = 0;
        double estimated_sum_x = 0;
        double estimated_sum_y = 0;
        double estimated_sum_z = 0;

        for (size_t index = 0; index < point_count; ++index) {
            ground_truth_sum_x += ground_truth_x[index];
            ground_truth_sum_y += ground_truth_y[index];
            ground_truth_sum_z += ground_truth_z[index];
            estimated_sum_x += estimated_x[index];
            estimated_sum_y += estimated_y[index];
            estimated_sum_z += estimated_z[index];
        }

        ground_truth_centroid_x = ground_truth_sum_x / static_cast<double>(point_count);
        ground_truth_centroid_y = ground_truth_sum_y / static_cast<double>(point_count);
        ground_truth_centroid_z = ground_truth_sum_z / static_cast<double>(point_count);
        estimated_centroid_x = estimated_sum_x / static_cast<double>(point_count);
        estimated_centroid_y = estimated_sum_y / static_cast<double>(point_count);
        estimated_centroid_z = estimated_sum_z / static_cast<double>(point_count);
    }
}

void compute_covariance_matrix(
    const double* const ground_truth_x,
    const double* const ground_truth_y,
    const double* const ground_truth_z,
    const double* const estimated_x,
    const double* const estimated_y,
    const double* const estimated_z,
    const size_t point_count,
    const double ground_truth_centroid_x,
    const double ground_truth_centroid_y,
    const double ground_truth_centroid_z,
    const double estimated_centroid_x,
    const double estimated_centroid_y,
    const double estimated_centroid_z,
    double covariance_matrix[3][3]
) {
    for (size_t row = 0; row < 3; ++row) {
        for (size_t column = 0; column < 3; ++column) {
            covariance_matrix[row][column] = 0;
        }
    }

    for (size_t index = 0; index < point_count; ++index) {
        const double delta_estimated_x = estimated_x[index] - estimated_centroid_x;
        const double delta_estimated_y = estimated_y[index] - estimated_centroid_y;
        const double delta_estimated_z = estimated_z[index] - estimated_centroid_z;

        const double delta_ground_truth_x = ground_truth_x[index] - ground_truth_centroid_x;
        const double delta_ground_truth_y = ground_truth_y[index] - ground_truth_centroid_y;
        const double delta_ground_truth_z = ground_truth_z[index] - ground_truth_centroid_z;

        covariance_matrix[0][0] += delta_estimated_x * delta_ground_truth_x;
        covariance_matrix[0][1] += delta_estimated_x * delta_ground_truth_y;
        covariance_matrix[0][2] += delta_estimated_x * delta_ground_truth_z;
        covariance_matrix[1][0] += delta_estimated_y * delta_ground_truth_x;
        covariance_matrix[1][1] += delta_estimated_y * delta_ground_truth_y;
        covariance_matrix[1][2] += delta_estimated_y * delta_ground_truth_z;
        covariance_matrix[2][0] += delta_estimated_z * delta_ground_truth_x;
        covariance_matrix[2][1] += delta_estimated_z * delta_ground_truth_y;
        covariance_matrix[2][2] += delta_estimated_z * delta_ground_truth_z;
    }
}

void compute_rotation_matrix(
    double covariance_matrix[3][3],
    double optimal_rotation[3][3]
) {
    double singular_value_u[3][3];
    double singular_value_s[3][3];
    double singular_value_vt[3][3];

    matrix::decompose_singular_value(
        &covariance_matrix[0][0], 
        3, 
        3, 
        &singular_value_u[0][0], 
        &singular_value_s[0][0], 
        &singular_value_vt[0][0]
    );

    double singular_value_v[3][3];
    for (size_t row = 0; row < 3; ++row) {
        for (size_t column = 0; column < 3; ++column) {
            singular_value_v[row][column] = singular_value_vt[column][row];
        }
    }

    double singular_value_ut[3][3];
    for (size_t row = 0; row < 3; ++row) {
        for (size_t column = 0; column < 3; ++column) {
            singular_value_ut[row][column] = singular_value_u[column][row];
        }
    }

    double matrix_vut[3][3];
    for (size_t row = 0; row < 3; ++row) {
        for (size_t column = 0; column < 3; ++column) {
            matrix_vut[row][column] = 0;
            for (size_t k = 0; k < 3; ++k) {
                matrix_vut[row][column] += singular_value_v[row][k] * singular_value_ut[k][column];
            }
        }
    }

    const double determinant = 
        matrix_vut[0][0] * (matrix_vut[1][1] * matrix_vut[2][2] - matrix_vut[2][1] * matrix_vut[1][2]) -
        matrix_vut[0][1] * (matrix_vut[1][0] * matrix_vut[2][2] - matrix_vut[1][2] * matrix_vut[2][0]) +
        matrix_vut[0][2] * (matrix_vut[1][0] * matrix_vut[2][1] - matrix_vut[1][1] * matrix_vut[2][0]);

    const double reflection_matrix[3][3] = { 
        { 1.0, 0.0, 0.0 }, 
        { 0.0, 1.0, 0.0 }, 
        { 0.0, 0.0, (determinant > 0.0) ? 1.0 : -1.0 } 
    };

    double matrix_vr[3][3];
    for (size_t row = 0; row < 3; ++row) {
        for (size_t column = 0; column < 3; ++column) {
            matrix_vr[row][column] = 0;
            for (size_t k = 0; k < 3; ++k) {
                matrix_vr[row][column] += singular_value_v[row][k] * reflection_matrix[k][column];
            }
        }
    }

    for (size_t row = 0; row < 3; ++row) {
        for (size_t column = 0; column < 3; ++column) {
            optimal_rotation[row][column] = 0;
            for (size_t k = 0; k < 3; ++k) {
                optimal_rotation[row][column] += matrix_vr[row][k] * singular_value_ut[k][column];
            }
        }
    }
}

double compute_optimal_scale(
    const double* const estimated_x,
    const double* const estimated_y,
    const double* const estimated_z,
    const size_t point_count,
    const double estimated_centroid_x,
    const double estimated_centroid_y,
    const double estimated_centroid_z,
    const double covariance_matrix[3][3]
) {
    double singular_value_u[3][3];
    double singular_value_s[3][3];
    double singular_value_vt[3][3];
    double covariance_copy[3][3];
    for (size_t row = 0; row < 3; ++row) {
        for (size_t column = 0; column < 3; ++column) {
            covariance_copy[row][column] = covariance_matrix[row][column];
        }
    }

    matrix::decompose_singular_value(
        &covariance_copy[0][0], 
        3, 
        3, 
        &singular_value_u[0][0], 
        &singular_value_s[0][0], 
        &singular_value_vt[0][0]
    );

    double estimated_variance = 0;
    for (size_t index = 0; index < point_count; ++index) {
        const double delta_x = estimated_x[index] - estimated_centroid_x;
        const double delta_y = estimated_y[index] - estimated_centroid_y;
        const double delta_z = estimated_z[index] - estimated_centroid_z;
        estimated_variance += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    const double trace_s = singular_value_s[0][0] + singular_value_s[1][1] + singular_value_s[2][2];
    return (estimated_variance > 0.0) ? (trace_s / estimated_variance) : 1.0;
}

void compute_optimal_translation(
    const double ground_truth_centroid_x,
    const double ground_truth_centroid_y,
    const double ground_truth_centroid_z,
    const double estimated_centroid_x,
    const double estimated_centroid_y,
    const double estimated_centroid_z,
    const double optimal_scale,
    const double optimal_rotation[3][3],
    double optimal_translation[3]
) {
    optimal_translation[0] = ground_truth_centroid_x - optimal_scale * (optimal_rotation[0][0] * estimated_centroid_x + optimal_rotation[0][1] * estimated_centroid_y + optimal_rotation[0][2] * estimated_centroid_z);
    optimal_translation[1] = ground_truth_centroid_y - optimal_scale * (optimal_rotation[1][0] * estimated_centroid_x + optimal_rotation[1][1] * estimated_centroid_y + optimal_rotation[1][2] * estimated_centroid_z);
    optimal_translation[2] = ground_truth_centroid_z - optimal_scale * (optimal_rotation[2][0] * estimated_centroid_x + optimal_rotation[2][1] * estimated_centroid_y + optimal_rotation[2][2] * estimated_centroid_z);
}

bool absolute_orientation(
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
    const bool overlap_first = false
) {
    if ((ground_truth_size != estimated_size) || (ground_truth_size < 3)) {
        return false;
    }

    const size_t point_count = ground_truth_size;

    double ground_truth_centroid_x = 0;
    double ground_truth_centroid_y = 0;
    double ground_truth_centroid_z = 0;
    double estimated_centroid_x = 0;
    double estimated_centroid_y = 0;
    double estimated_centroid_z = 0;

    compute_centroids(
        ground_truth_x,
        ground_truth_y,
        ground_truth_z,
        estimated_x,
        estimated_y,
        estimated_z,
        point_count,
        overlap_first,
        ground_truth_centroid_x,
        ground_truth_centroid_y,
        ground_truth_centroid_z,
        estimated_centroid_x,
        estimated_centroid_y,
        estimated_centroid_z
    );

    double covariance_matrix[3][3];
    compute_covariance_matrix(
        ground_truth_x,
        ground_truth_y,
        ground_truth_z,
        estimated_x,
        estimated_y,
        estimated_z,
        point_count,
        ground_truth_centroid_x,
        ground_truth_centroid_y,
        ground_truth_centroid_z,
        estimated_centroid_x,
        estimated_centroid_y,
        estimated_centroid_z,
        covariance_matrix
    );

    compute_rotation_matrix(
        covariance_matrix, 
        optimal_rotation
    );

    optimal_scale = compute_optimal_scale(
        estimated_x,
        estimated_y,
        estimated_z,
        point_count,
        estimated_centroid_x,
        estimated_centroid_y,
        estimated_centroid_z,
        covariance_matrix
    );

    compute_optimal_translation(
        ground_truth_centroid_x,
        ground_truth_centroid_y,
        ground_truth_centroid_z,
        estimated_centroid_x,
        estimated_centroid_y,
        estimated_centroid_z,
        optimal_scale,
        optimal_rotation,
        optimal_translation
    );

    return true;
}

#endif

