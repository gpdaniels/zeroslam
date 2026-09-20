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
#ifndef ZEROSLAM_TOOLS_COMMON_METRICS_HPP
#define ZEROSLAM_TOOLS_COMMON_METRICS_HPP

#include "absolute_orientation.hpp"
#include "dataset.hpp"
#include "rotation.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace metrics {

    constexpr static const long long default_association_tolerance_nanoseconds = 20000000LL;

    constexpr static const double relative_distance_metres = 1.0;

    constexpr static const std::size_t interval_count = 2;
    constexpr static const std::size_t intervals[interval_count] = { 1, 10 };

    struct similarity {
        bool valid = false;
        double scale = 1.0;
        double rotation[3][3] = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } };
        double translation[3] = { 0.0, 0.0, 0.0 };

        void apply(const double x, const double y, const double z, double& out_x, double& out_y, double& out_z) const {
            out_x = (((this->rotation[0][0] * x) + (this->rotation[0][1] * y) + (this->rotation[0][2] * z)) * this->scale) + this->translation[0];
            out_y = (((this->rotation[1][0] * x) + (this->rotation[1][1] * y) + (this->rotation[1][2] * z)) * this->scale) + this->translation[1];
            out_z = (((this->rotation[2][0] * x) + (this->rotation[2][1] * y) + (this->rotation[2][2] * z)) * this->scale) + this->translation[2];
        }
    };

    struct interval_result {
        std::size_t interval = 0;
        std::size_t segments = 0;
        double rmse = 0.0;
        double mean = 0.0;
        double maximum = 0.0;
        double latest = 0.0;
        std::size_t measurable_segments = 0;
        double scale_median = 1.0;
        double scale_worst = 1.0;
    };

    struct result {
        bool valid = false;
        similarity transform;
        double alignment_conditioning = 0.0;
        bool alignment_degenerate = false;
        std::vector<std::pair<std::size_t, std::size_t>> pairs;
        std::vector<double> aligned_x;
        std::vector<double> aligned_y;
        std::vector<double> aligned_z;
        std::vector<double> errors;
        double ate_rmse = 0.0;
        double ate_mean = 0.0;
        double ate_median = 0.0;
        double ate_minimum = 0.0;
        double ate_maximum = 0.0;
        double ate_standard_deviation = 0.0;
        double ate_sum_squared = 0.0;
        double distance_ground_truth = 0.0;
        double distance_estimated = 0.0;
        interval_result relative[interval_count];

        struct distance_result {
            std::size_t segments = 0;
            double translation_percent_rmse = 0.0;
            double translation_percent_median = 0.0;
            double rotation_degrees_rmse = 0.0;
            double rotation_degrees_median = 0.0;
        };

        distance_result per_metre;
    };

    inline bool first_pair_rotation(const dataset::trajectory_pose& truth, const dataset::trajectory_pose& estimated, double* const rotation) {
        const double truth_xyzw[4] = { truth.quaternion_x, truth.quaternion_y, truth.quaternion_z, truth.quaternion_w };
        const double estimated_xyzw[4] = { estimated.quaternion_x, estimated.quaternion_y, estimated.quaternion_z, estimated.quaternion_w };
        for (const double* const quaternion : { &truth_xyzw[0], &estimated_xyzw[0] }) {
            const double length_squared = (quaternion[0] * quaternion[0]) + (quaternion[1] * quaternion[1]) + (quaternion[2] * quaternion[2]) + (quaternion[3] * quaternion[3]);
            if (std::fabs(length_squared - 1.0) > 1e-6) {
                return false;
            }
        }
        double truth_matrix[9];
        double estimated_matrix[9];
        rotation::quaternion_to_matrix(&truth_xyzw[0], &truth_matrix[0]);
        rotation::quaternion_to_matrix(&estimated_xyzw[0], &estimated_matrix[0]);
        for (std::size_t row = 0; row < 3; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                double sum = 0.0;
                for (std::size_t k = 0; k < 3; ++k) {
                    sum += truth_matrix[(3 * row) + k] * estimated_matrix[(3 * column) + k];
                }
                rotation[(3 * row) + column] = sum;
            }
        }
        return true;
    }

    inline result evaluate(
        const std::vector<dataset::trajectory_pose>& ground_truth,
        const std::vector<dataset::trajectory_pose>& estimated,
        const bool overlap_first_pose,
        const long long association_tolerance_nanoseconds = default_association_tolerance_nanoseconds
    ) {
        result output;
        dataset::associate_trajectories(ground_truth, estimated, association_tolerance_nanoseconds, output.pairs);
        const std::size_t count = output.pairs.size();
        if (count < 3) {
            return output;
        }

        std::vector<double> truth[3];
        std::vector<double> estimate[3];
        for (int axis = 0; axis < 3; ++axis) {
            truth[axis].reserve(count);
            estimate[axis].reserve(count);
        }
        for (const std::pair<std::size_t, std::size_t>& pair : output.pairs) {
            const dataset::trajectory_pose& reference = ground_truth[pair.first];
            const dataset::trajectory_pose& pose = estimated[pair.second];
            truth[0].push_back(reference.x_coordinate);
            truth[1].push_back(reference.y_coordinate);
            truth[2].push_back(reference.z_coordinate);
            estimate[0].push_back(pose.x_coordinate);
            estimate[1].push_back(pose.y_coordinate);
            estimate[2].push_back(pose.z_coordinate);
        }

        double prior_rotation[9];
        const bool prior = first_pair_rotation(ground_truth[output.pairs.front().first], estimated[output.pairs.front().second], &prior_rotation[0]);

        if (!alignment::absolute_orientation(truth[0].data(), truth[1].data(), truth[2].data(), count, estimate[0].data(), estimate[1].data(), estimate[2].data(), count, output.transform.scale, output.transform.rotation, output.transform.translation, overlap_first_pose, prior ? &prior_rotation[0] : nullptr, &output.alignment_conditioning)) {
            return output;
        }
        output.alignment_degenerate = (output.alignment_conditioning < alignment::degenerate_conditioning);
        output.transform.valid = true;
        output.valid = true;

        output.aligned_x.resize(count);
        output.aligned_y.resize(count);
        output.aligned_z.resize(count);
        output.errors.resize(count);
        double sum = 0.0;
        double sum_squared = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            output.transform.apply(estimate[0][i], estimate[1][i], estimate[2][i], output.aligned_x[i], output.aligned_y[i], output.aligned_z[i]);
            const double dx = output.aligned_x[i] - truth[0][i];
            const double dy = output.aligned_y[i] - truth[1][i];
            const double dz = output.aligned_z[i] - truth[2][i];
            output.errors[i] = std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
            sum += output.errors[i];
            sum_squared += output.errors[i] * output.errors[i];
        }
        {
            std::vector<double> sorted(output.errors);
            std::sort(sorted.begin(), sorted.end());
            output.ate_minimum = sorted.front();
            output.ate_maximum = sorted.back();
            output.ate_median = sorted[sorted.size() / 2];
        }
        output.ate_mean = sum / static_cast<double>(count);
        output.ate_rmse = std::sqrt(sum_squared / static_cast<double>(count));
        output.ate_sum_squared = sum_squared;
        output.ate_standard_deviation = std::sqrt(std::max(0.0, (sum_squared / static_cast<double>(count)) - (output.ate_mean * output.ate_mean)));

        const auto displacement = [&](const std::size_t i, const std::size_t interval, double estimated_delta[3], double truth_delta[3]) {
            estimated_delta[0] = output.aligned_x[i + interval] - output.aligned_x[i];
            estimated_delta[1] = output.aligned_y[i + interval] - output.aligned_y[i];
            estimated_delta[2] = output.aligned_z[i + interval] - output.aligned_z[i];
            for (int axis = 0; axis < 3; ++axis) {
                truth_delta[axis] = truth[axis][i + interval] - truth[axis][i];
            }
        };
        const auto length = [](const double delta[3]) {
            return std::sqrt((delta[0] * delta[0]) + (delta[1] * delta[1]) + (delta[2] * delta[2]));
        };

        for (std::size_t i = 0; i + 1 < count; ++i) {
            double estimated_delta[3];
            double truth_delta[3];
            displacement(i, 1, estimated_delta, truth_delta);
            output.distance_estimated += length(estimated_delta);
            output.distance_ground_truth += length(truth_delta);
        }

        {
            std::vector<double> cumulative(count, 0.0);
            for (std::size_t i = 1; i < count; ++i) {
                double estimated_delta[3];
                double truth_delta[3];
                displacement(i - 1, 1, estimated_delta, truth_delta);
                cumulative[i] = cumulative[i - 1] + length(truth_delta);
            }
            const auto rotation_of = [&](const dataset::trajectory_pose& pose, double (&matrix)[9]) {
                const double quaternion[4] = { pose.quaternion_x, pose.quaternion_y, pose.quaternion_z, pose.quaternion_w };
                rotation::quaternion_to_matrix(&quaternion[0], &matrix[0]);
            };
            const auto relative_angle_degrees = [](const double (&a)[9], const double (&b)[9]) {
                double trace = 0.0;
                for (int r = 0; r < 3; ++r) {
                    for (int k = 0; k < 3; ++k) {
                        trace += a[(k * 3) + r] * b[(k * 3) + r];
                    }
                }
                const double cosine = std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0));
                return std::acos(cosine) * (180.0 / 3.14159265358979323846);
            };
            std::vector<double> translation_percent;
            std::vector<double> rotation_degrees;
            std::size_t j = 0;
            for (std::size_t i = 0; i < count; ++i) {
                while ((j < count) && (cumulative[j] - cumulative[i] < relative_distance_metres)) {
                    ++j;
                }
                if (j >= count) {
                    break;
                }
                const double segment = cumulative[j] - cumulative[i];
                double estimated_delta[3];
                double truth_delta[3];
                displacement(i, j - i, estimated_delta, truth_delta);
                const double difference[3] = { estimated_delta[0] - truth_delta[0], estimated_delta[1] - truth_delta[1], estimated_delta[2] - truth_delta[2] };
                translation_percent.push_back(100.0 * length(difference) / segment);
                double estimated_i[9];
                double estimated_j[9];
                double truth_i[9];
                double truth_j[9];
                rotation_of(estimated[output.pairs[i].second], estimated_i);
                rotation_of(estimated[output.pairs[j].second], estimated_j);
                rotation_of(ground_truth[output.pairs[i].first], truth_i);
                rotation_of(ground_truth[output.pairs[j].first], truth_j);
                double estimated_relative[9];
                double truth_relative[9];
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        double sum_estimated = 0.0;
                        double sum_truth = 0.0;
                        for (int k = 0; k < 3; ++k) {
                            sum_estimated += estimated_i[(k * 3) + r] * estimated_j[(k * 3) + c];
                            sum_truth += truth_i[(k * 3) + r] * truth_j[(k * 3) + c];
                        }
                        estimated_relative[(r * 3) + c] = sum_estimated;
                        truth_relative[(r * 3) + c] = sum_truth;
                    }
                }
                rotation_degrees.push_back(relative_angle_degrees(estimated_relative, truth_relative));
            }
            output.per_metre.segments = translation_percent.size();
            if (!translation_percent.empty()) {
                double squared_translation = 0.0;
                double squared_rotation = 0.0;
                for (std::size_t k = 0; k < translation_percent.size(); ++k) {
                    squared_translation += translation_percent[k] * translation_percent[k];
                    squared_rotation += rotation_degrees[k] * rotation_degrees[k];
                }
                output.per_metre.translation_percent_rmse = std::sqrt(squared_translation / static_cast<double>(translation_percent.size()));
                output.per_metre.rotation_degrees_rmse = std::sqrt(squared_rotation / static_cast<double>(rotation_degrees.size()));
                std::sort(translation_percent.begin(), translation_percent.end());
                std::sort(rotation_degrees.begin(), rotation_degrees.end());
                output.per_metre.translation_percent_median = translation_percent[translation_percent.size() / 2];
                output.per_metre.rotation_degrees_median = rotation_degrees[rotation_degrees.size() / 2];
            }
        }

        for (std::size_t index = 0; index < interval_count; ++index) {
            interval_result& relative = output.relative[index];
            relative.interval = intervals[index];
            if (count <= relative.interval) {
                continue;
            }
            relative.segments = count - relative.interval;
            double squared = 0.0;
            std::vector<double> ratios;
            ratios.reserve(relative.segments);
            for (std::size_t i = 0; i < relative.segments; ++i) {
                double estimated_delta[3];
                double truth_delta[3];
                displacement(i, relative.interval, estimated_delta, truth_delta);
                const double difference[3] = { estimated_delta[0] - truth_delta[0], estimated_delta[1] - truth_delta[1], estimated_delta[2] - truth_delta[2] };
                const double error = length(difference);
                relative.mean += error;
                squared += error * error;
                relative.maximum = std::max(relative.maximum, error);
                relative.latest = error;
                const double truth_length = length(truth_delta);
                if (truth_length < 1.0e-6) {
                    continue;
                }
                const double ratio = length(estimated_delta) / truth_length;
                ratios.push_back(ratio);
                if (std::abs(ratio - 1.0) > std::abs(relative.scale_worst - 1.0)) {
                    relative.scale_worst = ratio;
                }
            }
            relative.mean /= static_cast<double>(relative.segments);
            relative.rmse = std::sqrt(squared / static_cast<double>(relative.segments));
            relative.measurable_segments = ratios.size();
            if (!ratios.empty()) {
                std::sort(ratios.begin(), ratios.end());
                relative.scale_median = ratios[ratios.size() / 2];
            }
        }
        return output;
    }
}

#endif // ZEROSLAM_TOOLS_COMMON_METRICS_HPP
