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
#ifndef SLAM_HPP
#define SLAM_HPP

#include "consensus.hpp"
#include "frame.hpp"
#include "geometry.hpp"
#include "image.hpp"
#include "map.hpp"
#include "matrix.hpp"
#include "pose_estimation.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdlib>
#include <vector>
#include <unordered_set>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

class slam {
public:
    map::map reconstruction;

private:
    static void ratio_test(std::vector<feature::match>& matches) {
        for (std::vector<feature::match>::iterator i = matches.begin(); (i != matches.end()) && ((i + 1) != matches.end());) {
            if (i->lhs_index == (i + 1)->lhs_index) {
                if (i->score < (i + 1)->score) {
                    if (i->score / (i + 1)->score > 0.75f) {
                        i = matches.erase(i, i + 1);
                    }
                    else {
                        ++i;
                        i = matches.erase(i);
                    }
                }
                else {
                    if ((i + 1)->score / (i)->score > 0.75f) {
                        i = matches.erase(i, i + 1);
                    }
                    else {
                        i = matches.erase(i);
                        ++i;
                    }
                }
            }
            else {
                ++i;
            }
        }
    }

    static void symmetry_test(const std::vector<feature::match>& lhs, const std::vector<feature::match>& rhs, std::vector<feature::match>& matches) {
        matches.clear();
        matches.reserve(std::min(lhs.size(), rhs.size()));
        std::unordered_set<size_t> rhs_index_set;
        rhs_index_set.reserve(rhs.size());
        for (size_t i = 0; i < rhs.size(); ++i) {
            const size_t key = (rhs[i].lhs_index << 32) | rhs[i].rhs_index;
            rhs_index_set.insert(key);
        }
        for (const auto& lhs_match : lhs) {
            const size_t key = (lhs_match.rhs_index << 32) | lhs_match.lhs_index;
            if (rhs_index_set.find(key) != rhs_index_set.end()) {
                matches.push_back(feature::match{ lhs_match.lhs_index, lhs_match.rhs_index, lhs_match.score });
            }
        }
    }

public:
    void process_frame(const matrix::matrix<double, 3, 3>& intrinsics, const image::image& image_grey) {
        {
            camera::pinhole camera_intrinsics(matrix::matrix<double, 1, 4>({ intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2] }).data(), 4);
            frame::frame frame(camera_intrinsics, image_grey);

            std::printf("Detected Features: ");
            for (size_t i = 0; i < frame.keypoint_pyramid.size(); ++i) {
                std::printf("%zu ", frame.keypoint_pyramid[i].size());
            }
            std::printf("features (at each pyramid level).\n");

            // Add the frame to the reconstruction.
            this->reconstruction.add_frame(frame);

            // Nothing to do for the first frame.
            if (frame.id == 0) {
                return;
            }
        }

        // Get the most recent pair of frames.
        const frame::frame& frame_previous = reconstruction.frames.at(frame::frame::id_generator - 2);
        frame::frame& frame_current = reconstruction.frames.at(frame::frame::id_generator - 1);

        std::printf("Previous Features: ");
        for (size_t i = 0; i < frame_previous.keypoint_pyramid.size(); ++i) {
            std::printf("%zu ", frame_previous.keypoint_pyramid[i].size());
        }
        std::printf("features (at each pyramid level).\n");

        // Compute matches.
        std::vector<feature::match> matches_cp(math::max(frame_current.descriptor_pyramid[0].size(), frame_previous.descriptor_pyramid[0].size()) * 2);
        const size_t found_cp = feature::find_matches(
            frame_current.descriptor_pyramid[0].data(),
            frame_current.descriptor_pyramid[0].size(),
            frame_previous.descriptor_pyramid[0].data(),
            frame_previous.descriptor_pyramid[0].size(),
            32.0f,
            2,
            matches_cp.data(),
            matches_cp.size()
        );
        matches_cp.resize(found_cp);
        std::vector<feature::match> matches_pc(math::max(frame_current.descriptor_pyramid[0].size(), frame_previous.descriptor_pyramid[0].size()) * 2);
        const size_t found_pc = feature::find_matches(
            frame_previous.descriptor_pyramid[0].data(),
            frame_previous.descriptor_pyramid[0].size(),
            frame_current.descriptor_pyramid[0].data(),
            frame_current.descriptor_pyramid[0].size(),
            32.0f,
            2,
            matches_pc.data(),
            matches_pc.size()
        );
        matches_pc.resize(found_pc);
        // Filter matches.
        ratio_test(matches_cp);
        ratio_test(matches_pc);
        std::vector<feature::match> matches;
        symmetry_test(matches_cp, matches_pc, matches);
        // Create final arrays of good matches.
        std::vector<int> match_index_current;
        std::vector<int> match_index_previous;
        std::vector<matrix::matrix<double, 2, 1>> match_point_current;
        std::vector<matrix::matrix<double, 2, 1>> match_point_previous;
        for (const feature::match& m : matches) {
            match_index_current.push_back(static_cast<int>(m.lhs_index));
            match_point_current.push_back({ { static_cast<double>(frame_current.keypoint_pyramid[0][m.lhs_index].x), static_cast<double>(frame_current.keypoint_pyramid[0][m.lhs_index].y) } });
            match_index_previous.push_back(static_cast<int>(m.rhs_index));
            match_point_previous.push_back({ { static_cast<double>(frame_previous.keypoint_pyramid[0][m.rhs_index].x), static_cast<double>(frame_previous.keypoint_pyramid[0][m.rhs_index].y) } });
        }
        std::printf("Matched: %zu features to previous frame.\n", matches.size());
        std::printf("  (Starting Matches: %zu & %zu -> After Ratio: %zu & %zu -> After Symmetry: %zu)\n", found_cp, found_pc, matches_cp.size(), matches_pc.size(), matches.size());

        // Cache observations from the previous frame that are in this one.
        std::unordered_map<int, int> frame_previous_points; // key is kp_index and data is landmark_id.
        for (const auto& [landmark_id, landmark_observations] : this->reconstruction.observations) {
            for (const auto& [frame_id, kp_index] : landmark_observations) {
                if (frame_id == frame_previous.id) {
                    frame_previous_points[kp_index] = landmark_id;
                }
            }
        }

        // Pose estimation of new frame.
        if (frame_current.id < 2) {
            consensus::model_essential<double> model;
            std::vector<consensus::correspondence_2d_2d<double>> essential_correspondencies;
            for (const feature::match& m : matches) {
                const double lhs_point[2] = {
                    static_cast<double>(frame_current.keypoint_pyramid[0][m.lhs_index].x),
                    static_cast<double>(frame_current.keypoint_pyramid[0][m.lhs_index].y)
                };
                double lhs_ray[3];
                if (!frame_current.camera.unproject(&lhs_point[0], &lhs_ray[0])) {
                    continue;
                }
                const double rhs_point[2] = {
                    static_cast<double>(frame_previous.keypoint_pyramid[0][m.rhs_index].x),
                    static_cast<double>(frame_previous.keypoint_pyramid[0][m.rhs_index].y)
                };
                double rhs_ray[3];
                if (!frame_previous.camera.unproject(&rhs_point[0], &rhs_ray[0])) {
                    continue;
                }
                essential_correspondencies.push_back({ { lhs_ray[0] / lhs_ray[2], lhs_ray[1] / lhs_ray[2] }, { rhs_ray[0] / rhs_ray[2], rhs_ray[1] / rhs_ray[2] } });
            }
            std::vector<float> essential_residuals(essential_correspondencies.size());
            std::vector<size_t> essential_inliers(essential_correspondencies.size());
            size_t inliers = essential_correspondencies.size();
            if (!consensus::solve_ransac_essential(essential_correspondencies.data(), essential_correspondencies.size(), essential_residuals.data(), essential_inliers.data(), inliers, model)) {
                std::fprintf(stderr, "Failed to calculate the initial pose transform.\n");
                std::exit(1);
            }
            std::vector<matrix::matrix<double, 2, 1>> match_point_current_inlier;
            std::vector<matrix::matrix<double, 2, 1>> match_point_previous_inlier;
            match_point_current_inlier.reserve(inliers);
            match_point_previous_inlier.reserve(inliers);
            for (size_t inlier_index = 0; inlier_index < inliers; ++inlier_index) {
                match_point_current_inlier.push_back({ { essential_correspondencies[essential_inliers[inlier_index]].lhs.x, essential_correspondencies[essential_inliers[inlier_index]].lhs.y } });
                match_point_previous_inlier.push_back({ { essential_correspondencies[essential_inliers[inlier_index]].rhs.x, essential_correspondencies[essential_inliers[inlier_index]].rhs.y } });
            }

            matrix::matrix<double, 3, 3> rotation;
            matrix::matrix<double, 3, 1> translation;
            std::vector<matrix::matrix<double, 3, 1>> match_point_triangulated_inlier(inliers);
            pose_estimation::essential_matrix<double>::recover_pose(&model.essential[0][0], match_point_current_inlier.data()->data(), match_point_previous_inlier.data()->data(), inliers, rotation.data(), translation.data(), match_point_triangulated_inlier.data()->data());

            // Recover pose returns pose 2 to pose 1 rather than pose 1 to pose 2, so invert it.
            rotation = matrix::transpose(rotation);
            translation = -rotation * translation;

            // Set the initial pose of the new frame.
            frame_current.rotation = rotation * frame_previous.rotation;
            frame_current.translation = (rotation * frame_previous.translation) + translation;

            std::printf("Inliers: %zu inliers in pose estimation.\n", inliers);
            std::printf("Initial pose:\n");
            std::printf(
                "{\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f },\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f },\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f },\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f }\n}\n",
                frame_current.rotation[0][0],
                frame_current.rotation[0][1],
                frame_current.rotation[0][2],
                frame_current.translation[0],
                frame_current.rotation[1][0],
                frame_current.rotation[1][1],
                frame_current.rotation[1][2],
                frame_current.translation[1],
                frame_current.rotation[2][0],
                frame_current.rotation[2][1],
                frame_current.rotation[2][2],
                frame_current.translation[2],
                0.0,
                0.0,
                0.0,
                1.0
            );
        }
        else {
            std::vector<consensus::correspondence_2d_3d<double>> pnp_correspondencies;
            for (size_t i = 0; i < match_index_previous.size(); ++i) {
                auto found_point = frame_previous_points.find(match_index_previous[i]);
                if (found_point != frame_previous_points.end()) {
                    consensus::correspondence_2d_3d<double> corr;
                    const double lhs_point[2] = {
                        static_cast<double>(frame_current.keypoint_pyramid[0][match_index_current[i]].x),
                        static_cast<double>(frame_current.keypoint_pyramid[0][match_index_current[i]].y)
                    };
                    double lhs_ray[3];
                    if (!frame_current.camera.unproject(&lhs_point[0], &lhs_ray[0])) {
                        continue;
                    }
                    corr.lhs.x = lhs_ray[0] / lhs_ray[2];
                    corr.lhs.y = lhs_ray[1] / lhs_ray[2];

                    const auto& landmark = this->reconstruction.landmarks.at(found_point->second);
                    corr.rhs.x = landmark.location[0];
                    corr.rhs.y = landmark.location[1];
                    corr.rhs.z = landmark.location[2];

                    pnp_correspondencies.push_back(corr);
                }
            }

            bool pnp_success = false;
            if (pnp_correspondencies.size() >= 3) {
                consensus::model_p3p<double> model;
                std::vector<float> pnp_residuals(pnp_correspondencies.size());
                std::vector<size_t> pnp_inliers(pnp_correspondencies.size());
                size_t inliers_size = 0;

                if (consensus::solve_ransac_p3p(pnp_correspondencies.data(), pnp_correspondencies.size(), pnp_residuals.data(), pnp_inliers.data(), inliers_size, model)) {
                    frame_current.rotation = matrix::matrix<double, 3, 3>({
                        { model.rotation[0][0], model.rotation[0][1], model.rotation[0][2] },
                        { model.rotation[1][0], model.rotation[1][1], model.rotation[1][2] },
                        { model.rotation[2][0], model.rotation[2][1], model.rotation[2][2] }
                    });
                    frame_current.translation = matrix::matrix<double, 3, 1>({
                        { model.translation[0] },
                        { model.translation[1] },
                        { model.translation[2] }
                    });
                    pnp_success = true;
                    std::printf("Inliers: %zu inliers in PnP pose estimation out of %zu correspondencies.\n", inliers_size, pnp_correspondencies.size());
                }
            }

            if (!pnp_success) {
                // Set the initial pose of the new frame from the previous frame.
                frame_current.rotation = frame_previous.rotation;
                frame_current.translation = frame_previous.translation;
            }
        }
        // Find matches that are already in the map as landmarks.
        int observations_of_landmarks = 0;
        for (size_t i = 0; i < match_index_previous.size(); ++i) {
            auto found_point = frame_previous_points.find(match_index_previous[i]);
            if (found_point != frame_previous_points.end()) {
                // Add observations to features that match existing landmarks.
                this->reconstruction.add_observation(frame_current, this->reconstruction.landmarks[found_point->second], match_index_current[i]);
                ++observations_of_landmarks;
            }
        }
        std::printf("Matched: %d features to previous frame landmarks.\n", observations_of_landmarks);
        // Optimise the pose of the new frame using the points that match the current map.
        this->reconstruction.optimise(1, true, 50);
        this->reconstruction.cull();
        if (observations_of_landmarks > 0) {
            std::printf("Initial Optimised pose:\n");
            std::printf(
                "{\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f },\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f },\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f },\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f }\n}\n",
                frame_current.rotation[0][0],
                frame_current.rotation[0][1],
                frame_current.rotation[0][2],
                frame_current.translation[0],
                frame_current.rotation[1][0],
                frame_current.rotation[1][1],
                frame_current.rotation[1][2],
                frame_current.translation[1],
                frame_current.rotation[2][0],
                frame_current.rotation[2][1],
                frame_current.rotation[2][2],
                frame_current.translation[2],
                0.0,
                0.0,
                0.0,
                1.0
            );
        }
        // Search for matches to other frames by projection.
        int observations_of_map = 0;
        for (const auto& [landmark_id, landmark] : this->reconstruction.landmarks) {
            // Check landmark is infront of frame.
            const matrix::matrix<double, 3, 1> mapped = (frame_current.rotation * landmark.location) + frame_current.translation;
            matrix::matrix<double, 2, 1> reprojected = {};
            if (!frame_current.camera.project(mapped.data(), reprojected.data())) {
                continue;
            }
            if ((reprojected[0] < 10) || (reprojected[0] > image_grey.get_cols() - 11) || (reprojected[1] < 10) || (reprojected[1] > image_grey.get_rows() - 11)) {
                continue;
            }
            // Check it has not already been matched.
            const std::vector<map::map::observation<int, size_t>>& landmark_observations = this->reconstruction.observations.at(landmark_id);
            bool landmark_found = false;
            for (const map::map::observation<int, size_t>& landmark_observation : landmark_observations) {
                if (landmark_observation.first == frame_current.id) {
                    landmark_found = true;
                    break;
                }
            }
            if (landmark_found) {
                continue;
            }
            // Find all detected features that are near the landmark reprojection.
            for (size_t i = 0; i < match_point_current.size(); ++i) {
                // Is it near enough?
                if ((match_point_current[i] - reprojected).get_length_squared() > 4) {
                    continue;
                }
                // Has it already been matched?
                auto found_point = frame_previous_points.find(match_index_previous[i]);
                if (found_point != frame_previous_points.end()) {
                    continue;
                }
                // Check similarity.
                const feature::descriptor& des_landmark = this->reconstruction.frames.at(landmark_observations[0].first).descriptor_pyramid[0][static_cast<size_t>(landmark_observations[0].second)];
                const feature::descriptor& des_current = frame_current.descriptor_pyramid[0][static_cast<size_t>(match_index_current[i])];
                if (feature::distance<256>(des_landmark.data, des_current.data) < 64) {
                    this->reconstruction.add_observation(frame_current, landmark, match_index_current[i]);
                    frame_previous_points[match_index_previous[i]] = -1;
                    ++observations_of_map;
                }
            }
        }
        std::printf("Matched: %d features to map landmarks.\n", observations_of_map);
        // Optimise the pose of the new frame using reprojected points from the current map.
        this->reconstruction.optimise(1, true, 50);
        this->reconstruction.cull();
        // Add all the remaing matched features.
        int potential_landmarks = 0;
        int invalid_landmarks = 0;
        int behind_landmarks = 0;
        int unprojectable_landmarks = 0;
        int poor_landmarks = 0;
        int new_landmarks = 0;
        for (size_t i = 0; i < match_index_previous.size(); ++i) {
            auto found_point = frame_previous_points.find(match_index_previous[i]);
            if (found_point != frame_previous_points.end()) {
                // Previous landmark.
                continue;
            }
            // New landmark to be triangulated.
            ++potential_landmarks;
            // Normalise the points.
            const matrix::matrix<double, 2, 1> point_previous = { { static_cast<double>(frame_previous.keypoint_pyramid[0][static_cast<size_t>(match_index_previous[i])].x), static_cast<double>(frame_previous.keypoint_pyramid[0][static_cast<size_t>(match_index_previous[i])].y) } };
            matrix::matrix<double, 3, 1> ray_previous;
            if (!frame_previous.camera.unproject(point_previous.data(), ray_previous.data())) {
                ++unprojectable_landmarks;
                continue;
            }
            const matrix::matrix<double, 2, 1> point_current = { { static_cast<double>(frame_current.keypoint_pyramid[0][static_cast<size_t>(match_index_current[i])].x), static_cast<double>(frame_current.keypoint_pyramid[0][static_cast<size_t>(match_index_current[i])].y) } };
            matrix::matrix<double, 3, 1> ray_current;
            if (!frame_current.camera.unproject(point_current.data(), ray_current.data())) {
                ++unprojectable_landmarks;
                continue;
            }
            // Check triangulation is valid.
            matrix::matrix<double, 3, 1> point;
            if (!geometry::triangulate(
                    ray_current,
                    frame_current.get_pose(),
                    ray_previous,
                    frame_previous.get_pose(),
                    point
                )) {
                ++invalid_landmarks;
                continue;
            }
            // Check triangulated point reprojects infront of both cameras.
            const matrix::matrix<double, 3, 1> mapped_previous = (frame_previous.rotation * point) + frame_previous.translation;
            if (mapped_previous[2] < 0) {
                ++behind_landmarks;
                continue;
            }
            const matrix::matrix<double, 3, 1> mapped_current = (frame_current.rotation * point) + frame_current.translation;
            if (mapped_current[2] < 0) {
                ++behind_landmarks;
                continue;
            }
            // Check reprojection error of point is low compared to the detected feature in both cameras.
            matrix::matrix<double, 2, 1> reprojected_previous;
            if (!frame_previous.camera.project(mapped_previous.data(), reprojected_previous.data())) {
                ++unprojectable_landmarks;
                continue;
            }
            if ((match_point_previous[i] - reprojected_previous).get_length_squared() > (2 * 2)) {
                ++poor_landmarks;
                continue;
            }
            matrix::matrix<double, 2, 1> reprojected_current;
            if (!frame_current.camera.project(mapped_current.data(), reprojected_current.data())) {
                ++unprojectable_landmarks;
                continue;
            }
            if ((match_point_current[i] - reprojected_current).get_length_squared() > (2 * 2)) {
                ++poor_landmarks;
                continue;
            }
            // Add it.
            const float colour = static_cast<float>(image_grey.get_data()[static_cast<size_t>(frame_current.keypoint_pyramid[0][static_cast<size_t>(match_index_current[i])].y) * image_grey.get_cols() + static_cast<size_t>(frame_current.keypoint_pyramid[0][static_cast<size_t>(match_index_current[i])].x)]) / 255.0f;
            landmark::point landmark(point, matrix::matrix<double, 3, 1>{ { colour, colour, colour } });
            this->reconstruction.add_landmark(landmark);
            this->reconstruction.add_observation(frame_previous, landmark, match_index_previous[i]);
            this->reconstruction.add_observation(frame_current, landmark, match_index_current[i]);
            ++new_landmarks;
        }
        std::printf("Created: %d/%d new landmarks [invalid: %d, behind: %d, unprojectable: %d, poor: %d]\n", new_landmarks, potential_landmarks, invalid_landmarks, behind_landmarks, unprojectable_landmarks, poor_landmarks);
        // Optimise the pose of the new frame again.
        this->reconstruction.optimise(1, true, 50);
        this->reconstruction.cull();
        // Optimise the whole map.
        this->reconstruction.optimise(10, false, 50);
        this->reconstruction.cull();
        // Print the map status and pose.
        std::printf("Map status: %zu frames, %zu landmarks.\n", this->reconstruction.frames.size(), this->reconstruction.landmarks.size());
        std::printf("Current pose:\n");
        std::printf(
            "{\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f },\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f },\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f },\n  { % 10.8f, % 10.8f, % 10.8f, % 10.8f }\n}\n",
            frame_current.rotation[0][0],
            frame_current.rotation[0][1],
            frame_current.rotation[0][2],
            frame_current.translation[0],
            frame_current.rotation[1][0],
            frame_current.rotation[1][1],
            frame_current.rotation[1][2],
            frame_current.translation[1],
            frame_current.rotation[2][0],
            frame_current.rotation[2][1],
            frame_current.rotation[2][2],
            frame_current.translation[2],
            0.0,
            0.0,
            0.0,
            1.0
        );
    }
};

#endif // SLAM_HPP
