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

#include "mapping/loop_closure.hpp"

#include "core/logger.hpp"
#include "estimation/correspondence_3d_3d.hpp"
#include "estimation/robust/solver/similarity.hpp"
#include "match/distance/hamming.hpp"
#include "match/matcher/bruteforce.hpp"
#include "match/matcher/gms.hpp"
#include "match/pair.hpp"
#include "math/math.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/similarity_reprojection.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/loss.hpp"
#include "optimisation/losses/huber.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/similarity.hpp"

namespace mapping {
    loop_closure::loop_closure()
        : recognition()
        , keyframes() {
    }

    loop_closure::result loop_closure::detect(const int keyframe_id, const math::se3<double>& pose, const sensor::model& camera, const covisibility& graph, const record* const keyframe_records, const size_t keyframe_records_size, const int submap_start_id) const {
        result outcome;
        outcome.found = false;
        outcome.keyframe_id = -1;
        outcome.correspondences = 0;
        outcome.inliers = 0;
        outcome.correction = math::sim3<double>::identity();
        outcome.relative = math::sim3<double>::identity();
        if (keyframe_records_size == 0) {
            return outcome;
        }

        std::vector<feature::descriptor::binary<256>> query(keyframe_records_size);
        for (size_t i = 0; i < keyframe_records_size; ++i) {
            query[i] = keyframe_records[i].descriptor;
        }
        const std::vector<place_recognition::candidate> candidates = this->recognition.get_candidates(query.data(), query.size(), keyframe_id, loop_closure::max_candidates);
        const keyframe* candidate_keyframe = nullptr;
        int candidate_id = -1;
        for (const place_recognition::candidate& candidate : candidates) {
            if ((candidate.keyframe_id >= submap_start_id) && (candidate.keyframe_id > keyframe_id - loop_closure::min_keyframe_gap)) {
                continue;
            }
            const std::unordered_map<int, keyframe>::const_iterator found = this->keyframes.find(candidate.keyframe_id);
            if (found == this->keyframes.end()) {
                continue;
            }
            const bool covisible_candidate = (graph.weight(keyframe_id, candidate.keyframe_id) >= loop_closure::max_covisible_landmarks);
            const std::vector<record>& candidate_records = found->second.records;
            std::unordered_map<int, size_t> recorded_by_landmark;
            recorded_by_landmark.reserve(candidate_records.size());
            for (size_t i = 0; i < candidate_records.size(); ++i) {
                recorded_by_landmark[candidate_records[i].landmark_id] = i;
            }
            std::vector<estimation::correspondence_3d_3d<double>> correspondences;
            std::vector<correspondence> pairs;
            std::vector<std::pair<size_t, size_t>> pair_records;
            correspondences.reserve(keyframe_records_size);
            pairs.reserve(keyframe_records_size);
            pair_records.reserve(keyframe_records_size);
            for (size_t i = 0; i < keyframe_records_size; ++i) {
                const std::unordered_map<int, size_t>::const_iterator recorded = recorded_by_landmark.find(keyframe_records[i].landmark_id);
                if (recorded == recorded_by_landmark.end()) {
                    continue;
                }
                correspondences.push_back({ keyframe_records[i].location, candidate_records[recorded->second].location });
                pairs.push_back({ keyframe_records[i].landmark_id, candidate_records[recorded->second].landmark_id });
                pair_records.push_back({ i, recorded->second });
            }
            const size_t shared_by_id = correspondences.size();
            if (covisible_candidate || (shared_by_id >= loop_closure::max_covisible_landmarks)) {
                if (this->covisible_revisit_loop(keyframe_id, pose, camera, keyframe_records, candidate.keyframe_id, found->second, correspondences, pairs, pair_records, outcome)) {
                    return outcome;
                }
                core::logger::log(core::logger::level::debug, "Loop candidate keyframe %d -> %d not a material covisible revisit (%zu landmarks shared by id).", keyframe_id, candidate.keyframe_id, shared_by_id);
                continue;
            }
            candidate_id = candidate.keyframe_id;
            candidate_keyframe = &found->second;
            break;
        }
        if (candidate_keyframe == nullptr) {
            return outcome;
        }
        const std::vector<record>& candidate_records = candidate_keyframe->records;

        std::unordered_map<int, size_t> recorded_by_landmark;
        recorded_by_landmark.reserve(candidate_records.size());
        for (size_t i = 0; i < candidate_records.size(); ++i) {
            recorded_by_landmark[candidate_records[i].landmark_id] = i;
        }
        std::vector<estimation::correspondence_3d_3d<double>> correspondences;
        std::vector<correspondence> pairs;
        std::vector<std::pair<size_t, size_t>> pair_records;
        std::vector<unsigned char> current_paired(keyframe_records_size, 0);
        std::vector<unsigned char> recorded_paired(candidate_records.size(), 0);
        correspondences.reserve(keyframe_records_size);
        pairs.reserve(keyframe_records_size);
        pair_records.reserve(keyframe_records_size);
        for (size_t i = 0; i < keyframe_records_size; ++i) {
            const std::unordered_map<int, size_t>::const_iterator recorded = recorded_by_landmark.find(keyframe_records[i].landmark_id);
            if (recorded == recorded_by_landmark.end()) {
                continue;
            }
            correspondences.push_back({ keyframe_records[i].location, candidate_records[recorded->second].location });
            pairs.push_back({ keyframe_records[i].landmark_id, candidate_records[recorded->second].landmark_id });
            pair_records.push_back({ i, recorded->second });
            current_paired[i] = 1;
            recorded_paired[recorded->second] = 1;
        }
        const size_t shared_by_id = correspondences.size();

        std::vector<feature::descriptor::binary<256>> recorded_descriptors(candidate_records.size());
        for (size_t i = 0; i < candidate_records.size(); ++i) {
            recorded_descriptors[i] = candidate_records[i].descriptor;
        }
        std::vector<match::pair> forward(2 * keyframe_records_size);
        const size_t forward_count = match::matcher::bruteforce::find_matches(query.data(), query.size(), recorded_descriptors.data(), recorded_descriptors.size(), loop_closure::match_hamming_maximum * this->hamming_scale, 2, forward.data(), forward.size());
        std::vector<match::pair> backward(candidate_records.size());
        const size_t backward_count = match::matcher::bruteforce::find_matches(recorded_descriptors.data(), recorded_descriptors.size(), query.data(), query.size(), loop_closure::match_hamming_maximum * this->hamming_scale, 1, backward.data(), backward.size());
        std::vector<size_t> best_current_of_recorded(candidate_records.size(), static_cast<size_t>(-1));
        for (size_t i = 0; i < backward_count; ++i) {
            best_current_of_recorded[backward[i].lhs_index] = backward[i].rhs_index;
        }
        std::vector<match::pair> descriptor_pairs;
        for (size_t i = 0; i < forward_count; ++i) {
            const match::pair& best = forward[i];
            const bool has_second = (i + 1 < forward_count) && (forward[i + 1].lhs_index == best.lhs_index);
            if (has_second) {
                ++i;
                if (!(best.score < loop_closure::match_ratio * forward[i].score)) {
                    continue;
                }
            }
            if ((current_paired[best.lhs_index] != 0) || (recorded_paired[best.rhs_index] != 0) || (best_current_of_recorded[best.rhs_index] != best.lhs_index)) {
                continue;
            }
            descriptor_pairs.push_back(best);
        }
        if (descriptor_pairs.size() >= loop_closure::gms_minimum_pairs) {
            std::vector<feature::point> current_points(keyframe_records_size);
            std::vector<feature::point> recorded_points(candidate_records.size());
            float current_extent[2] = { 1.0f, 1.0f };
            float recorded_extent[2] = { 1.0f, 1.0f };
            for (size_t i = 0; i < keyframe_records_size; ++i) {
                current_points[i] = feature::point{ keyframe_records[i].pixel_x, keyframe_records[i].pixel_y, 0.0f, 0.0f, 0 };
                current_extent[0] = math::max(current_extent[0], keyframe_records[i].pixel_x + 1.0f);
                current_extent[1] = math::max(current_extent[1], keyframe_records[i].pixel_y + 1.0f);
            }
            for (size_t i = 0; i < candidate_records.size(); ++i) {
                recorded_points[i] = feature::point{ candidate_records[i].pixel_x, candidate_records[i].pixel_y, 0.0f, 0.0f, 0 };
                recorded_extent[0] = math::max(recorded_extent[0], candidate_records[i].pixel_x + 1.0f);
                recorded_extent[1] = math::max(recorded_extent[1], candidate_records[i].pixel_y + 1.0f);
            }
            const size_t before = descriptor_pairs.size();
            descriptor_pairs.resize(match::matcher::gms::filter(current_points.data(), current_extent[0], current_extent[1], recorded_points.data(), recorded_extent[0], recorded_extent[1], descriptor_pairs.data(), descriptor_pairs.size()));
            core::logger::log(core::logger::level::debug, "Loop candidate keyframe %d -> %d: %zu of %zu descriptor pairs pass the motion statistics.", keyframe_id, candidate_id, descriptor_pairs.size(), before);
        }
        for (const match::pair& best : descriptor_pairs) {
            correspondences.push_back({ keyframe_records[best.lhs_index].location, candidate_records[best.rhs_index].location });
            pairs.push_back({ keyframe_records[best.lhs_index].landmark_id, candidate_records[best.rhs_index].landmark_id });
            pair_records.push_back({ best.lhs_index, best.rhs_index });
            current_paired[best.lhs_index] = 1;
            recorded_paired[best.rhs_index] = 1;
        }
        outcome.correspondences = correspondences.size();
        if ((shared_by_id > 0) && (static_cast<double>(shared_by_id) >= loop_closure::max_covisible_fraction * static_cast<double>(correspondences.size()))) {
            core::logger::log(core::logger::level::debug, "Loop candidate keyframe %d -> %d skipped, %zu of %zu correspondences shared by id.", keyframe_id, candidate_id, shared_by_id, correspondences.size());
            return outcome;
        }
        if (correspondences.size() < 3) {
            core::logger::log(core::logger::level::debug, "Loop candidate keyframe %d -> %d rejected, %zu shared landmarks (%zu by id).", keyframe_id, candidate_id, correspondences.size(), shared_by_id);
            return outcome;
        }

        math::matrix<double, 3, 1> centroid = math::matrix<double, 3, 1>::zero();
        for (const estimation::correspondence_3d_3d<double>& matched : correspondences) {
            centroid = centroid + matched.lhs;
        }
        centroid = centroid * (1.0 / static_cast<double>(correspondences.size()));
        double radius_squared_sum = 0.0;
        for (const estimation::correspondence_3d_3d<double>& matched : correspondences) {
            radius_squared_sum += (matched.lhs - centroid).get_length_squared();
        }
        const double rms_radius = math::sqrt(radius_squared_sum / static_cast<double>(correspondences.size()));
        const float inlier_threshold = static_cast<float>(math::max(loop_closure::inlier_radius_fraction * rms_radius, 1.0e-6));

        std::vector<float> residuals(correspondences.size());
        std::vector<size_t> inliers(correspondences.size());
        size_t inliers_size = 0;
        estimation::robust::solver::similarity<double>::model_type model{};
        const bool solved = estimation::robust::solver::similarity<double>::solve(correspondences.data(), correspondences.size(), inlier_threshold, residuals.data(), inliers.data(), inliers_size, model);
        if (!solved) {
            core::logger::log(core::logger::level::debug, "Loop candidate keyframe %d -> %d rejected, no similarity fits its %zu shared landmarks (%zu by id).", keyframe_id, candidate_id, correspondences.size(), shared_by_id);
            return outcome;
        }

        const math::matrix<double, 3, 3> rotation = { { { model.rotation[0][0], model.rotation[0][1], model.rotation[0][2] },
                                                        { model.rotation[1][0], model.rotation[1][1], model.rotation[1][2] },
                                                        { model.rotation[2][0], model.rotation[2][1], model.rotation[2][2] } } };
        const math::matrix<double, 3, 1> translation = { { model.translation[0], model.translation[1], model.translation[2] } };
        const bool foreign_submap = candidate_id < submap_start_id;
        if (!foreign_submap && ((model.scale > loop_closure::max_scale_ratio) || (model.scale < 1.0 / loop_closure::max_scale_ratio))) {
            core::logger::log(core::logger::level::note, "Loop candidate keyframe %d -> %d rejected, scale %.4f is not plausible.", keyframe_id, candidate_id, model.scale);
            return outcome;
        }
        math::sim3<double> correction(math::se3<double>(math::so3<double>(rotation), translation), model.scale);

        const size_t paired = correspondences.size();
        size_t guided = 0;
        {
            const auto project_into = [](const sensor::model& into_camera, const math::se3<double>& into_pose, const math::matrix<double, 3, 1>& world, double (&pixel)[2]) -> bool {
                const math::matrix<double, 3, 1> in_camera = into_pose * world;
                return into_camera.project(in_camera.data(), &pixel[0]);
            };
            const unsigned int guided_bound = static_cast<unsigned int>(loop_closure::guided_hamming_maximum * this->hamming_scale);
            const double radius_squared = loop_closure::guided_search_radius * loop_closure::guided_search_radius;
            const auto nearest = [&](const feature::descriptor::binary<256>& descriptor, const record* const searched, const size_t searched_size, const std::vector<unsigned char>& claimed, const double (&predicted)[2]) -> size_t {
                size_t best = static_cast<size_t>(-1);
                unsigned int best_distance = guided_bound;
                for (size_t i = 0; i < searched_size; ++i) {
                    if (claimed[i] != 0) {
                        continue;
                    }
                    const double offset_x = static_cast<double>(searched[i].pixel_x) - predicted[0];
                    const double offset_y = static_cast<double>(searched[i].pixel_y) - predicted[1];
                    if (((offset_x * offset_x) + (offset_y * offset_y)) > radius_squared) {
                        continue;
                    }
                    const unsigned int distance = match::distance::hamming::distance(descriptor, searched[i].descriptor);
                    if (distance < best_distance) {
                        best_distance = distance;
                        best = i;
                    }
                }
                return best;
            };
            const math::sim3<double> into_recorded = correction;
            const math::sim3<double> into_current = correction.inverse();
            std::vector<size_t> current_to_recorded(keyframe_records_size, static_cast<size_t>(-1));
            std::vector<size_t> recorded_to_current(candidate_records.size(), static_cast<size_t>(-1));
            for (size_t i = 0; i < keyframe_records_size; ++i) {
                double predicted[2] = {};
                if ((current_paired[i] != 0) || !project_into(candidate_keyframe->camera, candidate_keyframe->pose, into_recorded * keyframe_records[i].location, predicted)) {
                    continue;
                }
                current_to_recorded[i] = nearest(keyframe_records[i].descriptor, candidate_records.data(), candidate_records.size(), recorded_paired, predicted);
            }
            for (size_t j = 0; j < candidate_records.size(); ++j) {
                double predicted[2] = {};
                if ((recorded_paired[j] != 0) || !project_into(camera, pose, into_current * candidate_records[j].location, predicted)) {
                    continue;
                }
                recorded_to_current[j] = nearest(candidate_records[j].descriptor, keyframe_records, keyframe_records_size, current_paired, predicted);
            }
            for (size_t i = 0; i < keyframe_records_size; ++i) {
                const size_t j = current_to_recorded[i];
                if ((j == static_cast<size_t>(-1)) || (recorded_to_current[j] != i)) {
                    continue;
                }
                correspondences.push_back({ keyframe_records[i].location, candidate_records[j].location });
                pairs.push_back({ keyframe_records[i].landmark_id, candidate_records[j].landmark_id });
                pair_records.push_back({ i, j });
                current_paired[i] = 1;
                recorded_paired[j] = 1;
                ++guided;
            }
            outcome.correspondences = correspondences.size();
        }

        {
            optimisation::factor_graph refinement;
            double parameters[8] = { correction.transformation().translation()[0], correction.transformation().translation()[1], correction.transformation().translation()[2], correction.transformation().rotation().get_quaternion()[1], correction.transformation().rotation().get_quaternion()[2], correction.transformation().rotation().get_quaternion()[3], correction.transformation().rotation().get_quaternion()[0], correction.scale() };
            optimisation::vertex similarity_vertex{ optimisation::vertices::similarity() };
            similarity_vertex.set_parameters(&parameters[0], 8);
            optimisation::vertex* const vertex = refinement.add_vertex(static_cast<optimisation::vertex&&>(similarity_vertex));
            const optimisation::loss lossfunction(optimisation::losses::huber(math::sqrt(loop_closure::reprojection_inlier_bound_squared)));
            const auto add_edge = [&](const math::matrix<double, 3, 1>& location, const sensor::model& observer_camera, const math::se3<double>& observer_pose, const bool inverted, const float pixel_x, const float pixel_y) -> optimisation::edge* {
                optimisation::edge factor{ optimisation::edges::similarity_reprojection(sensor::camera::model<double>(observer_camera), location, observer_pose, inverted) };
                factor.add_vertex(vertex);
                const double observed[2] = { static_cast<double>(pixel_x), static_cast<double>(pixel_y) };
                factor.set_observation(math::matrix<double, 0, 0>(2, 1, &observed[0]));
                factor.set_loss(lossfunction);
                return refinement.add_edge(static_cast<optimisation::edge&&>(factor));
            };
            std::vector<std::pair<optimisation::edge*, optimisation::edge*>> factors;
            factors.reserve(correspondences.size());
            for (size_t i = 0; i < correspondences.size(); ++i) {
                const record& current_record = keyframe_records[pair_records[i].first];
                const record& recorded_record = candidate_records[pair_records[i].second];
                factors.push_back({ add_edge(correspondences[i].lhs, candidate_keyframe->camera, candidate_keyframe->pose, false, recorded_record.pixel_x, recorded_record.pixel_y), add_edge(correspondences[i].rhs, camera, pose, true, current_record.pixel_x, current_record.pixel_y) });
            }
            static_cast<void>(refinement.solve(loop_closure::refine_rounds, true));
            static_cast<void>(refinement.get_current_chi(true));
            size_t dropped = 0;
            for (const std::pair<optimisation::edge*, optimisation::edge*>& factor : factors) {
                if ((factor.first == nullptr) || (factor.second == nullptr)) {
                    continue;
                }
                if ((factor.first->chi2() > loop_closure::reprojection_inlier_bound_squared) || (factor.second->chi2() > loop_closure::reprojection_inlier_bound_squared)) {
                    static_cast<void>(refinement.remove_edge(factor.first));
                    static_cast<void>(refinement.remove_edge(factor.second));
                    ++dropped;
                }
            }
            if ((dropped > 0) && (dropped < factors.size())) {
                static_cast<void>(refinement.solve(loop_closure::refine_rounds, true));
            }
            const double* const refined = vertex->get_parameters();
            correction = math::sim3<double>(math::se3<double>(math::so3<double>(refined[6], refined[3], refined[4], refined[5]), { { refined[0], refined[1], refined[2] } }), refined[7]);
            if (!foreign_submap && ((correction.scale() > loop_closure::max_scale_ratio) || (correction.scale() < 1.0 / loop_closure::max_scale_ratio))) {
                core::logger::log(core::logger::level::note, "Loop candidate keyframe %d -> %d rejected, the refined scale %.4f is not plausible.", keyframe_id, candidate_id, correction.scale());
                return outcome;
            }
        }

        const math::sim3<double> correction_inverse = correction.inverse();
        size_t paired_inliers = 0;
        outcome.matches.reserve(correspondences.size());
        for (size_t i = 0; i < correspondences.size(); ++i) {
            const record& current_record = keyframe_records[pair_records[i].first];
            const record& recorded_record = candidate_records[pair_records[i].second];
            const math::matrix<double, 3, 1> current_in_loop = candidate_keyframe->pose * (correction * correspondences[i].lhs);
            const math::matrix<double, 3, 1> recorded_in_current = pose * (correction_inverse * correspondences[i].rhs);
            double projected_in_loop[2] = {};
            double projected_in_current[2] = {};
            if (!candidate_keyframe->camera.project(current_in_loop.data(), &projected_in_loop[0]) || !camera.project(recorded_in_current.data(), &projected_in_current[0])) {
                continue;
            }
            const double loop_error_x = projected_in_loop[0] - static_cast<double>(recorded_record.pixel_x);
            const double loop_error_y = projected_in_loop[1] - static_cast<double>(recorded_record.pixel_y);
            const double current_error_x = projected_in_current[0] - static_cast<double>(current_record.pixel_x);
            const double current_error_y = projected_in_current[1] - static_cast<double>(current_record.pixel_y);
            if (((loop_error_x * loop_error_x) + (loop_error_y * loop_error_y) > loop_closure::reprojection_inlier_bound_squared) || ((current_error_x * current_error_x) + (current_error_y * current_error_y) > loop_closure::reprojection_inlier_bound_squared)) {
                continue;
            }
            outcome.matches.push_back(pairs[i]);
            paired_inliers += (i < paired) ? 1 : 0;
        }
        outcome.inliers = outcome.matches.size();
        if ((outcome.inliers < loop_closure::min_inliers) || (static_cast<double>(paired_inliers) < loop_closure::min_inlier_fraction * static_cast<double>(paired))) {
            core::logger::log(core::logger::level::debug, "Loop candidate keyframe %d -> %d rejected, %zu of %zu shared landmarks (%zu by id, %zu found by projection) reproject through the refined similarity (%zu fitted it in 3D).", keyframe_id, candidate_id, outcome.inliers, correspondences.size(), shared_by_id, guided, inliers_size);
            outcome.matches.clear();
            return outcome;
        }

        outcome.found = true;
        outcome.keyframe_id = candidate_id;
        outcome.correction = correction;
        outcome.relative = math::sim3<double>(candidate_keyframe->pose, 1.0) * outcome.correction * math::sim3<double>(pose.inverse(), 1.0);
        core::logger::log(core::logger::level::note, "Loop detected keyframe %d -> %d, %zu of %zu shared landmarks (%zu by id, %zu found by projection) reproject through the refined similarity, scale %.4f, translation %.4f.", keyframe_id, candidate_id, outcome.inliers, correspondences.size(), shared_by_id, guided, correction.scale(), math::sqrt(correction.transformation().translation().get_length_squared()));
        return outcome;
    }

    bool loop_closure::covisible_revisit_loop(const int keyframe_id, const math::se3<double>& pose, const sensor::model& camera, const record* const keyframe_records, const int candidate_id, const keyframe& candidate, const std::vector<estimation::correspondence_3d_3d<double>>& correspondences, const std::vector<correspondence>& pairs, const std::vector<std::pair<size_t, size_t>>& pair_records, result& outcome) const {
        math::matrix<double, 3, 1> centroid = math::matrix<double, 3, 1>::zero();
        for (const estimation::correspondence_3d_3d<double>& matched : correspondences) {
            centroid = centroid + matched.lhs;
        }
        centroid = centroid * (1.0 / static_cast<double>(correspondences.size()));
        double radius_squared_sum = 0.0;
        for (const estimation::correspondence_3d_3d<double>& matched : correspondences) {
            radius_squared_sum += (matched.lhs - centroid).get_length_squared();
        }
        const double rms_radius = math::sqrt(radius_squared_sum / static_cast<double>(correspondences.size()));
        const float inlier_threshold = static_cast<float>(math::max(loop_closure::inlier_radius_fraction * rms_radius, 1.0e-6));

        std::vector<float> residuals(correspondences.size());
        std::vector<size_t> inliers(correspondences.size());
        size_t inliers_size = 0;
        estimation::robust::solver::similarity<double>::model_type model{};
        if (!estimation::robust::solver::similarity<double>::solve(correspondences.data(), correspondences.size(), inlier_threshold, residuals.data(), inliers.data(), inliers_size, model)) {
            core::logger::log(core::logger::level::debug, "Loop candidate keyframe %d -> %d skipped, no similarity fits its %zu covisible landmarks.", keyframe_id, candidate_id, correspondences.size());
            return false;
        }
        const math::matrix<double, 3, 3> rotation = { { { model.rotation[0][0], model.rotation[0][1], model.rotation[0][2] },
                                                        { model.rotation[1][0], model.rotation[1][1], model.rotation[1][2] },
                                                        { model.rotation[2][0], model.rotation[2][1], model.rotation[2][2] } } };
        const math::matrix<double, 3, 1> translation = { { model.translation[0], model.translation[1], model.translation[2] } };
        if ((model.scale > loop_closure::max_scale_ratio) || (model.scale < 1.0 / loop_closure::max_scale_ratio)) {
            core::logger::log(core::logger::level::debug, "Loop candidate keyframe %d -> %d skipped, the covisible drift rescales by %.4f.", keyframe_id, candidate_id, model.scale);
            return false;
        }
        math::sim3<double> correction(math::se3<double>(math::so3<double>(rotation), translation), 1.0);
        const double drift = math::sqrt(translation.get_length_squared());
        if ((drift < loop_closure::covisible_loop_min_translation) || (drift < loop_closure::covisible_loop_min_relative_translation * rms_radius)) {
            core::logger::log(core::logger::level::debug, "Loop candidate keyframe %d -> %d skipped, covisible visit drifted %.3f m (%.3f of the cloud's radius).", keyframe_id, candidate_id, drift, drift / rms_radius);
            return false;
        }

        const math::sim3<double> correction_inverse = correction.inverse();
        outcome.matches.reserve(correspondences.size());
        for (size_t i = 0; i < correspondences.size(); ++i) {
            const record& current_record = keyframe_records[pair_records[i].first];
            const record& recorded_record = candidate.records[pair_records[i].second];
            const math::matrix<double, 3, 1> current_in_loop = candidate.pose * (correction * correspondences[i].lhs);
            const math::matrix<double, 3, 1> recorded_in_current = pose * (correction_inverse * correspondences[i].rhs);
            double projected_in_loop[2] = {};
            double projected_in_current[2] = {};
            if (!candidate.camera.project(current_in_loop.data(), &projected_in_loop[0]) || !camera.project(recorded_in_current.data(), &projected_in_current[0])) {
                continue;
            }
            const double loop_error_x = projected_in_loop[0] - static_cast<double>(recorded_record.pixel_x);
            const double loop_error_y = projected_in_loop[1] - static_cast<double>(recorded_record.pixel_y);
            const double current_error_x = projected_in_current[0] - static_cast<double>(current_record.pixel_x);
            const double current_error_y = projected_in_current[1] - static_cast<double>(current_record.pixel_y);
            if (((loop_error_x * loop_error_x) + (loop_error_y * loop_error_y) > loop_closure::reprojection_inlier_bound_squared) || ((current_error_x * current_error_x) + (current_error_y * current_error_y) > loop_closure::reprojection_inlier_bound_squared)) {
                continue;
            }
            outcome.matches.push_back(pairs[i]);
        }
        outcome.inliers = outcome.matches.size();
        if ((outcome.inliers < loop_closure::min_inliers) || (static_cast<double>(outcome.inliers) < loop_closure::min_inlier_fraction * static_cast<double>(correspondences.size()))) {
            core::logger::log(core::logger::level::debug, "Loop candidate keyframe %d -> %d skipped, %zu of %zu covisible landmarks reproject through the drift.", keyframe_id, candidate_id, outcome.inliers, correspondences.size());
            outcome.matches.clear();
            outcome.inliers = 0;
            return false;
        }

        outcome.found = true;
        outcome.keyframe_id = candidate_id;
        outcome.correspondences = correspondences.size();
        outcome.correction = correction;
        outcome.relative = math::sim3<double>(candidate.pose, 1.0) * outcome.correction * math::sim3<double>(pose.inverse(), 1.0);
        core::logger::log(core::logger::level::note, "Loop keyframe %d -> %d closed by covisible revisit, %zu of %zu shared landmarks reproject through the drift similarity, drift %.3f m, fitted scale %.4f.", keyframe_id, candidate_id, outcome.inliers, correspondences.size(), drift, correction.scale());
        return true;
    }

    void loop_closure::add_keyframe(const int keyframe_id, const math::se3<double>& pose, const sensor::model& camera, const record* const keyframe_records, const size_t keyframe_records_size) {
        if (keyframe_records_size == 0) {
            return;
        }
        std::vector<feature::descriptor::binary<256>> descriptors(keyframe_records_size);
        for (size_t i = 0; i < keyframe_records_size; ++i) {
            descriptors[i] = keyframe_records[i].descriptor;
        }
        this->recognition.add_keyframe(keyframe_id, descriptors.data(), descriptors.size());
        keyframe& stored = this->keyframes[keyframe_id];
        stored.pose = pose;
        stored.camera = camera;
        stored.records.assign(keyframe_records, keyframe_records + keyframe_records_size);
    }

    void loop_closure::set_hamming_scale(const float scale) {
        this->hamming_scale = scale;
    }

    void loop_closure::remove_keyframe(const int keyframe_id) {
        this->keyframes.erase(keyframe_id);
    }

    size_t loop_closure::num_keyframes() const {
        return this->recognition.num_keyframes();
    }

    std::vector<int> loop_closure::recall(const feature::descriptor::binary<256>* const descriptors, const size_t descriptors_size, const size_t max_recalled) const {
        std::vector<int> recalled;
        if (descriptors_size == 0) {
            return recalled;
        }
        for (const place_recognition::candidate& candidate : this->recognition.get_candidates(descriptors, descriptors_size, -1, max_recalled)) {
            recalled.push_back(candidate.keyframe_id);
        }
        return recalled;
    }

    const loop_closure::record* loop_closure::records_of(const int keyframe_id, size_t& records_size) const {
        const std::unordered_map<int, keyframe>::const_iterator found = this->keyframes.find(keyframe_id);
        if (found == this->keyframes.end()) {
            records_size = 0;
            return nullptr;
        }
        records_size = found->second.records.size();
        return found->second.records.data();
    }
}
