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
#include "core/random_pcg.hpp"
#include "estimation/correspondence_3d_3d.hpp"
#include "estimation/minimal/similarity_3_point.hpp"
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

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <string>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mapping {
    loop_closure::loop_closure()
        : recognition()
        , keyframes() {
    }

    void loop_closure::shared_landmarks(const record* const keyframe_records, const size_t keyframe_records_size, const keyframe& candidate, std::vector<estimation::correspondence_3d_3d<double>>& correspondences, std::vector<correspondence>& pairs, std::vector<std::pair<size_t, size_t>>& pair_records) {
        const std::vector<record>& candidate_records = candidate.records;
        std::unordered_map<int, size_t> recorded_by_landmark;
        recorded_by_landmark.reserve(candidate_records.size());
        for (size_t i = 0; i < candidate_records.size(); ++i) {
            recorded_by_landmark[candidate_records[i].landmark_id] = i;
        }
        correspondences.clear();
        pairs.clear();
        pair_records.clear();
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
    }

    loop_closure::result loop_closure::detect(const int keyframe_id, const math::se3<double>& pose, const sensor::model& camera, const covisibility& graph, const record* const keyframe_records, const size_t keyframe_records_size, const int submap_start_id, const bool seek_foreign) {
        const auto unfound = []() {
            result empty;
            empty.found = false;
            empty.keyframe_id = -1;
            empty.correspondences = 0;
            empty.inliers = 0;
            empty.correction = math::sim3<double>::identity();
            empty.relative = math::sim3<double>::identity();
            return empty;
        };
        result outcome = unfound();
        if (keyframe_records_size == 0) {
            return outcome;
        }

        std::vector<feature::descriptor::stored> query(keyframe_records_size);
        for (size_t i = 0; i < keyframe_records_size; ++i) {
            query[i] = keyframe_records[i].descriptor;
        }
        const bool islands = this->recognition.get_engine() == place_recognition::engine::ibow;
        const auto recent = [keyframe_id, submap_start_id](const int candidate_id) {
            return (candidate_id >= submap_start_id) && (candidate_id > keyframe_id - loop_closure::min_keyframe_gap);
        };
        const auto excluded = [&](const int candidate_id) {
            return recent(candidate_id) || (graph.weight(keyframe_id, candidate_id) >= loop_closure::max_covisible_landmarks) || (this->keyframes.count(candidate_id) == 0);
        };
        const std::vector<place_recognition::candidate> candidates = islands ? this->recognition.get_loop_candidates(query.data(), query.size(), keyframe_id, loop_closure::max_verified_candidates, excluded) : this->recognition.get_candidates(query.data(), query.size(), keyframe_id, loop_closure::max_candidates);
        if (core::logger::enabled(core::logger::level::debug)) {
            // The ranked candidates (keyframe id and votes) and the keyframes covisible with the query, which the outcome lines
            // of each candidate below follow, so the detector can be scored offline against ground truth.
            std::string ranked;
            for (const place_recognition::candidate& candidate : candidates) {
                ranked += " " + std::to_string(candidate.keyframe_id) + ":" + std::to_string(candidate.votes);
            }
            std::string covisible;
            for (const int neighbour : graph.neighbours(keyframe_id, loop_closure::max_covisible_landmarks)) {
                covisible += " " + std::to_string(neighbour);
            }
            // The ranking with the recent and covisible keyframes left out before the list is cut, for comparison.
            std::string distant;
            size_t distant_count = 0;
            for (const place_recognition::candidate& candidate : this->recognition.get_candidates(query.data(), query.size(), keyframe_id, this->keyframes.size())) {
                if (((candidate.keyframe_id >= submap_start_id) && (candidate.keyframe_id > keyframe_id - loop_closure::min_keyframe_gap)) || (graph.weight(keyframe_id, candidate.keyframe_id) >= loop_closure::max_covisible_landmarks)) {
                    continue;
                }
                distant += " " + std::to_string(candidate.keyframe_id) + ":" + std::to_string(candidate.votes);
                if (++distant_count >= loop_closure::max_candidates) {
                    break;
                }
            }
            core::logger::log(core::logger::level::debug, "Loop query keyframe %d: %zu records, %zu keyframes indexed, candidates%s; covisible%s; distant%s.", keyframe_id, keyframe_records_size, this->keyframes.size(), ranked.c_str(), covisible.c_str(), distant.c_str());
        }
        std::vector<estimation::correspondence_3d_3d<double>> correspondences;
        std::vector<correspondence> pairs;
        std::vector<std::pair<size_t, size_t>> pair_records;
        if (islands) {
            // The islands' best keyframes first: a loop the map does not know of yet is worth more than a revisit.
            for (const place_recognition::candidate& candidate : candidates) {
                const keyframe& found = this->keyframes.at(candidate.keyframe_id);
                loop_closure::shared_landmarks(keyframe_records, keyframe_records_size, found, correspondences, pairs, pair_records);
                // Records that share many landmarks with the keyframe make it covisible whatever the graph says.
                if (correspondences.size() >= loop_closure::max_covisible_landmarks) {
                    if (this->covisible_revisits && this->covisible_revisit_loop(keyframe_id, pose, camera, keyframe_records, candidate.keyframe_id, found, correspondences, pairs, pair_records, outcome)) {
                        return outcome;
                    }
                    core::logger::log(core::logger::level::debug, "Loop candidate keyframe %d -> %d not a material covisible revisit (%zu landmarks shared by id).", keyframe_id, candidate.keyframe_id, correspondences.size());
                    continue;
                }
                result attempt = unfound();
                if (this->verify_candidate(keyframe_id, pose, camera, keyframe_records, keyframe_records_size, query, submap_start_id, candidate.keyframe_id, found, attempt)) {
                    return attempt;
                }
                if (&candidate == &candidates.front()) {
                    outcome.correspondences = attempt.correspondences;
                    outcome.inliers = attempt.inliers;
                }
            }
            if (this->covisible_revisits) {
                for (const place_recognition::candidate& candidate : this->recognition.get_candidates(query.data(), query.size(), keyframe_id, loop_closure::max_candidates)) {
                    const std::unordered_map<int, keyframe>::const_iterator found = this->keyframes.find(candidate.keyframe_id);
                    if (recent(candidate.keyframe_id) || (found == this->keyframes.end()) || (graph.weight(keyframe_id, candidate.keyframe_id) < loop_closure::max_covisible_landmarks)) {
                        continue;
                    }
                    loop_closure::shared_landmarks(keyframe_records, keyframe_records_size, found->second, correspondences, pairs, pair_records);
                    if (this->covisible_revisit_loop(keyframe_id, pose, camera, keyframe_records, candidate.keyframe_id, found->second, correspondences, pairs, pair_records, outcome)) {
                        return outcome;
                    }
                    core::logger::log(core::logger::level::debug, "Loop candidate keyframe %d -> %d not a material covisible revisit (%zu landmarks shared by id).", keyframe_id, candidate.keyframe_id, correspondences.size());
                }
            }
        }
        else {
            size_t verified = 0;
            for (const place_recognition::candidate& candidate : candidates) {
                if ((candidate.keyframe_id >= submap_start_id) && (candidate.keyframe_id > keyframe_id - loop_closure::min_keyframe_gap)) {
                    continue;
                }
                const std::unordered_map<int, keyframe>::const_iterator found = this->keyframes.find(candidate.keyframe_id);
                if (found == this->keyframes.end()) {
                    continue;
                }
                const bool covisible_candidate = (graph.weight(keyframe_id, candidate.keyframe_id) >= loop_closure::max_covisible_landmarks);
                loop_closure::shared_landmarks(keyframe_records, keyframe_records_size, found->second, correspondences, pairs, pair_records);
                const size_t shared_by_id = correspondences.size();
                if (covisible_candidate || (shared_by_id >= loop_closure::max_covisible_landmarks)) {
                    if (this->covisible_revisits && this->covisible_revisit_loop(keyframe_id, pose, camera, keyframe_records, candidate.keyframe_id, found->second, correspondences, pairs, pair_records, outcome)) {
                        return outcome;
                    }
                    core::logger::log(core::logger::level::debug, "Loop candidate keyframe %d -> %d not a material covisible revisit (%zu landmarks shared by id).", keyframe_id, candidate.keyframe_id, shared_by_id);
                    continue;
                }
                // A candidate that fails verification leaves the next one in rank to be tried, so an alias ranked first cannot hide the true loop.
                if (verified >= loop_closure::max_verified_candidates) {
                    break;
                }
                result attempt = unfound();
                if (this->verify_candidate(keyframe_id, pose, camera, keyframe_records, keyframe_records_size, query, submap_start_id, candidate.keyframe_id, found->second, attempt)) {
                    return attempt;
                }
                // When nothing verifies, the counts reported are those of the best ranked candidate.
                if (verified == 0) {
                    outcome.correspondences = attempt.correspondences;
                    outcome.inliers = attempt.inliers;
                }
                ++verified;
            }
        }
        if (seek_foreign && (submap_start_id > 0)) {
            size_t foreign_verified = 0;
            const std::vector<place_recognition::candidate> foreign = this->recognition.get_candidates(query.data(), query.size(), keyframe_id, loop_closure::max_candidates, submap_start_id);
            if (core::logger::enabled(core::logger::level::debug)) {
                std::string ranked;
                for (const place_recognition::candidate& candidate : foreign) {
                    ranked += " " + std::to_string(candidate.keyframe_id) + ":" + std::to_string(candidate.votes);
                }
                core::logger::log(core::logger::level::debug, "Loop query keyframe %d before keyframe %d: candidates%s.", keyframe_id, submap_start_id, ranked.c_str());
            }
            for (const place_recognition::candidate& candidate : foreign) {
                if (foreign_verified >= loop_closure::max_verified_candidates) {
                    break;
                }
                const std::unordered_map<int, keyframe>::const_iterator found = this->keyframes.find(candidate.keyframe_id);
                if (found == this->keyframes.end()) {
                    continue;
                }
                result attempt = unfound();
                if (this->verify_candidate(keyframe_id, pose, camera, keyframe_records, keyframe_records_size, query, submap_start_id, candidate.keyframe_id, found->second, attempt)) {
                    return attempt;
                }
                ++foreign_verified;
            }
        }
        return outcome;
    }

    bool loop_closure::reprojection_consensus(const math::se3<double>& pose, const sensor::model& camera, const keyframe& candidate, const record* const keyframe_records, const std::vector<estimation::correspondence_3d_3d<double>>& correspondences, const std::vector<std::pair<size_t, size_t>>& pair_records, double (&rotation)[3][3], double (&translation)[3], double& scale, size_t& supporters) {
        const size_t count = correspondences.size();
        supporters = 0;
        if (count < 3) {
            return false;
        }
        const auto support = [&](const double (&hypothesis_rotation)[9], const double (&hypothesis_translation)[3], const double hypothesis_scale, std::vector<size_t>* const inliers) {
            if (!(hypothesis_scale > 0.0) || !math::isfinite(hypothesis_scale)) {
                return static_cast<size_t>(0);
            }
            const math::matrix<double, 3, 3> rotation_matrix = { { { hypothesis_rotation[0], hypothesis_rotation[1], hypothesis_rotation[2] }, { hypothesis_rotation[3], hypothesis_rotation[4], hypothesis_rotation[5] }, { hypothesis_rotation[6], hypothesis_rotation[7], hypothesis_rotation[8] } } };
            const math::matrix<double, 3, 1> translation_vector = { { hypothesis_translation[0], hypothesis_translation[1], hypothesis_translation[2] } };
            const math::matrix<double, 3, 3> rotation_inverse = math::transpose(rotation_matrix);
            size_t supported = 0;
            for (size_t i = 0; i < count; ++i) {
                const record& current_record = keyframe_records[pair_records[i].first];
                const record& recorded_record = candidate.records[pair_records[i].second];
                const math::matrix<double, 3, 1> in_loop = candidate.pose * ((rotation_matrix * correspondences[i].lhs) * hypothesis_scale + translation_vector);
                const math::matrix<double, 3, 1> in_current = pose * ((rotation_inverse * (correspondences[i].rhs - translation_vector)) * (1.0 / hypothesis_scale));
                double projected_in_loop[2] = {};
                double projected_in_current[2] = {};
                if (!(in_loop[2] > 0.0) || !(in_current[2] > 0.0) || !candidate.camera.project(in_loop.data(), &projected_in_loop[0]) || !camera.project(in_current.data(), &projected_in_current[0])) {
                    continue;
                }
                const double loop_x = projected_in_loop[0] - static_cast<double>(recorded_record.pixel_x);
                const double loop_y = projected_in_loop[1] - static_cast<double>(recorded_record.pixel_y);
                const double current_x = projected_in_current[0] - static_cast<double>(current_record.pixel_x);
                const double current_y = projected_in_current[1] - static_cast<double>(current_record.pixel_y);
                if (((loop_x * loop_x) + (loop_y * loop_y) > loop_closure::reprojection_inlier_bound_squared) || ((current_x * current_x) + (current_y * current_y) > loop_closure::reprojection_inlier_bound_squared)) {
                    continue;
                }
                ++supported;
                if (inliers != nullptr) {
                    inliers->push_back(i);
                }
            }
            return supported;
        };
        // Deterministic samples, as many as the support found so far leaves needed to draw a supported triple with 99%
        // probability, between 10 and 300.
        core::random_pcg random(0x5113ull + static_cast<unsigned long long>(count));
        double best_rotation[9] = {};
        double best_translation[3] = {};
        double best_scale = 0.0;
        size_t best = 0;
        size_t needed = 300;
        for (size_t iteration = 0; (iteration < needed) && (iteration < 300); ++iteration) {
            size_t picks[3] = { 0, 0, 0 };
            picks[0] = random.get_random(0u, static_cast<unsigned int>(count - 1));
            do {
                picks[1] = random.get_random(0u, static_cast<unsigned int>(count - 1));
            } while (picks[1] == picks[0]);
            do {
                picks[2] = random.get_random(0u, static_cast<unsigned int>(count - 1));
            } while ((picks[2] == picks[0]) || (picks[2] == picks[1]));
            double source[9];
            double target[9];
            for (size_t p = 0; p < 3; ++p) {
                for (size_t axis = 0; axis < 3; ++axis) {
                    source[(p * 3) + axis] = correspondences[picks[p]].lhs[axis];
                    target[(p * 3) + axis] = correspondences[picks[p]].rhs[axis];
                }
            }
            double hypothesis_rotation[9];
            double hypothesis_translation[3];
            double hypothesis_scale = 0.0;
            if (!estimation::minimal::similarity_3_point<double>::solve(&source[0], &target[0], 3, hypothesis_rotation, hypothesis_translation, hypothesis_scale)) {
                continue;
            }
            const size_t supported = support(hypothesis_rotation, hypothesis_translation, hypothesis_scale, nullptr);
            if (supported <= best) {
                continue;
            }
            best = supported;
            for (size_t i = 0; i < 9; ++i) {
                best_rotation[i] = hypothesis_rotation[i];
            }
            for (size_t i = 0; i < 3; ++i) {
                best_translation[i] = hypothesis_translation[i];
            }
            best_scale = hypothesis_scale;
            const double fraction = static_cast<double>(best) / static_cast<double>(count);
            const double all_supported = fraction * fraction * fraction;
            needed = (all_supported >= 1.0) ? 10 : math::max<size_t>(10, static_cast<size_t>(math::log(0.01) / math::log(1.0 - all_supported)) + 1);
        }
        if (best < 3) {
            return false;
        }
        std::vector<size_t> inliers;
        support(best_rotation, best_translation, best_scale, &inliers);
        std::vector<double> source(inliers.size() * 3);
        std::vector<double> target(inliers.size() * 3);
        for (size_t i = 0; i < inliers.size(); ++i) {
            for (size_t axis = 0; axis < 3; ++axis) {
                source[(i * 3) + axis] = correspondences[inliers[i]].lhs[axis];
                target[(i * 3) + axis] = correspondences[inliers[i]].rhs[axis];
            }
        }
        double refit_rotation[9];
        double refit_translation[3];
        double refit_scale = 0.0;
        if (estimation::minimal::similarity_3_point<double>::solve(source.data(), target.data(), inliers.size(), refit_rotation, refit_translation, refit_scale)) {
            const size_t refit_supported = support(refit_rotation, refit_translation, refit_scale, nullptr);
            if (refit_supported >= best) {
                best = refit_supported;
                for (size_t i = 0; i < 9; ++i) {
                    best_rotation[i] = refit_rotation[i];
                }
                for (size_t i = 0; i < 3; ++i) {
                    best_translation[i] = refit_translation[i];
                }
                best_scale = refit_scale;
            }
        }
        for (size_t row = 0; row < 3; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                rotation[row][column] = best_rotation[(row * 3) + column];
            }
            translation[row] = best_translation[row];
        }
        scale = best_scale;
        supporters = best;
        return true;
    }

    size_t loop_closure::guided_pairs(const math::se3<double>& pose, const sensor::model& camera, const record* const keyframe_records, const size_t keyframe_records_size, const keyframe& candidate, const math::sim3<double>& correction, std::vector<unsigned char>& current_paired, std::vector<unsigned char>& recorded_paired, std::vector<estimation::correspondence_3d_3d<double>>& correspondences, std::vector<correspondence>& pairs, std::vector<std::pair<size_t, size_t>>& pair_records) const {
        const std::vector<record>& candidate_records = candidate.records;
        const auto project_into = [](const sensor::model& into_camera, const math::se3<double>& into_pose, const math::matrix<double, 3, 1>& world, double (&pixel)[2]) -> bool {
            const math::matrix<double, 3, 1> in_camera = into_pose * world;
            return into_camera.project(in_camera.data(), &pixel[0]);
        };
        const unsigned int guided_bound = static_cast<unsigned int>(loop_closure::guided_hamming_maximum * this->hamming_scale);
        const double radius_squared = loop_closure::guided_search_radius * loop_closure::guided_search_radius;
        const auto nearest = [&](const feature::descriptor::stored& descriptor, const record* const searched, const size_t searched_size, const std::vector<unsigned char>& claimed, const double (&predicted)[2]) -> size_t {
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
            if ((current_paired[i] != 0) || !project_into(candidate.camera, candidate.pose, into_recorded * keyframe_records[i].location, predicted)) {
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
        size_t added = 0;
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
            ++added;
        }
        return added;
    }

    bool loop_closure::reprojects(const math::se3<double>& pose, const sensor::model& camera, const record& current_record, const keyframe& candidate, const record& recorded_record, const estimation::correspondence_3d_3d<double>& correspondence, const math::sim3<double>& similarity, const math::sim3<double>& similarity_inverse, const double bound_squared) {
        const math::matrix<double, 3, 1> current_in_loop = candidate.pose * (similarity * correspondence.lhs);
        const math::matrix<double, 3, 1> recorded_in_current = pose * (similarity_inverse * correspondence.rhs);
        double projected_in_loop[2] = {};
        double projected_in_current[2] = {};
        if (!candidate.camera.project(current_in_loop.data(), &projected_in_loop[0]) || !camera.project(recorded_in_current.data(), &projected_in_current[0])) {
            return false;
        }
        const double loop_error_x = projected_in_loop[0] - static_cast<double>(recorded_record.pixel_x);
        const double loop_error_y = projected_in_loop[1] - static_cast<double>(recorded_record.pixel_y);
        const double current_error_x = projected_in_current[0] - static_cast<double>(current_record.pixel_x);
        const double current_error_y = projected_in_current[1] - static_cast<double>(current_record.pixel_y);
        return ((loop_error_x * loop_error_x) + (loop_error_y * loop_error_y) <= bound_squared) && ((current_error_x * current_error_x) + (current_error_y * current_error_y) <= bound_squared);
    }

    math::sim3<double> loop_closure::refine_similarity(const math::se3<double>& pose, const sensor::model& camera, const record* const keyframe_records, const keyframe& candidate, const std::vector<estimation::correspondence_3d_3d<double>>& correspondences, const std::vector<std::pair<size_t, size_t>>& pair_records, const std::vector<unsigned char>& seeded, const size_t seeded_count, const math::sim3<double>& initial) {
        optimisation::factor_graph refinement;
        double parameters[8] = { initial.transformation().translation()[0], initial.transformation().translation()[1], initial.transformation().translation()[2], initial.transformation().rotation().get_quaternion()[1], initial.transformation().rotation().get_quaternion()[2], initial.transformation().rotation().get_quaternion()[3], initial.transformation().rotation().get_quaternion()[0], initial.scale() };
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
            if (seeded[i] == 0) {
                factors.push_back({ nullptr, nullptr });
                continue;
            }
            const record& current_record = keyframe_records[pair_records[i].first];
            const record& recorded_record = candidate.records[pair_records[i].second];
            factors.push_back({ add_edge(correspondences[i].lhs, candidate.camera, candidate.pose, false, recorded_record.pixel_x, recorded_record.pixel_y), add_edge(correspondences[i].rhs, camera, pose, true, current_record.pixel_x, current_record.pixel_y) });
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
        if ((dropped > 0) && (dropped < seeded_count)) {
            static_cast<void>(refinement.solve(loop_closure::refine_rounds, true));
        }
        const double* const refined = vertex->get_parameters();
        return math::sim3<double>(math::se3<double>(math::so3<double>(refined[6], refined[3], refined[4], refined[5]), { { refined[0], refined[1], refined[2] } }), refined[7]);
    }

    bool loop_closure::verify_candidate(const int keyframe_id, const math::se3<double>& pose, const sensor::model& camera, const record* const keyframe_records, const size_t keyframe_records_size, const std::vector<feature::descriptor::stored>& query, const int submap_start_id, const int candidate_id, const keyframe& candidate, result& outcome) const {
        const keyframe* const candidate_keyframe = &candidate;
        const std::vector<record>& candidate_records = candidate_keyframe->records;
        core::logger::log(core::logger::level::debug, "Loop candidate keyframe %d -> %d: %zu records against %zu.", keyframe_id, candidate_id, candidate_records.size(), keyframe_records_size);

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

        std::vector<feature::descriptor::stored> recorded_descriptors(candidate_records.size());
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
            return false;
        }
        if (correspondences.size() < 3) {
            core::logger::log(core::logger::level::debug, "Loop candidate keyframe %d -> %d rejected, %zu shared landmarks (%zu by id).", keyframe_id, candidate_id, correspondences.size(), shared_by_id);
            return false;
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
        const bool solved = this->reprojection_hypotheses ? loop_closure::reprojection_consensus(pose, camera, candidate, keyframe_records, correspondences, pair_records, model.rotation, model.translation, model.scale, inliers_size) : estimation::robust::solver::similarity<double>::solve(correspondences.data(), correspondences.size(), inlier_threshold, residuals.data(), inliers.data(), inliers_size, model);
        if (!solved) {
            core::logger::log(core::logger::level::debug, "Loop candidate keyframe %d -> %d rejected, no similarity fits its %zu shared landmarks (%zu by id).", keyframe_id, candidate_id, correspondences.size(), shared_by_id);
            return false;
        }

        const math::matrix<double, 3, 3> rotation = { { { model.rotation[0][0], model.rotation[0][1], model.rotation[0][2] },
                                                        { model.rotation[1][0], model.rotation[1][1], model.rotation[1][2] },
                                                        { model.rotation[2][0], model.rotation[2][1], model.rotation[2][2] } } };
        const math::matrix<double, 3, 1> translation = { { model.translation[0], model.translation[1], model.translation[2] } };
        const bool foreign_submap = candidate_id < submap_start_id;
        if (!foreign_submap && ((model.scale > loop_closure::max_scale_ratio) || (model.scale < 1.0 / loop_closure::max_scale_ratio))) {
            core::logger::log(core::logger::level::note, "Loop candidate keyframe %d -> %d rejected, scale %.4f is not plausible.", keyframe_id, candidate_id, model.scale);
            return false;
        }
        math::sim3<double> correction(math::se3<double>(math::so3<double>(rotation), translation), model.scale);

        const size_t paired = correspondences.size();
        const size_t guided = this->guided_pairs(pose, camera, keyframe_records, keyframe_records_size, candidate, correction, current_paired, recorded_paired, correspondences, pairs, pair_records);
        outcome.correspondences = correspondences.size();

        const auto pair_reprojects = [&](const size_t i, const math::sim3<double>& similarity, const math::sim3<double>& similarity_inverse, const double bound_squared) {
            return loop_closure::reprojects(pose, camera, keyframe_records[pair_records[i].first], candidate, candidate_records[pair_records[i].second], correspondences[i], similarity, similarity_inverse, bound_squared);
        };

        // The pairs the refinement starts from: with refine_from_hypothesis those the first similarity reprojects within the
        // guided search's radius, so that the initial pairs it does not explain cannot pull it away from its inliers when
        // they are most of the pairs.
        std::vector<unsigned char> seeded(correspondences.size(), static_cast<unsigned char>(1));
        size_t seeded_count = correspondences.size();
        if (this->refine_from_hypothesis) {
            const math::sim3<double> correction_inverse = correction.inverse();
            const double radius_squared = loop_closure::guided_search_radius * loop_closure::guided_search_radius;
            size_t explained = 0;
            for (size_t i = 0; i < correspondences.size(); ++i) {
                seeded[i] = pair_reprojects(i, correction, correction_inverse, radius_squared) ? 1 : 0;
                explained += seeded[i];
            }
            if (explained >= 3) {
                seeded_count = explained;
            }
            else {
                seeded.assign(correspondences.size(), static_cast<unsigned char>(1));
            }
        }

        correction = loop_closure::refine_similarity(pose, camera, keyframe_records, candidate, correspondences, pair_records, seeded, seeded_count, correction);
        if (!foreign_submap && ((correction.scale() > loop_closure::max_scale_ratio) || (correction.scale() < 1.0 / loop_closure::max_scale_ratio))) {
            core::logger::log(core::logger::level::note, "Loop candidate keyframe %d -> %d rejected, the refined scale %.4f is not plausible.", keyframe_id, candidate_id, correction.scale());
            return false;
        }

        const math::sim3<double> correction_inverse = correction.inverse();
        size_t paired_inliers = 0;
        outcome.matches.reserve(correspondences.size());
        for (size_t i = 0; i < correspondences.size(); ++i) {
            if (!pair_reprojects(i, correction, correction_inverse, loop_closure::reprojection_inlier_bound_squared)) {
                continue;
            }
            outcome.matches.push_back(pairs[i]);
            paired_inliers += (i < paired) ? 1 : 0;
        }
        outcome.inliers = outcome.matches.size();
        // Between submaps every pair is a descriptor match across a loss, which keeps fewer of them, while a submap left apart
        // costs the map its consistency until it joins; such a join may keep a smaller share of its pairs if it keeps more.
        const size_t inliers_required = foreign_submap ? loop_closure::foreign_min_inliers : loop_closure::min_inliers;
        const double fraction_required = foreign_submap ? loop_closure::foreign_min_inlier_fraction : loop_closure::min_inlier_fraction;
        const bool enough_share = static_cast<double>(paired_inliers) >= fraction_required * static_cast<double>(paired);
        const bool enough_alone = this->accept_by_inliers && (outcome.inliers >= loop_closure::min_inliers_any_share);
        if ((outcome.inliers < inliers_required) || (!enough_share && !enough_alone)) {
            core::logger::log(core::logger::level::debug, "Loop candidate keyframe %d -> %d rejected, %zu of %zu shared landmarks (%zu by id, %zu found by projection) reproject through the refined similarity (%zu fitted it in 3D).", keyframe_id, candidate_id, outcome.inliers, correspondences.size(), shared_by_id, guided, inliers_size);
            outcome.matches.clear();
            return false;
        }

        outcome.found = true;
        outcome.keyframe_id = candidate_id;
        outcome.correction = correction;
        outcome.relative = math::sim3<double>(candidate_keyframe->pose, 1.0) * outcome.correction * math::sim3<double>(pose.inverse(), 1.0);
        core::logger::log(core::logger::level::note, "Loop detected keyframe %d -> %d, %zu of %zu shared landmarks (%zu by id, %zu found by projection) reproject through the refined similarity, scale %.4f, translation %.4f.", keyframe_id, candidate_id, outcome.inliers, correspondences.size(), shared_by_id, guided, correction.scale(), math::sqrt(correction.transformation().translation().get_length_squared()));
        return true;
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
        if (!estimation::robust::solver::similarity<double>::solve(correspondences.data(), correspondences.size(), inlier_threshold, residuals.data(), inliers.data(), inliers_size, model) || (inliers_size == 0)) {
            core::logger::log(core::logger::level::debug, "Loop candidate keyframe %d -> %d skipped, no similarity fits its %zu covisible landmarks.", keyframe_id, candidate_id, correspondences.size());
            return false;
        }
        const math::matrix<double, 3, 3> rotation = { { { model.rotation[0][0], model.rotation[0][1], model.rotation[0][2] },
                                                        { model.rotation[1][0], model.rotation[1][1], model.rotation[1][2] },
                                                        { model.rotation[2][0], model.rotation[2][1], model.rotation[2][2] } } };
        // The drift correction keeps the scale at one, and the fitted translation assumed the fitted scale; with the scale at one the best translation is between the inliers' centroids under the same rotation.
        math::matrix<double, 3, 1> centroid_lhs = math::matrix<double, 3, 1>::zero();
        math::matrix<double, 3, 1> centroid_rhs = math::matrix<double, 3, 1>::zero();
        for (size_t i = 0; i < inliers_size; ++i) {
            centroid_lhs = centroid_lhs + correspondences[inliers[i]].lhs;
            centroid_rhs = centroid_rhs + correspondences[inliers[i]].rhs;
        }
        centroid_lhs = centroid_lhs * (1.0 / static_cast<double>(inliers_size));
        centroid_rhs = centroid_rhs * (1.0 / static_cast<double>(inliers_size));
        const math::matrix<double, 3, 1> translation = centroid_rhs - (rotation * centroid_lhs);
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
        std::vector<feature::descriptor::stored> descriptors(keyframe_records_size);
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
        this->recognition.set_distance_threshold(static_cast<unsigned int>((static_cast<float>(place_recognition::default_distance_threshold) * scale) + 0.5f));
    }

    void loop_closure::set_place_recognition(const place_recognition::engine engine) {
        if (engine == this->recognition.get_engine()) {
            return;
        }
        this->recognition.set_engine(engine);
        // The keyframes already held are indexed again by the new engine, in the order they came.
        std::vector<int> keyframe_ids;
        keyframe_ids.reserve(this->keyframes.size());
        for (const auto& [keyframe_id, stored] : this->keyframes) {
            static_cast<void>(stored);
            keyframe_ids.push_back(keyframe_id);
        }
        std::sort(keyframe_ids.begin(), keyframe_ids.end());
        std::vector<feature::descriptor::stored> descriptors;
        for (const int keyframe_id : keyframe_ids) {
            const std::vector<record>& records = this->keyframes.at(keyframe_id).records;
            descriptors.resize(records.size());
            for (size_t i = 0; i < records.size(); ++i) {
                descriptors[i] = records[i].descriptor;
            }
            this->recognition.add_keyframe(keyframe_id, descriptors.data(), descriptors.size());
        }
    }

    place_recognition::engine loop_closure::get_place_recognition() const {
        return this->recognition.get_engine();
    }

    void loop_closure::set_accept_by_inliers(const bool enabled) {
        this->accept_by_inliers = enabled;
    }

    void loop_closure::set_refine_from_hypothesis(const bool enabled) {
        this->refine_from_hypothesis = enabled;
    }

    void loop_closure::set_reprojection_hypotheses(const bool enabled) {
        this->reprojection_hypotheses = enabled;
    }

    void loop_closure::set_covisible_revisits(const bool enabled) {
        this->covisible_revisits = enabled;
    }

    void loop_closure::refresh(const std::function<bool(const int, math::se3<double>&)>& pose_of, const std::function<bool(const int, math::matrix<double, 3, 1>&)>& location_of) {
        for (auto& [keyframe_id, stored] : this->keyframes) {
            math::se3<double> pose;
            if (pose_of(keyframe_id, pose)) {
                stored.pose = pose;
            }
            size_t kept = 0;
            for (size_t i = 0; i < stored.records.size(); ++i) {
                math::matrix<double, 3, 1> location;
                if (!location_of(stored.records[i].landmark_id, location)) {
                    continue;
                }
                stored.records[kept] = stored.records[i];
                stored.records[kept].location = location;
                ++kept;
            }
            stored.records.resize(kept);
        }
    }

    void loop_closure::remove_keyframe(const int keyframe_id) {
        this->recognition.remove_keyframe(keyframe_id);
        this->keyframes.erase(keyframe_id);
    }

    size_t loop_closure::num_keyframes() const {
        return this->recognition.num_keyframes();
    }

    std::vector<int> loop_closure::recall(const feature::descriptor::stored* const descriptors, const size_t descriptors_size, const size_t max_recalled) const {
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
