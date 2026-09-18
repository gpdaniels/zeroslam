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

#include "feature/tracker/tracker.hpp"

#include "core/arena.hpp"
#include "core/arena_allocator.hpp"
#include "core/logger.hpp"
#include "core/sort.hpp"
#include "core/thread_pool.hpp"
#include "feature/tracker/dominant_flow.hpp"
#include "feature/tracker/optical_flow.hpp"
#include "feature/tracker/patch_flow.hpp"
#include "feature/tracker/wavelet_flow.hpp"
#include "match/distance/hamming.hpp"
#include "math/math.hpp"

namespace feature::tracker {
    float tracker::distance_squared(float ax, float ay, float bx, float by) {
        const float dx = ax - bx;
        const float dy = ay - by;
        return dx * dx + dy * dy;
    }

    bool tracker::near_existing_track(float x, float y) const {
        const float limit = this->settings.min_spawn_distance * this->settings.min_spawn_distance;
        for (size_t i = 0; i < this->track_list.size(); ++i) {
            if (tracker::distance_squared(x, y, this->track_list[i].x, this->track_list[i].y) < limit) {
                return true;
            }
        }
        return false;
    }

    patch_flow::options tracker::anchor_options() const {
        patch_flow::options built;
        built.model = this->settings.anchor_model;
        built.half_window = this->settings.klt_half_window;
        built.iterations = this->settings.klt_iterations;
        built.min_eigenvalue = static_cast<double>(this->settings.klt_min_eigenvalue);
        built.max_error = this->settings.klt_max_error;
        built.damped_steps = this->settings.klt_damped_steps;
        return built;
    }

    wavelet_flow::options tracker::wavelet_options() const {
        wavelet_flow::options built;
        built.half_window = this->settings.wavelet_half_window;
        built.coarsest_level = this->settings.wavelet_levels;
        built.fb_threshold = this->settings.klt_fb_threshold;
        built.robust = this->settings.wavelet_robust;
        built.guess_first = (this->settings.wavelet_seed != wavelet_seed_kind::rest);
        return built;
    }

    void tracker::build_wavelet_anchors(const image::quaternion_wavelet& transform) {
        std::vector<size_t> pending;
        for (size_t i = 0; i < this->track_list.size(); ++i) {
            if (!this->track_list[i].wavelet_anchored) {
                pending.push_back(i);
            }
        }
        if (pending.empty()) {
            return;
        }
        const wavelet_flow::options build_options = this->wavelet_options();
        for (size_t p = 0; p < pending.size(); ++p) {
            this->track_list[pending[p]].wavelet_anchored.reset(new wavelet_anchoring());
        }
        core::thread_pool::instance().parallel_for(pending.size(), 16, [&](const size_t p) {
            track& current = this->track_list[pending[p]];
            static_cast<void>(wavelet_flow::build_anchor(transform, current.x, current.y, build_options, current.wavelet_anchored->anchor));
        });
    }

    void tracker::build_anchors(const image::pyramid& pyramid) {
        std::vector<size_t> pending;
        for (size_t i = 0; i < this->track_list.size(); ++i) {
            if (!this->track_list[i].anchored) {
                pending.push_back(i);
            }
        }
        if (pending.empty()) {
            return;
        }
        patch_flow::options build_options = this->anchor_options();
        build_options.levels = static_cast<int>(pyramid.size());
        for (size_t p = 0; p < pending.size(); ++p) {
            this->track_list[pending[p]].anchored.reset(new anchoring());
        }
        core::thread_pool::instance().parallel_for(pending.size(), 16, [&](const size_t p) {
            track& current = this->track_list[pending[p]];
            if (!patch_flow::build_anchor(pyramid, current.x, current.y, build_options, current.anchored->anchor)) {
                current.anchored->anchor.levels = 0;
            }
        });
    }

    void tracker::discard_anchors() {
        for (size_t i = 0; i < this->track_list.size(); ++i) {
            this->track_list[i].anchored.reset();
            this->track_list[i].wavelet_anchored.reset();
        }
    }

    void tracker::prune_collisions() {
        if (!(this->settings.collision_distance > 0.0f)) {
            return;
        }
        const size_t count = this->track_list.size();
        if (count < 2) {
            return;
        }
        std::vector<size_t> ranked;
        ranked.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            if (this->track_list[i].active) {
                ranked.push_back(i);
            }
        }
        if (ranked.size() < 2) {
            return;
        }
        const std::vector<track>& entries = this->track_list;
        core::sort::heap(ranked.data(), ranked.size(), [&entries](const size_t lhs, const size_t rhs) {
            if (entries[lhs].length != entries[rhs].length) {
                return entries[lhs].length > entries[rhs].length;
            }
            return entries[lhs].id < entries[rhs].id;
        });
        float minimum_x = this->track_list[ranked[0]].x;
        float minimum_y = this->track_list[ranked[0]].y;
        float maximum_x = minimum_x;
        float maximum_y = minimum_y;
        for (size_t r = 1; r < ranked.size(); ++r) {
            const track& current = this->track_list[ranked[r]];
            minimum_x = math::min(minimum_x, static_cast<float>(current.x));
            minimum_y = math::min(minimum_y, static_cast<float>(current.y));
            maximum_x = math::max(maximum_x, static_cast<float>(current.x));
            maximum_y = math::max(maximum_y, static_cast<float>(current.y));
        }
        constexpr static const int maximum_cells_per_axis = 1024;
        const float span_x = maximum_x - minimum_x;
        const float span_y = maximum_y - minimum_y;
        const float cell_size = math::max(this->settings.collision_distance, math::max(span_x, span_y) / static_cast<float>(maximum_cells_per_axis));
        const int columns = math::max(1, math::min(maximum_cells_per_axis, static_cast<int>(span_x / cell_size) + 1));
        const int rows = math::max(1, math::min(maximum_cells_per_axis, static_cast<int>(span_y / cell_size) + 1));
        std::vector<unsigned int> cell_head(static_cast<size_t>(columns) * static_cast<size_t>(rows), 0u);
        std::vector<unsigned int> chain_next(count, 0u);
        std::vector<unsigned char> dropped(count, static_cast<unsigned char>(0));
        const float limit_squared = this->settings.collision_distance * this->settings.collision_distance;
        size_t collisions = 0;
        for (size_t r = 0; r < ranked.size(); ++r) {
            const size_t index = ranked[r];
            const track& current = this->track_list[index];
            const int cell_x = math::min(columns - 1, static_cast<int>((current.x - minimum_x) / cell_size));
            const int cell_y = math::min(rows - 1, static_cast<int>((current.y - minimum_y) / cell_size));
            bool collides = false;
            for (int offset_y = -1; (offset_y <= 1) && !collides; ++offset_y) {
                for (int offset_x = -1; (offset_x <= 1) && !collides; ++offset_x) {
                    const int neighbour_x = cell_x + offset_x;
                    const int neighbour_y = cell_y + offset_y;
                    if ((neighbour_x < 0) || (neighbour_y < 0) || (neighbour_x >= columns) || (neighbour_y >= rows)) {
                        continue;
                    }
                    const size_t cell = (static_cast<size_t>(neighbour_y) * static_cast<size_t>(columns)) + static_cast<size_t>(neighbour_x);
                    for (unsigned int item = cell_head[cell]; item != 0u; item = chain_next[item - 1u]) {
                        const track& kept = this->track_list[item - 1u];
                        if (tracker::distance_squared(current.x, current.y, kept.x, kept.y) < limit_squared) {
                            collides = true;
                            break;
                        }
                    }
                }
            }
            if (collides) {
                dropped[index] = static_cast<unsigned char>(1);
                ++collisions;
                continue;
            }
            const size_t cell = (static_cast<size_t>(cell_y) * static_cast<size_t>(columns)) + static_cast<size_t>(cell_x);
            chain_next[index] = cell_head[cell];
            cell_head[cell] = static_cast<unsigned int>(index + 1u);
        }
        if (collisions == 0) {
            return;
        }
        size_t write = 0;
        for (size_t read = 0; read < count; ++read) {
            if (dropped[read]) {
                continue;
            }
            if (write != read) {
                this->track_list[write] = static_cast<track&&>(this->track_list[read]);
            }
            ++write;
        }
        this->track_list.resize(write);
        core::logger::log(core::logger::level::debug, "Tracker: %zu collided tracks dropped.", collisions);
    }

    tracker::track& tracker::spawn(int frame_id, const feature::point& keypoint, const descriptor::binary<256>& descriptor) {
        track created;
        created.id = this->next_id++;
        created.landmark_id = -1;
        created.active = true;
        created.x = keypoint.x;
        created.y = keypoint.y;
        created.octave = keypoint.octave;
        created.start_frame_id = frame_id;
        created.last_frame_id = frame_id;
        created.length = 1;
        created.missed = 0;
        created.outliers = 0;
        created.outlier_frame_id = -1;
        created.descriptor = descriptor;
        created.history.push_back(observation{ frame_id, keypoint.x, keypoint.y });
        this->track_list.push_back(static_cast<track&&>(created));
        return this->track_list.back();
    }

    void tracker::observe(track& existing, int frame_id, float x, float y) {
        existing.x = x;
        existing.y = y;
        existing.active = true;
        existing.missed = 0;
        existing.last_frame_id = frame_id;
        existing.length += 1;
        existing.history.push_back(observation{ frame_id, x, y });
    }

    tracker::tracker()
        : tracker(options()) {
    }

    tracker::tracker(const options& opts)
        : settings(opts)
        , track_list()
        , pyramid_previous()
        , wavelet_previous()
        , flow_previous()
        , carry_ratio_previous(1.0f)
        , has_previous(false)
        , next_id(0) {
    }

    void tracker::set_options(const options& opts) {
        const bool geometry_changed = (opts.anchor_model != this->settings.anchor_model) || (opts.klt_half_window != this->settings.klt_half_window) || (opts.anchored_patches != this->settings.anchored_patches) || (opts.flow != this->settings.flow) || (opts.wavelet_half_window != this->settings.wavelet_half_window) || (opts.wavelet_levels != this->settings.wavelet_levels) || (opts.wavelet_undecimated != this->settings.wavelet_undecimated);
        this->settings = opts;
        if (geometry_changed) {
            this->discard_anchors();
        }
    }

    const tracker::options& tracker::get_options() const {
        return this->settings;
    }

    void tracker::update(
        int frame_id,
        const image::pyramid& pyramid,
        const std::vector<feature::point>& keypoints,
        const std::vector<descriptor::binary<256>>& descriptors
    ) {
        core::arena::scope scratch;
        const size_t detection_count = math::min(keypoints.size(), descriptors.size());

        std::vector<unsigned char, core::arena_allocator<unsigned char>> claimed(detection_count, static_cast<unsigned char>(0));

        if (!this->has_previous) {
            for (size_t d = 0; d < detection_count; ++d) {
                if (this->near_existing_track(keypoints[d].x, keypoints[d].y)) {
                    continue;
                }
                this->spawn(frame_id, keypoints[d], descriptors[d]);
                claimed[d] = static_cast<unsigned char>(1);
            }
            const bool intensity = (this->settings.flow == flow_kind::intensity);
            if (this->settings.anchored_patches && intensity) {
                this->build_anchors(pyramid);
            }
            this->pyramid_previous = pyramid;
            if (!intensity) {
                this->wavelet_previous = image::quaternion_wavelet(pyramid[0], static_cast<size_t>(math::max(2, this->settings.wavelet_levels)), 2, this->settings.wavelet_undecimated);
                if (this->settings.anchored_patches) {
                    this->build_wavelet_anchors(this->wavelet_previous);
                }
            }
            this->has_previous = true;
            return;
        }

        const bool wavelet_path = (this->settings.flow == flow_kind::wavelet);
        if (this->settings.anchored_patches && !wavelet_path) {
            this->build_anchors(this->pyramid_previous);
        }

        image::quaternion_wavelet wavelet;
        if (wavelet_path) {
            const size_t depth = static_cast<size_t>(math::max(2, this->settings.wavelet_levels));
            wavelet = image::quaternion_wavelet(pyramid[0], depth, 2, this->settings.wavelet_undecimated);
            if ((this->wavelet_previous.size() == 0) && (this->pyramid_previous.size() > 0)) {
                this->wavelet_previous = image::quaternion_wavelet(this->pyramid_previous[0], depth, 2, this->settings.wavelet_undecimated);
            }
            if (this->settings.anchored_patches) {
                this->build_wavelet_anchors(this->wavelet_previous);
            }
        }

        constexpr static const float cell_size = 16.0f;
        std::vector<int, core::arena_allocator<int>> cell_of(detection_count);
        int cell_columns = 1;
        int cell_rows = 1;
        {
            float max_x = 0.0f;
            float max_y = 0.0f;
            for (size_t d = 0; d < detection_count; ++d) {
                max_x = math::max(max_x, keypoints[d].x);
                max_y = math::max(max_y, keypoints[d].y);
            }
            cell_columns = static_cast<int>(max_x / cell_size) + 1;
            cell_rows = static_cast<int>(max_y / cell_size) + 1;
        }
        std::vector<int, core::arena_allocator<int>> cell_start(static_cast<size_t>(cell_columns * cell_rows) + 1, 0);
        std::vector<size_t, core::arena_allocator<size_t>> cell_items(detection_count);
        for (size_t d = 0; d < detection_count; ++d) {
            const int cx = math::max(0, math::min(cell_columns - 1, static_cast<int>(keypoints[d].x / cell_size)));
            const int cy = math::max(0, math::min(cell_rows - 1, static_cast<int>(keypoints[d].y / cell_size)));
            cell_of[d] = (cy * cell_columns) + cx;
            ++cell_start[static_cast<size_t>(cell_of[d]) + 1];
        }
        for (size_t c = 1; c < cell_start.size(); ++c) {
            cell_start[c] += cell_start[c - 1];
        }
        {
            std::vector<int, core::arena_allocator<int>> fill(cell_start.begin(), cell_start.end() - 1);
            for (size_t d = 0; d < detection_count; ++d) {
                cell_items[static_cast<size_t>(fill[static_cast<size_t>(cell_of[d])]++)] = d;
            }
        }

        struct candidate final {
            size_t detection;
            unsigned int hamming;
            float distance_squared;
        };

        const auto search = [&](const float x, const float y, const float radius, const int octave, const descriptor::binary<256>& query, const bool skip_claimed, candidate& best, candidate& second) {
            best = candidate{ detection_count, 256u, 0.0f };
            second = candidate{ detection_count, 256u, 0.0f };
            const float radius_squared = radius * radius;
            const int cx0 = math::max(0, static_cast<int>((x - radius) / cell_size));
            const int cx1 = math::min(cell_columns - 1, static_cast<int>((x + radius) / cell_size));
            const int cy0 = math::max(0, static_cast<int>((y - radius) / cell_size));
            const int cy1 = math::min(cell_rows - 1, static_cast<int>((y + radius) / cell_size));
            for (int cy = cy0; cy <= cy1; ++cy) {
                for (int cx = cx0; cx <= cx1; ++cx) {
                    const size_t cell = static_cast<size_t>((cy * cell_columns) + cx);
                    for (int item = cell_start[cell]; item < cell_start[cell + 1]; ++item) {
                        const size_t d = cell_items[static_cast<size_t>(item)];
                        if (skip_claimed && claimed[d]) {
                            continue;
                        }
                        const int octave_offset = keypoints[d].octave - octave;
                        if (((octave_offset < 0) ? -octave_offset : octave_offset) > tracker::octave_match_window) {
                            continue;
                        }
                        const float candidate_distance_squared = tracker::distance_squared(x, y, keypoints[d].x, keypoints[d].y);
                        if (candidate_distance_squared > radius_squared) {
                            continue;
                        }
                        const unsigned int hamming = ::match::distance::hamming::distance(query, descriptors[d]);
                        if ((hamming < best.hamming) || ((hamming == best.hamming) && (candidate_distance_squared < best.distance_squared))) {
                            second = best;
                            best = candidate{ d, hamming, candidate_distance_squared };
                        }
                        else if (hamming < second.hamming) {
                            second = candidate{ d, hamming, candidate_distance_squared };
                        }
                    }
                }
            }
        };
        const bool predict = this->settings.association != association_kind::optical_flow;
        const unsigned int hamming_bound = (this->settings.association == association_kind::descriptor) ? this->settings.match_hamming : this->settings.reacquire_hamming;
        const auto accepted = [&](const candidate& best, const candidate& second) {
            if (!predict) {
                return (best.detection < detection_count) && (best.hamming <= hamming_bound);
            }
            return (best.detection < detection_count) && (best.hamming <= hamming_bound) && ((second.detection == detection_count) || (static_cast<float>(best.hamming) < this->settings.match_ratio * static_cast<float>(second.hamming)));
        };

        const bool follow_image = this->settings.association != association_kind::descriptor;
        std::vector<unsigned char, core::arena_allocator<unsigned char>> observed(this->track_list.size(), static_cast<unsigned char>(0));
        std::vector<unsigned char, core::arena_allocator<unsigned char>> was_active(this->track_list.size(), static_cast<unsigned char>(0));
        for (size_t i = 0; i < this->track_list.size(); ++i) {
            was_active[i] = this->track_list[i].active ? static_cast<unsigned char>(1) : static_cast<unsigned char>(0);
        }
        std::vector<float, core::arena_allocator<float>> flow_from_x;
        std::vector<float, core::arena_allocator<float>> flow_from_y;
        std::vector<float, core::arena_allocator<float>> flow_to_x;
        std::vector<float, core::arena_allocator<float>> flow_to_y;

        if (follow_image) {
            std::vector<size_t, core::arena_allocator<size_t>> active_indices;
            std::vector<float, core::arena_allocator<float>> seeds_x;
            std::vector<float, core::arena_allocator<float>> seeds_y;
            std::vector<float, core::arena_allocator<float>> guesses_x;
            std::vector<float, core::arena_allocator<float>> guesses_y;
            for (size_t i = 0; i < this->track_list.size(); ++i) {
                if (this->track_list[i].active) {
                    active_indices.push_back(i);
                    seeds_x.push_back(this->track_list[i].x);
                    seeds_y.push_back(this->track_list[i].y);
                    float predicted_x = this->track_list[i].x;
                    float predicted_y = this->track_list[i].y;
                    if (this->settings.klt_predict || (this->settings.flow == flow_kind::wavelet)) {
                        this->flow_previous.apply(this->track_list[i].x, this->track_list[i].y, predicted_x, predicted_y);
                    }
                    guesses_x.push_back(predicted_x - this->track_list[i].x);
                    guesses_y.push_back(predicted_y - this->track_list[i].y);
                }
            }
            if (!active_indices.empty()) {
                std::vector<optical_flow::result, core::arena_allocator<optical_flow::result>> results(active_indices.size());
                if (this->settings.anchored_patches && wavelet_path) {
                    const wavelet_flow::options align_options = this->wavelet_options();
                    core::thread_pool::instance().parallel_for(active_indices.size(), 16, [&](const size_t k) {
                        track& current = this->track_list[active_indices[k]];
                        if (!current.wavelet_anchored || current.wavelet_anchored->anchor.levels.empty()) {
                            results[k].x = current.x;
                            results[k].y = current.y;
                            results[k].error = align_options.max_error + 1.0f;
                            results[k].tracked = false;
                            return;
                        }
                        wavelet_flow::result outcome;
                        bool aligned = wavelet_flow::align(wavelet, current.wavelet_anchored->anchor, align_options, current.wavelet_anchored->displacement_x, current.wavelet_anchored->displacement_y, outcome);
                        if (!aligned && ((guesses_x[k] != 0.0f) || (guesses_y[k] != 0.0f))) {
                            float retry_x = current.wavelet_anchored->displacement_x + guesses_x[k];
                            float retry_y = current.wavelet_anchored->displacement_y + guesses_y[k];
                            wavelet_flow::result retried;
                            if (wavelet_flow::align(wavelet, current.wavelet_anchored->anchor, align_options, retry_x, retry_y, retried)) {
                                current.wavelet_anchored->displacement_x = retry_x;
                                current.wavelet_anchored->displacement_y = retry_y;
                                outcome = retried;
                                aligned = true;
                            }
                        }
                        results[k].x = outcome.x;
                        results[k].y = outcome.y;
                        results[k].error = outcome.error;
                        results[k].tracked = aligned;
                    });
                }
                else if (this->settings.anchored_patches) {
                    const patch_flow::options align_options = this->anchor_options();
                    core::thread_pool::instance().parallel_for(active_indices.size(), 16, [&](const size_t k) {
                        track& current = this->track_list[active_indices[k]];
                        if (!current.anchored || (current.anchored->anchor.levels < 1)) {
                            results[k].x = current.x;
                            results[k].y = current.y;
                            results[k].error = align_options.max_error + 1.0f;
                            results[k].tracked = false;
                            return;
                        }
                        patch_flow::result outcome;
                        static_cast<void>(patch_flow::align(pyramid, current.anchored->anchor, align_options, current.anchored->warp, outcome));
                        results[k].x = outcome.x;
                        results[k].y = outcome.y;
                        results[k].error = outcome.error;
                        results[k].tracked = outcome.tracked;
                    });
                }
                else if (wavelet_path) {
                    std::vector<wavelet_flow::result, core::arena_allocator<wavelet_flow::result>> phases(active_indices.size());
                    std::vector<optical_flow::result, core::arena_allocator<optical_flow::result>> reached;
                    if (this->settings.wavelet_seed != wavelet_seed_kind::rest) {
                        reached.resize(active_indices.size());
                        optical_flow::track(this->pyramid_previous, pyramid, seeds_x.data(), seeds_y.data(), active_indices.size(), reached.data(), this->settings.klt_half_window, this->settings.klt_iterations, this->settings.klt_min_eigenvalue, this->settings.klt_max_error, true, this->settings.klt_fb_threshold, nullptr, nullptr, this->settings.klt_damped_steps);
                        for (size_t k = 0; k < active_indices.size(); ++k) {
                            if (reached[k].tracked) {
                                guesses_x[k] = reached[k].x - seeds_x[k];
                                guesses_y[k] = reached[k].y - seeds_y[k];
                            }
                        }
                    }
                    wavelet_flow::track(
                        this->wavelet_previous,
                        wavelet,
                        seeds_x.data(),
                        seeds_y.data(),
                        active_indices.size(),
                        phases.data(),
                        this->wavelet_options(),
                        guesses_x.data(),
                        guesses_y.data()
                    );
                    for (size_t k = 0; k < active_indices.size(); ++k) {
                        results[k].x = phases[k].x;
                        results[k].y = phases[k].y;
                        results[k].error = phases[k].error;
                        results[k].tracked = phases[k].tracked;
                        if (!phases[k].tracked && (this->settings.wavelet_seed == wavelet_seed_kind::klt_fallback) && !reached.empty() && reached[k].tracked) {
                            results[k] = reached[k];
                        }
                    }
                }
                else {
                    optical_flow::track(
                        this->pyramid_previous,
                        pyramid,
                        seeds_x.data(),
                        seeds_y.data(),
                        active_indices.size(),
                        results.data(),
                        ((this->settings.klt_half_window_blur > 0) && (this->carry_ratio_previous < this->settings.blur_carry_ratio)) ? this->settings.klt_half_window_blur : this->settings.klt_half_window,
                        this->settings.klt_iterations,
                        this->settings.klt_min_eigenvalue,
                        this->settings.klt_max_error,
                        true,
                        this->settings.klt_fb_threshold,
                        guesses_x.data(),
                        guesses_y.data(),
                        this->settings.klt_damped_steps
                    );
                }
                const bool check_drift = this->settings.association == association_kind::both;
                size_t carried = 0;
                for (size_t k = 0; k < active_indices.size(); ++k) {
                    track& current = this->track_list[active_indices[k]];
                    if (!results[k].tracked) {
                        current.active = false;
                        continue;
                    }
                    if (check_drift) {
                        candidate best;
                        candidate second;
                        search(results[k].x, results[k].y, this->settings.drift_radius, current.octave, current.descriptor, false, best, second);
                        if (best.detection < detection_count) {
                            if (best.hamming <= this->settings.match_hamming) {
                                current.descriptor = descriptors[best.detection];
                            }
                            else if ((this->settings.drift_hamming > 0) && (best.hamming > this->settings.drift_hamming)) {
                                current.active = false;
                                continue;
                            }
                        }
                    }
                    flow_from_x.push_back(current.x);
                    flow_from_y.push_back(current.y);
                    flow_to_x.push_back(results[k].x);
                    flow_to_y.push_back(results[k].y);
                    this->observe(current, frame_id, results[k].x, results[k].y);
                    if (this->settings.anchored_patches && (this->settings.anchor_refresh_error > 0.0f) && (results[k].error > this->settings.anchor_refresh_error)) {
                        current.anchored.reset();
                        current.wavelet_anchored.reset();
                    }
                    observed[active_indices[k]] = static_cast<unsigned char>(1);
                    ++carried;
                }
                this->carry_ratio_previous = static_cast<float>(carried) / static_cast<float>(active_indices.size());
                core::logger::log(core::logger::level::debug, "Carried %zu of %zu active tracks (%.0f%%).", carried, active_indices.size(), 100.0 * static_cast<double>(this->carry_ratio_previous));
            }
        }
        else {
            for (size_t i = 0; i < this->track_list.size(); ++i) {
                const track& current = this->track_list[i];
                if (!current.active) {
                    continue;
                }
                float coarse_x = current.x;
                float coarse_y = current.y;
                this->flow_previous.apply(current.x, current.y, coarse_x, coarse_y);
                candidate best;
                candidate second;
                search(coarse_x, coarse_y, this->settings.match_search_radius, current.octave, current.descriptor, false, best, second);
                if (accepted(best, second)) {
                    flow_from_x.push_back(current.x);
                    flow_from_y.push_back(current.y);
                    flow_to_x.push_back(keypoints[best.detection].x);
                    flow_to_y.push_back(keypoints[best.detection].y);
                }
            }
        }
        dominant_flow::affine flow;
        {
            dominant_flow::options fit_options;
            fit_options.sigma = 3.0f;
            if (!dominant_flow::fit(flow_from_x.data(), flow_from_y.data(), flow_to_x.data(), flow_to_y.data(), flow_from_x.size(), fit_options, flow)) {
                flow = this->flow_previous;
            }
            this->flow_previous = flow;
        }

        size_t matched_by_descriptor = 0;
        size_t lost = 0;
        for (size_t i = 0; i < this->track_list.size(); ++i) {
            track& current = this->track_list[i];
            if (observed[i]) {
                continue;
            }
            float predicted_x = current.x;
            float predicted_y = current.y;
            if (predict) {
                flow.apply(current.x, current.y, predicted_x, predicted_y);
            }
            const float radius = (predict && was_active[i]) ? this->settings.match_radius : this->settings.reacquire_radius;
            candidate best;
            candidate second;
            search(predicted_x, predicted_y, radius, current.octave, current.descriptor, true, best, second);
            if (!accepted(best, second)) {
                lost += was_active[i];
                current.active = false;
                continue;
            }
            ++matched_by_descriptor;
            this->observe(current, frame_id, keypoints[best.detection].x, keypoints[best.detection].y);
            if (this->settings.anchored_patches) {
                current.anchored.reset();
                current.wavelet_anchored.reset();
            }
            if (!follow_image) {
                current.descriptor = descriptors[best.detection];
            }
            claimed[best.detection] = static_cast<unsigned char>(1);
            observed[i] = static_cast<unsigned char>(1);
        }

        core::logger::log(core::logger::level::debug, "Tracker: %zu flow pairs, flow (%.1f, %.1f), %zu tracks matched by descriptor, %zu active tracks lost, %zu detections.", flow_from_x.size(), static_cast<double>(flow.bx), static_cast<double>(flow.by), matched_by_descriptor, lost, detection_count);

        this->prune_collisions();

        size_t write = 0;
        for (size_t read = 0; read < this->track_list.size(); ++read) {
            track& current = this->track_list[read];
            if (!current.active) {
                current.missed += 1;
                if (current.missed > this->settings.max_missed) {
                    continue;
                }
            }
            if (write != read) {
                this->track_list[write] = static_cast<track&&>(current);
            }
            ++write;
        }
        this->track_list.resize(write);

        for (size_t d = 0; d < detection_count; ++d) {
            if (claimed[d]) {
                continue;
            }
            if (this->near_existing_track(keypoints[d].x, keypoints[d].y)) {
                continue;
            }
            this->spawn(frame_id, keypoints[d], descriptors[d]);
            claimed[d] = static_cast<unsigned char>(1);
        }

        if (this->settings.anchored_patches) {
            if (wavelet_path) {
                this->build_wavelet_anchors(wavelet);
            }
            else {
                this->build_anchors(pyramid);
            }
        }

        this->pyramid_previous = pyramid;
        this->wavelet_previous = static_cast<image::quaternion_wavelet&&>(wavelet);
    }

    const std::vector<tracker::track>& tracker::tracks() const {
        return this->track_list;
    }

    std::vector<tracker::track*> tracker::active_tracks() {
        std::vector<track*> result;
        for (size_t i = 0; i < this->track_list.size(); ++i) {
            if (this->track_list[i].active) {
                result.push_back(&this->track_list[i]);
            }
        }
        return result;
    }

    std::vector<tracker::track*> tracker::all_tracks() {
        std::vector<track*> result;
        for (size_t i = 0; i < this->track_list.size(); ++i) {
            result.push_back(&this->track_list[i]);
        }
        return result;
    }

    tracker::track* tracker::find(int track_id) {
        for (size_t i = 0; i < this->track_list.size(); ++i) {
            if (this->track_list[i].id == track_id) {
                return &this->track_list[i];
            }
        }
        return nullptr;
    }
}
