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

#include "feature/tracker/optical_flow.hpp"
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

    void tracker::spawn(int frame_id, const feature::point& keypoint, const descriptor::binary<256>& descriptor) {
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
        created.descriptor = descriptor;
        created.history.push_back(observation{ frame_id, keypoint.x, keypoint.y });
        this->track_list.push_back(static_cast<track&&>(created));
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
        , has_previous(false)
        , next_id(0) {
    }

    void tracker::set_options(const options& opts) {
        this->settings = opts;
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
        const size_t detection_count = math::min(keypoints.size(), descriptors.size());

        std::vector<unsigned char> claimed(detection_count, static_cast<unsigned char>(0));

        if (!this->has_previous) {
            for (size_t d = 0; d < detection_count; ++d) {
                if (this->near_existing_track(keypoints[d].x, keypoints[d].y)) {
                    continue;
                }
                this->spawn(frame_id, keypoints[d], descriptors[d]);
                claimed[d] = static_cast<unsigned char>(1);
            }
            this->pyramid_previous = pyramid;
            this->has_previous = true;
            return;
        }

        std::vector<size_t> active_indices;
        std::vector<float> seeds_x;
        std::vector<float> seeds_y;
        for (size_t i = 0; i < this->track_list.size(); ++i) {
            if (this->track_list[i].active) {
                active_indices.push_back(i);
                seeds_x.push_back(this->track_list[i].x);
                seeds_y.push_back(this->track_list[i].y);
            }
        }
        if (!active_indices.empty()) {
            std::vector<optical_flow::result> results(active_indices.size());
            optical_flow::track(
                this->pyramid_previous,
                pyramid,
                seeds_x.data(),
                seeds_y.data(),
                active_indices.size(),
                results.data(),
                this->settings.klt_half_window,
                this->settings.klt_iterations,
                this->settings.klt_min_eigenvalue,
                this->settings.klt_max_error,
                true,
                this->settings.klt_fb_threshold
            );
            for (size_t k = 0; k < active_indices.size(); ++k) {
                track& current = this->track_list[active_indices[k]];
                if (results[k].tracked) {
                    this->observe(current, frame_id, results[k].x, results[k].y);
                }
                else {
                    current.active = false;
                }
            }
        }

        const float radius_squared = this->settings.reacquire_radius * this->settings.reacquire_radius;
        for (size_t i = 0; i < this->track_list.size(); ++i) {
            track& current = this->track_list[i];
            if (current.active) {
                continue;
            }
            size_t best_detection = detection_count;
            unsigned int best_hamming = this->settings.reacquire_hamming + 1u;
            float best_distance_squared = radius_squared;
            for (size_t d = 0; d < detection_count; ++d) {
                if (claimed[d]) {
                    continue;
                }
                const int octave_offset = keypoints[d].octave - current.octave;
                const int octave_distance = (octave_offset < 0) ? -octave_offset : octave_offset;
                if (octave_distance > tracker::octave_match_window) {
                    continue;
                }
                const float candidate_distance_squared = tracker::distance_squared(current.x, current.y, keypoints[d].x, keypoints[d].y);
                if (candidate_distance_squared > radius_squared) {
                    continue;
                }
                const unsigned int hamming = ::match::distance::hamming::distance(current.descriptor, descriptors[d]);
                if (hamming > this->settings.reacquire_hamming) {
                    continue;
                }
                if ((hamming < best_hamming) || ((hamming == best_hamming) && (candidate_distance_squared < best_distance_squared))) {
                    best_hamming = hamming;
                    best_distance_squared = candidate_distance_squared;
                    best_detection = d;
                }
            }
            if (best_detection < detection_count) {
                this->observe(current, frame_id, keypoints[best_detection].x, keypoints[best_detection].y);
                claimed[best_detection] = static_cast<unsigned char>(1);
            }
        }

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

        this->pyramid_previous = pyramid;
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
