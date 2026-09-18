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

#include "feature/tracker/line.hpp"

#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::tracker {
    float line::midpoint_distance_squared(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2) {
        const float amx = (ax1 + ax2) * 0.5f;
        const float amy = (ay1 + ay2) * 0.5f;
        const float bmx = (bx1 + bx2) * 0.5f;
        const float bmy = (by1 + by2) * 0.5f;
        const float dx = amx - bmx;
        const float dy = amy - bmy;
        return (dx * dx) + (dy * dy);
    }

    float line::angle_degrees(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2) {
        const float adx = ax2 - ax1;
        const float ady = ay2 - ay1;
        const float bdx = bx2 - bx1;
        const float bdy = by2 - by1;
        const float alength = math::sqrt((adx * adx) + (ady * ady));
        const float blength = math::sqrt((bdx * bdx) + (bdy * bdy));
        if ((alength < 1.0e-6f) || (blength < 1.0e-6f)) {
            return 180.0f;
        }
        const float cosine = math::min(1.0f, math::abs((adx * bdx) + (ady * bdy)) / (alength * blength));
        return math::acos(cosine) * (180.0f / math::pi<float>());
    }

    float line::overlap_fraction(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2) {
        const float adx = ax2 - ax1;
        const float ady = ay2 - ay1;
        const float alength = math::sqrt((adx * adx) + (ady * ady));
        const float blength = math::sqrt(((bx2 - bx1) * (bx2 - bx1)) + ((by2 - by1) * (by2 - by1)));
        if ((alength < 1.0e-6f) || (blength < 1.0e-6f)) {
            return 0.0f;
        }
        const float ux = adx / alength;
        const float uy = ady / alength;
        const float a0 = 0.0f;
        const float a1 = alength;
        const float b0 = ((bx1 - ax1) * ux) + ((by1 - ay1) * uy);
        const float b1 = ((bx2 - ax1) * ux) + ((by2 - ay1) * uy);
        const float min_b = math::min(b0, b1);
        const float max_b = math::max(b0, b1);
        const float overlap = math::min(a1, max_b) - math::max(a0, min_b);
        if (overlap < 0.0f) {
            return 0.0f;
        }
        return overlap / blength;
    }

    void line::spawn(int frame_id, const detector::elsed::segment& segment) {
        track created;
        created.id = this->next_id++;
        created.landmark_id = -1;
        created.active = true;
        created.x1 = segment.x1;
        created.y1 = segment.y1;
        created.x2 = segment.x2;
        created.y2 = segment.y2;
        created.start_frame_id = frame_id;
        created.last_frame_id = frame_id;
        created.length = 1;
        created.missed = 0;
        created.history.push_back(observation{ frame_id, segment.x1, segment.y1, segment.x2, segment.y2 });
        this->track_list.push_back(static_cast<track&&>(created));
    }

    void line::observe(track& existing, int frame_id, const detector::elsed::segment& segment) {
        existing.x1 = segment.x1;
        existing.y1 = segment.y1;
        existing.x2 = segment.x2;
        existing.y2 = segment.y2;
        existing.active = true;
        existing.missed = 0;
        existing.last_frame_id = frame_id;
        existing.length += 1;
        existing.history.push_back(observation{ frame_id, segment.x1, segment.y1, segment.x2, segment.y2 });
    }

    line::line()
        : line(options()) {
    }

    line::line(const options& opts)
        : settings(opts)
        , track_list()
        , next_id(0) {
    }

    void line::set_options(const options& opts) {
        this->settings = opts;
    }

    const line::options& line::get_options() const {
        return this->settings;
    }

    void line::update(int frame_id, const std::vector<detector::elsed::segment>& segments) {
        if (this->track_list.empty()) {
            for (size_t d = 0; d < segments.size(); ++d) {
                if (segments[d].length < this->settings.min_length) {
                    continue;
                }
                this->spawn(frame_id, segments[d]);
            }
            return;
        }

        const size_t detection_count = segments.size();
        std::vector<unsigned char> claimed(detection_count, static_cast<unsigned char>(0));

        for (size_t t = 0; t < this->track_list.size(); ++t) {
            this->track_list[t].active = false;
        }

        struct candidate final {
            size_t track_index;
            size_t detection_index;
            float distance_squared;
        };

        std::vector<candidate> candidates;
        for (size_t t = 0; t < this->track_list.size(); ++t) {
            const track& current = this->track_list[t];
            for (size_t d = 0; d < detection_count; ++d) {
                const detector::elsed::segment& segment = segments[d];
                if (segment.length < this->settings.min_length) {
                    continue;
                }
                const float midpoint_distance_squared = line::midpoint_distance_squared(current.x1, current.y1, current.x2, current.y2, segment.x1, segment.y1, segment.x2, segment.y2);
                if (midpoint_distance_squared > (this->settings.match_midpoint_distance * this->settings.match_midpoint_distance)) {
                    continue;
                }
                const float angle = line::angle_degrees(current.x1, current.y1, current.x2, current.y2, segment.x1, segment.y1, segment.x2, segment.y2);
                if (angle > this->settings.match_angle_tolerance) {
                    continue;
                }
                const float overlap_a = line::overlap_fraction(current.x1, current.y1, current.x2, current.y2, segment.x1, segment.y1, segment.x2, segment.y2);
                if (overlap_a < this->settings.match_overlap) {
                    continue;
                }
                const float overlap_b = line::overlap_fraction(segment.x1, segment.y1, segment.x2, segment.y2, current.x1, current.y1, current.x2, current.y2);
                if (overlap_b < this->settings.match_overlap) {
                    continue;
                }
                candidates.push_back(candidate{ t, d, midpoint_distance_squared });
            }
        }
        std::stable_sort(candidates.begin(), candidates.end(), [](const candidate& lhs, const candidate& rhs) {
            if (lhs.distance_squared != rhs.distance_squared) {
                return lhs.distance_squared < rhs.distance_squared;
            }
            if (lhs.track_index != rhs.track_index) {
                return lhs.track_index < rhs.track_index;
            }
            return lhs.detection_index < rhs.detection_index;
        });
        for (const candidate& match : candidates) {
            if (claimed[match.detection_index]) {
                continue;
            }
            track& current = this->track_list[match.track_index];
            if (current.active) {
                continue;
            }
            this->observe(current, frame_id, segments[match.detection_index]);
            claimed[match.detection_index] = static_cast<unsigned char>(1);
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
            if (segments[d].length < this->settings.min_length) {
                continue;
            }
            this->spawn(frame_id, segments[d]);
        }
    }

    const std::vector<line::track>& line::tracks() const {
        return this->track_list;
    }

    std::vector<line::track*> line::active_tracks() {
        std::vector<track*> result;
        for (size_t i = 0; i < this->track_list.size(); ++i) {
            if (this->track_list[i].active) {
                result.push_back(&this->track_list[i]);
            }
        }
        return result;
    }

    std::vector<line::track*> line::all_tracks() {
        std::vector<track*> result;
        for (size_t i = 0; i < this->track_list.size(); ++i) {
            result.push_back(&this->track_list[i]);
        }
        return result;
    }

    line::track* line::find(int track_id) {
        for (size_t i = 0; i < this->track_list.size(); ++i) {
            if (this->track_list[i].id == track_id) {
                return &this->track_list[i];
            }
        }
        return nullptr;
    }
}
