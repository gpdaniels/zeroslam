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

#include "feature/tracker/extrema.hpp"

#include "core/arena.hpp"
#include "core/arena_allocator.hpp"
#include "core/coordinates.hpp"
#include "feature/angle/orb.hpp"
#include "feature/descriptor/orb.hpp"
#include "image/blur.hpp"
#include "image/derivative.hpp"
#include "image/resize.hpp"
#include "match/matcher/bruteforce.hpp"
#include "match/pair.hpp"
#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::tracker {
    namespace {
        template <typename vector_type>
        void ratio_filter(vector_type& matches, const float ratio) {
            size_t write = 0;
            size_t read = 0;
            while (read < matches.size()) {
                size_t group_end = read + 1;
                while ((group_end < matches.size()) && (matches[group_end].lhs_index == matches[read].lhs_index)) {
                    ++group_end;
                }
                if (group_end - read == 1) {
                    matches[write++] = matches[read];
                }
                else if (group_end - read == 2) {
                    const match::pair& best = (matches[read].score <= matches[read + 1].score) ? matches[read] : matches[read + 1];
                    const match::pair& worst = (matches[read].score <= matches[read + 1].score) ? matches[read + 1] : matches[read];
                    if ((worst.score > 0.0f) && (best.score <= ratio * worst.score)) {
                        matches[write++] = best;
                    }
                }
                read = group_end;
            }
            matches.resize(write);
        }
    }

    void extrema::observe(tracker::track& existing, const int frame_id, const float x, const float y) {
        existing.x = core::to_pixel_centre(x);
        existing.y = core::to_pixel_centre(y);
        existing.active = true;
        existing.missed = 0;
        existing.last_frame_id = frame_id;
        existing.length += 1;
        existing.history.push_back(tracker::observation{ frame_id, existing.x, existing.y });
    }

    void extrema::hill_climb(
        const std::int64_t* __restrict const kappa,
        const int width,
        const int height,
        const int border,
        const double sign,
        const double scale,
        const double lambda,
        const double sigma_squared,
        const float predicted_x,
        const float predicted_y,
        const int iteration_limit,
        int& cell_x,
        int& cell_y
    ) {
        const auto evaluate = [&](const int x, const int y) -> double {
            const double normalised = sign * static_cast<double>(kappa[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)]) / scale;
            const double du = static_cast<double>(x) - static_cast<double>(predicted_x);
            const double dv = static_cast<double>(y) - static_cast<double>(predicted_y);
            return normalised + lambda * (sigma_squared / (sigma_squared + du * du + dv * dv));
        };
        cell_x = math::max(border, math::min(width - border - 1, math::round(predicted_x)));
        cell_y = math::max(border, math::min(height - border - 1, math::round(predicted_y)));
        double best = evaluate(cell_x, cell_y);
        constexpr static const int neighbour_x[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
        constexpr static const int neighbour_y[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
        for (int step = 0; step < iteration_limit; ++step) {
            int best_neighbour = -1;
            for (int n = 0; n < 8; ++n) {
                const int candidate_x = cell_x + neighbour_x[n];
                const int candidate_y = cell_y + neighbour_y[n];
                if ((candidate_x < border) || (candidate_x >= width - border) || (candidate_y < border) || (candidate_y >= height - border)) {
                    continue;
                }
                const double value = evaluate(candidate_x, candidate_y);
                if (value > best) {
                    best = value;
                    best_neighbour = n;
                }
            }
            if (best_neighbour < 0) {
                break;
            }
            cell_x += neighbour_x[best_neighbour];
            cell_y += neighbour_y[best_neighbour];
        }
    }

    dominant_flow::affine extrema::small_to_full(const dominant_flow::affine& small, const float ratio_x, const float ratio_y) {
        const float centre_x = (ratio_x - 1.0f) / 2.0f;
        const float centre_y = (ratio_y - 1.0f) / 2.0f;
        dominant_flow::affine full;
        full.a00 = small.a00;
        full.a01 = small.a01 * (ratio_x / ratio_y);
        full.a10 = small.a10 * (ratio_y / ratio_x);
        full.a11 = small.a11;
        full.bx = ratio_x * small.bx + (1.0f - small.a00) * centre_x - small.a01 * (ratio_x / ratio_y) * centre_y;
        full.by = ratio_y * small.by + (1.0f - small.a11) * centre_y - small.a10 * (ratio_y / ratio_x) * centre_x;
        return full;
    }

    extrema::extrema()
        : extrema(options()) {
    }

    extrema::extrema(const options& opts)
        : settings(opts)
        , entries()
        , next_id(0)
        , has_previous(false)
        , small_points_previous()
        , small_descriptors_previous()
        , flow_previous()
        , kappa_previous()
        , response_scale_previous(1)
        , width_previous(0)
        , height_previous(0)
        , last_diagnostics() {
    }

    void extrema::update(const int frame_id, const image::image& image_level0) {
        const int width = static_cast<int>(image_level0.get_cols());
        const int height = static_cast<int>(image_level0.get_rows());
        const size_t plane_size = static_cast<size_t>(width) * static_cast<size_t>(height);
        diagnostics current_diagnostics;
        core::arena::scope scratch;

        std::vector<unsigned char, core::arena_allocator<unsigned char>> blurred(plane_size);
        image::blur::gaussian_7x7(image_level0.get_data(), width, height, width, blurred.data());
        std::vector<std::int16_t, core::arena_allocator<std::int16_t>> dx(plane_size);
        std::vector<std::int16_t, core::arena_allocator<std::int16_t>> dy(plane_size);
        std::vector<std::int16_t, core::arena_allocator<std::int16_t>> dxx(plane_size);
        std::vector<std::int16_t, core::arena_allocator<std::int16_t>> dyy(plane_size);
        std::vector<std::int16_t, core::arena_allocator<std::int16_t>> dxy(plane_size);
        std::vector<std::int64_t> kappa(plane_size);
        image::derivative::curvature(blurred.data(), width, height, width, dx.data(), dy.data(), dxx.data(), dyy.data(), dxy.data(), kappa.data());

        detector::curvature_extrema::options detection = this->settings.detection;
        detection.border = math::max(detection.border, 5);
        std::vector<detector::curvature_extrema::extremum> found_extrema;
        std::int64_t response_threshold = 0;
        detector::curvature_extrema::detect(kappa.data(), width, height, width, detection, found_extrema, &response_threshold);
        current_diagnostics.detected_extrema = found_extrema.size();
        const std::int64_t response_scale = math::max(static_cast<std::int64_t>(1), response_threshold);
        current_diagnostics.response_scale = response_scale;

        const int small_width = math::max(8, width / math::max(1, this->settings.flow_scale));
        const int small_height = math::max(8, height / math::max(1, this->settings.flow_scale));
        std::vector<unsigned char, core::arena_allocator<unsigned char>> small_image(static_cast<size_t>(small_width) * static_cast<size_t>(small_height));
        image::resize::linear(blurred.data(), static_cast<size_t>(width), static_cast<size_t>(height), static_cast<size_t>(small_width), static_cast<size_t>(small_height), small_image.data());
        std::vector<feature::point> small_points;
        std::vector<descriptor::binary<256>> small_descriptors;
        {
            detector::curvature_extrema::options small_detection;
            small_detection.quantile = 0.5f;
            small_detection.subpixel = false;
            small_detection.border = 19;
            small_detection.maximum = this->settings.flow_maximum_features;
            std::vector<std::int64_t> small_kappa;
            std::vector<detector::curvature_extrema::extremum> small_extrema;
            detector::curvature_extrema::detect(small_image.data(), small_width, small_height, small_width, small_detection, small_kappa, small_extrema);
            small_points.reserve(small_extrema.size());
            small_descriptors.reserve(small_extrema.size());
            for (const detector::curvature_extrema::extremum& found : small_extrema) {
                const unsigned char* const at = small_image.data() + static_cast<size_t>(found.cell_y) * static_cast<size_t>(small_width) + static_cast<size_t>(found.cell_x);
                const float angle = feature::angle::orb::dominant_angle(at, small_width);
                feature::point point;
                point.x = static_cast<float>(found.cell_x);
                point.y = static_cast<float>(found.cell_y);
                point.response = static_cast<float>(found.response);
                point.angle = angle;
                point.octave = 0;
                descriptor::binary<256> described;
                descriptor::orb::describe(at, small_width, angle, described);
                small_points.push_back(point);
                small_descriptors.push_back(described);
            }
        }

        dominant_flow::affine flow_full;
        if (this->has_previous) {
            std::vector<match::pair, core::arena_allocator<match::pair>> matches(2 * math::max<size_t>(1, this->small_descriptors_previous.size()));
            const size_t match_count = match::matcher::bruteforce::find_matches(
                this->small_descriptors_previous.data(),
                this->small_descriptors_previous.size(),
                small_descriptors.data(),
                small_descriptors.size(),
                this->settings.flow_match_threshold,
                2,
                matches.data(),
                matches.size()
            );
            matches.resize(match_count);
            ratio_filter(matches, this->settings.flow_ratio);
            current_diagnostics.flow_matches = matches.size();
            std::vector<float, core::arena_allocator<float>> from_x(matches.size());
            std::vector<float, core::arena_allocator<float>> from_y(matches.size());
            std::vector<float, core::arena_allocator<float>> to_x(matches.size());
            std::vector<float, core::arena_allocator<float>> to_y(matches.size());
            for (size_t i = 0; i < matches.size(); ++i) {
                from_x[i] = this->small_points_previous[matches[i].lhs_index].x;
                from_y[i] = this->small_points_previous[matches[i].lhs_index].y;
                to_x[i] = small_points[matches[i].rhs_index].x;
                to_y[i] = small_points[matches[i].rhs_index].y;
            }
            dominant_flow::options fit_options;
            fit_options.sigma = this->settings.flow_sigma;
            dominant_flow::affine flow_small;
            if (dominant_flow::fit(from_x.data(), from_y.data(), to_x.data(), to_y.data(), matches.size(), fit_options, flow_small)) {
                const float ratio_x = static_cast<float>(width) / static_cast<float>(small_width);
                const float ratio_y = static_cast<float>(height) / static_cast<float>(small_height);
                flow_full = extrema::small_to_full(flow_small, ratio_x, ratio_y);
            }
            else {
                flow_full = this->flow_previous;
                current_diagnostics.flow_fallback = true;
            }
        }
        current_diagnostics.flow = flow_full;

        const int border = math::max(2, detection.border);
        std::vector<unsigned char, core::arena_allocator<unsigned char>> claimed(plane_size, static_cast<unsigned char>(0));
        const double lambda = static_cast<double>(this->settings.climb_lambda);
        const double sigma_squared = static_cast<double>(this->settings.climb_sigma) * static_cast<double>(this->settings.climb_sigma);
        const double scale = static_cast<double>(response_scale);
        dominant_flow::affine flow_inverse;
        bool bidirectional_available = false;
        if (this->settings.climb_bidirectional && (this->width_previous == width) && (this->height_previous == height) && (!this->kappa_previous.empty())) {
            bidirectional_available = flow_full.invert(flow_inverse);
        }
        const double scale_previous = static_cast<double>(this->response_scale_previous);
        for (entry& current : this->entries) {
            if (!current.track.active) {
                continue;
            }
            float predicted_x;
            float predicted_y;
            flow_full.apply(core::to_pixel_index_position(current.track.x), core::to_pixel_index_position(current.track.y), predicted_x, predicted_y);
            if ((predicted_x < static_cast<float>(border)) || (predicted_x >= static_cast<float>(width - border)) ||
                (predicted_y < static_cast<float>(border)) || (predicted_y >= static_cast<float>(height - border))) {
                current.track.active = false;
                continue;
            }
            const double sign = static_cast<double>(current.sign);
            int cell_x = 0;
            int cell_y = 0;
            extrema::hill_climb(kappa.data(), width, height, border, sign, scale, lambda, sigma_squared, predicted_x, predicted_y, this->settings.climb_iteration_limit, cell_x, cell_y);
            const float correction_x = static_cast<float>(cell_x) - predicted_x;
            const float correction_y = static_cast<float>(cell_y) - predicted_y;
            if (correction_x * correction_x + correction_y * correction_y > this->settings.maximum_correction * this->settings.maximum_correction) {
                current.track.active = false;
                continue;
            }
            const size_t cell_index = static_cast<size_t>(cell_y) * static_cast<size_t>(width) + static_cast<size_t>(cell_x);
            if (claimed[cell_index]) {
                current.track.active = false;
                continue;
            }
            if (bidirectional_available) {
                float reverse_x;
                float reverse_y;
                flow_inverse.apply(static_cast<float>(cell_x), static_cast<float>(cell_y), reverse_x, reverse_y);
                bool consistent = true;
                if ((reverse_x >= static_cast<float>(border)) && (reverse_x < static_cast<float>(width - border)) &&
                    (reverse_y >= static_cast<float>(border)) && (reverse_y < static_cast<float>(height - border))) {
                    int reverse_cell_x = 0;
                    int reverse_cell_y = 0;
                    extrema::hill_climb(this->kappa_previous.data(), width, height, border, sign, scale_previous, lambda, sigma_squared, reverse_x, reverse_y, this->settings.climb_iteration_limit, reverse_cell_x, reverse_cell_y);
                    float reverse_final_x = static_cast<float>(reverse_cell_x);
                    float reverse_final_y = static_cast<float>(reverse_cell_y);
                    if (this->settings.detection.subpixel) {
                        float offset_x = 0.0f;
                        float offset_y = 0.0f;
                        if (detector::curvature_extrema::refine_subpixel(this->kappa_previous.data(), width, reverse_cell_x, reverse_cell_y, offset_x, offset_y)) {
                            reverse_final_x += offset_x;
                            reverse_final_y += offset_y;
                        }
                    }
                    const float error_x = reverse_final_x - core::to_pixel_index_position(current.track.x);
                    const float error_y = reverse_final_y - core::to_pixel_index_position(current.track.y);
                    consistent = (error_x * error_x + error_y * error_y) <= (this->settings.bidirectional_threshold * this->settings.bidirectional_threshold);
                }
                if (!consistent) {
                    current.track.active = false;
                    ++current_diagnostics.bidirectional_rejections;
                    continue;
                }
            }
            claimed[cell_index] = static_cast<unsigned char>(1);
            float final_x = static_cast<float>(cell_x);
            float final_y = static_cast<float>(cell_y);
            if (this->settings.detection.subpixel) {
                float offset_x = 0.0f;
                float offset_y = 0.0f;
                if (detector::curvature_extrema::refine_subpixel(kappa.data(), width, cell_x, cell_y, offset_x, offset_y)) {
                    final_x += offset_x;
                    final_y += offset_y;
                }
            }
            extrema::observe(current.track, frame_id, final_x, final_y);
        }

        size_t write = 0;
        for (size_t read = 0; read < this->entries.size(); ++read) {
            entry& current = this->entries[read];
            if (!current.track.active) {
                current.track.missed += 1;
                if (current.track.missed > this->settings.maximum_missed) {
                    ++current_diagnostics.lost;
                    continue;
                }
            }
            else if (this->has_previous) {
                ++current_diagnostics.tracked;
            }
            if (write != read) {
                this->entries[write] = static_cast<entry&&>(current);
            }
            ++write;
        }
        this->entries.resize(write);

        {
            const float spacing = math::max(1.0f, this->settings.minimum_spawn_distance);
            const int grid_width = static_cast<int>(static_cast<float>(width) / spacing) + 1;
            const int grid_height = static_cast<int>(static_cast<float>(height) / spacing) + 1;
            std::vector<std::uint32_t, core::arena_allocator<std::uint32_t>> grid(static_cast<size_t>(grid_width) * static_cast<size_t>(grid_height), 0u);
            std::vector<float, core::arena_allocator<float>> occupant_x;
            std::vector<float, core::arena_allocator<float>> occupant_y;
            std::vector<std::uint32_t, core::arena_allocator<std::uint32_t>> occupant_link;
            const auto occupy = [&](const float x, const float y) {
                const int grid_x = math::min(grid_width - 1, static_cast<int>(x / spacing));
                const int grid_y = math::min(grid_height - 1, static_cast<int>(y / spacing));
                const size_t index = static_cast<size_t>(grid_y) * static_cast<size_t>(grid_width) + static_cast<size_t>(grid_x);
                occupant_x.push_back(x);
                occupant_y.push_back(y);
                occupant_link.push_back(grid[index]);
                grid[index] = static_cast<std::uint32_t>(occupant_x.size());
            };
            const auto near_occupied = [&](const float x, const float y) -> bool {
                const int grid_x = math::min(grid_width - 1, static_cast<int>(x / spacing));
                const int grid_y = math::min(grid_height - 1, static_cast<int>(y / spacing));
                for (int gy = math::max(0, grid_y - 1); gy <= math::min(grid_height - 1, grid_y + 1); ++gy) {
                    for (int gx = math::max(0, grid_x - 1); gx <= math::min(grid_width - 1, grid_x + 1); ++gx) {
                        std::uint32_t occupant = grid[static_cast<size_t>(gy) * static_cast<size_t>(grid_width) + static_cast<size_t>(gx)];
                        while (occupant != 0u) {
                            const float du = occupant_x[occupant - 1] - x;
                            const float dv = occupant_y[occupant - 1] - y;
                            if (du * du + dv * dv < spacing * spacing) {
                                return true;
                            }
                            occupant = occupant_link[occupant - 1];
                        }
                    }
                }
                return false;
            };
            for (const entry& current : this->entries) {
                if (current.track.active) {
                    occupy(core::to_pixel_index_position(current.track.x), core::to_pixel_index_position(current.track.y));
                }
            }
            std::vector<size_t, core::arena_allocator<size_t>> order(found_extrema.size());
            for (size_t i = 0; i < order.size(); ++i) {
                order[i] = i;
            }
            std::sort(order.begin(), order.end(), [&found_extrema](const size_t lhs, const size_t rhs) {
                if (found_extrema[lhs].response != found_extrema[rhs].response) {
                    return found_extrema[lhs].response > found_extrema[rhs].response;
                }
                return lhs < rhs;
            });
            for (const size_t index : order) {
                if (this->entries.size() >= this->settings.maximum_tracks) {
                    break;
                }
                const detector::curvature_extrema::extremum& found = found_extrema[index];
                if (claimed[static_cast<size_t>(found.cell_y) * static_cast<size_t>(width) + static_cast<size_t>(found.cell_x)]) {
                    continue;
                }
                if (near_occupied(found.x, found.y)) {
                    continue;
                }
                entry created;
                created.track.id = this->next_id++;
                created.track.landmark_id = -1;
                created.track.active = true;
                created.track.x = core::to_pixel_centre(found.x);
                created.track.y = core::to_pixel_centre(found.y);
                created.track.octave = 0;
                created.track.start_frame_id = frame_id;
                created.track.last_frame_id = frame_id;
                created.track.length = 1;
                created.track.missed = 0;
                created.track.outliers = 0;
                created.track.outlier_frame_id = -1;
                created.sign = found.sign;
                for (size_t byte = 0; byte < sizeof(created.track.descriptor.data); ++byte) {
                    created.track.descriptor.data[byte] = 0;
                }
                const int distance_from_border = this->settings.descriptor_border;
                if ((found.cell_x >= distance_from_border) && (found.cell_x < width - distance_from_border) &&
                    (found.cell_y >= distance_from_border) && (found.cell_y < height - distance_from_border)) {
                    const unsigned char* const at = image_level0.get_data() + static_cast<size_t>(found.cell_y) * static_cast<size_t>(width) + static_cast<size_t>(found.cell_x);
                    const float angle = feature::angle::orb::dominant_angle(at, width);
                    descriptor::orb::describe(at, width, angle, created.track.descriptor);
                }
                created.track.history.push_back(tracker::observation{ frame_id, created.track.x, created.track.y });
                occupy(found.x, found.y);
                this->entries.push_back(static_cast<entry&&>(created));
                ++current_diagnostics.spawned;
            }
        }

        this->small_points_previous = static_cast<std::vector<feature::point>&&>(small_points);
        this->small_descriptors_previous = static_cast<std::vector<descriptor::binary<256>>&&>(small_descriptors);
        this->flow_previous = flow_full;
        if (this->settings.climb_bidirectional) {
            this->kappa_previous = static_cast<std::vector<std::int64_t>&&>(kappa);
            this->response_scale_previous = response_scale;
            this->width_previous = width;
            this->height_previous = height;
        }
        else {
            this->kappa_previous.clear();
            this->width_previous = 0;
            this->height_previous = 0;
        }
        this->has_previous = true;
        this->last_diagnostics = current_diagnostics;
    }

    void extrema::set_options(const options& opts) {
        this->settings = opts;
    }

    const extrema::options& extrema::get_options() const {
        return this->settings;
    }

    const extrema::diagnostics& extrema::get_diagnostics() const {
        return this->last_diagnostics;
    }

    int extrema::sign_of(const int track_id) const {
        for (const entry& current : this->entries) {
            if (current.track.id == track_id) {
                return current.sign;
            }
        }
        return 0;
    }

    std::vector<tracker::track*> extrema::active_tracks() {
        std::vector<tracker::track*> result;
        for (entry& current : this->entries) {
            if (current.track.active) {
                result.push_back(&current.track);
            }
        }
        return result;
    }

    std::vector<tracker::track*> extrema::all_tracks() {
        std::vector<tracker::track*> result;
        for (entry& current : this->entries) {
            result.push_back(&current.track);
        }
        return result;
    }

    tracker::track* extrema::find(const int track_id) {
        for (entry& current : this->entries) {
            if (current.track.id == track_id) {
                return &current.track;
            }
        }
        return nullptr;
    }

    size_t extrema::size() const {
        return this->entries.size();
    }
}
