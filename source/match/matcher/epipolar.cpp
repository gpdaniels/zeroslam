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

#include "match/matcher/epipolar.hpp"

#include "core/assert.hpp"
#include "geometry/fundamental.hpp"
#include "match/distance/hamming.hpp"
#include "math/math.hpp"

namespace match::matcher {
    double epipolar::reduce_angle(double angle) {
        constexpr static const double pi = math::pi<double>();
        double reduced = math::fmod(angle, pi);
        if (reduced < 0.0) {
            reduced += pi;
        }
        if (!(reduced < pi)) {
            reduced = 0.0;
        }
        return reduced;
    }

    double epipolar::line_angle(double line_a, double line_b) {
        return epipolar::reduce_angle(math::atan2(line_a, -line_b));
    }

    epipolar::index::index()
        : root(index::npos)
        , fundamental_matrix{ { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } }
        , epipole_x(0.0)
        , epipole_y(0.0)
        , normal_x(0.0)
        , normal_y(0.0)
        , epipole_infinite(false)
        , valid(false) {
    }

    void epipolar::index::reserve(const size_t points_size) {
        this->point_x.reserve(points_size);
        this->point_y.reserve(points_size);
        this->point_tolerance.reserve(points_size);
        this->interval_start.reserve(points_size * 2);
        this->interval_end.reserve(points_size * 2);
        this->interval_point.reserve(points_size * 2);
        this->order.reserve(points_size * 2);
        this->partition_scratch.reserve(points_size * 2);
        this->nodes.reserve(points_size * 2);
        this->always.reserve(points_size);
    }

    bool epipolar::index::build(
        const feature::point* __restrict const points,
        const size_t points_size,
        const math::matrix<double, 3, 3>& fundamental,
        const float tolerance,
        const float* __restrict const tolerances
    ) {
        this->clear();

        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                this->fundamental_matrix[row][col] = fundamental[row][col];
            }
        }

        double homogeneous_x = 0.0;
        double homogeneous_y = 0.0;
        double homogeneous_z = 0.0;
        if (!geometry::fundamental<double>::epipole_rhs(fundamental, homogeneous_x, homogeneous_y, homogeneous_z)) {
            return false;
        }

        const double homogeneous_norm = math::sqrt(homogeneous_x * homogeneous_x + homogeneous_y * homogeneous_y + homogeneous_z * homogeneous_z);
        if (math::abs(homogeneous_z) <= (epipolar::infinity_tolerance * homogeneous_norm)) {
            // Every epipolar line is orthogonal to the direction of the epipole at infinity, so they share one normal.
            const double direction_norm = math::sqrt(homogeneous_x * homogeneous_x + homogeneous_y * homogeneous_y);
            if (direction_norm <= 0.0) {
                return false;
            }
            this->epipole_infinite = true;
            this->normal_x = -homogeneous_y / direction_norm;
            this->normal_y = +homogeneous_x / direction_norm;
        }
        else {
            this->epipole_infinite = false;
            this->epipole_x = homogeneous_x / homogeneous_z;
            this->epipole_y = homogeneous_y / homogeneous_z;
        }

        this->point_x.resize(points_size);
        this->point_y.resize(points_size);
        this->point_tolerance.resize(points_size);
        for (size_t i = 0; i < points_size; ++i) {
            this->point_x[i] = static_cast<double>(points[i].x);
            this->point_y[i] = static_cast<double>(points[i].y);
            const double tolerance_value = static_cast<double>((tolerances != nullptr) ? tolerances[i] : tolerance);
            if (!(tolerance_value > 0.0)) {
                this->clear();
                return false;
            }
            this->point_tolerance[i] = tolerance_value;
        }

        if (!this->epipole_infinite) {
            constexpr static const double pi = math::pi<double>();
            for (size_t i = 0; i < points_size; ++i) {
                const double delta_x = this->point_x[i] - this->epipole_x;
                const double delta_y = this->point_y[i] - this->epipole_y;
                const double radius = math::sqrt(delta_x * delta_x + delta_y * delta_y);
                if (radius <= this->point_tolerance[i]) {
                    // The tolerance disc contains the epipole, so every epipolar line intersects it.
                    this->always.push_back(i);
                    continue;
                }
                const double half_angle = math::asin(this->point_tolerance[i] / radius);
                ASSERT(half_angle < (pi / 2.0), "Angular half-width must be below pi/2 away from the epipole.");
                const double centre = epipolar::reduce_angle(math::atan2(delta_y, delta_x));
                const double start = centre - half_angle;
                const double end = centre + half_angle;
                // An interval running off either end of [0, pi) is split into its two disjoint pieces.
                if (start < 0.0) {
                    this->push_interval(0.0, end, i);
                    this->push_interval(start + pi, pi, i);
                }
                else if (end > pi) {
                    this->push_interval(start, pi, i);
                    this->push_interval(0.0, end - pi, i);
                }
                else {
                    this->push_interval(start, end, i);
                }
            }
        }
        else {
            for (size_t i = 0; i < points_size; ++i) {
                const double offset = (this->normal_x * this->point_x[i]) + (this->normal_y * this->point_y[i]);
                this->push_interval(offset - this->point_tolerance[i], offset + this->point_tolerance[i], i);
            }
        }

        const size_t interval_count = this->interval_start.size();
        this->order.resize(interval_count);
        for (size_t i = 0; i < interval_count; ++i) {
            this->order[i] = i;
        }
        this->partition_scratch.resize(interval_count);
        this->sort_order_by_midpoint();
        this->root = this->build_subtree(0, interval_count);

        this->valid = true;
        return true;
    }

    bool epipolar::index::is_valid() const {
        return this->valid;
    }

    bool epipolar::index::epipole_at_infinity() const {
        return this->epipole_infinite;
    }

    void epipolar::index::epipole(double& x, double& y) const {
        x = this->epipole_x;
        y = this->epipole_y;
    }

    size_t epipolar::index::size() const {
        return this->point_x.size();
    }

    size_t epipolar::index::query(
        const float query_x,
        const float query_y,
        size_t* __restrict const candidates,
        const size_t candidates_size
    ) const {
        if (!this->valid || (candidates_size == 0)) {
            return 0;
        }

        double line[3];
        this->epipolar_line(query_x, query_y, line);
        const double line_normal = math::sqrt(line[0] * line[0] + line[1] * line[1]);
        const double line_norm = math::sqrt(line_normal * line_normal + line[2] * line[2]);
        if (line_normal <= (epipolar::line_tolerance * line_norm)) {
            return this->all_candidates(candidates, candidates_size);
        }

        double key;
        if (!this->epipole_infinite) {
            key = epipolar::line_angle(line[0], line[1]);
        }
        else {
            const double lambda = (line[0] * this->normal_x) + (line[1] * this->normal_y);
            if (lambda == 0.0) {
                return this->all_candidates(candidates, candidates_size);
            }
            key = -line[2] / lambda;
        }

        size_t count = 0;
        size_t current = this->root;
        while (current != index::npos) {
            const node& current_node = this->nodes[current];
            if ((key >= current_node.min_start) && (key <= current_node.max_end)) {
                for (size_t i = current_node.bucket_begin; i < current_node.bucket_end; ++i) {
                    const size_t interval = this->order[i];
                    if ((key >= this->interval_start[interval]) && (key <= this->interval_end[interval])) {
                        if (count < candidates_size) {
                            candidates[count] = this->interval_point[interval];
                            ++count;
                        }
                    }
                }
            }
            if (key < current_node.split) {
                current = current_node.left;
            }
            else if (key > current_node.split) {
                current = current_node.right;
            }
            else {
                // A key equal to the split is contained only by intervals in this bucket.
                break;
            }
        }

        for (size_t i = 0; i < this->always.size(); ++i) {
            if (count < candidates_size) {
                candidates[count] = this->always[i];
                ++count;
            }
        }

        index::sort_ascending(candidates, count);
        return count;
    }

    size_t epipolar::index::query_linear(
        const float query_x,
        const float query_y,
        size_t* __restrict const candidates,
        const size_t candidates_size
    ) const {
        if (!this->valid || (candidates_size == 0)) {
            return 0;
        }

        double line[3];
        this->epipolar_line(query_x, query_y, line);
        const double line_normal = math::sqrt(line[0] * line[0] + line[1] * line[1]);
        const double line_norm = math::sqrt(line_normal * line_normal + line[2] * line[2]);
        if (line_normal <= (epipolar::line_tolerance * line_norm)) {
            return this->all_candidates(candidates, candidates_size);
        }

        size_t count = 0;
        for (size_t i = 0; i < this->point_x.size(); ++i) {
            const double distance = math::abs((line[0] * this->point_x[i]) + (line[1] * this->point_y[i]) + line[2]) / line_normal;
            if (distance <= this->point_tolerance[i]) {
                if (count < candidates_size) {
                    candidates[count] = i;
                    ++count;
                }
            }
        }
        return count;
    }

    void epipolar::index::epipolar_line(const float query_x, const float query_y, double* __restrict const line) const {
        const double point[3] = { static_cast<double>(query_x), static_cast<double>(query_y), 1.0 };
        for (size_t row = 0; row < 3; ++row) {
            line[row] = (this->fundamental_matrix[row][0] * point[0]) + (this->fundamental_matrix[row][1] * point[1]) + (this->fundamental_matrix[row][2] * point[2]);
        }
    }

    void epipolar::index::clear() {
        this->point_x.clear();
        this->point_y.clear();
        this->point_tolerance.clear();
        this->interval_start.clear();
        this->interval_end.clear();
        this->interval_point.clear();
        this->order.clear();
        this->partition_scratch.clear();
        this->nodes.clear();
        this->always.clear();
        this->root = index::npos;
        this->epipole_x = 0.0;
        this->epipole_y = 0.0;
        this->normal_x = 0.0;
        this->normal_y = 0.0;
        this->epipole_infinite = false;
        this->valid = false;
    }

    void epipolar::index::push_interval(const double start, const double end, const size_t point_index) {
        this->interval_start.push_back(start);
        this->interval_end.push_back(end);
        this->interval_point.push_back(point_index);
    }

    double epipolar::index::midpoint(const size_t interval) const {
        return (this->interval_start[interval] + this->interval_end[interval]) * 0.5;
    }

    size_t epipolar::index::all_candidates(size_t* __restrict const candidates, const size_t candidates_size) const {
        const size_t count = math::min(this->point_x.size(), candidates_size);
        for (size_t i = 0; i < count; ++i) {
            candidates[i] = i;
        }
        return count;
    }

    void epipolar::index::sift_down(size_t* __restrict const heap, const size_t heap_size, size_t heap_root) {
        for (;;) {
            size_t largest = heap_root;
            const size_t left = (heap_root * 2) + 1;
            const size_t right = left + 1;
            if ((left < heap_size) && (heap[left] > heap[largest])) {
                largest = left;
            }
            if ((right < heap_size) && (heap[right] > heap[largest])) {
                largest = right;
            }
            if (largest == heap_root) {
                return;
            }
            const size_t swap = heap[heap_root];
            heap[heap_root] = heap[largest];
            heap[largest] = swap;
            heap_root = largest;
        }
    }

    void epipolar::index::sort_ascending(size_t* __restrict const values, const size_t values_size) {
        constexpr static const size_t insertion_sort_threshold = 24;
        if (values_size <= insertion_sort_threshold) {
            for (size_t i = 1; i < values_size; ++i) {
                const size_t value = values[i];
                size_t j = i;
                while ((j > 0) && (values[j - 1] > value)) {
                    values[j] = values[j - 1];
                    --j;
                }
                values[j] = value;
            }
            return;
        }
        for (size_t i = values_size / 2; i > 0; --i) {
            index::sift_down(values, values_size, i - 1);
        }
        for (size_t i = values_size; i > 1; --i) {
            const size_t swap = values[0];
            values[0] = values[i - 1];
            values[i - 1] = swap;
            index::sift_down(values, i - 1, 0);
        }
    }

    void epipolar::index::sort_order_by_midpoint() {
        // Bottom up stable merge sort through the scratch buffer, so a rebuild reproduces the same tree.
        const size_t count = this->order.size();
        for (size_t width = 1; width < count; width *= 2) {
            for (size_t begin = 0; begin < count; begin += (width * 2)) {
                const size_t middle = math::min(begin + width, count);
                const size_t end = math::min(begin + (width * 2), count);
                size_t lhs = begin;
                size_t rhs = middle;
                size_t write = begin;
                while ((lhs < middle) && (rhs < end)) {
                    if (this->midpoint(this->order[rhs]) < this->midpoint(this->order[lhs])) {
                        this->partition_scratch[write++] = this->order[rhs++];
                    }
                    else {
                        this->partition_scratch[write++] = this->order[lhs++];
                    }
                }
                while (lhs < middle) {
                    this->partition_scratch[write++] = this->order[lhs++];
                }
                while (rhs < end) {
                    this->partition_scratch[write++] = this->order[rhs++];
                }
                for (size_t i = begin; i < end; ++i) {
                    this->order[i] = this->partition_scratch[i];
                }
            }
        }
    }

    size_t epipolar::index::build_subtree(const size_t begin, const size_t end) {
        if (begin == end) {
            return index::npos;
        }

        // The split is the midpoint of the median interval, which always contains it, so the bucket is never empty.
        const size_t median = begin + ((end - begin) / 2);
        const double split = this->midpoint(this->order[median]);

        size_t left_count = 0;
        size_t bucket_count = 0;
        for (size_t i = begin; i < end; ++i) {
            const size_t interval = this->order[i];
            if (this->interval_end[interval] < split) {
                ++left_count;
            }
            else if (!(this->interval_start[interval] > split)) {
                ++bucket_count;
            }
        }
        size_t write_left = begin;
        size_t write_bucket = begin + left_count;
        size_t write_right = begin + left_count + bucket_count;
        for (size_t i = begin; i < end; ++i) {
            const size_t interval = this->order[i];
            if (this->interval_end[interval] < split) {
                this->partition_scratch[write_left++] = interval;
            }
            else if (this->interval_start[interval] > split) {
                this->partition_scratch[write_right++] = interval;
            }
            else {
                this->partition_scratch[write_bucket++] = interval;
            }
        }
        for (size_t i = begin; i < end; ++i) {
            this->order[i] = this->partition_scratch[i];
        }

        const size_t bucket_begin = begin + left_count;
        const size_t bucket_end = bucket_begin + bucket_count;
        ASSERT(bucket_count > 0, "A centered interval tree node must own at least one interval.");

        double min_start = this->interval_start[this->order[bucket_begin]];
        double max_end = this->interval_end[this->order[bucket_begin]];
        for (size_t i = bucket_begin + 1; i < bucket_end; ++i) {
            const size_t interval = this->order[i];
            min_start = math::min(min_start, this->interval_start[interval]);
            max_end = math::max(max_end, this->interval_end[interval]);
        }

        const size_t current = this->nodes.size();
        this->nodes.push_back(node{ split, min_start, max_end, bucket_begin, bucket_end, index::npos, index::npos });
        const size_t left = this->build_subtree(begin, bucket_begin);
        const size_t right = this->build_subtree(bucket_end, end);
        this->nodes[current].left = left;
        this->nodes[current].right = right;
        return current;
    }

    size_t epipolar::find_matches(
        const feature::point* __restrict const lhs_points,
        const feature::descriptor::binary<256>* __restrict const lhs_descriptors,
        const size_t lhs_descriptors_size,
        const feature::descriptor::binary<256>* __restrict const rhs_descriptors,
        const size_t rhs_descriptors_size,
        const index& rhs_index,
        const float threshold,
        const size_t matches_count,
        pair* __restrict const matches,
        const size_t matches_size
    ) {
        if ((matches_count == 0) || (matches_size == 0)) {
            return 0;
        }
        if (!rhs_index.is_valid()) {
            return 0;
        }
        ASSERT(rhs_index.size() == rhs_descriptors_size, "The index must be built over the same right-hand keypoints as these descriptors.");

        std::vector<size_t> candidates(rhs_descriptors_size);

        size_t count = 0;
        for (size_t lhs_index = 0; lhs_index < lhs_descriptors_size; ++lhs_index) {
            if (count + matches_count > matches_size) {
                break;
            }
            for (size_t matches_index = 0; matches_index < matches_count; ++matches_index) {
                matches[count + matches_index].lhs_index = lhs_index;
                matches[count + matches_index].score = threshold;
            }
            const size_t candidates_count = rhs_index.query(lhs_points[lhs_index].x, lhs_points[lhs_index].y, candidates.data(), candidates.size());
            for (size_t candidate_index = 0; candidate_index < candidates_count; ++candidate_index) {
                const size_t rhs_index_value = candidates[candidate_index];
                const float score = static_cast<float>(distance::hamming::distance(lhs_descriptors[lhs_index], rhs_descriptors[rhs_index_value]));
                for (size_t matches_index = 0; matches_index < matches_count; ++matches_index) {
                    if (score < matches[count + matches_index].score) {
                        for (size_t shift_index = matches_count - 1; shift_index > matches_index; --shift_index) {
                            matches[count + shift_index].score = matches[count + shift_index - 1].score;
                            matches[count + shift_index].rhs_index = matches[count + shift_index - 1].rhs_index;
                        }
                        matches[count + matches_index].score = score;
                        matches[count + matches_index].rhs_index = rhs_index_value;
                        break;
                    }
                }
            }
            const size_t save_index = count;
            for (size_t matches_index = 0; matches_index < matches_count; ++matches_index) {
                if (matches[save_index + matches_index].score < threshold) {
                    ++count;
                }
            }
        }
        return count;
    }

    size_t epipolar::find_matches(
        const feature::point* __restrict const lhs_points,
        const feature::descriptor::binary<256>* __restrict const lhs_descriptors,
        const size_t lhs_descriptors_size,
        const feature::point* __restrict const rhs_points,
        const feature::descriptor::binary<256>* __restrict const rhs_descriptors,
        const size_t rhs_descriptors_size,
        const math::matrix<double, 3, 3>& fundamental,
        const float tolerance,
        const float* __restrict const tolerances,
        const float threshold,
        const size_t matches_count,
        pair* __restrict const matches,
        const size_t matches_size
    ) {
        epipolar::index rhs_index;
        if (!rhs_index.build(rhs_points, rhs_descriptors_size, fundamental, tolerance, tolerances)) {
            return 0;
        }
        return epipolar::find_matches(
            lhs_points,
            lhs_descriptors,
            lhs_descriptors_size,
            rhs_descriptors,
            rhs_descriptors_size,
            rhs_index,
            threshold,
            matches_count,
            matches,
            matches_size
        );
    }
}
