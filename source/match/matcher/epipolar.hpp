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
#ifndef ZEROSLAM_MATCH_MATCHER_EPIPOLAR_HPP
#define ZEROSLAM_MATCH_MATCHER_EPIPOLAR_HPP

#include "feature/descriptor/binary.hpp"
#include "feature/point.hpp"
#include "match/pair.hpp"
#include "math/matrix.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace match::matcher {
    class epipolar final {
    public:
        constexpr static const double infinity_tolerance = 1e-9;
        constexpr static const double line_tolerance = 1e-12;

    public:
        // Reduce an angle to the domain [0, pi) of undirected lines.
        static double reduce_angle(double angle);

        // The direction of the line (a, b, c) on the domain [0, pi).
        static double line_angle(double line_a, double line_b);

    public:
        // Interval tree over the keypoints of the image the epipolar lines land in, answering which keypoints lie within their tolerance of a line.
        class index final {
        private:
            constexpr static const size_t npos = static_cast<size_t>(-1);

            class node final {
            public:
                double split;
                double min_start;
                double max_end;
                size_t bucket_begin;
                size_t bucket_end;
                size_t left;
                size_t right;
            };

        private:
            std::vector<double> point_x;
            std::vector<double> point_y;
            std::vector<double> point_tolerance;
            std::vector<double> interval_start;
            std::vector<double> interval_end;
            std::vector<size_t> interval_point;
            std::vector<size_t> order;
            std::vector<size_t> partition_scratch;
            std::vector<node> nodes;
            size_t root;
            std::vector<size_t> always;
            double fundamental_matrix[3][3];
            double epipole_x;
            double epipole_y;
            double normal_x;
            double normal_y;
            bool epipole_infinite;
            bool valid;

        public:
            index();

        public:
            void reserve(const size_t points_size);

            // Keypoint coordinates, tolerances (pixels) and the fundamental matrix must share the same pyramid level units; false leaves the index unusable.
            bool build(
                const feature::point* __restrict const points,
                const size_t points_size,
                const math::matrix<double, 3, 3>& fundamental,
                const float tolerance,
                const float* __restrict const tolerances = nullptr
            );

            bool is_valid() const;
            bool epipole_at_infinity() const;
            void epipole(double& x, double& y) const;
            size_t size() const;

            // Candidate indices ascending without duplicates, at most candidates_size of them; returns how many were written.
            size_t query(
                const float query_x,
                const float query_y,
                size_t* __restrict const candidates,
                const size_t candidates_size
            ) const;

            // The same candidate set by direct point to line distance over every keypoint.
            size_t query_linear(
                const float query_x,
                const float query_y,
                size_t* __restrict const candidates,
                const size_t candidates_size
            ) const;

            // The epipolar line in this image of a point in the other image, as (a, b, c).
            void epipolar_line(const float query_x, const float query_y, double* __restrict const line) const;

        private:
            void clear();
            void push_interval(const double start, const double end, const size_t point_index);
            double midpoint(const size_t interval) const;
            size_t all_candidates(size_t* __restrict const candidates, const size_t candidates_size) const;
            static void sift_down(size_t* __restrict const heap, const size_t heap_size, size_t heap_root);
            static void sort_ascending(size_t* __restrict const values, const size_t values_size);
            void sort_order_by_midpoint();
            size_t build_subtree(const size_t begin, const size_t end);
        };

    public:
        // Match every left keypoint against the right keypoints its epipolar line passes near; scoring and tie breaks are those of the brute force matcher.
        static size_t find_matches(
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
        );

        // The same, building the index over the right keypoints from the fundamental matrix first.
        static size_t find_matches(
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
        );
    };
}

#endif // ZEROSLAM_MATCH_MATCHER_EPIPOLAR_HPP
