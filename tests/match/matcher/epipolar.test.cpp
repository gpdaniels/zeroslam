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

#include "core/random_pcg.hpp"
#include "core/timestamp.hpp"
#include "geometry/fundamental.hpp"
#include "match/distance/hamming.hpp"
#include "match/matcher/bruteforce.hpp"
#include "math/lie.hpp"
#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static inline math::matrix<double, 3, 3> make_intrinsics(double fx, double fy, double cx, double cy) {
    return math::matrix<double, 3, 3>{ { { fx, 0.0, cx }, { 0.0, fy, cy }, { 0.0, 0.0, 1.0 } } };
}

static inline math::matrix<double, 3, 3> make_rotation(double axis_x, double axis_y, double axis_z, double angle_radians) {
    const double scale = angle_radians / std::sqrt((axis_x * axis_x) + (axis_y * axis_y) + (axis_z * axis_z));
    return math::so3<double>::rotation(axis_x * scale, axis_y * scale, axis_z * scale).get_matrix();
}

static inline double point_line_distance(const double* line, double x, double y) {
    return std::abs((line[0] * x) + (line[1] * y) + line[2]) / std::sqrt((line[0] * line[0]) + (line[1] * line[1]));
}

static inline double paper_line_angle(const double* line, double epipole_x, double epipole_y, double centre_x, double centre_y) {
    const double scale = ((line[0] * centre_x) + (line[1] * centre_y) + line[2]) / ((line[0] * line[0]) + (line[1] * line[1]));
    const double closest_x = centre_x - (scale * line[0]);
    const double closest_y = centre_y - (scale * line[1]);
    return match::matcher::epipolar::reduce_angle(std::atan2(closest_y - epipole_y, closest_x - epipole_x));
}

static inline double angle_difference(double lhs, double rhs) {
    const double pi = math::pi<double>();
    const double difference = std::abs(lhs - rhs);
    return math::min(difference, pi - difference);
}

static inline bool angular_member(double centre, double half_width, double angle) {
    const double pi = math::pi<double>();
    for (int shift = -3; shift <= 3; ++shift) {
        const double shifted = angle + (static_cast<double>(shift) * pi);
        if ((shifted >= (centre - half_width)) && (shifted <= (centre + half_width))) {
            return true;
        }
    }
    return false;
}

static inline void fill_descriptor(feature::descriptor::binary<256>& value, unsigned int key) {
    for (size_t i = 0; i < sizeof(value.data); ++i) {
        value.data[i] = static_cast<unsigned char>(((key * 2654435761u) + (static_cast<unsigned int>(i) * 40503u) + (key >> 3u)) % 256u);
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    const double pi = math::pi<double>();
    const double image_width = 640.0;
    const double image_height = 480.0;
    const math::matrix<double, 3, 3> intrinsics = make_intrinsics(525.0, 525.0, 320.0, 240.0);

    // Line direction against the closest point formula.
    {
        core::random_pcg random(0x5eed0002ull);
        double worst = 0.0;
        size_t samples = 0;
        for (int trial = 0; trial < 16; ++trial) {
            const math::matrix<double, 3, 3> rotation = make_rotation(random.get_random(-1.0, 1.0), random.get_random(-1.0, 1.0), random.get_random(-1.0, 1.0), random.get_random(-0.6, 0.6));
            const math::matrix<double, 3, 1> translation{ { random.get_random(-0.5, 0.5), random.get_random(-0.5, 0.5), random.get_random(0.2, 0.8) } };
            math::matrix<double, 3, 3> fundamental;
            REQUIRE(geometry::fundamental<double>::from_poses(intrinsics, intrinsics, rotation, translation, fundamental));

            double epipole_x = 0.0;
            double epipole_y = 0.0;
            double epipole_z = 0.0;
            REQUIRE(geometry::fundamental<double>::epipole_rhs(fundamental, epipole_x, epipole_y, epipole_z));
            if (std::abs(epipole_z) <= 0.0) {
                continue;
            }
            epipole_x /= epipole_z;
            epipole_y /= epipole_z;

            std::vector<feature::point> points(1);
            points[0].x = 100.0f;
            points[0].y = 100.0f;
            match::matcher::epipolar::index index;
            REQUIRE(index.build(points.data(), points.size(), fundamental, 3.0f));

            for (int sample = 0; sample < 512; ++sample) {
                const float query_x = static_cast<float>(random.get_random(0.0, image_width));
                const float query_y = static_cast<float>(random.get_random(0.0, image_height));
                double line[3];
                index.epipolar_line(query_x, query_y, line);
                const double closed_form = match::matcher::epipolar::line_angle(line[0], line[1]);
                const double from_paper = paper_line_angle(line, epipole_x, epipole_y, 320.0, 240.0);
                worst = math::max(worst, angle_difference(closed_form, from_paper));
                ++samples;
            }
        }
        REQUIRE(samples > 4096);
        REQUIRE(worst < 1e-9);
    }

    // Candidate sets: tree, linear scan and angular membership agree.
    {
        const size_t point_counts[5] = { 0, 1, 2, 50, 2000 };
        const float tolerances[4] = { 1.0f, 3.0f, 50.0f, 200.0f };
        size_t wrapped_intervals = 0;
        size_t near_epipole_points = 0;
        size_t compared = 0;

        const double translations[4][3] = {
            { 0.0, 0.0, 1.0 },
            { 0.35, -0.15, 0.35 },
            { 0.4, -0.2, 0.05 },
            { 0.4, -0.2, 0.0005 }
        };

        for (size_t count_index = 0; count_index < 5; ++count_index) {
            for (size_t tolerance_index = 0; tolerance_index < 4; ++tolerance_index) {
                for (size_t translation_index = 0; translation_index < 4; ++translation_index) {
                    const size_t point_count = point_counts[count_index];
                    const float tolerance = tolerances[tolerance_index];
                    core::random_pcg random(0x5eed0000ull + (count_index * 977ull) + (tolerance_index * 31ull) + (translation_index * 7919ull));

                    const math::matrix<double, 3, 3> rotation = make_rotation(0.3, -0.8, 0.5, 0.25);
                    const math::matrix<double, 3, 1> translation{ { translations[translation_index][0], translations[translation_index][1], translations[translation_index][2] } };
                    math::matrix<double, 3, 3> fundamental;
                    REQUIRE(geometry::fundamental<double>::from_poses(intrinsics, intrinsics, rotation, translation, fundamental));

                    std::vector<feature::point> points(point_count);
                    for (size_t i = 0; i < point_count; ++i) {
                        points[i].x = static_cast<float>(random.get_random(0.0, image_width));
                        points[i].y = static_cast<float>(random.get_random(0.0, image_height));
                    }

                    match::matcher::epipolar::index index;
                    REQUIRE(index.build(points.data(), points.size(), fundamental, tolerance));
                    REQUIRE(index.is_valid());
                    REQUIRE(index.size() == point_count);

                    double epipole_x = 0.0;
                    double epipole_y = 0.0;
                    index.epipole(epipole_x, epipole_y);
                    const bool at_infinity = index.epipole_at_infinity();

                    if (!at_infinity) {
                        for (size_t i = 0; i < point_count; ++i) {
                            const double delta_x = static_cast<double>(points[i].x) - epipole_x;
                            const double delta_y = static_cast<double>(points[i].y) - epipole_y;
                            const double radius = std::sqrt((delta_x * delta_x) + (delta_y * delta_y));
                            if (radius <= static_cast<double>(tolerance)) {
                                ++near_epipole_points;
                                continue;
                            }
                            const double half_angle = std::asin(static_cast<double>(tolerance) / radius);
                            const double centre = match::matcher::epipolar::reduce_angle(std::atan2(delta_y, delta_x));
                            if (((centre - half_angle) < 0.0) || ((centre + half_angle) > pi)) {
                                ++wrapped_intervals;
                            }
                        }
                    }

                    std::vector<size_t> from_tree(point_count + 1);
                    std::vector<size_t> from_scan(point_count + 1);
                    for (int sample = 0; sample < 64; ++sample) {
                        const float query_x = static_cast<float>(random.get_random(0.0, image_width));
                        const float query_y = static_cast<float>(random.get_random(0.0, image_height));

                        const size_t tree_count = index.query(query_x, query_y, from_tree.data(), from_tree.size());
                        const size_t scan_count = index.query_linear(query_x, query_y, from_scan.data(), from_scan.size());

                        for (size_t i = 1; i < tree_count; ++i) {
                            REQUIRE(from_tree[i - 1] < from_tree[i]);
                        }

                        double line[3];
                        index.epipolar_line(query_x, query_y, line);

                        size_t tree_cursor = 0;
                        size_t scan_cursor = 0;
                        for (size_t i = 0; i < point_count; ++i) {
                            const double distance = point_line_distance(line, static_cast<double>(points[i].x), static_cast<double>(points[i].y));
                            const bool in_tree = (tree_cursor < tree_count) && (from_tree[tree_cursor] == i);
                            const bool in_scan = (scan_cursor < scan_count) && (from_scan[scan_cursor] == i);
                            if (in_tree) {
                                ++tree_cursor;
                            }
                            if (in_scan) {
                                ++scan_cursor;
                            }
                            if (std::abs(distance - static_cast<double>(tolerance)) <= (1e-9 * static_cast<double>(tolerance))) {
                                continue;
                            }
                            const bool expected = (distance <= static_cast<double>(tolerance));
                            REQUIRE(in_tree == expected);
                            REQUIRE(in_scan == expected);

                            if (!at_infinity) {
                                const double delta_x = static_cast<double>(points[i].x) - epipole_x;
                                const double delta_y = static_cast<double>(points[i].y) - epipole_y;
                                const double radius = std::sqrt((delta_x * delta_x) + (delta_y * delta_y));
                                bool angular = true;
                                if (radius > static_cast<double>(tolerance)) {
                                    const double half_angle = std::asin(static_cast<double>(tolerance) / radius);
                                    const double centre = std::atan2(delta_y, delta_x);
                                    angular = angular_member(centre, half_angle, match::matcher::epipolar::line_angle(line[0], line[1]));
                                }
                                REQUIRE(angular == expected);
                            }
                            ++compared;
                        }
                        REQUIRE(tree_cursor == tree_count);
                        REQUIRE(scan_cursor == scan_count);
                    }
                }
            }
        }
        REQUIRE(wrapped_intervals > 0);
        REQUIRE(near_epipole_points > 0);
        REQUIRE(compared > 100000);
    }

    // Guided matching reproduces the brute force matcher on the candidate set, tie breaks included.
    {
        core::random_pcg random(0x5eed0007ull);
        const math::matrix<double, 3, 3> rotation = make_rotation(0.2, 0.9, -0.3, 0.2);
        const math::matrix<double, 3, 1> translation{ { 0.3, 0.1, 0.4 } };
        math::matrix<double, 3, 3> fundamental;
        REQUIRE(geometry::fundamental<double>::from_poses(intrinsics, intrinsics, rotation, translation, fundamental));

        for (int variant = 0; variant < 2; ++variant) {
            const bool force_ties = (variant == 1);
            const size_t lhs_size = 120;
            const size_t rhs_size = 400;
            const float tolerance = 40.0f;
            const float threshold = 200.0f;
            const size_t matches_count = 2;

            std::vector<feature::point> lhs_points(lhs_size);
            std::vector<feature::descriptor::binary<256>> lhs_descriptors(lhs_size);
            for (size_t i = 0; i < lhs_size; ++i) {
                lhs_points[i].x = static_cast<float>(random.get_random(0.0, image_width));
                lhs_points[i].y = static_cast<float>(random.get_random(0.0, image_height));
                fill_descriptor(lhs_descriptors[i], static_cast<unsigned int>(i * 7u) + 1u);
            }
            std::vector<feature::point> rhs_points(rhs_size);
            std::vector<feature::descriptor::binary<256>> rhs_descriptors(rhs_size);
            for (size_t i = 0; i < rhs_size; ++i) {
                rhs_points[i].x = static_cast<float>(random.get_random(0.0, image_width));
                rhs_points[i].y = static_cast<float>(random.get_random(0.0, image_height));
                fill_descriptor(rhs_descriptors[i], force_ties ? 99u : (static_cast<unsigned int>(i * 3u) + 5u));
            }

            match::matcher::epipolar::index index;
            REQUIRE(index.build(rhs_points.data(), rhs_points.size(), fundamental, tolerance));

            std::vector<match::pair> guided(lhs_size * matches_count);
            const size_t guided_count = match::matcher::epipolar::find_matches(
                lhs_points.data(),
                lhs_descriptors.data(),
                lhs_size,
                rhs_descriptors.data(),
                rhs_size,
                index,
                threshold,
                matches_count,
                guided.data(),
                guided.size()
            );

            std::vector<match::pair> expected;
            std::vector<size_t> candidates(rhs_size);
            std::vector<feature::descriptor::binary<256>> compacted(rhs_size);
            std::vector<match::pair> block(matches_count);
            size_t ties_seen = 0;
            for (size_t i = 0; i < lhs_size; ++i) {
                const size_t candidate_count = index.query_linear(lhs_points[i].x, lhs_points[i].y, candidates.data(), candidates.size());
                for (size_t c = 0; c < candidate_count; ++c) {
                    compacted[c] = rhs_descriptors[candidates[c]];
                }
                const size_t block_count = match::matcher::bruteforce::find_matches(
                    &lhs_descriptors[i],
                    1,
                    compacted.data(),
                    candidate_count,
                    threshold,
                    matches_count,
                    block.data(),
                    block.size()
                );
                for (size_t b = 0; b < block_count; ++b) {
                    expected.push_back(match::pair{ i, candidates[block[b].rhs_index], block[b].score });
                }
                if ((block_count >= 2) && (block[0].score == block[1].score)) {
                    ++ties_seen;
                }
            }

            REQUIRE(guided_count == expected.size());
            for (size_t i = 0; i < guided_count; ++i) {
                REQUIRE(guided[i].lhs_index == expected[i].lhs_index);
                REQUIRE(guided[i].rhs_index == expected[i].rhs_index);
                REQUIRE(guided[i].score == expected[i].score);
            }
            REQUIRE(guided_count > 0);
            REQUIRE(ties_seen > 0);

            if (force_ties) {
                for (size_t i = 0; i < lhs_size; ++i) {
                    const size_t candidate_count = index.query_linear(lhs_points[i].x, lhs_points[i].y, candidates.data(), candidates.size());
                    if (candidate_count == 0) {
                        continue;
                    }
                    for (size_t g = 0; g < guided_count; ++g) {
                        if (guided[g].lhs_index == i) {
                            REQUIRE(guided[g].rhs_index == candidates[0]);
                            break;
                        }
                    }
                }
            }
        }
    }

    // A whole image tolerance is identical to the brute force matcher.
    {
        core::random_pcg random(0x5eed0008ull);
        const math::matrix<double, 3, 3> rotation = make_rotation(-0.4, 0.5, 0.7, 0.3);
        const math::matrix<double, 3, 1> translation{ { -0.2, 0.35, 0.5 } };
        math::matrix<double, 3, 3> fundamental;
        REQUIRE(geometry::fundamental<double>::from_poses(intrinsics, intrinsics, rotation, translation, fundamental));

        const size_t lhs_size = 200;
        const size_t rhs_size = 200;
        const float threshold = 256.0f;
        const size_t matches_count = 2;

        std::vector<feature::point> lhs_points(lhs_size);
        std::vector<feature::descriptor::binary<256>> lhs_descriptors(lhs_size);
        for (size_t i = 0; i < lhs_size; ++i) {
            lhs_points[i].x = static_cast<float>(random.get_random(0.0, image_width));
            lhs_points[i].y = static_cast<float>(random.get_random(0.0, image_height));
            fill_descriptor(lhs_descriptors[i], static_cast<unsigned int>(i * 11u) + 3u);
        }
        std::vector<feature::point> rhs_points(rhs_size);
        std::vector<feature::descriptor::binary<256>> rhs_descriptors(rhs_size);
        for (size_t i = 0; i < rhs_size; ++i) {
            rhs_points[i].x = static_cast<float>(random.get_random(0.0, image_width));
            rhs_points[i].y = static_cast<float>(random.get_random(0.0, image_height));
            fill_descriptor(rhs_descriptors[i], static_cast<unsigned int>(i * 5u) + 17u);
        }

        double epipole_x = 0.0;
        double epipole_y = 0.0;
        double epipole_z = 0.0;
        REQUIRE(geometry::fundamental<double>::epipole_rhs(fundamental, epipole_x, epipole_y, epipole_z));
        epipole_x /= epipole_z;
        epipole_y /= epipole_z;
        double furthest = 0.0;
        for (size_t i = 0; i < rhs_size; ++i) {
            const double delta_x = static_cast<double>(rhs_points[i].x) - epipole_x;
            const double delta_y = static_cast<double>(rhs_points[i].y) - epipole_y;
            furthest = math::max(furthest, std::sqrt((delta_x * delta_x) + (delta_y * delta_y)));
        }
        const float tolerance = static_cast<float>((furthest * 10.0) + 1.0);

        std::vector<match::pair> guided(lhs_size * matches_count);
        const size_t guided_count = match::matcher::epipolar::find_matches(
            lhs_points.data(),
            lhs_descriptors.data(),
            lhs_size,
            rhs_points.data(),
            rhs_descriptors.data(),
            rhs_size,
            fundamental,
            tolerance,
            nullptr,
            threshold,
            matches_count,
            guided.data(),
            guided.size()
        );

        std::vector<match::pair> ungated(lhs_size * matches_count);
        const size_t ungated_count = match::matcher::bruteforce::find_matches(
            lhs_descriptors.data(),
            lhs_size,
            rhs_descriptors.data(),
            rhs_size,
            threshold,
            matches_count,
            ungated.data(),
            ungated.size()
        );

        REQUIRE(guided_count == ungated_count);
        REQUIRE(guided_count > 0);
        for (size_t i = 0; i < guided_count; ++i) {
            REQUIRE(guided[i].lhs_index == ungated[i].lhs_index);
            REQUIRE(guided[i].rhs_index == ungated[i].rhs_index);
            REQUIRE(guided[i].score == ungated[i].score);
        }
    }

    // Per keypoint tolerances.
    {
        core::random_pcg random(0x5eed0009ull);
        const math::matrix<double, 3, 3> rotation = make_rotation(0.6, -0.1, 0.4, 0.35);
        const math::matrix<double, 3, 1> translation{ { 0.25, 0.3, 0.45 } };
        math::matrix<double, 3, 3> fundamental;
        REQUIRE(geometry::fundamental<double>::from_poses(intrinsics, intrinsics, rotation, translation, fundamental));

        const size_t point_count = 500;
        const float narrow = 2.0f;
        const float wide = 20.0f;
        std::vector<feature::point> points(point_count);
        std::vector<float> tolerances(point_count);
        for (size_t i = 0; i < point_count; ++i) {
            points[i].x = static_cast<float>(random.get_random(0.0, image_width));
            points[i].y = static_cast<float>(random.get_random(0.0, image_height));
            tolerances[i] = ((i % 2u) == 0u) ? narrow : wide;
        }

        match::matcher::epipolar::index uniform_index;
        match::matcher::epipolar::index mixed_index;
        REQUIRE(uniform_index.build(points.data(), points.size(), fundamental, narrow));
        REQUIRE(mixed_index.build(points.data(), points.size(), fundamental, narrow, tolerances.data()));

        std::vector<size_t> uniform_candidates(point_count);
        std::vector<size_t> mixed_candidates(point_count);
        size_t strictly_larger = 0;
        size_t checked = 0;
        for (int sample = 0; sample < 256; ++sample) {
            const float query_x = static_cast<float>(random.get_random(0.0, image_width));
            const float query_y = static_cast<float>(random.get_random(0.0, image_height));
            const size_t uniform_count = uniform_index.query(query_x, query_y, uniform_candidates.data(), uniform_candidates.size());
            const size_t mixed_count = mixed_index.query(query_x, query_y, mixed_candidates.data(), mixed_candidates.size());

            size_t mixed_cursor = 0;
            for (size_t i = 0; i < uniform_count; ++i) {
                while ((mixed_cursor < mixed_count) && (mixed_candidates[mixed_cursor] < uniform_candidates[i])) {
                    ++mixed_cursor;
                }
                REQUIRE(mixed_cursor < mixed_count);
                REQUIRE(mixed_candidates[mixed_cursor] == uniform_candidates[i]);
            }
            if (mixed_count > uniform_count) {
                ++strictly_larger;
            }

            double line[3];
            mixed_index.epipolar_line(query_x, query_y, line);
            size_t cursor = 0;
            for (size_t i = 0; i < point_count; ++i) {
                const double distance = point_line_distance(line, static_cast<double>(points[i].x), static_cast<double>(points[i].y));
                const bool in_mixed = (cursor < mixed_count) && (mixed_candidates[cursor] == i);
                if (in_mixed) {
                    ++cursor;
                }
                if (std::abs(distance - static_cast<double>(tolerances[i])) <= (1e-9 * static_cast<double>(tolerances[i]))) {
                    continue;
                }
                REQUIRE(in_mixed == (distance <= static_cast<double>(tolerances[i])));
                ++checked;
            }
        }
        REQUIRE(strictly_larger > 0);
        REQUIRE(checked > 10000);
    }

    // Epipole at infinity.
    {
        core::random_pcg random(0x5eed0010ull);
        size_t compared = 0;
        for (int trial = 0; trial < 8; ++trial) {
            const math::matrix<double, 3, 3> rotation = make_rotation(random.get_random(-1.0, 1.0), random.get_random(-1.0, 1.0), random.get_random(-1.0, 1.0), random.get_random(-0.5, 0.5));
            const math::matrix<double, 3, 1> translation{ { random.get_random(0.2, 0.8), random.get_random(-0.8, 0.8), 0.0 } };
            math::matrix<double, 3, 3> fundamental;
            REQUIRE(geometry::fundamental<double>::from_poses(intrinsics, intrinsics, rotation, translation, fundamental));

            const size_t point_count = 600;
            const float tolerance = 5.0f;
            std::vector<feature::point> points(point_count);
            for (size_t i = 0; i < point_count; ++i) {
                points[i].x = static_cast<float>(random.get_random(0.0, image_width));
                points[i].y = static_cast<float>(random.get_random(0.0, image_height));
            }

            match::matcher::epipolar::index index;
            REQUIRE(index.build(points.data(), points.size(), fundamental, tolerance));
            REQUIRE(index.epipole_at_infinity());

            std::vector<size_t> from_tree(point_count);
            for (int sample = 0; sample < 64; ++sample) {
                const float query_x = static_cast<float>(random.get_random(0.0, image_width));
                const float query_y = static_cast<float>(random.get_random(0.0, image_height));
                const size_t tree_count = index.query(query_x, query_y, from_tree.data(), from_tree.size());
                double line[3];
                index.epipolar_line(query_x, query_y, line);
                size_t cursor = 0;
                for (size_t i = 0; i < point_count; ++i) {
                    const double distance = point_line_distance(line, static_cast<double>(points[i].x), static_cast<double>(points[i].y));
                    const bool in_tree = (cursor < tree_count) && (from_tree[cursor] == i);
                    if (in_tree) {
                        ++cursor;
                    }
                    if (std::abs(distance - static_cast<double>(tolerance)) <= (1e-9 * static_cast<double>(tolerance))) {
                        continue;
                    }
                    REQUIRE(in_tree == (distance <= static_cast<double>(tolerance)));
                    ++compared;
                }
                REQUIRE(cursor == tree_count);
            }
        }
        REQUIRE(compared > 100000);
    }

    // Degenerate configurations.
    {
        const math::matrix<double, 3, 3> rotation = make_rotation(0.1, 0.2, 0.9, 0.2);
        const math::matrix<double, 3, 1> translation{ { 0.2, 0.1, 0.5 } };
        math::matrix<double, 3, 3> fundamental;
        REQUIRE(geometry::fundamental<double>::from_poses(intrinsics, intrinsics, rotation, translation, fundamental));

        std::vector<feature::point> points(8);
        for (size_t i = 0; i < points.size(); ++i) {
            points[i].x = static_cast<float>(100 + (i * 40));
            points[i].y = static_cast<float>(120 + (i * 20));
        }
        std::vector<size_t> candidates(points.size());

        {
            match::matcher::epipolar::index index;
            REQUIRE(!index.is_valid());
            REQUIRE(index.query(100.0f, 100.0f, candidates.data(), candidates.size()) == 0);
            REQUIRE(index.query_linear(100.0f, 100.0f, candidates.data(), candidates.size()) == 0);
            REQUIRE(index.size() == 0);
        }

        {
            match::matcher::epipolar::index index;
            const math::matrix<double, 3, 3> zero = math::matrix<double, 3, 3>::zero();
            REQUIRE(!index.build(points.data(), points.size(), zero, 3.0f));
            REQUIRE(!index.is_valid());
            REQUIRE(index.query(100.0f, 100.0f, candidates.data(), candidates.size()) == 0);
        }

        {
            match::matcher::epipolar::index index;
            const double lhs[3] = { 0.3, -0.7, 1.1 };
            const double rhs[3] = { 2.0, 0.5, -1.5 };
            math::matrix<double, 3, 3> rank_one;
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    rank_one[row][col] = lhs[row] * rhs[col];
                }
            }
            REQUIRE(!index.build(points.data(), points.size(), rank_one, 3.0f));
        }

        {
            match::matcher::epipolar::index index;
            REQUIRE(!index.build(points.data(), points.size(), fundamental, 0.0f));
            REQUIRE(!index.build(points.data(), points.size(), fundamental, -1.0f));
            std::vector<float> tolerances(points.size(), 3.0f);
            tolerances[3] = 0.0f;
            REQUIRE(!index.build(points.data(), points.size(), fundamental, 3.0f, tolerances.data()));
        }

        {
            match::matcher::epipolar::index index;
            REQUIRE(index.build(nullptr, 0, fundamental, 3.0f));
            REQUIRE(index.is_valid());
            REQUIRE(index.size() == 0);
            REQUIRE(index.query(100.0f, 100.0f, candidates.data(), candidates.size()) == 0);
            REQUIRE(index.query_linear(100.0f, 100.0f, candidates.data(), candidates.size()) == 0);

            match::pair matches[4];
            std::vector<feature::point> lhs_points(2);
            std::vector<feature::descriptor::binary<256>> lhs_descriptors(2);
            REQUIRE(match::matcher::epipolar::find_matches(lhs_points.data(), lhs_descriptors.data(), 0, nullptr, 0, index, 100.0f, 2, &matches[0], 4) == 0);
            REQUIRE(match::matcher::epipolar::find_matches(lhs_points.data(), lhs_descriptors.data(), 2, nullptr, 0, index, 100.0f, 2, &matches[0], 4) == 0);
            REQUIRE(match::matcher::epipolar::find_matches(lhs_points.data(), lhs_descriptors.data(), 2, nullptr, 0, index, 100.0f, 0, &matches[0], 4) == 0);
            REQUIRE(match::matcher::epipolar::find_matches(lhs_points.data(), lhs_descriptors.data(), 2, nullptr, 0, index, 100.0f, 2, &matches[0], 0) == 0);
        }

        {
            const math::matrix<double, 3, 1> forward{ { 0.0, 0.0, 1.0 } };
            math::matrix<double, 3, 3> forward_fundamental;
            REQUIRE(geometry::fundamental<double>::from_poses(intrinsics, intrinsics, math::matrix<double, 3, 3>::identity(), forward, forward_fundamental));
            double epipole_x = 0.0;
            double epipole_y = 0.0;
            double epipole_z = 0.0;
            REQUIRE(geometry::fundamental<double>::epipole_rhs(forward_fundamental, epipole_x, epipole_y, epipole_z));
            epipole_x /= epipole_z;
            epipole_y /= epipole_z;
            REQUIRE(std::abs(epipole_x - 320.0) < 1e-6);
            REQUIRE(std::abs(epipole_y - 240.0) < 1e-6);

            std::vector<feature::point> on_epipole(4);
            for (size_t i = 0; i < on_epipole.size(); ++i) {
                on_epipole[i].x = static_cast<float>(epipole_x + (static_cast<double>(i) * 0.25));
                on_epipole[i].y = static_cast<float>(epipole_y);
            }
            match::matcher::epipolar::index index;
            REQUIRE(index.build(on_epipole.data(), on_epipole.size(), forward_fundamental, 3.0f));
            REQUIRE(!index.epipole_at_infinity());
            std::vector<size_t> found(on_epipole.size());
            for (int sample = 0; sample < 16; ++sample) {
                const float query_x = static_cast<float>(40 + (sample * 30));
                const float query_y = static_cast<float>(60 + (sample * 20));
                const size_t count = index.query(query_x, query_y, found.data(), found.size());
                REQUIRE(count == on_epipole.size());
                for (size_t i = 0; i < count; ++i) {
                    REQUIRE(found[i] == i);
                }
            }
        }

        {
            const math::matrix<double, 3, 1> centre = math::transpose(rotation) * translation;
            const math::matrix<double, 3, 1> epipole_lhs = intrinsics * math::matrix<double, 3, 1>{ { -centre[0], -centre[1], -centre[2] } };
            REQUIRE(std::abs(epipole_lhs[2]) > 1e-12);
            const float query_x = static_cast<float>(epipole_lhs[0] / epipole_lhs[2]);
            const float query_y = static_cast<float>(epipole_lhs[1] / epipole_lhs[2]);

            match::matcher::epipolar::index index;
            REQUIRE(index.build(points.data(), points.size(), fundamental, 3.0f));
            double line[3];
            index.epipolar_line(query_x, query_y, line);
            const double line_normal = std::sqrt((line[0] * line[0]) + (line[1] * line[1]));
            const double line_norm = std::sqrt((line_normal * line_normal) + (line[2] * line[2]));
            if (line_normal <= (match::matcher::epipolar::line_tolerance * line_norm)) {
                const size_t count = index.query(query_x, query_y, candidates.data(), candidates.size());
                REQUIRE(count == points.size());
                REQUIRE(index.query_linear(query_x, query_y, candidates.data(), candidates.size()) == points.size());
            }
        }

        {
            match::matcher::epipolar::index index;
            REQUIRE(index.build(points.data(), points.size(), fundamental, 1.0e6f));
            std::vector<size_t> small(3);
            const size_t count = index.query(200.0f, 200.0f, small.data(), small.size());
            REQUIRE(count == small.size());
            for (size_t i = 1; i < count; ++i) {
                REQUIRE(small[i - 1] < small[i]);
            }
            REQUIRE(index.query(200.0f, 200.0f, small.data(), 0) == 0);
        }
    }

    // Candidate reduction and timing.
    {
        const math::matrix<double, 3, 3> rotation = make_rotation(0.2, 0.8, 0.1, 0.15);
        const math::matrix<double, 3, 1> translation{ { 0.15, 0.05, 0.3 } };
        math::matrix<double, 3, 3> fundamental;
        REQUIRE(geometry::fundamental<double>::from_poses(intrinsics, intrinsics, rotation, translation, fundamental));

        const size_t counts[2] = { 850, 2000 };
        const float tolerance = 3.0f;
        for (size_t index_of_count = 0; index_of_count < 2; ++index_of_count) {
            const size_t count = counts[index_of_count];
            core::random_pcg random(0x5eed0012ull + index_of_count);
            std::vector<feature::point> points(count);
            std::vector<feature::descriptor::binary<256>> descriptors(count);
            for (size_t i = 0; i < count; ++i) {
                points[i].x = static_cast<float>(random.get_random(0.0, image_width));
                points[i].y = static_cast<float>(random.get_random(0.0, image_height));
                fill_descriptor(descriptors[i], static_cast<unsigned int>(i) + 1u);
            }

            const long long int build_start = core::timestamp();
            match::matcher::epipolar::index index;
            REQUIRE(index.build(points.data(), points.size(), fundamental, tolerance));
            const long long int build_end = core::timestamp();

            std::vector<size_t> candidates(count);
            size_t total_candidates = 0;
            const long long int query_start = core::timestamp();
            for (size_t i = 0; i < count; ++i) {
                total_candidates += index.query(points[i].x, points[i].y, candidates.data(), candidates.size());
            }
            const long long int query_end = core::timestamp();

            size_t linear_candidates = 0;
            const long long int linear_start = core::timestamp();
            for (size_t i = 0; i < count; ++i) {
                linear_candidates += index.query_linear(points[i].x, points[i].y, candidates.data(), candidates.size());
            }
            const long long int linear_end = core::timestamp();

            REQUIRE(total_candidates == linear_candidates);

            volatile unsigned int sink = 0;
            const long long int brute_start = core::timestamp();
            for (size_t i = 0; i < count; ++i) {
                for (size_t j = 0; j < count; ++j) {
                    sink = match::distance::hamming::distance(descriptors[i], descriptors[j]);
                }
            }
            const long long int brute_end = core::timestamp();
            static_cast<void>(sink);

            std::printf("Epipolar index (%zu x %zu keypoints, tolerance %.0f px):\n", count, count, static_cast<double>(tolerance));
            std::printf("  Candidates per query:    %.1f of %zu (%.2f%%)\n", static_cast<double>(total_candidates) / static_cast<double>(count), count, (100.0 * static_cast<double>(total_candidates)) / static_cast<double>(count * count));
            std::printf("  Index build:             %lld us\n", (build_end - build_start) / 1000ll);
            std::printf("  Tree queries:            %lld us\n", (query_end - query_start) / 1000ll);
            std::printf("  Linear scan queries:     %lld us\n", (linear_end - linear_start) / 1000ll);
            std::printf("  Brute force descriptors: %lld us\n", (brute_end - brute_start) / 1000ll);
        }
    }

    return EXIT_SUCCESS;
}
