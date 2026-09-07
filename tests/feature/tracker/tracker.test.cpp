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

#include "feature/angle/orb.hpp"
#include "feature/descriptor/orb.hpp"
#include "image/pyramid.hpp"
#include "match/distance/hamming.hpp"

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

static inline double texture(double x, double y) {
    return 128.0 +
           55.0 * std::sin(0.21 * x) * std::sin(0.19 * y) +
           35.0 * std::cos(0.15 * x + 0.10 * y) +
           25.0 * std::sin(0.30 * x - 0.12 * y);
}

template <typename function_type>
static inline image::image make_image(size_t rows, size_t cols, function_type function) {
    image::image result(rows, cols);
    for (size_t y = 0; y < rows; ++y) {
        for (size_t x = 0; x < cols; ++x) {
            double value = function(static_cast<double>(x), static_cast<double>(y));
            if (value < 0.0) {
                value = 0.0;
            }
            if (value > 255.0) {
                value = 255.0;
            }
            result.get_data()[y * cols + x] = static_cast<unsigned char>(value + 0.5);
        }
    }
    return result;
}

static inline image::image make_frame(size_t dimension, int k, int shift_x, int shift_y) {
    return make_image(dimension, dimension, [=](double x, double y) {
        return texture(x - static_cast<double>(k * shift_x), y - static_cast<double>(k * shift_y));
    });
}

static inline feature::descriptor::binary<256> describe_at(const image::image& img, int x, int y) {
    const int stride = static_cast<int>(img.get_cols());
    const unsigned char* ptr = img.get_data() + static_cast<size_t>(y) * img.get_cols() + static_cast<size_t>(x);
    const float angle = feature::angle::orb::dominant_angle(ptr, stride);
    feature::descriptor::binary<256> descriptor;
    feature::descriptor::orb::describe(ptr, stride, angle, descriptor);
    return descriptor;
}

static inline void occlude(image::image& img, int cx, int cy, int radius) {
    const int cols = static_cast<int>(img.get_cols());
    const int rows = static_cast<int>(img.get_rows());
    for (int y = cy - radius; y <= cy + radius; ++y) {
        if ((y < 0) || (y >= rows)) {
            continue;
        }
        for (int x = cx - radius; x <= cx + radius; ++x) {
            if ((x < 0) || (x >= cols)) {
                continue;
            }
            img.get_data()[static_cast<size_t>(y) * img.get_cols() + static_cast<size_t>(x)] = 128;
        }
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    constexpr static const size_t dimension = 200;
    constexpr static const int shift_x = 2;
    constexpr static const int shift_y = 1;

    {
        constexpr static const int base_x[12] = { 50, 80, 110, 140, 50, 80, 110, 140, 50, 80, 110, 140 };
        constexpr static const int base_y[12] = { 50, 50, 50, 50, 90, 90, 90, 90, 130, 130, 130, 130 };
        constexpr static const size_t point_count = 12;
        constexpr static const int frame_count = 6;

        feature::tracker::tracker manager;

        {
            const image::image image0 = make_frame(dimension, 0, shift_x, shift_y);
            std::vector<feature::point> keypoints;
            std::vector<feature::descriptor::binary<256>> descriptors;
            for (size_t p = 0; p < point_count; ++p) {
                keypoints.push_back(feature::point{ static_cast<float>(base_x[p]), static_cast<float>(base_y[p]), 0.0f, 0.0f, 0 });
                descriptors.push_back(describe_at(image0, base_x[p], base_y[p]));
            }
            manager.update(0, image::pyramid(image0), keypoints, descriptors);
        }
        REQUIRE(manager.tracks().size() == point_count);

        int point_ids[point_count];
        for (size_t p = 0; p < point_count; ++p) {
            point_ids[p] = manager.tracks()[p].id;
        }

        for (int k = 1; k < frame_count; ++k) {
            const image::image image_k = make_frame(dimension, k, shift_x, shift_y);
            std::vector<feature::point> keypoints;
            std::vector<feature::descriptor::binary<256>> descriptors;
            for (size_t p = 0; p < point_count; ++p) {
                const int x = base_x[p] + k * shift_x;
                const int y = base_y[p] + k * shift_y;
                keypoints.push_back(feature::point{ static_cast<float>(x), static_cast<float>(y), 0.0f, 0.0f, 0 });
                descriptors.push_back(describe_at(image_k, x, y));
            }
            manager.update(k, image::pyramid(image_k), keypoints, descriptors);

            for (size_t p = 0; p < point_count; ++p) {
                feature::tracker::tracker::track* const followed = manager.find(point_ids[p]);
                REQUIRE(followed != nullptr);
                REQUIRE(followed->active);
                const float expected_x = static_cast<float>(base_x[p] + k * shift_x);
                const float expected_y = static_cast<float>(base_y[p] + k * shift_y);
                REQUIRE(std::abs(static_cast<double>(followed->x - expected_x)) < 1.5);
                REQUIRE(std::abs(static_cast<double>(followed->y - expected_y)) < 1.5);
            }
        }

        for (size_t p = 0; p < point_count; ++p) {
            feature::tracker::tracker::track* const followed = manager.find(point_ids[p]);
            REQUIRE(followed != nullptr);
            REQUIRE(followed->length == frame_count);
            REQUIRE(followed->history.size() == static_cast<size_t>(frame_count));
        }

        REQUIRE(manager.tracks().size() <= point_count + 2);
        REQUIRE(static_cast<size_t>(point_count + 2) < point_count * static_cast<size_t>(frame_count));
        REQUIRE(manager.tracks().size() == point_count);
    }

    {
        constexpr static const int target_x = 100;
        constexpr static const int target_y = 100;
        constexpr static const int other_x[3] = { 50, 150, 50 };
        constexpr static const int other_y[3] = { 50, 50, 150 };
        constexpr static const size_t other_count = 3;
        constexpr static const size_t total_count = other_count + 1;

        feature::tracker::tracker manager;

        {
            const image::image image0 = make_frame(dimension, 0, shift_x, shift_y);
            std::vector<feature::point> keypoints;
            std::vector<feature::descriptor::binary<256>> descriptors;
            keypoints.push_back(feature::point{ static_cast<float>(target_x), static_cast<float>(target_y), 0.0f, 0.0f, 0 });
            descriptors.push_back(describe_at(image0, target_x, target_y));
            for (size_t p = 0; p < other_count; ++p) {
                keypoints.push_back(feature::point{ static_cast<float>(other_x[p]), static_cast<float>(other_y[p]), 0.0f, 0.0f, 0 });
                descriptors.push_back(describe_at(image0, other_x[p], other_y[p]));
            }
            manager.update(0, image::pyramid(image0), keypoints, descriptors);
        }
        REQUIRE(manager.tracks().size() == total_count);
        const int target_id = manager.tracks()[0].id;

        for (int k = 1; k <= 4; ++k) {
            image::image image_k = make_frame(dimension, k, shift_x, shift_y);
            const bool omit_target = (k == 3);
            if (omit_target) {
                occlude(image_k, target_x + k * shift_x, target_y + k * shift_y, 22);
            }
            std::vector<feature::point> keypoints;
            std::vector<feature::descriptor::binary<256>> descriptors;
            if (!omit_target) {
                const int x = target_x + k * shift_x;
                const int y = target_y + k * shift_y;
                keypoints.push_back(feature::point{ static_cast<float>(x), static_cast<float>(y), 0.0f, 0.0f, 0 });
                descriptors.push_back(describe_at(image_k, x, y));
            }
            for (size_t p = 0; p < other_count; ++p) {
                const int x = other_x[p] + k * shift_x;
                const int y = other_y[p] + k * shift_y;
                keypoints.push_back(feature::point{ static_cast<float>(x), static_cast<float>(y), 0.0f, 0.0f, 0 });
                descriptors.push_back(describe_at(image_k, x, y));
            }
            manager.update(k, image::pyramid(image_k), keypoints, descriptors);

            if (omit_target) {
                feature::tracker::tracker::track* const lost = manager.find(target_id);
                REQUIRE(lost != nullptr);
                REQUIRE(lost->active == false);
                REQUIRE(lost->missed == 1);
            }
        }

        feature::tracker::tracker::track* const reacquired = manager.find(target_id);
        REQUIRE(reacquired != nullptr);
        REQUIRE(reacquired->active);
        REQUIRE(reacquired->length == 4);
        REQUIRE(manager.tracks().size() == total_count);
        REQUIRE(std::abs(static_cast<double>(reacquired->x - static_cast<float>(target_x + 4 * shift_x))) < 1.5);
        REQUIRE(std::abs(static_cast<double>(reacquired->y - static_cast<float>(target_y + 4 * shift_y))) < 1.5);
    }

    {
        feature::tracker::tracker::options opts;
        opts.max_missed = 2;
        feature::tracker::tracker manager(opts);

        constexpr static const int target_x = 100;
        constexpr static const int target_y = 100;
        constexpr static const int other_x[3] = { 50, 150, 50 };
        constexpr static const int other_y[3] = { 50, 50, 150 };
        constexpr static const size_t other_count = 3;

        {
            const image::image image0 = make_frame(dimension, 0, shift_x, shift_y);
            std::vector<feature::point> keypoints;
            std::vector<feature::descriptor::binary<256>> descriptors;
            keypoints.push_back(feature::point{ static_cast<float>(target_x), static_cast<float>(target_y), 0.0f, 0.0f, 0 });
            descriptors.push_back(describe_at(image0, target_x, target_y));
            for (size_t p = 0; p < other_count; ++p) {
                keypoints.push_back(feature::point{ static_cast<float>(other_x[p]), static_cast<float>(other_y[p]), 0.0f, 0.0f, 0 });
                descriptors.push_back(describe_at(image0, other_x[p], other_y[p]));
            }
            manager.update(0, image::pyramid(image0), keypoints, descriptors);
        }
        const int target_id = manager.tracks()[0].id;

        for (int k = 1; k <= 3; ++k) {
            image::image image_k = make_frame(dimension, k, shift_x, shift_y);
            occlude(image_k, target_x + k * shift_x, target_y + k * shift_y, 22);
            std::vector<feature::point> keypoints;
            std::vector<feature::descriptor::binary<256>> descriptors;
            for (size_t p = 0; p < other_count; ++p) {
                const int x = other_x[p] + k * shift_x;
                const int y = other_y[p] + k * shift_y;
                keypoints.push_back(feature::point{ static_cast<float>(x), static_cast<float>(y), 0.0f, 0.0f, 0 });
                descriptors.push_back(describe_at(image_k, x, y));
            }
            manager.update(k, image::pyramid(image_k), keypoints, descriptors);
        }
        REQUIRE(manager.find(target_id) == nullptr);

        {
            const int k = 4;
            const image::image image_k = make_frame(dimension, k, shift_x, shift_y);
            std::vector<feature::point> keypoints;
            std::vector<feature::descriptor::binary<256>> descriptors;
            const int tx = target_x + k * shift_x;
            const int ty = target_y + k * shift_y;
            keypoints.push_back(feature::point{ static_cast<float>(tx), static_cast<float>(ty), 0.0f, 0.0f, 0 });
            descriptors.push_back(describe_at(image_k, tx, ty));
            for (size_t p = 0; p < other_count; ++p) {
                const int x = other_x[p] + k * shift_x;
                const int y = other_y[p] + k * shift_y;
                keypoints.push_back(feature::point{ static_cast<float>(x), static_cast<float>(y), 0.0f, 0.0f, 0 });
                descriptors.push_back(describe_at(image_k, x, y));
            }
            manager.update(k, image::pyramid(image_k), keypoints, descriptors);

            const float reappear_x = static_cast<float>(tx);
            const float reappear_y = static_cast<float>(ty);
            std::vector<feature::tracker::tracker::track*> active = manager.active_tracks();
            bool found_fresh = false;
            for (size_t i = 0; i < active.size(); ++i) {
                if ((std::abs(static_cast<double>(active[i]->x - reappear_x)) < 2.0) &&
                    (std::abs(static_cast<double>(active[i]->y - reappear_y)) < 2.0)) {
                    found_fresh = true;
                    REQUIRE(active[i]->id != target_id);
                    REQUIRE(active[i]->start_frame_id == k);
                }
            }
            REQUIRE(found_fresh);
        }
    }

    {
        constexpr static const int target_x = 100;
        constexpr static const int target_y = 100;
        constexpr static const int other_x[3] = { 50, 150, 50 };
        constexpr static const int other_y[3] = { 50, 50, 150 };
        constexpr static const size_t other_count = 3;
        constexpr static const size_t total_count = other_count + 1;

        feature::tracker::tracker manager;

        feature::descriptor::binary<256> target_reference;
        {
            const image::image image0 = make_frame(dimension, 0, shift_x, shift_y);
            std::vector<feature::point> keypoints;
            std::vector<feature::descriptor::binary<256>> descriptors;
            target_reference = describe_at(image0, target_x, target_y);
            keypoints.push_back(feature::point{ static_cast<float>(target_x), static_cast<float>(target_y), 0.0f, 0.0f, 0 });
            descriptors.push_back(target_reference);
            for (size_t p = 0; p < other_count; ++p) {
                keypoints.push_back(feature::point{ static_cast<float>(other_x[p]), static_cast<float>(other_y[p]), 0.0f, 0.0f, 0 });
                descriptors.push_back(describe_at(image0, other_x[p], other_y[p]));
            }
            manager.update(0, image::pyramid(image0), keypoints, descriptors);
        }
        REQUIRE(manager.tracks().size() == total_count);
        const int target_id = manager.tracks()[0].id;

        for (int k = 1; k <= 2; ++k) {
            const image::image image_k = make_frame(dimension, k, shift_x, shift_y);
            std::vector<feature::point> keypoints;
            std::vector<feature::descriptor::binary<256>> descriptors;
            const int tx = target_x + k * shift_x;
            const int ty = target_y + k * shift_y;
            keypoints.push_back(feature::point{ static_cast<float>(tx), static_cast<float>(ty), 0.0f, 0.0f, 0 });
            descriptors.push_back(describe_at(image_k, tx, ty));
            for (size_t p = 0; p < other_count; ++p) {
                const int x = other_x[p] + k * shift_x;
                const int y = other_y[p] + k * shift_y;
                keypoints.push_back(feature::point{ static_cast<float>(x), static_cast<float>(y), 0.0f, 0.0f, 0 });
                descriptors.push_back(describe_at(image_k, x, y));
            }
            manager.update(k, image::pyramid(image_k), keypoints, descriptors);
        }
        REQUIRE(manager.find(target_id) != nullptr);
        REQUIRE(manager.find(target_id)->active);

        {
            const int k = 3;
            image::image image_k = make_frame(dimension, k, shift_x, shift_y);
            const int tx = target_x + k * shift_x;
            const int ty = target_y + k * shift_y;
            occlude(image_k, tx, ty, 22);

            feature::descriptor::binary<256> mismatched = target_reference;
            for (size_t b = 0; b < 10; ++b) {
                mismatched.data[b] = static_cast<unsigned char>(mismatched.data[b] ^ 0xFF);
            }
            REQUIRE(match::distance::hamming::distance(target_reference, mismatched) > 50u);

            std::vector<feature::point> keypoints;
            std::vector<feature::descriptor::binary<256>> descriptors;
            keypoints.push_back(feature::point{ static_cast<float>(tx), static_cast<float>(ty), 0.0f, 0.0f, 0 });
            descriptors.push_back(mismatched);
            for (size_t p = 0; p < other_count; ++p) {
                const int x = other_x[p] + k * shift_x;
                const int y = other_y[p] + k * shift_y;
                keypoints.push_back(feature::point{ static_cast<float>(x), static_cast<float>(y), 0.0f, 0.0f, 0 });
                descriptors.push_back(describe_at(image_k, x, y));
            }
            manager.update(k, image::pyramid(image_k), keypoints, descriptors);

            feature::tracker::tracker::track* const pooled = manager.find(target_id);
            REQUIRE(pooled != nullptr);
            REQUIRE(pooled->active == false);
            REQUIRE(pooled->missed == 1);
            REQUIRE(manager.tracks().size() == total_count);
            int tracks_near_target = 0;
            for (size_t i = 0; i < manager.tracks().size(); ++i) {
                const feature::tracker::tracker::track& t = manager.tracks()[i];
                const double dx = static_cast<double>(t.x) - static_cast<double>(tx);
                const double dy = static_cast<double>(t.y) - static_cast<double>(ty);
                if ((dx * dx + dy * dy) < (8.0 * 8.0)) {
                    ++tracks_near_target;
                }
            }
            REQUIRE(tracks_near_target == 1);
        }
    }

    {
        const image::image image0 = make_frame(dimension, 0, shift_x, shift_y);
        feature::tracker::tracker manager;

        std::vector<feature::point> keypoints;
        std::vector<feature::descriptor::binary<256>> descriptors;
        keypoints.push_back(feature::point{ 60.0f, 60.0f, 0.0f, 0.0f, 0 });
        descriptors.push_back(describe_at(image0, 60, 60));
        keypoints.push_back(feature::point{ 63.0f, 60.0f, 0.0f, 0.0f, 0 });
        descriptors.push_back(describe_at(image0, 63, 60));
        keypoints.push_back(feature::point{ 120.0f, 120.0f, 0.0f, 0.0f, 0 });
        descriptors.push_back(describe_at(image0, 120, 120));

        manager.update(0, image::pyramid(image0), keypoints, descriptors);

        REQUIRE(manager.tracks().size() == 2);
        bool has_a = false;
        bool has_b = false;
        bool has_c = false;
        for (size_t i = 0; i < manager.tracks().size(); ++i) {
            const feature::tracker::tracker::track& t = manager.tracks()[i];
            if ((std::abs(static_cast<double>(t.x - 60.0f)) < 0.5) && (std::abs(static_cast<double>(t.y - 60.0f)) < 0.5)) {
                has_a = true;
            }
            if ((std::abs(static_cast<double>(t.x - 63.0f)) < 0.5) && (std::abs(static_cast<double>(t.y - 60.0f)) < 0.5)) {
                has_b = true;
            }
            if ((std::abs(static_cast<double>(t.x - 120.0f)) < 0.5) && (std::abs(static_cast<double>(t.y - 120.0f)) < 0.5)) {
                has_c = true;
            }
        }
        REQUIRE(has_a);
        REQUIRE(has_b == false);
        REQUIRE(has_c);
    }

    {
        feature::tracker::tracker manager;
        constexpr static const int px = 90;
        constexpr static const int py = 90;

        {
            const image::image image0 = make_frame(dimension, 0, shift_x, shift_y);
            std::vector<feature::point> keypoints;
            std::vector<feature::descriptor::binary<256>> descriptors;
            keypoints.push_back(feature::point{ static_cast<float>(px), static_cast<float>(py), 0.0f, 0.0f, 0 });
            descriptors.push_back(describe_at(image0, px, py));
            manager.update(0, image::pyramid(image0), keypoints, descriptors);
        }
        REQUIRE(manager.tracks().size() == 1);
        const int track_id = manager.tracks()[0].id;

        feature::tracker::tracker::track* const before = manager.find(track_id);
        REQUIRE(before != nullptr);
        REQUIRE(before->landmark_id == -1);
        before->landmark_id = 42;

        {
            const int k = 1;
            const image::image image_k = make_frame(dimension, k, shift_x, shift_y);
            std::vector<feature::point> keypoints;
            std::vector<feature::descriptor::binary<256>> descriptors;
            const int x = px + k * shift_x;
            const int y = py + k * shift_y;
            keypoints.push_back(feature::point{ static_cast<float>(x), static_cast<float>(y), 0.0f, 0.0f, 0 });
            descriptors.push_back(describe_at(image_k, x, y));
            manager.update(k, image::pyramid(image_k), keypoints, descriptors);
        }

        feature::tracker::tracker::track* const after = manager.find(track_id);
        REQUIRE(after != nullptr);
        REQUIRE(after->active);
        REQUIRE(after->landmark_id == 42);
    }

    {
        const image::image image0 = make_frame(dimension, 0, shift_x, shift_y);
        feature::tracker::tracker manager;
        std::vector<feature::point> keypoints;
        std::vector<feature::descriptor::binary<256>> descriptors;
        feature::point fine{ 60.0f, 60.0f, 0.0f, 0.0f, 0 };
        feature::point coarse{ 120.0f, 120.0f, 0.0f, 0.0f, 3 };
        keypoints.push_back(fine);
        descriptors.push_back(describe_at(image0, 60, 60));
        keypoints.push_back(coarse);
        descriptors.push_back(describe_at(image0, 120, 120));
        manager.update(0, image::pyramid(image0), keypoints, descriptors);
        REQUIRE(manager.tracks().size() == 2);
        REQUIRE(manager.tracks()[0].octave == 0);
        REQUIRE(manager.tracks()[1].octave == 3);
    }

    {
        constexpr static const int target_x = 100;
        constexpr static const int target_y = 100;

        feature::tracker::tracker manager;

        const image::image image0 = make_frame(dimension, 0, shift_x, shift_y);
        {
            std::vector<feature::point> keypoints;
            std::vector<feature::descriptor::binary<256>> descriptors;
            keypoints.push_back(feature::point{ static_cast<float>(target_x), static_cast<float>(target_y), 0.0f, 0.0f, 0 });
            descriptors.push_back(describe_at(image0, target_x, target_y));
            manager.update(0, image::pyramid(image0), keypoints, descriptors);
        }
        REQUIRE(manager.tracks().size() == 1);
        REQUIRE(manager.tracks()[0].octave == 0);
        const int target_id = manager.tracks()[0].id;

        {
            image::image image1 = make_frame(dimension, 1, shift_x, shift_y);
            occlude(image1, target_x, target_y, 20);
            const std::vector<feature::point> keypoints;
            const std::vector<feature::descriptor::binary<256>> descriptors;
            manager.update(1, image::pyramid(image1), keypoints, descriptors);
        }
        feature::tracker::tracker::track* const lost = manager.find(target_id);
        REQUIRE(lost != nullptr);
        REQUIRE(!lost->active);

        {
            const image::image image2 = make_frame(dimension, 2, shift_x, shift_y);
            const int x = target_x + 2 * shift_x;
            const int y = target_y + 2 * shift_y;
            std::vector<feature::point> keypoints;
            std::vector<feature::descriptor::binary<256>> descriptors;
            feature::point detection{ static_cast<float>(x), static_cast<float>(y), 0.0f, 0.0f, 4 };
            keypoints.push_back(detection);
            descriptors.push_back(describe_at(image2, x, y));
            manager.update(2, image::pyramid(image2), keypoints, descriptors);
        }
        feature::tracker::tracker::track* const recovered = manager.find(target_id);
        REQUIRE(recovered != nullptr);
        REQUIRE(!recovered->active);
        REQUIRE(manager.tracks().size() == 1);
    }

    return EXIT_SUCCESS;
}
