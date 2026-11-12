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

#include "mapping/map.hpp"

#include "geometry/plucker.hpp"
#include "mapping/line.hpp"
#include "math/lie.hpp"
#include "sensor/camera.hpp"

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

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // Basic insert / lookup tests.
    {
        mapping::map m;

        mapping::frame f;
        f.id = 1;
        {
            std::vector<feature::point> kps;
            for (int k = 0; k < 6; ++k) {
                kps.push_back(feature::point{ static_cast<float>(10 * k), static_cast<float>(10 * k + 1), 0.0f, 0.0f, 0 });
            }
            f.keypoints = kps;
        }
        REQUIRE(m.frames.find(f.id) == m.frames.end());
        m.add_frame(f);
        auto itf = m.frames.find(f.id);
        REQUIRE(itf != m.frames.end());
        REQUIRE(itf->second.id == f.id);

        mapping::point p;
        p.id = 10;
        REQUIRE(m.landmarks.find(p.id) == m.landmarks.end());
        m.add_landmark(p);
        auto itp = m.landmarks.find(p.id);
        REQUIRE(itp != m.landmarks.end());
        REQUIRE(itp->second.id == p.id);

        const size_t kp_index = 3;
        m.add_observation(f, p, kp_index);
        auto ito = m.observations.find(p.id);
        REQUIRE(ito != m.observations.end());
        REQUIRE(ito->second.size() == 1);
        REQUIRE(ito->second[0].frame_id == f.id);
        REQUIRE(ito->second[0].kp_index == kp_index);
        REQUIRE(ito->second[0].point[0] == 30.0);
        REQUIRE(ito->second[0].point[1] == 31.0);

        mapping::frame f2;
        f2.id = 2;
        {
            std::vector<feature::point> kps;
            for (int k = 0; k < 6; ++k) {
                kps.push_back(feature::point{ static_cast<float>(10 * k), static_cast<float>(10 * k + 1), 0.0f, 0.0f, 0 });
            }
            f2.keypoints = kps;
        }
        m.add_frame(f2);

        const size_t kp_index2 = 5;
        m.add_observation(f2, p, kp_index2);
        m.add_observation(99, p, 1.0, 2.0);
        m.add_observation(98, p, 1.0, 2.0);
        m.observations.at(p.id).resize(2);

        auto ito2 = m.observations.find(p.id);
        REQUIRE(ito2 != m.observations.end());
        REQUIRE(ito2->second.size() == 2);
        REQUIRE(ito2->second[0].frame_id == f.id);
        REQUIRE(ito2->second[0].kp_index == kp_index);
        REQUIRE(ito2->second[1].frame_id == f2.id);
        REQUIRE(ito2->second[1].kp_index == kp_index2);
        REQUIRE(ito2->second[1].point[0] == 50.0);
        REQUIRE(ito2->second[1].point[1] == 51.0);

        REQUIRE(m.frames.size() >= 2);
        REQUIRE(m.landmarks.find(p.id) != m.landmarks.end());
    }
    {
        mapping::map m;
        mapping::frame f;
        f.id = 100;
        m.add_frame(f);

        mapping::frame f_modified;
        f_modified.id = 100;
        m.add_frame(f_modified);
        REQUIRE(m.frames.find(100) != m.frames.end());
        REQUIRE(m.frames.at(100).id == 100);

        mapping::point p;
        p.id = 200;
        m.add_landmark(p);
        mapping::point p_modified;
        p_modified.id = 200;
        m.add_landmark(p_modified);
        REQUIRE(m.landmarks.find(200) != m.landmarks.end());
        REQUIRE(m.landmarks.at(200).id == 200);
    }
    {
        mapping::map m;
        mapping::frame f1;
        f1.id = 0;
        f1.keypoints = { feature::point{ 0.0f, 0.0f, 0.0f, 0.0f, 0 } };
        mapping::frame f2;
        f2.id = 1;
        f2.keypoints = { feature::point{ 0.0f, 0.0f, 0.0f, 0.0f, 0 } };
        mapping::point l1;
        l1.id = 0;
        m.add_frame(f1);
        m.add_frame(f2);
        m.add_landmark(l1);
        m.add_observation(f1, l1, 0);
        m.add_observation(f2, l1, 0);
        m.optimise(2, false, 3);
    }
    {
        mapping::map m;
        mapping::frame f1;
        f1.id = 0;
        f1.keypoints = { feature::point{ 0.0f, 0.0f, 0.0f, 0.0f, 0 } };
        mapping::point l1;
        l1.id = 0;
        m.add_frame(f1);
        m.add_landmark(l1);
        m.add_observation(f1, l1, 0);
        const size_t before = m.landmarks.size();
        m.cull();
        const size_t after = m.landmarks.size();
        REQUIRE(before > after);
    }
    {
        mapping::map m;
        REQUIRE(m.frames.empty());
        m.optimise(10, false, 50);
        m.optimise(1, true, 50);
        REQUIRE(m.frames.empty());
    }
    {
        mapping::map m;
        mapping::point kept;
        kept.id = 1;
        mapping::point dropped;
        dropped.id = 2;
        m.add_landmark(kept);
        m.add_landmark(dropped);
        m.add_observation(3, kept, 1.0, 1.0);
        m.add_observation(9, kept, 9.0, 9.0);
        m.add_observation(5, dropped, 5.0, 5.0);
        m.add_observation(9, dropped, 90.0, 90.0);
        m.add_observation(12, dropped, 12.0, 12.0);
        REQUIRE(!m.merge_landmark(1, 1));
        REQUIRE(!m.merge_landmark(1, 7));
        REQUIRE(m.merge_landmark(1, 2));
        REQUIRE(m.landmarks.count(2) == 0);
        REQUIRE(m.observations.count(2) == 0);
        const std::vector<mapping::map::observation>& merged = m.observations.at(1);
        REQUIRE(merged.size() == 4);
        REQUIRE(merged[0].frame_id == 3);
        REQUIRE(merged[1].frame_id == 5);
        REQUIRE(merged[2].frame_id == 9);
        REQUIRE(merged[2].point[0] == 9.0);
        REQUIRE(merged[3].frame_id == 12);
        REQUIRE(!m.merge_landmark(1, 2));
    }
    {
        mapping::map m;
        mapping::point p;
        p.id = 42;
        m.add_landmark(p);
        REQUIRE(m.observations.find(p.id) == m.observations.end());
        REQUIRE(m.landmarks.find(p.id) != m.landmarks.end());
        m.cull();
        REQUIRE(m.landmarks.find(p.id) == m.landmarks.end());
    }

    // A line's error is the mean pixel distance of the observed endpoints from its projection, also when fx differs from fy.
    {
        const double parameters[sensor::model::parameter_count] = { 400.0, 600.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        mapping::frame f;
        f.id = 7;
        f.camera = sensor::model(&parameters[0], sensor::model::parameter_count);
        f.rotation = math::matrix<double, 3, 3>::identity();
        f.translation = math::matrix<double, 3, 1>::zero();
        const math::matrix<double, 3, 1> a = { { -1.0, 0.5, 5.0 } };
        const math::matrix<double, 3, 1> b = { { 1.0, -0.2, 6.0 } };
        geometry::plucker line;
        REQUIRE(geometry::plucker::from_points(a, b, line));
        const mapping::line landmark(3, line, a, b);
        const double pa[2] = { (400.0 * a[0] / a[2]) + 320.0, (600.0 * a[1] / a[2]) + 240.0 };
        const double pb[2] = { (400.0 * b[0] / b[2]) + 320.0, (600.0 * b[1] / b[2]) + 240.0 };
        const double along[2] = { pb[0] - pa[0], pb[1] - pa[1] };
        const double length = std::sqrt((along[0] * along[0]) + (along[1] * along[1]));
        const double normal[2] = { -along[1] / length, along[0] / length };
        const double offsets[2] = { 1.5, -0.5 };
        const double fractions[2] = { 0.3, 0.8 };
        double ends[2][2] = {};
        for (int end = 0; end < 2; ++end) {
            ends[end][0] = pa[0] + (fractions[end] * along[0]) + (offsets[end] * normal[0]);
            ends[end][1] = pa[1] + (fractions[end] * along[1]) + (offsets[end] * normal[1]);
        }
        const mapping::map::line_observation observed{ f.id, ends[0][0], ends[0][1], ends[1][0], ends[1][1] };
        double error = 0.0;
        REQUIRE(mapping::map::line_reprojection_error(f, landmark, observed, error));
        REQUIRE(std::abs(error - 1.0) < 1.0e-9);
    }

    {
        // An adjustment takes no observation whose landmark does not project within a focal length of it. Here one landmark of
        // sixty one has been left somewhere absurd, in front of every camera and 89 degrees off their axes, where the distorted
        // camera puts it some 1e19 pixels from each of its observations. With that in the solve the first pass is spent on it
        // and the adjustment is undone or takes good observations with it, so the rest of the map is never adjusted.
        const double parameters[sensor::model::parameter_count] = { 517.3, 516.5, 318.6, 255.3, 0.2624, -0.9531, -0.0054, 0.0026, 1.1633, 0.0, 0.0, 0.0 };
        const sensor::model camera(&parameters[0], sensor::model::parameter_count);
        constexpr static const int frame_count = 6;
        constexpr static const int landmark_count = 60;
        constexpr static const int absurd_id = landmark_count;
        mapping::map m;
        for (int frame_id = 0; frame_id < frame_count; ++frame_id) {
            mapping::frame f;
            f.id = frame_id;
            f.camera = camera;
            f.translation = math::matrix<double, 3, 1>({ -0.2 * frame_id, 0.0, 0.0 });
            m.add_frame(f);
        }
        const auto observe = [&](const mapping::point& landmark, const math::matrix<double, 3, 1>& truth) {
            size_t observations = 0;
            for (int frame_id = 0; frame_id < frame_count; ++frame_id) {
                const mapping::frame& f = m.frames.at(frame_id);
                const math::matrix<double, 3, 1> in_camera = (f.rotation * truth) + f.translation;
                double pixel[2] = {};
                if (camera.project(in_camera.data(), &pixel[0]) && (pixel[0] > 0.0) && (pixel[0] < 640.0) && (pixel[1] > 0.0) && (pixel[1] < 480.0)) {
                    m.add_observation(frame_id, landmark, pixel[0], pixel[1]);
                    ++observations;
                }
            }
            return observations;
        };
        unsigned int state = 12345u;
        const auto uniform = [&state]() {
            state = (state * 1664525u) + 1013904223u;
            return static_cast<double>(state >> 8) / static_cast<double>(1u << 24);
        };
        size_t good_observations = 0;
        for (int landmark_id = 0; landmark_id < landmark_count; ++landmark_id) {
            const math::matrix<double, 3, 1> truth({ -1.0 + (3.0 * uniform()), -0.8 + (1.6 * uniform()), 4.0 + (4.0 * uniform()) });
            // The map holds each landmark a little off, by a pixel or two, which is what the adjustment is there to put right.
            const mapping::point landmark(landmark_id, truth + math::matrix<double, 3, 1>({ 0.02 * (uniform() - 0.5), 0.02 * (uniform() - 0.5), 0.1 * (uniform() - 0.5) }), math::matrix<double, 3, 1>::zero());
            m.add_landmark(landmark);
            const size_t observations = observe(landmark, truth);
            REQUIRE(observations >= 2);
            good_observations += observations;
        }
        const mapping::point absurd(absurd_id, math::matrix<double, 3, 1>({ 60.0, -50.0, 0.3 }), math::matrix<double, 3, 1>::zero());
        m.add_landmark(absurd);
        REQUIRE(observe(absurd, math::matrix<double, 3, 1>({ 0.5, 0.2, 6.0 })) == static_cast<size_t>(frame_count));
        {
            math::matrix<double, 2, 1> projected;
            REQUIRE(mapping::map::project_landmark(m.frames.at(0), m.landmarks.at(absurd_id), projected));
            REQUIRE((projected - m.observations.at(absurd_id)[0].point).get_length_squared() > 1.0e20);
        }

        m.optimise(0, false, 20);

        // The absurd landmark has lost its observations to the outlier removal and gone, and nothing else has: every other
        // observation is still there and now fits to a hundredth of a pixel.
        REQUIRE(m.landmarks.count(absurd_id) == 0);
        REQUIRE(m.observations.count(absurd_id) == 0);
        size_t remaining_observations = 0;
        for (const auto& [landmark_id, landmark_observations] : m.observations) {
            for (const mapping::map::observation& obs : landmark_observations) {
                math::matrix<double, 2, 1> projected;
                REQUIRE(mapping::map::project_landmark(m.frames.at(obs.frame_id), m.landmarks.at(landmark_id), projected));
                REQUIRE((projected - obs.point).get_length_squared() < 1.0e-4);
                ++remaining_observations;
            }
        }
        REQUIRE(remaining_observations == good_observations);
    }

    {
        // Each frame is adjusted with its own camera, as each is given one when it is processed. The adjustment took the camera
        // of whichever frame came first in its table for every observation, so with two cameras in a map the observations of
        // one of them were some eighty pixels from where its landmarks project.
        const double narrow_parameters[sensor::model::parameter_count] = { 517.3, 516.5, 318.6, 255.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        const double wide_parameters[sensor::model::parameter_count] = { 380.0, 381.0, 322.0, 238.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        const sensor::model narrow(&narrow_parameters[0], sensor::model::parameter_count);
        const sensor::model wide(&wide_parameters[0], sensor::model::parameter_count);
        constexpr static const int frame_count = 6;
        constexpr static const int landmark_count = 60;
        mapping::map m;
        for (int frame_id = 0; frame_id < frame_count; ++frame_id) {
            mapping::frame f;
            f.id = frame_id;
            f.camera = ((frame_id % 2) == 0) ? narrow : wide;
            f.translation = math::matrix<double, 3, 1>({ -0.2 * frame_id, 0.0, 0.0 });
            m.add_frame(f);
        }
        unsigned int state = 54321u;
        const auto uniform = [&state]() {
            state = (state * 1664525u) + 1013904223u;
            return static_cast<double>(state >> 8) / static_cast<double>(1u << 24);
        };
        size_t observations = 0;
        for (int landmark_id = 0; landmark_id < landmark_count; ++landmark_id) {
            const math::matrix<double, 3, 1> truth({ -1.0 + (3.0 * uniform()), -0.8 + (1.6 * uniform()), 4.0 + (4.0 * uniform()) });
            const mapping::point landmark(landmark_id, truth + math::matrix<double, 3, 1>({ 0.02 * (uniform() - 0.5), 0.02 * (uniform() - 0.5), 0.1 * (uniform() - 0.5) }), math::matrix<double, 3, 1>::zero());
            m.add_landmark(landmark);
            size_t seen_by = 0;
            for (int frame_id = 0; frame_id < frame_count; ++frame_id) {
                const mapping::frame& f = m.frames.at(frame_id);
                const math::matrix<double, 3, 1> in_camera = (f.rotation * truth) + f.translation;
                double pixel[2] = {};
                if (f.camera.project(in_camera.data(), &pixel[0]) && (pixel[0] > 0.0) && (pixel[0] < 640.0) && (pixel[1] > 0.0) && (pixel[1] < 480.0)) {
                    m.add_observation(frame_id, landmark, pixel[0], pixel[1]);
                    ++seen_by;
                }
            }
            REQUIRE(seen_by >= 4);
            observations += seen_by;
        }

        m.optimise(0, false, 20);

        // Every observation is still there and fits to a hundredth of a pixel through the camera of its own frame.
        size_t remaining_observations = 0;
        for (const auto& [landmark_id, landmark_observations] : m.observations) {
            for (const mapping::map::observation& obs : landmark_observations) {
                math::matrix<double, 2, 1> projected;
                REQUIRE(mapping::map::project_landmark(m.frames.at(obs.frame_id), m.landmarks.at(landmark_id), projected));
                REQUIRE((projected - obs.point).get_length_squared() < 1.0e-4);
                ++remaining_observations;
            }
        }
        REQUIRE(remaining_observations == observations);
    }

    return EXIT_SUCCESS;
}
