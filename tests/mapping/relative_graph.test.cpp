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

#include "mapping/relative_graph.hpp"

#include "geometry/plucker.hpp"
#include "mapping/frame.hpp"
#include "mapping/line.hpp"
#include "mapping/map.hpp"
#include "mapping/point.hpp"
#include "math/lie.hpp"
#include "sensor/camera.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

namespace {
    class generator final {
    private:
        std::uint64_t state = 0x2545F4914F6CDD1Dull;

    public:
        double uniform(const double low, const double high) {
            this->state ^= this->state << 13;
            this->state ^= this->state >> 7;
            this->state ^= this->state << 17;
            return low + ((high - low) * static_cast<double>(this->state >> 11) / static_cast<double>(1ull << 53));
        }
    };

    class scene final {
    public:
        mapping::map reconstruction;
        std::vector<math::se3<double>> truth;
        std::vector<math::matrix<double, 3, 1>> points;
        std::vector<std::pair<int, size_t>> keyframes;
    };

    // Keyframes moving sideways past a wall of points, the first six in one submap and the last two in another.
    scene make_scene(const int frames = 8) {
        scene result;
        const double parameters[sensor::model::parameter_count] = { 500.0, 500.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        const sensor::model camera(&parameters[0], sensor::model::parameter_count);
        for (int index = 0; index < frames; ++index) {
            const math::matrix<double, 3, 1> centre = { { 0.3 * static_cast<double>(index), 0.05 * static_cast<double>(index % 3), 0.0 } };
            const math::so3<double> rotation = math::so3<double>::exp({ { 0.01 * static_cast<double>(index % 2), -0.03 * static_cast<double>(index), 0.005 * static_cast<double>(index) } });
            const math::se3<double> pose(rotation, -(rotation * centre));
            result.truth.push_back(pose);
            mapping::frame frame;
            frame.id = index;
            frame.camera = camera;
            frame.measurement_sigma = 1.0;
            frame.rotation = pose.rotation().get_matrix();
            frame.translation = pose.translation();
            result.reconstruction.add_frame(frame);
            result.keyframes.push_back({ index, (index < 6) ? 0u : 1u });
        }
        result.reconstruction.next_frame_id = frames;
        generator random;
        for (int index = 0; index < 150; ++index) {
            const math::matrix<double, 3, 1> location = { { random.uniform(-2.0, 4.5), random.uniform(-1.5, 1.5), random.uniform(4.0, 8.0) } };
            mapping::point landmark(index, location, math::matrix<double, 3, 1>::zero());
            int observers = 0;
            for (int frame_id = 0; frame_id < frames; ++frame_id) {
                const mapping::frame& frame = result.reconstruction.frames.at(frame_id);
                const math::matrix<double, 3, 1> in_frame = (frame.rotation * location) + frame.translation;
                double pixel[2];
                if (!frame.camera.project(in_frame.data(), &pixel[0]) || (pixel[0] < 0.0) || (pixel[0] > 640.0) || (pixel[1] < 0.0) || (pixel[1] > 480.0)) {
                    continue;
                }
                if (observers == 0) {
                    REQUIRE(landmark.anchor(frame.rotation, frame.translation));
                }
                result.reconstruction.add_observation(frame_id, landmark, pixel[0], pixel[1]);
                ++observers;
            }
            if (observers >= 2) {
                landmark.update_location_from_inverse_depth();
                result.reconstruction.add_landmark(landmark);
                result.points.push_back(location);
            }
            else {
                result.reconstruction.observations.erase(index);
            }
        }
        result.reconstruction.next_landmark_id = 150;
        return result;
    }

    // The largest reprojection error of the graph's points in the frames observing them, through the graph itself.
    double worst_graph_error(mapping::relative_graph& graph, const mapping::map& reconstruction) {
        double worst = 0.0;
        for (const auto& [landmark_id, landmark_observations] : reconstruction.observations) {
            if (graph.points.count(landmark_id) == 0) {
                continue;
            }
            for (const mapping::map::observation& obs : landmark_observations) {
                // Only frames joined to a point's base see it through the graph.
                math::matrix<double, 3, 1> point;
                if (!graph.point_in_frame(landmark_id, obs.frame_id, point)) {
                    const std::vector<int> joined = graph.component(graph.points.at(landmark_id).base);
                    REQUIRE(!std::binary_search(joined.begin(), joined.end(), obs.frame_id));
                    continue;
                }
                double pixel[2];
                REQUIRE(reconstruction.frames.at(obs.frame_id).camera.project(point.data(), &pixel[0]));
                worst = std::fmax(worst, std::hypot(pixel[0] - obs.point[0], pixel[1] - obs.point[1]));
            }
        }
        return worst;
    }

    // The largest reprojection error of the map's landmarks in the frames observing them, through the map's poses.
    double worst_map_error(const mapping::map& reconstruction) {
        double worst = 0.0;
        for (const auto& [landmark_id, landmark_observations] : reconstruction.observations) {
            const mapping::point& landmark = reconstruction.landmarks.at(landmark_id);
            for (const mapping::map::observation& obs : landmark_observations) {
                math::matrix<double, 2, 1> pixel;
                REQUIRE(mapping::map::project_landmark(reconstruction.frames.at(obs.frame_id), landmark, pixel));
                worst = std::fmax(worst, std::hypot(pixel[0] - obs.point[0], pixel[1] - obs.point[1]));
            }
        }
        return worst;
    }

    // How far the pose of one frame relative to another is from the truth, in its rotation and translation.
    double relative_pose_error(const scene& setup, const int first, const int second) {
        const mapping::frame& a = setup.reconstruction.frames.at(first);
        const mapping::frame& b = setup.reconstruction.frames.at(second);
        const math::se3<double> estimated = math::se3<double>(a.rotation, a.translation) * math::se3<double>(b.rotation, b.translation).inverse();
        const math::se3<double> truth = setup.truth[static_cast<size_t>(first)] * setup.truth[static_cast<size_t>(second)].inverse();
        const math::matrix<double, 6, 1> difference = (truth.inverse() * estimated).log();
        return std::sqrt(difference.get_length_squared());
    }
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // Synchronising builds a chain through each submap and bases each point in front of its first observer.
    {
        scene setup = make_scene();
        mapping::relative_graph graph;
        graph.synchronise(setup.reconstruction, setup.keyframes);
        REQUIRE(graph.nodes.size() == 8);
        REQUIRE(graph.links.size() == 6);
        REQUIRE(graph.points.size() == setup.reconstruction.landmarks.size());
        REQUIRE(graph.nodes.at(0).parent_link < 0);
        REQUIRE(graph.nodes.at(6).parent_link < 0);
        for (int frame_id = 1; frame_id < 8; ++frame_id) {
            if (frame_id == 6) {
                continue;
            }
            const mapping::relative_graph::link& joined = graph.links[static_cast<size_t>(graph.nodes.at(frame_id).parent_link)];
            REQUIRE((joined.parent == frame_id - 1) && (joined.child == frame_id) && !joined.similarity);
            REQUIRE(joined.transform.scale() == 1.0);
        }
        REQUIRE(graph.component(2) == std::vector<int>({ 0, 1, 2, 3, 4, 5 }));
        REQUIRE(graph.component(7) == std::vector<int>({ 6, 7 }));
        for (const auto& [landmark_id, based] : graph.points) {
            REQUIRE(based.base == setup.reconstruction.observations.at(landmark_id).front().frame_id);
        }
        REQUIRE(worst_graph_error(graph, setup.reconstruction) < 1.0e-6);

        // The steps from an observer to an older base are the inverses of the links, to a newer base the links.
        std::vector<optimisation::relative_bundle_adjustment::step> steps;
        REQUIRE(graph.path(3, 1, steps));
        REQUIRE(steps.size() == 2);
        REQUIRE(steps[0].inverse && steps[1].inverse);
        REQUIRE(steps[0].transform == graph.nodes.at(3).parent_link);
        REQUIRE(graph.path(1, 3, steps));
        REQUIRE(steps.size() == 2);
        REQUIRE(!steps[0].inverse && !steps[1].inverse);
        REQUIRE(steps[0].transform == graph.nodes.at(2).parent_link);
        REQUIRE(graph.path(4, 4, steps) && steps.empty());
        REQUIRE(!graph.path(7, 0, steps));

        // Embedding from the newest frame of a component leaves a consistent map as it was.
        const mapping::map before = setup.reconstruction;
        graph.embed(setup.reconstruction, 5);
        for (int frame_id = 0; frame_id < 8; ++frame_id) {
            const mapping::frame& was = before.frames.at(frame_id);
            const mapping::frame& now = setup.reconstruction.frames.at(frame_id);
            for (size_t row = 0; row < 3; ++row) {
                REQUIRE(std::fabs(now.translation[row] - was.translation[row]) < 1.0e-9);
                for (size_t column = 0; column < 3; ++column) {
                    REQUIRE(std::fabs(now.rotation[row][column] - was.rotation[row][column]) < 1.0e-9);
                }
            }
        }
        for (const auto& [landmark_id, landmark] : setup.reconstruction.landmarks) {
            REQUIRE(std::sqrt((landmark.location - before.landmarks.at(landmark_id).location).get_length_squared()) < 1.0e-7);
            REQUIRE(landmark.inverse_depth);
        }
        REQUIRE(worst_map_error(setup.reconstruction) < 1.0e-6);

        // Synchronising again changes nothing.
        const std::vector<mapping::relative_graph::link> links = graph.links;
        graph.synchronise(setup.reconstruction, setup.keyframes);
        REQUIRE(graph.links.size() == links.size());
        for (size_t index = 0; index < links.size(); ++index) {
            REQUIRE(graph.links[index].transform == links[index].transform);
        }
    }

    // An adjustment from the newest frame repairs the transforms and points around it, which the embedding writes back.
    {
        scene setup = make_scene(6);
        for (std::pair<int, size_t>& keyframe : setup.keyframes) {
            keyframe.second = 0;
        }
        mapping::relative_graph graph;
        graph.synchronise(setup.reconstruction, setup.keyframes);
        // The newest frames' poses and the points' depths are disturbed, as tracking and triangulation would leave them.
        generator random;
        for (int frame_id = 3; frame_id < 6; ++frame_id) {
            mapping::relative_graph::link& joined = graph.links[static_cast<size_t>(graph.nodes.at(frame_id).parent_link)];
            const math::matrix<double, 7, 1> delta = { { random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01), random.uniform(-0.02, 0.02), random.uniform(-0.02, 0.02), random.uniform(-0.02, 0.02), 0.0 } };
            joined.transform = joined.transform * math::sim3<double>::exp(delta);
        }
        for (auto& [landmark_id, based] : graph.points) {
            static_cast<void>(landmark_id);
            based.parameters[2] *= random.uniform(0.95, 1.05);
        }
        REQUIRE(worst_graph_error(graph, setup.reconstruction) > 2.0);
        const mapping::relative_graph::summary summary = graph.adjust(setup.reconstruction, { 5 }, {}, 50);
        REQUIRE(!summary.diverged);
        REQUIRE(summary.active_frames >= 5);
        REQUIRE(summary.active_links >= 4);
        REQUIRE(summary.final_cost < 1.0e-6 * summary.initial_cost);
        REQUIRE(summary.removed == 0);
        REQUIRE(worst_graph_error(graph, setup.reconstruction) < 1.0e-3);
        for (int frame_id = 0; frame_id < 6; ++frame_id) {
            REQUIRE(graph.nodes.at(frame_id).error >= 0.0);
        }
        graph.embed(setup.reconstruction, 5);
        REQUIRE(worst_map_error(setup.reconstruction) < 1.0e-3);
        // The scale is held by the link from the first frame, so the relative poses come back to the truth.
        for (int frame_id = 0; frame_id < 5; ++frame_id) {
            REQUIRE(relative_pose_error(setup, frame_id, 5) < 1.0e-4);
        }

        // An adjustment of an undisturbed graph keeps it.
        const mapping::relative_graph::summary again = graph.adjust(setup.reconstruction, { 5 }, {}, 50);
        REQUIRE(again.final_cost <= again.initial_cost);
        REQUIRE(again.final_cost < 1.0e-6);
    }

    // A gross outlier leaves the map with the adjustment.
    {
        scene setup = make_scene(6);
        for (std::pair<int, size_t>& keyframe : setup.keyframes) {
            keyframe.second = 0;
        }
        int corrupted = -1;
        for (auto& [landmark_id, landmark_observations] : setup.reconstruction.observations) {
            if ((landmark_observations.size() >= 4) && ((corrupted < 0) || (landmark_id < corrupted))) {
                corrupted = landmark_id;
            }
        }
        REQUIRE(corrupted >= 0);
        const size_t observed = setup.reconstruction.observations.at(corrupted).size();
        setup.reconstruction.observations.at(corrupted).back().point[0] += 40.0;
        mapping::relative_graph graph;
        graph.synchronise(setup.reconstruction, setup.keyframes);
        const mapping::relative_graph::summary summary = graph.adjust(setup.reconstruction, { 5 }, {}, 50);
        REQUIRE(summary.removed >= 1);
        REQUIRE(setup.reconstruction.observations.at(corrupted).size() == observed - 1);
    }

    // Removing a keyframe links its neighbours through it and bases its points elsewhere, and predicts the same pixels.
    {
        scene setup = make_scene();
        mapping::relative_graph graph;
        graph.synchronise(setup.reconstruction, setup.keyframes);
        // Frame 2 is demoted, its observations leaving the map as the keyframe cull removes them.
        for (auto& [landmark_id, landmark_observations] : setup.reconstruction.observations) {
            static_cast<void>(landmark_id);
            std::vector<mapping::map::observation> kept;
            for (const mapping::map::observation& obs : landmark_observations) {
                if (obs.frame_id != 2) {
                    kept.push_back(obs);
                }
            }
            landmark_observations = kept;
        }
        std::vector<std::pair<int, size_t>> remaining;
        for (const std::pair<int, size_t>& keyframe : setup.keyframes) {
            if (keyframe.first != 2) {
                remaining.push_back(keyframe);
            }
        }
        graph.synchronise(setup.reconstruction, remaining);
        REQUIRE(graph.nodes.count(2) == 0);
        REQUIRE(graph.nodes.size() == 7);
        const mapping::relative_graph::link& bridged = graph.links[static_cast<size_t>(graph.nodes.at(3).parent_link)];
        REQUIRE((bridged.parent == 1) && (bridged.child == 3) && !bridged.similarity);
        REQUIRE(graph.component(0) == std::vector<int>({ 0, 1, 3, 4, 5 }));
        for (const auto& [landmark_id, based] : graph.points) {
            static_cast<void>(landmark_id);
            REQUIRE(based.base != 2);
        }
        REQUIRE(worst_graph_error(graph, setup.reconstruction) < 1.0e-6);

        // Removing the first frame of a component makes its child the first.
        remaining.erase(remaining.begin());
        graph.synchronise(setup.reconstruction, remaining);
        REQUIRE(graph.nodes.at(1).parent_link < 0);
        REQUIRE(graph.component(1) == std::vector<int>({ 1, 3, 4, 5 }));
        REQUIRE(worst_graph_error(graph, setup.reconstruction) < 1.0e-6);
    }

    // A frame something else moves takes the transform from its parent from the map, and a moved landmark is based again.
    {
        scene setup = make_scene();
        mapping::relative_graph graph;
        graph.synchronise(setup.reconstruction, setup.keyframes);
        mapping::frame& moved = setup.reconstruction.frames.at(4);
        const math::se3<double> shifted = math::se3<double>(math::so3<double>::identity(), { { 0.1, 0.0, 0.0 } }) * math::se3<double>(moved.rotation, moved.translation);
        moved.rotation = shifted.rotation().get_matrix();
        moved.translation = shifted.translation();
        mapping::point& landmark = setup.reconstruction.landmarks.begin()->second;
        const int landmark_id = landmark.id;
        landmark.location = landmark.location + math::matrix<double, 3, 1>({ 0.0, 0.0, 0.5 });
        landmark.inverse_depth = false;
        graph.synchronise(setup.reconstruction, setup.keyframes);
        const mapping::relative_graph::link& joined = graph.links[static_cast<size_t>(graph.nodes.at(4).parent_link)];
        const mapping::frame& parent = setup.reconstruction.frames.at(3);
        const math::se3<double> expected = math::se3<double>(parent.rotation, parent.translation) * shifted.inverse();
        REQUIRE(std::sqrt((joined.transform.transformation().translation() - expected.translation()).get_length_squared()) < 1.0e-12);
        const mapping::relative_graph::based_point& based = graph.points.at(landmark_id);
        REQUIRE(based.location == landmark.location);
        math::matrix<double, 3, 1> in_base;
        REQUIRE(graph.point_in_frame(landmark_id, based.base, in_base));
        const mapping::frame& base_frame = setup.reconstruction.frames.at(based.base);
        const math::matrix<double, 3, 1> expected_in_base = (base_frame.rotation * landmark.location) + base_frame.translation;
        REQUIRE(std::fabs((in_base[0] / in_base[2]) - (expected_in_base[0] / expected_in_base[2])) < 1.0e-12);
        REQUIRE(std::fabs((1.0 / based.parameters[2]) - expected_in_base[2]) < 1.0e-9);
    }

    // A loop is a similarity that shortens the paths across it, and relaxing an inconsistent one spreads its error.
    {
        scene setup = make_scene(6);
        for (std::pair<int, size_t>& keyframe : setup.keyframes) {
            keyframe.second = 0;
        }
        mapping::relative_graph graph;
        graph.synchronise(setup.reconstruction, setup.keyframes);
        const math::sim3<double> truth(setup.truth[0] * setup.truth[5].inverse(), 1.0);
        const int loop = graph.add_loop(0, 5, truth);
        REQUIRE(loop == 5);
        REQUIRE(graph.links[5].similarity);
        REQUIRE(graph.add_loop(3, 3, truth) < 0);
        REQUIRE(graph.add_loop(0, 42, truth) < 0);
        std::vector<optimisation::relative_bundle_adjustment::step> steps;
        REQUIRE(graph.path(5, 0, steps));
        REQUIRE((steps.size() == 1) && (steps[0].transform == loop) && steps[0].inverse);
        REQUIRE(graph.path(0, 4, steps));
        REQUIRE((steps.size() == 2) && (steps[0].transform == loop) && !steps[0].inverse && steps[1].inverse);
        REQUIRE(worst_graph_error(graph, setup.reconstruction) < 1.0e-6);

        // A loop that disagrees with the chain by a scale and a shift is relaxed into a view between the two.
        graph.links[static_cast<size_t>(loop)].transform = math::sim3<double>(math::se3<double>(math::so3<double>::identity(), { { 0.05, 0.0, 0.0 } }), 1.0) * truth * math::sim3<double>(math::se3<double>::identity(), 1.03);
        REQUIRE(graph.relax(setup.reconstruction, 50));
        for (const auto& [frame_id, frame] : setup.reconstruction.frames) {
            static_cast<void>(frame_id);
            for (size_t row = 0; row < 3; ++row) {
                REQUIRE(std::isfinite(frame.translation[row]));
            }
        }
        // The gauge frame is held where it joined the graph, and the newest frame moves towards where the loop puts it.
        const mapping::frame& gauge = setup.reconstruction.frames.at(setup.reconstruction.gauge_frame_id);
        const math::se3<double> joined = setup.truth[static_cast<size_t>(setup.reconstruction.gauge_frame_id)];
        REQUIRE(std::sqrt((gauge.translation - joined.translation()).get_length_squared()) < 1.0e-9);
        REQUIRE(relative_pose_error(setup, 0, 5) > 1.0e-4);
        REQUIRE(relative_pose_error(setup, 0, 5) < 0.06);
    }

    // A line moves with the first frame that sees it.
    {
        scene setup = make_scene(6);
        for (std::pair<int, size_t>& keyframe : setup.keyframes) {
            keyframe.second = 0;
        }
        const math::matrix<double, 3, 1> a = { { 0.0, 0.0, 5.0 } };
        const math::matrix<double, 3, 1> b = { { 1.0, 0.0, 5.0 } };
        geometry::plucker plucker_line;
        REQUIRE(geometry::plucker::from_points(a, b, plucker_line));
        setup.reconstruction.add_line_landmark(mapping::line(0, plucker_line, a, b));
        setup.reconstruction.add_line_observation(1, setup.reconstruction.line_landmarks.at(0), 100.0, 100.0, 200.0, 100.0);
        mapping::relative_graph graph;
        graph.synchronise(setup.reconstruction, setup.keyframes);
        const mapping::frame frame_before = setup.reconstruction.frames.at(1);
        const math::matrix<double, 3, 1> in_frame_before = (frame_before.rotation * a) + frame_before.translation;
        // A shorter link into the newest frame moves every older frame, and the line with frame 1.
        mapping::relative_graph::link& joined = graph.links[static_cast<size_t>(graph.nodes.at(5).parent_link)];
        joined.transform = math::sim3<double>(math::se3<double>(math::so3<double>::identity(), { { 0.2, 0.0, 0.0 } }), 1.0) * joined.transform;
        graph.embed(setup.reconstruction, 5);
        const mapping::frame& frame_after = setup.reconstruction.frames.at(1);
        const mapping::line& moved = setup.reconstruction.line_landmarks.at(0);
        const math::matrix<double, 3, 1> in_frame_after = (frame_after.rotation * moved.locations[0]) + frame_after.translation;
        REQUIRE(std::sqrt((frame_after.translation - frame_before.translation).get_length_squared()) > 0.1);
        REQUIRE(std::sqrt((in_frame_after - in_frame_before).get_length_squared()) < 1.0e-9);
    }

    return EXIT_SUCCESS;
}
