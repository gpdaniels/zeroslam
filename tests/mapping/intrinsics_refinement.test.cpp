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

#include "mapping/intrinsics_refinement.hpp"

#include "mapping/calibration.hpp"
#include "mapping/frame.hpp"
#include "math/lie.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/edges/reprojection.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/point.hpp"
#include "optimisation/vertices/pose.hpp"
#include "sensor/camera.hpp"
#include "sensor/camera/pinhole_radial_tangential.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
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

    const double camera_parameters[sensor::camera::pinhole_radial_tangential<double>::parameter_count] = { 500.0, 500.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const double other_parameters[sensor::camera::pinhole_radial_tangential<double>::parameter_count] = { 400.0, 400.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const sensor::model camera(sensor::camera::pinhole_radial_tangential<double>(&camera_parameters[0], sensor::camera::pinhole_radial_tangential<double>::parameter_count));
    const sensor::model other(sensor::camera::pinhole_radial_tangential<double>(&other_parameters[0], sensor::camera::pinhole_radial_tangential<double>::parameter_count));

    // Keyframes turning a degree apart about the vertical, and one more frame with another camera.
    std::unordered_map<int, mapping::frame> frames;
    std::unordered_set<int> keyframe_ids;
    for (int frame_id = 0; frame_id < 12; ++frame_id) {
        mapping::frame frame;
        frame.id = frame_id;
        frame.camera = camera;
        frame.rotation = math::so3<double>::exp({ { 0.0, 0.0175 * static_cast<double>(frame_id), 0.0 } }).get_matrix();
        frames[frame_id] = frame;
        keyframe_ids.insert(frame_id);
    }
    mapping::frame odd;
    odd.id = 12;
    odd.camera = other;
    frames[12] = odd;

    {
        // No request while the calibration is off, while the keyframes span too little of a turn, or are too few.
        mapping::calibration calibration;
        static_cast<void>(calibration.camera_for(camera));
        mapping::calibration::refinement request;
        REQUIRE(!mapping::intrinsics_refinement::make_request(calibration, frames, keyframe_ids, request));
        calibration.set_level(mapping::calibration::level::focal);
        // Eleven degrees between the first and last, five and a half from their mean.
        REQUIRE(!mapping::intrinsics_refinement::make_request(calibration, frames, keyframe_ids, request));
        for (int frame_id = 0; frame_id < 12; ++frame_id) {
            frames[frame_id].rotation = math::so3<double>::exp({ { 0.0, 0.04 * static_cast<double>(frame_id), 0.0 } }).get_matrix();
        }
        REQUIRE(mapping::intrinsics_refinement::make_request(calibration, frames, keyframe_ids, request));
        const std::unordered_set<int> few = { 0, 11 };
        REQUIRE(!mapping::intrinsics_refinement::make_request(calibration, frames, few, request));
    }

    {
        // An adjustment's part: the frames of the camera take the vertex, the other's does not, the priors join once an
        // edge has, and the result goes to every frame of the camera and comes off them again when undone.
        mapping::calibration calibration;
        calibration.set_level(mapping::calibration::level::focal_centre);
        static_cast<void>(calibration.camera_for(camera));
        mapping::calibration::refinement request;
        REQUIRE(calibration.make_refinement(request));
        optimisation::factor_graph graph;
        std::unordered_map<int, optimisation::vertex*> camera_vertexes;
        for (const int frame_id : { 0, 5, 12 }) {
            optimisation::vertex pose{ optimisation::vertices::pose() };
            const double identity[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
            REQUIRE(pose.set_parameters(&identity[0], 7));
            camera_vertexes[frame_id] = graph.add_vertex(static_cast<optimisation::vertex&&>(pose));
        }
        mapping::intrinsics_refinement intrinsics(&request, graph, true);
        intrinsics.add_vertex(frames, camera_vertexes);
        REQUIRE(intrinsics.takes(0));
        REQUIRE(intrinsics.takes(5));
        REQUIRE(!intrinsics.takes(12));
        REQUIRE(!intrinsics.takes(3));
        optimisation::vertex point{ optimisation::vertices::point() };
        const double location[3] = { 0.1, 0.2, 4.0 };
        REQUIRE(point.set_parameters(&location[0], 3));
        point.set_marginalised(true);
        optimisation::vertex* const point_node = graph.add_vertex(static_cast<optimisation::vertex&&>(point));
        optimisation::edge factor{ optimisation::edges::reprojection(sensor::camera::model<double>(camera), true) };
        factor.set_observation(math::matrix<double, 0, 0>(2, 1, math::matrix<double, 2, 1>{ { 333.0, 266.0 } }.data()));
        REQUIRE(factor.add_vertex(camera_vertexes.at(0)));
        REQUIRE(factor.add_vertex(point_node));
        intrinsics.attach(factor);
        REQUIRE(factor.num_vertices() == 3);
        optimisation::edge* const added = graph.add_edge(static_cast<optimisation::edge&&>(factor));
        REQUIRE(added != nullptr);
        optimisation::vertex* const intrinsics_node = added->get_vertex(2);
        intrinsics.add_priors();
        // The edge and a prior on each free parameter, the focal lengths' two and the principal point's two.
        REQUIRE(graph.get_connected_edges(intrinsics_node).size() == 5);
        const double refined[sensor::camera::pinhole_radial_tangential<double>::parameter_count] = { 510.0, 510.0, 321.0, 239.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        REQUIRE(intrinsics_node->set_parameters(&refined[0], sensor::camera::pinhole_radial_tangential<double>::parameter_count));
        intrinsics.apply(frames);
        REQUIRE(request.refined);
        REQUIRE(request.parameters.size() == sensor::camera::pinhole_radial_tangential<double>::parameter_count);
        REQUIRE(request.parameters[0] == 510.0);
        for (int frame_id = 0; frame_id < 12; ++frame_id) {
            double parameters[sensor::camera::pinhole_radial_tangential<double>::parameter_count] = {};
            REQUIRE(frames.at(frame_id).camera.get_parameters(&parameters[0], sensor::camera::pinhole_radial_tangential<double>::parameter_count));
            REQUIRE(parameters[0] == 510.0);
            REQUIRE(parameters[3] == 239.0);
            double before[sensor::camera::pinhole_radial_tangential<double>::parameter_count] = {};
            REQUIRE(intrinsics.camera_before(frame_id, frames.at(frame_id)).get_parameters(&before[0], sensor::camera::pinhole_radial_tangential<double>::parameter_count));
            REQUIRE(before[0] == 500.0);
        }
        double kept[sensor::camera::pinhole_radial_tangential<double>::parameter_count] = {};
        REQUIRE(frames.at(12).camera.get_parameters(&kept[0], sensor::camera::pinhole_radial_tangential<double>::parameter_count));
        REQUIRE(kept[0] == 400.0);
        intrinsics.restore(frames);
        REQUIRE(!request.refined);
        for (int frame_id = 0; frame_id < 12; ++frame_id) {
            double parameters[sensor::camera::pinhole_radial_tangential<double>::parameter_count] = {};
            REQUIRE(frames.at(frame_id).camera.get_parameters(&parameters[0], sensor::camera::pinhole_radial_tangential<double>::parameter_count));
            REQUIRE(parameters[0] == 500.0);
        }
    }

    {
        // Without a request, or with the landmarks fixed, nothing is added and no frame takes the vertex.
        optimisation::factor_graph graph;
        std::unordered_map<int, optimisation::vertex*> camera_vertexes;
        optimisation::vertex pose{ optimisation::vertices::pose() };
        camera_vertexes[0] = graph.add_vertex(static_cast<optimisation::vertex&&>(pose));
        mapping::intrinsics_refinement none(nullptr, graph, true);
        none.add_vertex(frames, camera_vertexes);
        REQUIRE(!none.takes(0));
        mapping::calibration calibration;
        calibration.set_level(mapping::calibration::level::focal);
        static_cast<void>(calibration.camera_for(camera));
        mapping::calibration::refinement request;
        REQUIRE(calibration.make_refinement(request));
        mapping::intrinsics_refinement fixed(&request, graph, false);
        fixed.add_vertex(frames, camera_vertexes);
        REQUIRE(!fixed.takes(0));
        fixed.add_priors();
        fixed.apply(frames);
        REQUIRE(!request.refined);
        // A vertex no edge took leaves the graph.
        mapping::intrinsics_refinement unused(&request, graph, true);
        unused.add_vertex(frames, camera_vertexes);
        REQUIRE(unused.takes(0));
        unused.add_priors();
        REQUIRE(!unused.takes(0));
    }

    return EXIT_SUCCESS;
}
