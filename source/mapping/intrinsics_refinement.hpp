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
#ifndef ZEROSLAM_MAPPING_INTRINSICS_REFINEMENT_HPP
#define ZEROSLAM_MAPPING_INTRINSICS_REFINEMENT_HPP

#include "mapping/calibration.hpp"
#include "mapping/frame.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "sensor/camera.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace mapping {
    // The part of an adjustment that refines the camera a calibration request names (see mapping::calibration): one
    // intrinsics vertex in the adjustment's graph, which the edges of the frames taken with that camera take after their
    // pose and landmark, with a prior on each free parameter; after the solve the refined camera goes to every frame of the
    // map taken with the camera, and comes back off them if the adjustment is undone. Without a request, or with the
    // landmarks fixed, it does nothing and takes no frame.
    class intrinsics_refinement final {
    private:
        calibration::refinement* request;
        optimisation::factor_graph& graph;
        bool enabled;
        optimisation::vertex* node;
        std::vector<double> request_parameters;
        std::unordered_set<int> frame_ids;
        size_t edge_count;
        std::vector<int> refined_frame_ids;
        sensor::model previous_camera;

    public:
        intrinsics_refinement(calibration::refinement* const request_value, optimisation::factor_graph& graph_value, const bool landmarks_free);

        intrinsics_refinement(const intrinsics_refinement&) = delete;
        intrinsics_refinement& operator=(const intrinsics_refinement&) = delete;

        // The parameters of a camera, empty when it does not give them.
        template <typename camera_type>
        static std::vector<double> camera_parameters(const camera_type& camera) {
            std::vector<double> parameters(camera.get_parameter_count(), 0.0);
            if (!parameters.empty() && !camera.get_parameters(parameters.data(), parameters.size())) {
                parameters.clear();
            }
            return parameters;
        }

        // A request to refine the calibration's camera in an adjustment of the whole map, false when the calibration is off,
        // the keyframes do not yet span enough of a turn to observe it (see mapping::calibration), or the camera does not say
        // which of its parameters the level asks for.
        static bool make_request(const calibration& camera_calibration, const std::unordered_map<int, mapping::frame>& frames, const std::unordered_set<int>& keyframe_ids, calibration::refinement& request);

        // Adds the intrinsics vertex, and takes the frames of the adjustment whose camera is the request's.
        void add_vertex(const std::unordered_map<int, mapping::frame>& frames, const std::unordered_map<int, optimisation::vertex*>& camera_vertexes);

        // Whether the observations of the frame take the intrinsics vertex.
        bool takes(const int frame_id) const;

        // Gives the vertex to an edge made with its intrinsics, after its pose and landmark.
        void attach(optimisation::edge& factor);

        // The priors on the free parameters, once the edges are in; without an edge that took it the vertex leaves.
        void add_priors();

        // After the solve: the refined camera to every frame of the map taken with the camera, and to the request.
        void apply(std::unordered_map<int, mapping::frame>& frames);

        // The camera a frame had before the adjustment refined it.
        const sensor::model& camera_before(const int frame_id, const mapping::frame& frame) const;

        // Puts the cameras back, for an adjustment that is undone, and says so in the request.
        void restore(std::unordered_map<int, mapping::frame>& frames);
    };
}

#endif // ZEROSLAM_MAPPING_INTRINSICS_REFINEMENT_HPP
