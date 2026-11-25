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
#ifndef ZEROSLAM_MAPPING_CALIBRATION_DRIFT_REFINEMENT_HPP
#define ZEROSLAM_MAPPING_CALIBRATION_DRIFT_REFINEMENT_HPP

#include "mapping/calibration_drift.hpp"
#include "mapping/frame.hpp"
#include "mapping/rig.hpp"
#include "math/lie.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/factor_graph.hpp"
#include "optimisation/vertex.hpp"
#include "sensor/camera.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <map>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace mapping {
    // The part of an adjustment that holds the calibration of the frames whose calibration is followed over time (see
    // mapping::calibration_drift), each frame being in a chunk of time (mapping::frame::calibration_chunk) with a camera and,
    // for each of its views, a pose on the reference camera of its own.
    //
    // In every adjustment the observations of such a frame's views take an extrinsic vertex of their chunk in place of the
    // rig's, held fixed where the frames have it. Given a request, in an adjustment of the whole map with the landmarks free,
    // the chunks are solved as the paper has them: a chunk's observations take an intrinsics vertex of their camera in that
    // chunk, with only the focal lengths free, and the extrinsic vertex of a camera other than the reference is free; each
    // vertex is tied to that of the chunk before it by a random walk over the time between them, each focal length has a
    // prior on the calibration given and the two a prior on their ratio, and each extrinsic keeps its declared distance from
    // the reference. A vertex is made when the first observation takes it, so only chunks that hold observations have one;
    // after the solve every chunk the request came with, and every chunk a frame of the map is in, has its calibration, a
    // chunk without observations the way along between the chunks around it that have them: every frame of the map takes
    // its chunk's, keyframe or not, and the request takes them all. The frames' calibration comes back off them, and the
    // request's chunks are as it came with them, if the adjustment is undone.
    class calibration_drift_refinement final {
    private:
        // The vertices of one camera in one chunk, null until an observation takes them, and the observations that have.
        struct node final {
            optimisation::vertex* intrinsics = nullptr;
            optimisation::vertex* extrinsic = nullptr;
            size_t observations = 0;
        };

        calibration_drift::request* request;
        optimisation::factor_graph& graph;
        std::unordered_map<int, mapping::frame>& frames;
        bool following;
        // Each camera's nodes by chunk, the reference camera's first.
        std::vector<std::map<long long, node>> nodes;
        // The extrinsics handed to an observation's edge, its camera's alone.
        std::vector<optimisation::vertex*> handed;
        optimisation::vertex* reference_vertex;
        // The chunks the request came with, and every chunk as it was before the solve.
        std::map<long long, calibration_drift::chunk> known;
        std::map<long long, calibration_drift::chunk> before;

        // The frame's view through a camera when the frame is in a chunk, the frame itself for the reference, otherwise null.
        const mapping::frame* followed_view(const int frame_id, const int camera) const;

        node& node_of(const int camera, const long long chunk);

        // The calibration of a camera in a chunk after the solve, from what it was before it.
        calibration_drift::camera_state solved(const size_t camera, const long long chunk, const calibration_drift::camera_state& was) const;

    public:
        // Without a request, or with the landmarks fixed, the frames' calibration is held as it is.
        calibration_drift_refinement(calibration_drift::request* const request_value, optimisation::factor_graph& graph_value, std::unordered_map<int, mapping::frame>& frames_value, const bool landmarks_free);

        calibration_drift_refinement(const calibration_drift_refinement&) = delete;
        calibration_drift_refinement& operator=(const calibration_drift_refinement&) = delete;

        // A camera's pose on the reference camera as a frame holds it: its view's own in a chunk, otherwise the rig's.
        static math::se3<double> reference_from_camera(const mapping::rig& rig, const mapping::frame* const frame, const int camera);

        // A vertex holding that pose, fixed where it is the frame's own and otherwise as the rig gives it.
        static optimisation::vertex extrinsic_vertex(const mapping::rig& rig, const mapping::frame& frame, const int camera);

        // Whether the frame's observations through the camera take their chunk's intrinsics vertex, which is made if need be.
        bool takes(const int frame_id, const int camera);

        // Gives the chunk's intrinsics vertex to an edge made with its intrinsics, after its other vertices.
        void attach(optimisation::edge& factor, const int frame_id, const int camera);

        // The extrinsic vertices for an edge of the frame's observation through the camera (see
        // mapping::map::add_observation_vertices): the chunk's own for a view in a chunk, made if need be, otherwise the rig's.
        const std::vector<optimisation::vertex*>& extrinsics(const int frame_id, const int camera, const std::vector<optimisation::vertex*>& rig_extrinsics);

        // The walks between the chunks and the priors on each, once the observations' edges are in.
        void add_ties();

        // After the solve: every frame's calibration from its chunk, and every chunk's to the request.
        void apply();

        // The reference camera of a frame before the solve, the camera given for a frame that is in no chunk.
        const sensor::model& camera_before(const int frame_id, const sensor::model& otherwise) const;

        // Puts the frames' calibration back, for an adjustment that is undone, and says so in the request.
        void restore();
    };
}

#endif // ZEROSLAM_MAPPING_CALIBRATION_DRIFT_REFINEMENT_HPP
