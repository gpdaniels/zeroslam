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
#ifndef ZEROSLAM_MAPPING_PARALLEL_RIGIDITY_HPP
#define ZEROSLAM_MAPPING_PARALLEL_RIGIDITY_HPP

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace mapping {
    // The parts of a bundle adjustment's bipartite graph of frames and points that have a unique solution, after Manam
    // and Govindu, "Parallel Rigidity Matters for Bundle Adjustment" (CVPR 2026). An observation fixes only the direction
    // from a frame's centre to a point, so the centres and the points are fixed, up to one similarity, only within a
    // parallel rigid subgraph: a part joined to the rest by less can be scaled on its own. A four-loop, two frames that
    // both see two points, is rigid, and rigid subgraphs sharing two nodes are rigid together, so the subgraphs are grown
    // from four-loops through the covisibility of the frames, the map's counterpart of the paper's viewgraph (GPRBA). Two
    // frames are joined by an edge when they share two points, and two edges merge when they share a frame and a point
    // (the paper's type 1 interaction; two frames in common, type 2, is one edge). The edges' subgraphs then merge when
    // they share two points (type 3). As in the paper the subgraphs need not be maximal, since a frame or point joined to
    // a rigid subgraph at two nodes outside any four-loop is rigid with it but is not found.
    class parallel_rigidity final {
    public:
        class track final {
        public:
            int point = -1;
            // The frames observing the point, each once.
            std::vector<int> frames;
        };

        class component final {
        public:
            std::vector<int> frames;
            std::vector<int> points;
        };

        class result final {
        public:
            // The rigid subgraphs, the one with the most frames, then points, first.
            std::vector<component> components;
            // The observations, as (point, frame), in no four-loop: those of a point that fewer than two frames sharing
            // another point see, and of a frame sharing the point with no frame that shares another point with it (the
            // paper's hanging observations).
            std::vector<std::pair<int, int>> unlooped;
        };

    public:
        static result analyse(const std::vector<track>& tracks);
    };
}

#endif // ZEROSLAM_MAPPING_PARALLEL_RIGIDITY_HPP
