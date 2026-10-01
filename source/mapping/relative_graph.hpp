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
#ifndef ZEROSLAM_MAPPING_RELATIVE_GRAPH_HPP
#define ZEROSLAM_MAPPING_RELATIVE_GRAPH_HPP

#include "mapping/map.hpp"
#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/relative_bundle_adjustment.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace mapping {
    // The map as relative bundle adjustment holds it (Sibley, Mei, Reid and Newman, "Adaptive Relative Bundle Adjustment",
    // RSS 2009). The keyframes are nodes joined by relative transforms: a chain through each submap, and a similarity for
    // every loop, which absorbs the scale drift a monocular map accumulates around it. Each landmark is an inverse depth
    // point in the frame of one keyframe, its base. There is no world frame, the frames' poses and landmarks' locations
    // in the map are a view of the graph written by embedding it from one keyframe, so an adjustment only changes the
    // transforms and points near where the map changed, and closing a loop only adds a transform.
    class relative_graph final {
    public:
        class link final {
        public:
            int parent = -1;
            int child = -1;
            // X_parent = transform * X_child, in each frame's own units.
            math::sim3<double> transform = math::sim3<double>::identity();
            // A loop also solves for the scale between its frames; a link of the chain keeps a scale of one.
            bool similarity = false;
        };

        class node final {
        public:
            // The link to the previous keyframe of its submap, -1 for the first keyframe of a component.
            int parent_link = -1;
            std::vector<int> links;
            // World from node, in the node's units, as it was last embedded.
            math::sim3<double> embedding = math::sim3<double>::identity();
            // The embedding when the frame joined the graph, where relaxing holds the map's gauge frame.
            math::sim3<double> origin = math::sim3<double>::identity();
            // The frame's pose as it was last embedded, to tell when something else moves the frame.
            math::matrix<double, 3, 3> rotation = math::matrix<double, 3, 3>::identity();
            math::matrix<double, 3, 1> translation = math::matrix<double, 3, 1>::zero();
            // The frame's mean reprojection error, in pixels, after the last adjustment it was active in, negative before one.
            double error = -1.0;
        };

        class based_point final {
        public:
            int base = -1;
            // (x / z, y / z, 1 / z) in the base frame and its units.
            double parameters[3] = { 0.0, 0.0, 0.0 };
            // The landmark's location as it was last embedded, to tell when something else moves the landmark.
            math::matrix<double, 3, 1> location = math::matrix<double, 3, 1>::zero();
        };

        class summary final {
        public:
            int active_frames = 0;
            int static_frames = 0;
            int active_links = 0;
            int active_points = 0;
            int observations = 0;
            int priors = 0;
            double initial_cost = 0.0;
            double final_cost = 0.0;
            int accepted = 0;
            int removed = 0;
            // How many times the region grew and was solved again.
            int iterations = 0;
            bool diverged = false;
            // The points the adjustment moved, whose uncertainty is out of date.
            std::vector<int> adjusted_points;
        };

        // How an adjustment chooses its region.
        class region_settings final {
        public:
            // The change in a frame's mean reprojection error, in pixels, that makes it active (the paper's delta epsilon).
            double error_change = 0.05;
            // The frames nearest the seeds are always active, however little their error has changed.
            size_t frames_minimum = 5;
            // The most frames an adjustment solves for besides its seeds, which bounds its cost at a loop.
            size_t frames_maximum = 30;
            // The most times the region grows by the frames an adjustment disturbs and is solved again.
            int ripples_maximum = 4;
        };

        // Below this many active points seen by two static frames the static frames do not hold the scale.
        constexpr static const size_t scale_links_minimum = 30;
        // The information of a held translation length, relative to its square.
        constexpr static const double scale_prior_information = 1.0e6;
        // The reprojection error counted for an observation the graph cannot project.
        constexpr static const double unprojected_error = 20.0;

        region_settings region;

        std::unordered_map<int, node> nodes;
        // The nodes refer to their links by index, so a removed link keeps its place, with a parent and child of -1.
        std::vector<link> links;
        std::unordered_map<int, based_point> points;

    private:
        std::vector<int> search_queue;
        std::unordered_map<int, int> search_towards;

    public:
        // Brings the graph up to date with the map. Keyframes that have gone leave it, their neighbours linked through
        // them, and new keyframes, given ascending with their submaps, join it as children of the previous keyframe of
        // their submap. A frame or landmark something else moved is taken from the map again, as is a landmark whose base
        // has gone, based at the first keyframe that sees it in front.
        void synchronise(const mapping::map& reconstruction, const std::vector<std::pair<int, size_t>>& keyframes);

        // Links the frames of a loop by the similarity taking the child's coordinates into the parent's, returning its index.
        int add_loop(const int parent, const int child, const math::sim3<double>& child_to_parent);

        // Adapts the graph to the latest changes. The region is the frames nearest the seeds, and outwards from them every
        // frame whose reprojection error has changed since it was last adjusted. The transforms into the active frames, any
        // loop touching one, the links given and the points the active frames see are solved for, while every other frame
        // seeing those points holds them as it is. A static frame whose error the solution changes then joins the region,
        // which is solved again, until the changes die away. Observations beyond the inlier bound leave the map. The map's
        // view is not written, see embed.
        summary adjust(mapping::map& reconstruction, const std::vector<int>& seeds, const std::vector<int>& fresh_links, const int rounds);

        // Writes the view of the root's component into the map: the root keeps its pose and units, every other frame and
        // landmark is placed through the shortest chain of transforms from it, and the lines move with the first frame seeing them.
        void embed(mapping::map& reconstruction, const int root);

        // Writes a view of every component that agrees with all of its transforms as well as it can, from a pose graph over
        // the similarities, for the map to be saved or adjusted as a whole. The map's gauge frame is held where it joined
        // the graph, which keeps the world frame the absolute adjustment has, and any other component at its newest frame.
        bool relax(mapping::map& reconstruction, const int rounds);

        // The steps taking the base's coordinates into the observer's along a shortest chain, false when they are not joined.
        bool path(const int observer, const int base, std::vector<optimisation::relative_bundle_adjustment::step>& steps);

        // The point's coordinates in the observer's frame and units, scaled by its inverse depth, false when not joined.
        bool point_in_frame(const int point_id, const int observer, math::matrix<double, 3, 1>& point);

        // The frames of the graph that the given frame is joined to, itself included.
        std::vector<int> component(const int frame_id);

    private:
        int add_link(const int parent, const int child, const math::sim3<double>& transform, const bool similarity);
        void remove_link(const int index);
        void remove_node(const int frame_id);
        bool base_point(const mapping::point& landmark, const int base, based_point& based) const;
        // Breadth first from the source, recording the link towards it of each frame reached, until the sorted targets are
        // all reached, or through the whole component without them.
        void search(const int source, const std::vector<int>* const targets);
        void steps_to_source(const int from, std::vector<optimisation::relative_bundle_adjustment::step>& steps) const;
        void steps_from_source(const int to, std::vector<optimisation::relative_bundle_adjustment::step>& steps) const;
        double mean_error(const mapping::map& reconstruction, const int frame_id, const std::vector<std::pair<int, const mapping::map::observation*>>& observed);
        // The root's component placed through the shortest chains of transforms from the root's embedding.
        void embed_component(const int root, const math::sim3<double>& root_embedding, std::unordered_map<int, math::sim3<double>>& embeddings);
        void write(mapping::map& reconstruction, const std::unordered_map<int, math::sim3<double>>& embeddings);
    };
}

#endif // ZEROSLAM_MAPPING_RELATIVE_GRAPH_HPP
