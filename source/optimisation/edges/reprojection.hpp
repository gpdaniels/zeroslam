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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_REPROJECTION_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_REPROJECTION_HPP

#include "optimisation/edge.hpp"
#include "optimisation/vertices/intrinsics.hpp"
#include "sensor/camera/model.hpp"

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

namespace optimisation::edges {
    // The reprojection of a point landmark (vertices::point) into the frame of a pose (vertices::pose, world to camera).
    //
    // The camera is the model the edge is given, held fixed, unless the edge is made with_intrinsics: it then takes a third
    // vertex, a vertices::intrinsics shared by every edge of that camera, projects through that vertex's model and gives the
    // jacobian by its free parameters. Optional vertices follow the pose and the landmark, each at the next slot, so that
    // further optional blocks (a sensor's extrinsic) can follow in the same way.
    class reprojection final {
    public:
        constexpr static const char* name = "reprojection";
        constexpr static const int residual_count = 2;
        // The vertices without the optional ones, get_vertex_count gives those of an edge.
        constexpr static const int vertex_count = 2;

        // Every reprojection edge gives a point it cannot project, such as one behind the camera, the same residual in both
        // rows: the slope times the depth behind the camera, which pulls the point back in front and costs little just behind
        // it. A residual bounded away from zero made every such observation an outlier and a step in the cost that LM would
        // not cross, which lost tracking on TUM fr1 and EuRoC sequences, so the pipeline keeps this weak pull.
        constexpr static const double behind_camera_slope = 0.3;

    private:
        sensor::camera::model<double> camera;
        // The slot of the intrinsics vertex, -1 when the camera is held fixed.
        int intrinsics_slot;

    public:
        explicit reprojection(const sensor::camera::model<double>& camera_model, const bool with_intrinsics = false);

        int get_vertex_count() const;

        static double behind_camera_residual(const double depth);

        // The intrinsics vertex at the slot, null for none (a negative slot, or a vertex of another kind there).
        static const vertices::intrinsics* intrinsics_at(const edge& context, const int slot);

        // Writes the jacobian of a residual observed - projected by the intrinsics vertex: minus the projection's jacobian by
        // each free parameter, given by every parameter, or zero when the camera did not project the point (the behind camera
        // residual does not depend on the camera) or there is no such vertex.
        static void set_intrinsics_jacobian(const edge& context, const int slot, const bool projected, const double* const jacobian_parameters, math::matrix<double, 0, 0>& jacobian);

        // Whether the camera projects the point where the vertices put it, so that the residual is a distance in the image and
        // not the behind camera residual.
        bool measurable(const edge& context) const;

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;

        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_REPROJECTION_HPP
