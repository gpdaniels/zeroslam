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
#ifndef ZEROSLAM_OPTIMISATION_VERTICES_INTRINSICS_HPP
#define ZEROSLAM_OPTIMISATION_VERTICES_INTRINSICS_HPP

#include "optimisation/vertex.hpp"
#include "sensor/camera/model.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <memory>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace optimisation::vertices {
    // The parameters of a camera, one vertex shared by the edges of every frame taken with it.
    //
    // The vertex holds the whole parameter vector, as many values as the camera model has, and solves for some of them:
    // each parameter is held fixed or updated by one of the local dimensions. Parameters that share a local dimension move
    // together and keep their ratios as they were when the vertex was made, as the two focal lengths of a camera whose
    // pixels' aspect ratio is known do: then a motion that observes one of them (a turn about the vertical observes the
    // horizontal focal length, and no motion in a plane observes the vertical one) fixes both. A local dimension updates
    // its first parameter additively and the others in proportion.
    //
    // The vertex also holds the camera model set to its parameters, which the edges project through, so the model is set
    // once per step rather than once per edge. A step that the model refuses (a focal length that is not positive) leaves
    // the parameters where they were.
    class intrinsics final {
    public:
        constexpr static const char* name = "intrinsics";
        // The sizes of a vertex made from a camera are the camera's, given by get_parameter_count and get_local_count.
        constexpr static const int parameter_count = 1;
        constexpr static const int local_count = 1;

    private:
        // Held apart from the vertex, which is too small for a camera model.
        class state final {
        public:
            sensor::camera::model<double> camera;
            double weights[vertex::maximum_parameters] = {};
            signed char locals[vertex::maximum_parameters] = {};
        };

        std::unique_ptr<state> held;
        int count;
        int free_count;

    public:
        // A vertex over the camera's parameters, each given the local dimension that updates it or -1 to hold it fixed. The
        // local dimensions must be 0 to some count, each updating at least one parameter, or the vertex is invalid, as one
        // with no free parameter or more parameters than a vertex holds is.
        intrinsics(const sensor::camera::model<double>& camera_model, const int* const parameter_locals, const size_t parameter_locals_length);

        // A vertex with each parameter that the mask sets free updated by a local dimension of its own.
        intrinsics(const sensor::camera::model<double>& camera_model, const bool* const free_parameters, const size_t free_parameters_length);

        intrinsics(const intrinsics& other);
        intrinsics(intrinsics&& other);
        intrinsics& operator=(const intrinsics& other);
        intrinsics& operator=(intrinsics&& other);
        ~intrinsics();

        // An erased vertex over the camera with its parameters set from the camera, invalid if the camera or locals are.
        static vertex make(const sensor::camera::model<double>& camera_model, const int* const parameter_locals, const size_t parameter_locals_length);

        static vertex make(const sensor::camera::model<double>& camera_model, const bool* const free_parameters, const size_t free_parameters_length);

    public:
        int get_parameter_count() const;

        int get_local_count() const;

        void plus(double* const parameters, const double* const delta) const;

        void synchronise(const double* const parameters) const;

    public:
        // The camera model at the vertex's current parameters.
        const sensor::camera::model<double>& get_camera() const;

        // The first parameter a local dimension updates.
        int get_parameter_index(const int local_index) const;

        // The local dimension that updates a parameter, -1 for a parameter held fixed.
        int get_local_index(const int parameter_index) const;

        // How far a parameter moves for a unit step of its local dimension, zero for a parameter held fixed.
        double get_weight(const int parameter_index) const;

        // A row major jacobian by the local dimensions, scaled, from one by every parameter, rows by get_parameter_count.
        void select_free_columns(const double* const jacobian_parameters, const size_t rows, const double scale, double* const jacobian_local) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_VERTICES_INTRINSICS_HPP
