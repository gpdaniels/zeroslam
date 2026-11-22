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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_ODOMETRY_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_ODOMETRY_HPP

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "sensor/odometry/preintegration.hpp"

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
    // A preintegrated wheel odometry measurement between two body poses. The vertices are, in order:
    // - the body poses at the start and at the end of the interval, pose vertices holding the world to body transform T_bw (the
    //   convention of the camera poses, so that with a single camera the body is the camera);
    // - the odometry extrinsic, a pose vertex holding the body to odometry frame transform T_ob (p_odometry = T_ob * p_body), the
    //   sensor from body convention a camera's extrinsic takes too, updated on the left like a pose, so that its local dimensions
    //   are a rotation and a translation in the odometry frame; it is fixed when the extrinsic is not estimated;
    // - the calibration, a calibration vertex holding the model's parameters with a mask of the free ones.
    //
    // The odometry frame moves from T_oo = T_ob * T_bw(start) * T_bw(end)^-1 * T_ob^-1 over the interval. The measurement is the
    // planar motion (x, y, theta) at the current calibration, to first order from the parameters it was integrated with, lifted to
    // the rotation Rz(theta) and the translation (x, y, 0). The residual is the rotation error Log(Rz(theta)^T * R_oo) followed by
    // the translation error t_oo - (x, y, 0) in the start's odometry frame, so that its planar part, the third rotation and the
    // first two translation rows, is the preintegrated motion's and its covariance is the preintegration's, and the other three
    // rows, roll, pitch and height, hold the motion to the plane as softly as the noise's tilt and vertical densities say.
    //
    // The observation, built by observation(), is the preintegrated motion, the parameters it was integrated with, and its
    // Jacobian by them, (x, y, theta, p0[P], J[3 x P] row major), so that an edge already in a graph takes a preintegration
    // integrated again with set_observation and set_information.
    //
    // The planar motion of an odometer leaves the extrinsic's height above the plane unobservable: a translation of the odometry
    // frame along its z axis commutes with every rotation about that axis, so no planar motion moves it, and it needs a prior
    // (edges::extrinsic_prior), as does any extrinsic component a trajectory with too little turning leaves unseen.
    class odometry final {
    public:
        constexpr static const char* name = "odometry";
        constexpr static const int residual_count = 6;
        constexpr static const int vertex_count = 4;

        enum slot : size_t {
            first_pose = 0,
            second_pose = 1,
            extrinsic = 2,
            calibration = 3
        };

    public:
        // The observation and the information of a preintegrated motion.
        static math::matrix<double, 0, 0> observation(const sensor::odometry::preintegration& motion);
        static math::matrix<double, 0, 0> information(const sensor::odometry::preintegration& motion);

        // The planar motion an edge measures at its calibration vertex's current parameters.
        static math::matrix<double, 3, 1> measurement(const edge& context);

        // The transform a pose vertex holds, translation then quaternion (x, y, z, w).
        static math::se3<double> read_pose(const double* const parameters);
        // The parameters of a pose vertex holding the transform.
        static void write_pose(const math::se3<double>& pose, double* const parameters);

        // The adjoint of a transform on the (rotation, translation) tangent: T * exp(xi) * T^-1 = exp(adjoint(T) * xi).
        static math::matrix<double, 6, 6> adjoint(const math::se3<double>& pose);
        static math::matrix<double, 3, 3> skew(const math::matrix<double, 3, 1>& vector);

        // The motion of the odometry frame between two body poses (world to body) through the extrinsic (body to odometry).
        static math::se3<double> relative(const math::se3<double>& first_pose, const math::se3<double>& second_pose, const math::se3<double>& extrinsic);
        // A planar motion as a transform, the rotation Rz(theta) and the translation (x, y, 0).
        static math::se3<double> lift(const math::matrix<double, 3, 1>& planar);
        // The body pose (world to body) that follows a body pose by a planar motion of the odometry frame, for a prediction.
        static math::se3<double> predict(const math::se3<double>& first_pose, const math::se3<double>& extrinsic, const math::matrix<double, 3, 1>& planar);

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;
        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_ODOMETRY_HPP
