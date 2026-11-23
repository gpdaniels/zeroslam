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
#ifndef ZEROSLAM_OPTIMISATION_EDGES_INERTIAL_HPP
#define ZEROSLAM_OPTIMISATION_EDGES_INERTIAL_HPP

#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/edge.hpp"
#include "sensor/imu/preintegration.hpp"

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
    // A preintegrated imu measurement between two body poses, after Forster, Carlone, Dellaert and Scaramuzza (TRO 2017).
    //
    // The vertices are the body pose at i (a pose vertex, world to body), the imu's velocity at i in the world (a velocity vertex),
    // the imu bias at i (an imu_bias vertex, gyroscope then accelerometer), the body pose at j and the imu's velocity at j. The bias
    // at j is tied to the bias at i by a bias_random_walk edge. The imu sits on the body at a fixed transform, body_from_imu, which
    // is the identity when the body frame is the imu's, and gravity in the world is a fixed vector. The residual is
    //   r_R = Log(dR(b)^T R_i^T R_j),
    //   r_v = R_i^T (v_j - v_i - g dt) - dv(b),
    //   r_p = R_i^T (p_j - p_i - v_i dt - g dt^2 / 2) - dp(b),
    // with R and p the imu's orientation and position in the world, the deltas corrected to first order for the bias, and the
    // information the inverse of the preintegration's covariance. The jacobians are analytic.
    //
    // The measurement does not fit in the edge, so it is the edge's observation, observation(preintegration) laid out as
    // [dR (x, y, z, w), dv, dp, dt, b_g0, b_a0, J_R^g, J_v^g, J_v^a, J_p^g, J_p^a] with the jacobians row major, 62 values.
    //
    // The imu's place on the body is a fixed parameter, or, when the edge is made to take one, a sixth vertex: an extrinsic vertex
    // (optimisation::vertices::extrinsic, sensor to body, updated on the left) that an adjustment may refine, which the
    // parameter then only starts.
    class inertial final {
    public:
        constexpr static const char* name = "inertial";
        constexpr static const int residual_count = 9;
        constexpr static const int vertex_count = 5;
        constexpr static const int observation_count = 62;

        // The measurement read back from an edge's observation.
        class measurement final {
        public:
            math::so3<double> delta_rotation = math::so3<double>::identity();
            math::matrix<double, 3, 1> delta_velocity = math::matrix<double, 3, 1>::zero();
            math::matrix<double, 3, 1> delta_position = math::matrix<double, 3, 1>::zero();
            double delta_time = 0.0;
            sensor::imu::bias linearisation;
            math::matrix<double, 3, 3> rotation_gyroscope = math::matrix<double, 3, 3>::zero();
            math::matrix<double, 3, 3> velocity_gyroscope = math::matrix<double, 3, 3>::zero();
            math::matrix<double, 3, 3> velocity_accelerometer = math::matrix<double, 3, 3>::zero();
            math::matrix<double, 3, 3> position_gyroscope = math::matrix<double, 3, 3>::zero();
            math::matrix<double, 3, 3> position_accelerometer = math::matrix<double, 3, 3>::zero();

        public:
            static measurement from_observation(const math::matrix<double, 0, 0>& observation);
        };

        // The jacobians of the residual with respect to the imu's states: a right perturbation of each rotation, R * Exp(dphi), a
        // perturbation of each position and velocity in the world, the bias (gyroscope then accelerometer) and gravity.
        class state_jacobians final {
        public:
            math::matrix<double, 9, 3> rotation_i;
            math::matrix<double, 9, 3> position_i;
            math::matrix<double, 9, 3> velocity_i;
            math::matrix<double, 9, 6> bias;
            math::matrix<double, 9, 3> rotation_j;
            math::matrix<double, 9, 3> position_j;
            math::matrix<double, 9, 3> velocity_j;
            math::matrix<double, 9, 3> gravity;
        };

    private:
        math::matrix<double, 3, 1> gravity;
        math::se3<double> body_from_imu;
        bool extrinsic_vertex;

    public:
        explicit inertial(const math::matrix<double, 3, 1>& gravity_world, const math::se3<double>& imu_on_body = math::se3<double>::identity(), const bool with_extrinsic_vertex = false);

        // Five vertices, or six with the extrinsic vertex.
        int get_vertex_count() const;

    public:
        // The observation of a preintegration, and its information, the inverse of its covariance (false when it cannot be inverted).
        static math::matrix<double, 0, 0> observation(const sensor::imu::preintegration& preintegration);
        static bool information(const sensor::imu::preintegration& preintegration, math::matrix<double, 0, 0>& information_matrix);

        // The residual of a measurement between two imu states, and optionally its jacobians.
        static void evaluate(
            const measurement& preintegrated,
            const sensor::imu::preintegration::state& first,
            const sensor::imu::preintegration::state& second,
            const sensor::imu::bias& estimate,
            const math::matrix<double, 3, 1>& gravity_world,
            double* const residual,
            state_jacobians* const jacobians
        );

        // The imu's state from the parameters of a body pose vertex (world to body) and a velocity vertex, with the body's position
        // multiplied by a scale, as a monocular map's must be to be metric: R_wi = R_wb R_bi and p_wi = s p_wb + R_wb p_bi.
        static sensor::imu::preintegration::state imu_state(const double* const pose, const double* const velocity, const math::se3<double>& imu_on_body, const double scale = 1.0);

        // The jacobian with respect to a body pose vertex's update, exp(delta) * T_bw, from those with respect to the imu's rotation
        // and position at that pose.
        static math::matrix<double, 9, 6> pose_jacobian(const math::matrix<double, 9, 3>& rotation, const math::matrix<double, 9, 3>& position, const double* const pose, const math::se3<double>& imu_on_body, const double scale = 1.0);

        // The jacobian with respect to an extrinsic vertex's update, exp(delta) * T_bi, from those with respect to the imu's
        // rotation and position at a body pose.
        static math::matrix<double, 9, 6> extrinsic_jacobian(const math::matrix<double, 9, 3>& rotation, const math::matrix<double, 9, 3>& position, const double* const pose, const math::se3<double>& imu_on_body);

        void compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const;

        void compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const;
    };
}

#endif // ZEROSLAM_OPTIMISATION_EDGES_INERTIAL_HPP
