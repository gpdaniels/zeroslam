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

#include "optimisation/edges/planar_motion.hpp"

#include "math/lie.hpp"
#include "math/math.hpp"
#include "optimisation/edges/odometry.hpp"

namespace optimisation::edges {
    math::matrix<double, 0, 0> planar_motion::information(const double distance, const double tilt_density, const double vertical_density, const double minimum_tilt, const double minimum_vertical) {
        const double travelled = math::abs(distance);
        const double tilt = (tilt_density * tilt_density * travelled) + (minimum_tilt * minimum_tilt);
        const double vertical = (vertical_density * vertical_density * travelled) + (minimum_vertical * minimum_vertical);
        math::matrix<double, 0, 0> result = math::matrix<double, 0, 0>::zero(3, 3);
        result[0][0] = (tilt > 0.0) ? (1.0 / tilt) : 0.0;
        result[1][1] = result[0][0];
        result[2][2] = (vertical > 0.0) ? (1.0 / vertical) : 0.0;
        return result;
    }

    void planar_motion::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const math::se3<double> first = odometry::read_pose(context.get_vertex(0)->get_parameters());
        const math::se3<double> second = odometry::read_pose(context.get_vertex(1)->get_parameters());
        const math::se3<double> extrinsic = odometry::read_pose(context.get_vertex(2)->get_parameters());
        const math::se3<double> motion = odometry::relative(first, second, extrinsic);
        const math::matrix<double, 3, 3> rotation = motion.rotation().get_matrix();
        residual[0][0] = rotation[0][2];
        residual[1][0] = rotation[1][2];
        residual[2][0] = motion.translation()[2];
    }

    void planar_motion::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        const math::se3<double> first = odometry::read_pose(context.get_vertex(0)->get_parameters());
        const math::se3<double> second = odometry::read_pose(context.get_vertex(1)->get_parameters());
        const math::se3<double> extrinsic = odometry::read_pose(context.get_vertex(2)->get_parameters());
        const math::se3<double> motion = odometry::relative(first, second, extrinsic);
        const math::matrix<double, 3, 3> rotation = motion.rotation().get_matrix();
        const math::matrix<double, 3, 1> axis = { { rotation[0][2], rotation[1][2], rotation[2][2] } };
        const math::matrix<double, 3, 1>& translation = motion.translation();

        // The residual by a perturbation exp(phi) * T_oo of the motion: the axis R_oo * e_z turns by omega x axis, the height moves
        // by upsilon_z - ([t_oo]x * omega)_z.
        math::matrix<double, 3, 6> residual_motion = math::matrix<double, 3, 6>::zero();
        residual_motion[0][1] = axis[2];
        residual_motion[0][2] = -axis[1];
        residual_motion[1][0] = -axis[2];
        residual_motion[1][2] = axis[0];
        residual_motion[2][0] = translation[1];
        residual_motion[2][1] = -translation[0];
        residual_motion[2][5] = 1.0;

        const math::matrix<double, 3, 6> jacobian_first = residual_motion * odometry::adjoint(extrinsic);
        const math::matrix<double, 3, 6> jacobian_second = -(residual_motion * odometry::adjoint(motion * extrinsic));
        const math::matrix<double, 3, 6> jacobian_extrinsic = residual_motion * (math::matrix<double, 6, 6>::identity() - odometry::adjoint(motion));
        edge::set_jacobian(jacobians[0], 3, 6, jacobian_first.data());
        edge::set_jacobian(jacobians[1], 3, 6, jacobian_second.data());
        edge::set_jacobian(jacobians[2], 3, 6, jacobian_extrinsic.data());
    }
}
