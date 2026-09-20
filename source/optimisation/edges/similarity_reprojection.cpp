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

#include "optimisation/edges/similarity_reprojection.hpp"

namespace optimisation::edges {
    similarity_reprojection::similarity_reprojection(const sensor::camera::model<double>& camera_model, const math::matrix<double, 3, 1>& point_location, const math::se3<double>& observer_pose, const bool apply_inverse)
        : camera(camera_model)
        , location(point_location)
        , observer(observer_pose)
        , inverted(apply_inverse) {
    }

    void similarity_reprojection::compute_residual(const edge& context, math::matrix<double, 0, 0>& residual) const {
        const double* const parameters = context.get_vertex(0)->get_parameters();
        const math::sim3<double> similarity(math::se3<double>(math::so3<double>(parameters[6], parameters[3], parameters[4], parameters[5]), { { parameters[0], parameters[1], parameters[2] } }), parameters[7]);
        const math::matrix<double, 3, 1> transferred = this->inverted ? (similarity.inverse() * this->location) : (similarity * this->location);
        const math::matrix<double, 3, 1> in_camera = this->observer * transferred;
        math::matrix<double, 2, 1> projected = { { 0.0, 0.0 } };
        if (!this->camera.project(in_camera.data(), projected.data())) {
            const double penalty = -similarity_reprojection::behind_camera_penalty * in_camera[2];
            residual[0][0] = penalty;
            residual[1][0] = penalty;
            return;
        }
        residual[0][0] = context.get_observation()[0][0] - projected[0];
        residual[1][0] = context.get_observation()[1][0] - projected[1];
    }

    void similarity_reprojection::compute_jacobians(const edge& context, std::vector<math::matrix<double, 0, 0>>& jacobians) const {
        const double* const parameters = context.get_vertex(0)->get_parameters();
        const math::sim3<double> similarity(math::se3<double>(math::so3<double>(parameters[6], parameters[3], parameters[4], parameters[5]), { { parameters[0], parameters[1], parameters[2] } }), parameters[7]);
        const math::matrix<double, 3, 1> transferred = this->inverted ? (similarity.inverse() * this->location) : (similarity * this->location);
        const math::matrix<double, 3, 1> in_camera = this->observer * transferred;

        double projected[2];
        math::matrix<double, 2, 3> jacobian_camera;
        if (!this->camera.project(in_camera.data(), projected, jacobian_camera.data())) {
            jacobian_camera[0][0] = 0.0;
            jacobian_camera[0][1] = 0.0;
            jacobian_camera[0][2] = similarity_reprojection::behind_camera_penalty;
            jacobian_camera[1][0] = 0.0;
            jacobian_camera[1][1] = 0.0;
            jacobian_camera[1][2] = similarity_reprojection::behind_camera_penalty;
        }

        const math::matrix<double, 3, 1>& acted = this->inverted ? this->location : transferred;
        const double X = acted[0];
        const double Y = acted[1];
        const double Z = acted[2];
        math::matrix<double, 3, 7> jacobian_point_similarity;
        jacobian_point_similarity[0][0] = 0;
        jacobian_point_similarity[0][1] = Z;
        jacobian_point_similarity[0][2] = -Y;
        jacobian_point_similarity[1][0] = -Z;
        jacobian_point_similarity[1][1] = 0;
        jacobian_point_similarity[1][2] = X;
        jacobian_point_similarity[2][0] = Y;
        jacobian_point_similarity[2][1] = -X;
        jacobian_point_similarity[2][2] = 0;
        jacobian_point_similarity[0][3] = 1;
        jacobian_point_similarity[0][4] = 0;
        jacobian_point_similarity[0][5] = 0;
        jacobian_point_similarity[1][3] = 0;
        jacobian_point_similarity[1][4] = 1;
        jacobian_point_similarity[1][5] = 0;
        jacobian_point_similarity[2][3] = 0;
        jacobian_point_similarity[2][4] = 0;
        jacobian_point_similarity[2][5] = 1;
        jacobian_point_similarity[0][6] = X;
        jacobian_point_similarity[1][6] = Y;
        jacobian_point_similarity[2][6] = Z;

        const math::matrix<double, 3, 3> rotation_observer = this->observer.rotation().get_matrix();
        const math::matrix<double, 3, 3> jacobian_transfer = this->inverted ? ((rotation_observer * math::transpose(similarity.transformation().rotation().get_matrix())) * (-1.0 / similarity.scale())) : rotation_observer;

        const math::matrix<double, 2, 7> jacobian_similarity_res = -((jacobian_camera * jacobian_transfer) * jacobian_point_similarity);
        jacobians[0] = math::matrix<double, 0, 0>(2, 7, jacobian_similarity_res.data());
    }
}
