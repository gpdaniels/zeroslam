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

#include "estimation/robust/solver/generalised_p3p.hpp"

#include "estimation/correspondence_2d_3d.hpp"
#include "estimation/robust/solver/p3p.hpp"
#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace estimation::robust::solver {
    template <typename type>
    bool generalised_p3p<type>::solve(
        const correspondence* const correspondences,
        const size_t data_size,
        const math::se3<type>* const camera_from_reference,
        const size_t camera_count,
        size_t* const inliers,
        size_t& inliers_size,
        model& best_model,
        const float residual_threshold
    ) {
        inliers_size = 0;
        // Whether a point agrees with a world to reference pose, by the angle between its bearing and the reprojected one.
        const auto agrees = [&](const correspondence& match, const math::se3<type>& reference_from_world) {
            const math::matrix<type, 3, 1> world({ match.point[0], match.point[1], match.point[2] });
            const math::matrix<type, 3, 1> seen = camera_from_reference[match.camera] * (reference_from_world * world);
            const type seen_norm = math::sqrt(seen.get_length_squared());
            const type bearing_norm = math::sqrt((match.bearing[0] * match.bearing[0]) + (match.bearing[1] * match.bearing[1]) + type(1));
            if (!(seen_norm > type(0))) {
                return false;
            }
            const type cosine = ((seen[0] * match.bearing[0]) + (seen[1] * match.bearing[1]) + seen[2]) / (seen_norm * bearing_norm);
            return (type(1) - cosine) <= static_cast<type>(residual_threshold);
        };
        std::vector<correspondence_2d_3d<type>> camera_points;
        std::vector<float> residuals;
        std::vector<size_t> camera_inliers;
        bool found = false;
        size_t best_support = 0;
        for (size_t camera = 0; camera < camera_count; ++camera) {
            camera_points.clear();
            for (size_t i = 0; i < data_size; ++i) {
                if (correspondences[i].camera == camera) {
                    correspondence_2d_3d<type> entry;
                    entry.lhs[0] = correspondences[i].bearing[0];
                    entry.lhs[1] = correspondences[i].bearing[1];
                    entry.rhs[0] = correspondences[i].point[0];
                    entry.rhs[1] = correspondences[i].point[1];
                    entry.rhs[2] = correspondences[i].point[2];
                    camera_points.push_back(entry);
                }
            }
            if (camera_points.size() < generalised_p3p::camera_points_minimum) {
                continue;
            }
            residuals.assign(camera_points.size(), 0.0f);
            camera_inliers.assign(camera_points.size(), 0);
            size_t camera_inliers_size = 0;
            typename p3p<type>::model_type proposal;
            if (!p3p<type>::solve(camera_points.data(), camera_points.size(), residuals.data(), camera_inliers.data(), camera_inliers_size, proposal, residual_threshold)) {
                continue;
            }
            const math::matrix<type, 3, 3> rotation({ { proposal.rotation[0][0], proposal.rotation[0][1], proposal.rotation[0][2] }, { proposal.rotation[1][0], proposal.rotation[1][1], proposal.rotation[1][2] }, { proposal.rotation[2][0], proposal.rotation[2][1], proposal.rotation[2][2] } });
            const math::se3<type> camera_from_world(math::so3<type>(rotation), { { proposal.translation[0], proposal.translation[1], proposal.translation[2] } });
            const math::se3<type> reference_from_world = camera_from_reference[camera].inverse() * camera_from_world;
            size_t support = 0;
            for (size_t i = 0; i < data_size; ++i) {
                support += ((correspondences[i].camera < camera_count) && agrees(correspondences[i], reference_from_world)) ? 1u : 0u;
            }
            if (!found || (support > best_support)) {
                found = true;
                best_support = support;
                const math::matrix<type, 3, 3> best_rotation = reference_from_world.rotation().get_matrix();
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t column = 0; column < 3; ++column) {
                        best_model.rotation[row][column] = best_rotation[row][column];
                    }
                    best_model.translation[row] = reference_from_world.translation()[row];
                }
                best_model.camera = camera;
            }
        }
        if (!found) {
            return false;
        }
        const math::matrix<type, 3, 3> rotation({ { best_model.rotation[0][0], best_model.rotation[0][1], best_model.rotation[0][2] }, { best_model.rotation[1][0], best_model.rotation[1][1], best_model.rotation[1][2] }, { best_model.rotation[2][0], best_model.rotation[2][1], best_model.rotation[2][2] } });
        const math::se3<type> reference_from_world(math::so3<type>(rotation), { { best_model.translation[0], best_model.translation[1], best_model.translation[2] } });
        for (size_t i = 0; i < data_size; ++i) {
            if ((correspondences[i].camera < camera_count) && agrees(correspondences[i], reference_from_world)) {
                inliers[inliers_size++] = i;
            }
        }
        return true;
    }

    template class generalised_p3p<double>;
}
