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

#include "sensor/camera/pinhole_opencv.hpp"

#include "core/assert.hpp"
#include "math/math.hpp"
#include "math/polynomial.hpp"

namespace {
    // A radius of a million is within a microradian of ninety degrees, so any fold further out is never seen.
    constexpr static const double fold_search_limit = 1.0e12;

    // The squared radius u = r * r where the radial distortion r * N(u) / D(u) first stops increasing, infinite when it never does.
    // Its derivative has the sign of N * D + 2 * u * (N' * D - N * D'), and it also ends at a pole, where D reaches zero.
    double fold_radius_squared(const double* const radial_k) {
        const double numerator[4] = { 1.0, radial_k[0], radial_k[1], radial_k[2] };
        const double denominator[4] = { 1.0, radial_k[3], radial_k[4], radial_k[5] };
        double derivative[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                derivative[i + j] += (1.0 + (2.0 * static_cast<double>(i)) - (2.0 * static_cast<double>(j))) * numerator[i] * denominator[j];
            }
        }
        return math::min(math::polynomial::first_non_positive(&derivative[0], 6, fold_search_limit), math::polynomial::first_non_positive(&denominator[0], 3, fold_search_limit));
    }

    template <typename type>
    void multiply(const type* const lhs, const type* const rhs, type* const result) {
        for (size_t row = 0; row < 3; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                result[row * 3 + column] = (lhs[row * 3 + 0] * rhs[0 * 3 + column]) + (lhs[row * 3 + 1] * rhs[1 * 3 + column]) + (lhs[row * 3 + 2] * rhs[2 * 3 + column]);
            }
        }
    }

    // OpenCV's computeTiltProjectionMatrix: the rotation R = R_y(tau_y) R_x(tau_x) of the sensor followed by the projection
    // back onto the plane z = 1 along the optical axis, T = P_z(R) R, and its derivatives with respect to the two angles.
    template <typename type>
    void tilt_projection(const type tau_x, const type tau_y, type* const tilt, type* const tilt_by_x, type* const tilt_by_y) {
        const type cos_x = math::cos(tau_x);
        const type sin_x = math::sin(tau_x);
        const type cos_y = math::cos(tau_y);
        const type sin_y = math::sin(tau_y);
        const type rotation_x[9] = { type(1), type(0), type(0), type(0), cos_x, sin_x, type(0), -sin_x, cos_x };
        const type rotation_y[9] = { cos_y, type(0), -sin_y, type(0), type(1), type(0), sin_y, type(0), cos_y };
        type rotation[9];
        multiply(&rotation_y[0], &rotation_x[0], &rotation[0]);
        const type projection[9] = { rotation[8], type(0), -rotation[2], type(0), rotation[8], -rotation[5], type(0), type(0), type(1) };
        multiply(&projection[0], &rotation[0], tilt);
        if ((tilt_by_x == nullptr) || (tilt_by_y == nullptr)) {
            return;
        }
        const type rotation_x_derivative[9] = { type(0), type(0), type(0), type(0), -sin_x, cos_x, type(0), -cos_x, -sin_x };
        const type rotation_y_derivative[9] = { -sin_y, type(0), -cos_y, type(0), type(0), type(0), cos_y, type(0), -sin_y };
        type rotation_by[2][9];
        multiply(&rotation_y[0], &rotation_x_derivative[0], &rotation_by[0][0]);
        multiply(&rotation_y_derivative[0], &rotation_x[0], &rotation_by[1][0]);
        type* const results[2] = { tilt_by_x, tilt_by_y };
        for (size_t angle = 0; angle < 2; ++angle) {
            const type* const derivative = &rotation_by[angle][0];
            const type projection_derivative[9] = { derivative[8], type(0), -derivative[2], type(0), derivative[8], -derivative[5], type(0), type(0), type(0) };
            type first[9];
            type second[9];
            multiply(&projection_derivative[0], &rotation[0], &first[0]);
            multiply(&projection[0], derivative, &second[0]);
            for (size_t index = 0; index < 9; ++index) {
                results[angle][index] = first[index] + second[index];
            }
        }
    }

    // The point (x, y) through the homography h, (u, v, w) = h (x, y, 1) to (u / w, v / w), with its row major 2x2 jacobian,
    // false when w is not positive.
    template <typename type>
    bool apply_homography(const type* const h, const type* const point, type* const result, type* const jacobian) {
        const type u = (h[0] * point[0]) + (h[1] * point[1]) + h[2];
        const type v = (h[3] * point[0]) + (h[4] * point[1]) + h[5];
        const type w = (h[6] * point[0]) + (h[7] * point[1]) + h[8];
        if (!(w > type(1.0e-12))) {
            return false;
        }
        const type inverse_w = type(1) / w;
        result[0] = u * inverse_w;
        result[1] = v * inverse_w;
        if (jacobian != nullptr) {
            jacobian[0 * 2 + 0] = (h[0] - (result[0] * h[6])) * inverse_w;
            jacobian[0 * 2 + 1] = (h[1] - (result[0] * h[7])) * inverse_w;
            jacobian[1 * 2 + 0] = (h[3] - (result[1] * h[6])) * inverse_w;
            jacobian[1 * 2 + 1] = (h[4] - (result[1] * h[7])) * inverse_w;
        }
        return true;
    }
}

namespace sensor::camera {
    template <typename type>
    pinhole_opencv<type>::pinhole_opencv()
        : focal_lengths{ type(1), type(1) }
        , centre_points{ type(0.5), type(0.5) }
        , radial_k{ type(0), type(0), type(0), type(0), type(0), type(0) }
        , tangential_p{ type(0), type(0) }
        , prism_s{ type(0), type(0), type(0), type(0) }
        , tilt{ type(0), type(0) }
        , tilt_matrix{ type(1), type(0), type(0), type(0), type(1), type(0), type(0), type(0), type(1) }
        , maximum_radius_squared(math::inf<type>()) {
    }

    template <typename type>
    pinhole_opencv<type>::pinhole_opencv(const type* const parameters, const size_t parameters_length)
        : pinhole_opencv() {
        const bool valid = this->set_parameters(parameters, parameters_length);
        ASSERT(valid, "The OpenCV parameters must be eighteen finite values with positive focal lengths and tilts below a quarter turn.");
        static_cast<void>(valid);
    }

    template <typename type>
    size_t pinhole_opencv<type>::get_parameter_count() const {
        return pinhole_opencv::parameter_count;
    }

    template <typename type>
    bool pinhole_opencv<type>::set_parameters(const type* const parameters, const size_t parameters_length) {
        if ((parameters == nullptr) || (parameters_length != pinhole_opencv::parameter_count)) {
            return false;
        }
        for (size_t index = 0; index < pinhole_opencv::parameter_count; ++index) {
            if (!math::isfinite(parameters[index])) {
                return false;
            }
        }
        if (!(parameters[0] > type(0)) || !(parameters[1] > type(0))) {
            return false;
        }
        const type quarter_turn = type(0.5) * math::pi<type>();
        if (!(math::abs(parameters[16]) < quarter_turn) || !(math::abs(parameters[17]) < quarter_turn)) {
            return false;
        }
        this->focal_lengths[0] = parameters[0];
        this->focal_lengths[1] = parameters[1];
        this->centre_points[0] = parameters[2];
        this->centre_points[1] = parameters[3];
        this->radial_k[0] = parameters[4];
        this->radial_k[1] = parameters[5];
        this->tangential_p[0] = parameters[6];
        this->tangential_p[1] = parameters[7];
        this->radial_k[2] = parameters[8];
        this->radial_k[3] = parameters[9];
        this->radial_k[4] = parameters[10];
        this->radial_k[5] = parameters[11];
        for (size_t index = 0; index < 4; ++index) {
            this->prism_s[index] = parameters[12 + index];
        }
        this->tilt[0] = parameters[16];
        this->tilt[1] = parameters[17];
        tilt_projection<type>(this->tilt[0], this->tilt[1], &this->tilt_matrix[0], nullptr, nullptr);
        const double radial_k_as_double[6] = {
            static_cast<double>(this->radial_k[0]),
            static_cast<double>(this->radial_k[1]),
            static_cast<double>(this->radial_k[2]),
            static_cast<double>(this->radial_k[3]),
            static_cast<double>(this->radial_k[4]),
            static_cast<double>(this->radial_k[5])
        };
        this->maximum_radius_squared = static_cast<type>(fold_radius_squared(&radial_k_as_double[0]));
        return true;
    }

    template <typename type>
    bool pinhole_opencv<type>::get_parameters(type* const parameters, const size_t parameters_length) const {
        if ((parameters == nullptr) || (parameters_length != pinhole_opencv::parameter_count)) {
            return false;
        }
        parameters[0] = this->focal_lengths[0];
        parameters[1] = this->focal_lengths[1];
        parameters[2] = this->centre_points[0];
        parameters[3] = this->centre_points[1];
        parameters[4] = this->radial_k[0];
        parameters[5] = this->radial_k[1];
        parameters[6] = this->tangential_p[0];
        parameters[7] = this->tangential_p[1];
        parameters[8] = this->radial_k[2];
        parameters[9] = this->radial_k[3];
        parameters[10] = this->radial_k[4];
        parameters[11] = this->radial_k[5];
        for (size_t index = 0; index < 4; ++index) {
            parameters[12 + index] = this->prism_s[index];
        }
        parameters[16] = this->tilt[0];
        parameters[17] = this->tilt[1];
        return true;
    }

    template <typename type>
    bool pinhole_opencv<type>::get_pinhole(type* const pinhole_parameters, const size_t pinhole_parameters_length) const {
        if ((pinhole_parameters == nullptr) || (pinhole_parameters_length != 4)) {
            return false;
        }
        // The tilt keeps the optical axis at the origin and scales the plane there by T00 / T22 and T11 / T22, besides a small
        // shear the pinhole leaves out; the distortion is the identity to first order there.
        pinhole_parameters[0] = this->focal_lengths[0] * this->tilt_matrix[0] / this->tilt_matrix[8];
        pinhole_parameters[1] = this->focal_lengths[1] * this->tilt_matrix[4] / this->tilt_matrix[8];
        pinhole_parameters[2] = this->centre_points[0];
        pinhole_parameters[3] = this->centre_points[1];
        return true;
    }

    template <typename type>
    bool pinhole_opencv<type>::get_parameter_roles(parameter_role* const roles, const size_t roles_length) const {
        if ((roles == nullptr) || (roles_length != pinhole_opencv::parameter_count)) {
            return false;
        }
        // k1 k2 p1 p2 as the radial tangential model has them; k3, the rational k4 k5 k6, the thin prism and the tilt only a wide or unusual lens needs.
        const parameter_role layout[pinhole_opencv::parameter_count] = {
            parameter_role::focal_length,
            parameter_role::focal_length,
            parameter_role::principal_point,
            parameter_role::principal_point,
            parameter_role::distortion,
            parameter_role::distortion,
            parameter_role::distortion,
            parameter_role::distortion,
            parameter_role::distortion_higher,
            parameter_role::distortion_higher,
            parameter_role::distortion_higher,
            parameter_role::distortion_higher,
            parameter_role::distortion_higher,
            parameter_role::distortion_higher,
            parameter_role::distortion_higher,
            parameter_role::distortion_higher,
            parameter_role::distortion_higher,
            parameter_role::distortion_higher
        };
        for (size_t index = 0; index < pinhole_opencv::parameter_count; ++index) {
            roles[index] = layout[index];
        }
        return true;
    }

    template <typename type>
    bool pinhole_opencv<type>::distort(
        const type* const undistorted_xy,
        type* const distorted_xy,
        type* const jacobian_distortion,
        type* const jacobian_coefficients
    ) const {
        const type x = undistorted_xy[0];
        const type y = undistorted_xy[1];
        const type xx = x * x;
        const type yy = y * y;
        const type xy = x * y;
        const type r2 = xx + yy;
        if (r2 > this->maximum_radius_squared) {
            return false;
        }
        const type r4 = r2 * r2;
        const type numerator = type(1.0) + r2 * (this->radial_k[0] + r2 * (this->radial_k[1] + r2 * this->radial_k[2]));
        const type denominator = type(1.0) + r2 * (this->radial_k[3] + r2 * (this->radial_k[4] + r2 * this->radial_k[5]));
        if (!(denominator > type(1.0e-12)) && !(denominator < -type(1.0e-12))) {
            return false;
        }
        const type scale = numerator / denominator;

        distorted_xy[0] = x * scale + type(2.0) * this->tangential_p[0] * xy + this->tangential_p[1] * (r2 + type(2.0) * xx) + this->prism_s[0] * r2 + this->prism_s[1] * r4;
        distorted_xy[1] = y * scale + type(2.0) * this->tangential_p[1] * xy + this->tangential_p[0] * (r2 + type(2.0) * yy) + this->prism_s[2] * r2 + this->prism_s[3] * r4;

        if (jacobian_distortion != nullptr) {
            const type numerator_derivative = this->radial_k[0] + r2 * (type(2.0) * this->radial_k[1] + r2 * type(3.0) * this->radial_k[2]);
            const type denominator_derivative = this->radial_k[3] + r2 * (type(2.0) * this->radial_k[4] + r2 * type(3.0) * this->radial_k[5]);
            const type scale_derivative = (numerator_derivative * denominator - numerator * denominator_derivative) / (denominator * denominator);
            const type prism_x = type(2.0) * (this->prism_s[0] + type(2.0) * this->prism_s[1] * r2);
            const type prism_y = type(2.0) * (this->prism_s[2] + type(2.0) * this->prism_s[3] * r2);
            jacobian_distortion[0 * 2 + 0] = scale + type(2.0) * xx * scale_derivative + type(2.0) * this->tangential_p[0] * y + type(6.0) * this->tangential_p[1] * x + prism_x * x;
            jacobian_distortion[0 * 2 + 1] = type(2.0) * xy * scale_derivative + type(2.0) * this->tangential_p[0] * x + type(2.0) * this->tangential_p[1] * y + prism_x * y;
            jacobian_distortion[1 * 2 + 0] = type(2.0) * xy * scale_derivative + type(2.0) * this->tangential_p[0] * x + type(2.0) * this->tangential_p[1] * y + prism_y * x;
            jacobian_distortion[1 * 2 + 1] = scale + type(2.0) * yy * scale_derivative + type(2.0) * this->tangential_p[1] * x + type(6.0) * this->tangential_p[0] * y + prism_y * y;
        }

        if (jacobian_coefficients != nullptr) {
            const type r6 = r4 * r2;
            const type denominator_inverse = type(1.0) / denominator;
            const type rational = numerator * denominator_inverse * denominator_inverse;
            const type columns[2][12] = {
                { x * r2 * denominator_inverse, x * r4 * denominator_inverse, type(2.0) * xy, r2 + type(2.0) * xx, x * r6 * denominator_inverse, -x * r2 * rational, -x * r4 * rational, -x * r6 * rational, r2, r4, type(0), type(0) },
                { y * r2 * denominator_inverse, y * r4 * denominator_inverse, r2 + type(2.0) * yy, type(2.0) * xy, y * r6 * denominator_inverse, -y * r2 * rational, -y * r4 * rational, -y * r6 * rational, type(0), type(0), r2, r4 }
            };
            for (size_t row = 0; row < 2; ++row) {
                for (size_t column = 0; column < 12; ++column) {
                    jacobian_coefficients[row * 12 + column] = columns[row][column];
                }
            }
        }

        return true;
    }

    template <typename type>
    bool pinhole_opencv<type>::undistort(
        const type* const distorted_xy,
        type* const undistorted_xy,
        type* const jacobian_undistortion
    ) const {
        bool converged = false;
        undistorted_xy[0] = distorted_xy[0];
        undistorted_xy[1] = distorted_xy[1];
        // Every iterate must lie inside the fold, where the solution is unique, so start there too.
        const type start_squared = undistorted_xy[0] * undistorted_xy[0] + undistorted_xy[1] * undistorted_xy[1];
        if (start_squared > this->maximum_radius_squared) {
            const type shrink = type(0.5) * math::sqrt(this->maximum_radius_squared / start_squared);
            undistorted_xy[0] *= shrink;
            undistorted_xy[1] *= shrink;
        }
        type jacobian_distortion[2 * 2] = { type(1.0), type(0.0), type(0.0), type(1.0) };
        for (size_t iteration = 0; iteration < pinhole_opencv::maximum_undistortion_iterations; ++iteration) {
            type redistorted_xy[2];
            if (!this->distort(&undistorted_xy[0], &redistorted_xy[0], &jacobian_distortion[0])) {
                return false;
            }
            const type error[2] = { distorted_xy[0] - redistorted_xy[0], distorted_xy[1] - redistorted_xy[1] };
            const type chi_squared = error[0] * error[0] + error[1] * error[1];
            if (chi_squared < type(1.0e-12)) {
                converged = true;
                if (chi_squared < type(1.0e-24)) {
                    break;
                }
            }
            const type determinant = jacobian_distortion[0] * jacobian_distortion[3] - jacobian_distortion[1] * jacobian_distortion[2];
            if (!(determinant > type(1.0e-15)) && !(determinant < -type(1.0e-15))) {
                return false;
            }
            const type determinant_inverse = type(1.0) / determinant;
            undistorted_xy[0] += (jacobian_distortion[3] * error[0] - jacobian_distortion[1] * error[1]) * determinant_inverse;
            undistorted_xy[1] += (jacobian_distortion[0] * error[1] - jacobian_distortion[2] * error[0]) * determinant_inverse;
        }
        if (!converged) {
            return false;
        }
        if ((undistorted_xy[0] * undistorted_xy[0] + undistorted_xy[1] * undistorted_xy[1]) > this->maximum_radius_squared) {
            return false;
        }
        if (jacobian_undistortion != nullptr) {
            type redistorted_xy[2];
            if (!this->distort(&undistorted_xy[0], &redistorted_xy[0], &jacobian_distortion[0])) {
                return false;
            }
            const type determinant = jacobian_distortion[0] * jacobian_distortion[3] - jacobian_distortion[1] * jacobian_distortion[2];
            if (!(determinant > type(1.0e-15)) && !(determinant < -type(1.0e-15))) {
                return false;
            }
            const type determinant_inverse = type(1.0) / determinant;
            jacobian_undistortion[0 * 2 + 0] = +jacobian_distortion[3] * determinant_inverse;
            jacobian_undistortion[0 * 2 + 1] = -jacobian_distortion[1] * determinant_inverse;
            jacobian_undistortion[1 * 2 + 0] = -jacobian_distortion[2] * determinant_inverse;
            jacobian_undistortion[1 * 2 + 1] = +jacobian_distortion[0] * determinant_inverse;
        }
        return true;
    }

    template <typename type>
    bool pinhole_opencv<type>::project(
        const type* const point_xyz,
        type* const point_xy,
        type* const jacobian_projection,
        type* const jacobian_parameters
    ) const {
        if (point_xyz[2] < type(1.0e-12)) {
            return false;
        }
        const type inverse_z = type(1.0) / point_xyz[2];
        const type undistorted_xy[2] = { point_xyz[0] * inverse_z, point_xyz[1] * inverse_z };

        type untilted_xy[2];
        type jacobian_distortion[2 * 2];
        type jacobian_coefficients[2 * 12];
        if (!this->distort(&undistorted_xy[0], &untilted_xy[0], (jacobian_projection == nullptr) ? nullptr : &jacobian_distortion[0], (jacobian_parameters == nullptr) ? nullptr : &jacobian_coefficients[0])) {
            return false;
        }
        type distorted_xy[2];
        type jacobian_tilt[2 * 2];
        if (!apply_homography(&this->tilt_matrix[0], &untilted_xy[0], &distorted_xy[0], &jacobian_tilt[0])) {
            return false;
        }

        point_xy[0] = this->focal_lengths[0] * distorted_xy[0] + this->centre_points[0];
        point_xy[1] = this->focal_lengths[1] * distorted_xy[1] + this->centre_points[1];

        if (jacobian_projection != nullptr) {
            // d pixel / d point = f * J_tilt * J_distort * d(X / Z, Y / Z) / d point.
            const type plane_by_point[2][3] = {
                { inverse_z, type(0), -undistorted_xy[0] * inverse_z },
                { type(0), inverse_z, -undistorted_xy[1] * inverse_z }
            };
            for (size_t row = 0; row < 2; ++row) {
                const type chained[2] = {
                    (jacobian_tilt[row * 2 + 0] * jacobian_distortion[0 * 2 + 0]) + (jacobian_tilt[row * 2 + 1] * jacobian_distortion[1 * 2 + 0]),
                    (jacobian_tilt[row * 2 + 0] * jacobian_distortion[0 * 2 + 1]) + (jacobian_tilt[row * 2 + 1] * jacobian_distortion[1 * 2 + 1])
                };
                for (size_t column = 0; column < 3; ++column) {
                    jacobian_projection[row * 3 + column] = this->focal_lengths[row] * ((chained[0] * plane_by_point[0][column]) + (chained[1] * plane_by_point[1][column]));
                }
            }
        }

        if (jacobian_parameters != nullptr) {
            constexpr static const size_t count = pinhole_opencv::parameter_count;
            jacobian_parameters[0 * count + 0] = distorted_xy[0];
            jacobian_parameters[0 * count + 1] = type(0.0);
            jacobian_parameters[0 * count + 2] = type(1.0);
            jacobian_parameters[0 * count + 3] = type(0.0);
            jacobian_parameters[1 * count + 0] = type(0.0);
            jacobian_parameters[1 * count + 1] = distorted_xy[1];
            jacobian_parameters[1 * count + 2] = type(0.0);
            jacobian_parameters[1 * count + 3] = type(1.0);
            for (size_t index = 0; index < 12; ++index) {
                for (size_t row = 0; row < 2; ++row) {
                    jacobian_parameters[row * count + 4 + index] = this->focal_lengths[row] * ((jacobian_tilt[row * 2 + 0] * jacobian_coefficients[0 * 12 + index]) + (jacobian_tilt[row * 2 + 1] * jacobian_coefficients[1 * 12 + index]));
                }
            }
            // The tilt angles move the homography: d (u / w) = (du - (u / w) dw) / w.
            type tilt_unused[9];
            type tilt_by[2][9];
            tilt_projection<type>(this->tilt[0], this->tilt[1], &tilt_unused[0], &tilt_by[0][0], &tilt_by[1][0]);
            const type* const h = &this->tilt_matrix[0];
            const type inverse_w = type(1) / ((h[6] * untilted_xy[0]) + (h[7] * untilted_xy[1]) + h[8]);
            for (size_t angle = 0; angle < 2; ++angle) {
                const type* const dh = &tilt_by[angle][0];
                const type du = (dh[0] * untilted_xy[0]) + (dh[1] * untilted_xy[1]) + dh[2];
                const type dv = (dh[3] * untilted_xy[0]) + (dh[4] * untilted_xy[1]) + dh[5];
                const type dw = (dh[6] * untilted_xy[0]) + (dh[7] * untilted_xy[1]) + dh[8];
                jacobian_parameters[0 * count + 16 + angle] = this->focal_lengths[0] * (du - (distorted_xy[0] * dw)) * inverse_w;
                jacobian_parameters[1 * count + 16 + angle] = this->focal_lengths[1] * (dv - (distorted_xy[1] * dw)) * inverse_w;
            }
        }

        return true;
    }

    template <typename type>
    bool pinhole_opencv<type>::unproject(
        const type* const point_xy,
        type* const ray_xyz,
        type* const jacobian_unprojection
    ) const {
        const type distorted_xy[2] = {
            (point_xy[0] - this->centre_points[0]) / this->focal_lengths[0],
            (point_xy[1] - this->centre_points[1]) / this->focal_lengths[1]
        };

        // Undo the tilt through the inverse homography, the adjugate of T scaled by the sign that keeps w positive.
        const type* const h = &this->tilt_matrix[0];
        const type inverse_tilt[9] = {
            (h[4] * h[8]) - (h[5] * h[7]),
            (h[2] * h[7]) - (h[1] * h[8]),
            (h[1] * h[5]) - (h[2] * h[4]),
            (h[5] * h[6]) - (h[3] * h[8]),
            (h[0] * h[8]) - (h[2] * h[6]),
            (h[2] * h[3]) - (h[0] * h[5]),
            (h[3] * h[7]) - (h[4] * h[6]),
            (h[1] * h[6]) - (h[0] * h[7]),
            (h[0] * h[4]) - (h[1] * h[3])
        };
        type untilted_xy[2];
        type jacobian_untilt[2 * 2];
        if (!apply_homography(&inverse_tilt[0], &distorted_xy[0], &untilted_xy[0], &jacobian_untilt[0])) {
            return false;
        }

        type undistorted_xy[2];
        type jacobian_undistortion[2 * 2];
        if (!this->undistort(&untilted_xy[0], &undistorted_xy[0], (jacobian_unprojection == nullptr) ? nullptr : &jacobian_undistortion[0])) {
            return false;
        }

        ray_xyz[0] = undistorted_xy[0];
        ray_xyz[1] = undistorted_xy[1];
        ray_xyz[2] = type(1.0);

        if (jacobian_unprojection != nullptr) {
            const type inverse_f[2] = { type(1) / this->focal_lengths[0], type(1) / this->focal_lengths[1] };
            for (size_t row = 0; row < 2; ++row) {
                for (size_t column = 0; column < 2; ++column) {
                    jacobian_unprojection[row * 2 + column] = ((jacobian_undistortion[row * 2 + 0] * jacobian_untilt[0 * 2 + column]) + (jacobian_undistortion[row * 2 + 1] * jacobian_untilt[1 * 2 + column])) * inverse_f[column];
                }
            }
            jacobian_unprojection[2 * 2 + 0] = type(0.0);
            jacobian_unprojection[2 * 2 + 1] = type(0.0);
        }

        return true;
    }

    template class pinhole_opencv<float>;
    template class pinhole_opencv<double>;
}
