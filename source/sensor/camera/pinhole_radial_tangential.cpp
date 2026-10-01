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

#include "sensor/camera/pinhole_radial_tangential.hpp"

#include "core/assert.hpp"
#include "math/math.hpp"

namespace {
    double evaluate_polynomial(const double* const coefficients, const size_t degree, const double value) {
        double result = coefficients[degree];
        for (size_t index = degree; index > 0; --index) {
            result = (result * value) + coefficients[index - 1];
        }
        return result;
    }

    // The roots above lower and up to upper where the polynomial changes sign or reaches zero, in increasing order.
    // The roots of the derivative split the range into monotonic pieces, each holds at most one root, found by bisection.
    size_t polynomial_roots(const double* const coefficients, size_t degree, const double lower, const double upper, double* const roots) {
        while ((degree > 0) && (coefficients[degree] == 0.0)) {
            --degree;
        }
        if (degree == 0) {
            return 0;
        }
        double bounds[8];
        size_t bounds_count = 0;
        bounds[bounds_count++] = lower;
        if (degree > 1) {
            double derivative[6];
            for (size_t index = 1; index <= degree; ++index) {
                derivative[index - 1] = static_cast<double>(index) * coefficients[index];
            }
            double critical[6];
            const size_t critical_count = polynomial_roots(&derivative[0], degree - 1, lower, upper, &critical[0]);
            for (size_t index = 0; index < critical_count; ++index) {
                if (critical[index] < upper) {
                    bounds[bounds_count++] = critical[index];
                }
            }
        }
        bounds[bounds_count++] = upper;
        size_t count = 0;
        for (size_t index = 1; index < bounds_count; ++index) {
            double below = bounds[index - 1];
            double above = bounds[index];
            const double value_below = evaluate_polynomial(coefficients, degree, below);
            const double value_above = evaluate_polynomial(coefficients, degree, above);
            if (value_above == 0.0) {
                roots[count++] = above;
                continue;
            }
            if ((value_below == 0.0) || ((value_below < 0.0) == (value_above < 0.0))) {
                continue;
            }
            const bool rising = (value_above > 0.0);
            for (size_t iteration = 0; iteration < 256; ++iteration) {
                const double middle = 0.5 * (below + above);
                if (!(middle > below) || !(middle < above)) {
                    break;
                }
                if ((evaluate_polynomial(coefficients, degree, middle) > 0.0) == rising) {
                    above = middle;
                }
                else {
                    below = middle;
                }
            }
            roots[count++] = above;
        }
        return count;
    }

    // The first u past zero where the polynomial, one at zero, stops being positive, infinite when it never does.
    double first_non_positive(const double* const coefficients, const size_t degree) {
        // A radius of a million is within a microradian of ninety degrees, so any fold further out is never seen.
        const double search_limit = 1.0e12;
        double roots[6];
        const size_t count = polynomial_roots(coefficients, degree, 0.0, search_limit, &roots[0]);
        return (count > 0) ? roots[0] : math::inf<double>();
    }

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
        return math::min(first_non_positive(&derivative[0], 6), first_non_positive(&denominator[0], 3));
    }
}

namespace sensor::camera {
    template <typename type>
    pinhole_radial_tangential<type>::pinhole_radial_tangential()
        : focal_lengths{ type(1), type(1) }
        , centre_points{ type(0.5), type(0.5) }
        , radial_k{ type(0), type(0), type(0), type(0), type(0), type(0) }
        , tangential_p{ type(0), type(0) }
        , maximum_radius_squared(math::inf<type>()) {
    }

    template <typename type>
    pinhole_radial_tangential<type>::pinhole_radial_tangential(const type* const parameters, const size_t parameters_length)
        : pinhole_radial_tangential() {
        const bool valid = this->set_parameters(parameters, parameters_length);
        ASSERT(valid, "The radial-tangential parameters must be twelve values with positive focal lengths.");
        static_cast<void>(valid);
    }

    template <typename type>
    size_t pinhole_radial_tangential<type>::get_parameter_count() const {
        return pinhole_radial_tangential::parameter_count;
    }

    template <typename type>
    bool pinhole_radial_tangential<type>::set_parameters(const type* const parameters, const size_t parameters_length) {
        if ((parameters == nullptr) || (parameters_length != pinhole_radial_tangential::parameter_count)) {
            return false;
        }
        if (!(parameters[0] > type(0)) || !(parameters[1] > type(0))) {
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
    bool pinhole_radial_tangential<type>::get_parameters(type* const parameters, const size_t parameters_length) const {
        if ((parameters == nullptr) || (parameters_length != pinhole_radial_tangential::parameter_count)) {
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
        return true;
    }

    template <typename type>
    bool pinhole_radial_tangential<type>::distort(
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
        // Past the fold the distortion maps points from outside the field of view back into the image.
        if (r2 > this->maximum_radius_squared) {
            return false;
        }
        const type numerator = type(1.0) + r2 * (this->radial_k[0] + r2 * (this->radial_k[1] + r2 * this->radial_k[2]));
        const type denominator = type(1.0) + r2 * (this->radial_k[3] + r2 * (this->radial_k[4] + r2 * this->radial_k[5]));
        if (!(denominator > type(1.0e-12)) && !(denominator < -type(1.0e-12))) {
            return false;
        }
        const type scale = numerator / denominator;

        distorted_xy[0] = x * scale + type(2.0) * this->tangential_p[0] * xy + this->tangential_p[1] * (r2 + type(2.0) * xx);
        distorted_xy[1] = y * scale + type(2.0) * this->tangential_p[1] * xy + this->tangential_p[0] * (r2 + type(2.0) * yy);

        // The distortion jacobian, through ds/dr2.
        if (jacobian_distortion != nullptr) {
            const type numerator_derivative = this->radial_k[0] + r2 * (type(2.0) * this->radial_k[1] + r2 * type(3.0) * this->radial_k[2]);
            const type denominator_derivative = this->radial_k[3] + r2 * (type(2.0) * this->radial_k[4] + r2 * type(3.0) * this->radial_k[5]);
            const type scale_derivative = (numerator_derivative * denominator - numerator * denominator_derivative) / (denominator * denominator);
            jacobian_distortion[0 * 2 + 0] = scale + type(2.0) * xx * scale_derivative + type(2.0) * this->tangential_p[0] * y + type(6.0) * this->tangential_p[1] * x;
            jacobian_distortion[0 * 2 + 1] = type(2.0) * xy * scale_derivative + type(2.0) * this->tangential_p[0] * x + type(2.0) * this->tangential_p[1] * y;
            jacobian_distortion[1 * 2 + 0] = jacobian_distortion[0 * 2 + 1];
            jacobian_distortion[1 * 2 + 1] = scale + type(2.0) * yy * scale_derivative + type(2.0) * this->tangential_p[1] * x + type(6.0) * this->tangential_p[0] * y;
        }

        // The coefficient jacobian, d distorted_xy / d [k1 k2 p1 p2 k3 k4 k5 k6].
        if (jacobian_coefficients != nullptr) {
            const type r4 = r2 * r2;
            const type r6 = r4 * r2;
            const type denominator_inverse = type(1.0) / denominator;
            const type rational = numerator * denominator_inverse * denominator_inverse;
            jacobian_coefficients[0 * 8 + 0] = x * r2 * denominator_inverse;
            jacobian_coefficients[0 * 8 + 1] = x * r4 * denominator_inverse;
            jacobian_coefficients[0 * 8 + 2] = type(2.0) * xy;
            jacobian_coefficients[0 * 8 + 3] = r2 + type(2.0) * xx;
            jacobian_coefficients[0 * 8 + 4] = x * r6 * denominator_inverse;
            jacobian_coefficients[0 * 8 + 5] = -x * r2 * rational;
            jacobian_coefficients[0 * 8 + 6] = -x * r4 * rational;
            jacobian_coefficients[0 * 8 + 7] = -x * r6 * rational;
            jacobian_coefficients[1 * 8 + 0] = y * r2 * denominator_inverse;
            jacobian_coefficients[1 * 8 + 1] = y * r4 * denominator_inverse;
            jacobian_coefficients[1 * 8 + 2] = r2 + type(2.0) * yy;
            jacobian_coefficients[1 * 8 + 3] = type(2.0) * xy;
            jacobian_coefficients[1 * 8 + 4] = y * r6 * denominator_inverse;
            jacobian_coefficients[1 * 8 + 5] = -y * r2 * rational;
            jacobian_coefficients[1 * 8 + 6] = -y * r4 * rational;
            jacobian_coefficients[1 * 8 + 7] = -y * r6 * rational;
        }

        return true;
    }

    template <typename type>
    bool pinhole_radial_tangential<type>::undistort(
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
        for (size_t iteration = 0; iteration < pinhole_radial_tangential::maximum_undistortion_iterations; ++iteration) {
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
        // The last step is not distorted again, so check it stayed inside the fold.
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
    bool pinhole_radial_tangential<type>::project(
        const type* const point_xyz,
        type* const point_xy,
        type* const jacobian_projection,
        type* const jacobian_parameters
    ) const {
        // Cannot project points with zero depth, also return false for those behind the camera.
        // Note: This guard must match pinhole::project exactly: reprojection factors treat a false
        // return as "behind the camera" and switch to a penalty residual, so a mismatch here changes
        // the optimisation path and breaks the zero-distortion bit-identity with pinhole.
        if (point_xyz[2] < type(1.0e-12)) {
            return false;
        }

        // Project.
        const type inverse_z = type(1.0) / point_xyz[2];
        const type undistorted_xy[2] = {
            point_xyz[0] * inverse_z,
            point_xyz[1] * inverse_z
        };

        // Distort.
        type distorted_xy[2];
        type jacobian_distortion[2 * 2];
        type jacobian_coefficients[2 * 8];
        if (!this->distort(
                &undistorted_xy[0],
                &distorted_xy[0],
                (jacobian_projection == nullptr) ? nullptr : &jacobian_distortion[0],
                (jacobian_parameters == nullptr) ? nullptr : &jacobian_coefficients[0]
            )) {
            return false;
        }

        // Scale and offset.
        point_xy[0] = this->focal_lengths[0] * distorted_xy[0] + this->centre_points[0];
        point_xy[1] = this->focal_lengths[1] * distorted_xy[1] + this->centre_points[1];

        // Chain the projection jacobian: d pixel / d point = f * J_distort * d(X/Z, Y/Z)/d point.
        if (jacobian_projection != nullptr) {
            const type inverse_z_squared = inverse_z * inverse_z;
            jacobian_projection[0 * 3 + 0] = this->focal_lengths[0] * jacobian_distortion[0 * 2 + 0] * inverse_z;
            jacobian_projection[0 * 3 + 1] = this->focal_lengths[0] * jacobian_distortion[0 * 2 + 1] * inverse_z;
            jacobian_projection[0 * 3 + 2] = -this->focal_lengths[0] * (point_xyz[0] * jacobian_distortion[0 * 2 + 0] + point_xyz[1] * jacobian_distortion[0 * 2 + 1]) * inverse_z_squared;
            jacobian_projection[1 * 3 + 0] = this->focal_lengths[1] * jacobian_distortion[1 * 2 + 0] * inverse_z;
            jacobian_projection[1 * 3 + 1] = this->focal_lengths[1] * jacobian_distortion[1 * 2 + 1] * inverse_z;
            jacobian_projection[1 * 3 + 2] = -this->focal_lengths[1] * (point_xyz[0] * jacobian_distortion[1 * 2 + 0] + point_xyz[1] * jacobian_distortion[1 * 2 + 1]) * inverse_z_squared;
        }

        // The parameter jacobian: intrinsics directly, distortion coefficients through f.
        if (jacobian_parameters != nullptr) {
            jacobian_parameters[0 * pinhole_radial_tangential::parameter_count + 0] = distorted_xy[0];
            jacobian_parameters[0 * pinhole_radial_tangential::parameter_count + 1] = type(0.0);
            jacobian_parameters[0 * pinhole_radial_tangential::parameter_count + 2] = type(1.0);
            jacobian_parameters[0 * pinhole_radial_tangential::parameter_count + 3] = type(0.0);
            jacobian_parameters[1 * pinhole_radial_tangential::parameter_count + 0] = type(0.0);
            jacobian_parameters[1 * pinhole_radial_tangential::parameter_count + 1] = distorted_xy[1];
            jacobian_parameters[1 * pinhole_radial_tangential::parameter_count + 2] = type(0.0);
            jacobian_parameters[1 * pinhole_radial_tangential::parameter_count + 3] = type(1.0);
            for (size_t index = 0; index < 8; ++index) {
                jacobian_parameters[0 * pinhole_radial_tangential::parameter_count + index + 4] = this->focal_lengths[0] * jacobian_coefficients[0 * 8 + index];
                jacobian_parameters[1 * pinhole_radial_tangential::parameter_count + index + 4] = this->focal_lengths[1] * jacobian_coefficients[1 * 8 + index];
            }
        }

        return true;
    }

    template <typename type>
    bool pinhole_radial_tangential<type>::unproject(
        const type* const point_xy,
        type* const ray_xyz,
        type* const jacobian_unprojection
    ) const {
        // Scale and offset.
        const type distorted_xy[2] = {
            (point_xy[0] - this->centre_points[0]) / this->focal_lengths[0],
            (point_xy[1] - this->centre_points[1]) / this->focal_lengths[1]
        };

        // Undistort.
        type undistorted_xy[2];
        type jacobian_undistortion[2 * 2];
        if (!this->undistort(
                &distorted_xy[0],
                &undistorted_xy[0],
                (jacobian_unprojection == nullptr) ? nullptr : &jacobian_undistortion[0]
            )) {
            return false;
        }

        // Unproject.
        ray_xyz[0] = undistorted_xy[0];
        ray_xyz[1] = undistorted_xy[1];
        ray_xyz[2] = type(1.0);

        // Chain the unprojection jacobian through the focal scaling.
        if (jacobian_unprojection != nullptr) {
            jacobian_unprojection[0 * 2 + 0] = jacobian_undistortion[0 * 2 + 0] / this->focal_lengths[0];
            jacobian_unprojection[0 * 2 + 1] = jacobian_undistortion[0 * 2 + 1] / this->focal_lengths[1];
            jacobian_unprojection[1 * 2 + 0] = jacobian_undistortion[1 * 2 + 0] / this->focal_lengths[0];
            jacobian_unprojection[1 * 2 + 1] = jacobian_undistortion[1 * 2 + 1] / this->focal_lengths[1];
            jacobian_unprojection[2 * 2 + 0] = type(0.0);
            jacobian_unprojection[2 * 2 + 1] = type(0.0);
        }

        return true;
    }

    template class pinhole_radial_tangential<float>;
    template class pinhole_radial_tangential<double>;
}
