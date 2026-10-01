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
#ifndef ZEROSLAM_MATH_LIE_HPP
#define ZEROSLAM_MATH_LIE_HPP

#include "core/assert.hpp"
#include "math/math.hpp"
#include "math/matrix.hpp"

namespace math {
    template <typename type>
    class so3 final {
    private:
        type rotation_quaternion[4];

    public:
        constexpr so3();
        constexpr so3(const so3& other);
        constexpr so3(so3&& other);
        constexpr so3& operator=(const so3& other);
        constexpr so3& operator=(so3&& other);
        constexpr so3(
            type initial_rotation_quaternion_w,
            type initial_rotation_quaternion_x,
            type initial_rotation_quaternion_y,
            type initial_rotation_quaternion_z
        );
        constexpr so3(const type (&initial_rotation_quaternion_values)[4]);
        constexpr so3(const math::matrix<type, 3, 3>& initial_rotation_matrix);
        constexpr math::matrix<type, 4, 1> get_quaternion() const;
        constexpr bool is_unit() const;
        constexpr so3 normalised() const;
        constexpr static const type unit_tolerance = type(1e-6);
        constexpr math::matrix<type, 3, 3> get_matrix() const;
        static constexpr so3 identity();
        static constexpr so3 rotation(const type& x, const type& y, const type& z);
        constexpr so3 inverse() const;
        static constexpr math::matrix<type, 3, 3> generator(size_t parameter_index);
        static constexpr math::matrix<type, 3, 1> generator_field(size_t parameter_index, const math::matrix<type, 3, 1>& point);
        static constexpr so3 exp(const math::matrix<type, 3, 1>& omega);
        constexpr math::matrix<type, 3, 1> log() const;
        static math::matrix<type, 3, 3> left_jacobian(const math::matrix<type, 3, 1>& omega);
        static math::matrix<type, 3, 3> left_jacobian_inverse(const math::matrix<type, 3, 1>& omega);
        constexpr bool operator==(const so3& rhs) const;
        constexpr bool operator!=(const so3& rhs) const;
        constexpr so3 operator-() const;
        constexpr so3 operator*(const so3& rhs) const;
        constexpr math::matrix<type, 3, 1> operator*(const math::matrix<type, 3, 1>& point) const;
    };

    template <typename type>
    class se3 final {
    private:
        so3<type> rotation_so3;
        math::matrix<type, 3, 1> translation_vector;

    public:
        constexpr se3();
        constexpr se3(const so3<type>& initial_rotation_so3, const math::matrix<type, 3, 1>& initial_translation_vector);
        constexpr se3(const type (&initial_rotation_quaternion_values)[4], const type (&initial_translation_vector)[3]);
        constexpr const so3<type>& rotation() const;
        constexpr so3<type>& rotation();
        constexpr const math::matrix<type, 3, 1>& translation() const;
        constexpr math::matrix<type, 3, 1>& translation();
        static se3 identity();
        constexpr se3 inverse() const;
        static math::matrix<type, 4, 4> generator(size_t parameter_index);
        static math::matrix<type, 4, 1> generator_field(size_t parameter_index, const math::matrix<type, 4, 1>& point);
        static se3 exp(const math::matrix<type, 6, 1>& omega_upsilon);
        constexpr math::matrix<type, 6, 1> log() const;
        static math::matrix<type, 6, 6> left_jacobian(const math::matrix<type, 6, 1>& omega_upsilon);
        static math::matrix<type, 6, 6> left_jacobian_inverse(const math::matrix<type, 6, 1>& omega_upsilon);
        constexpr bool operator==(const se3& rhs) const;
        constexpr bool operator!=(const se3& rhs) const;
        constexpr se3 operator*(const se3& rhs) const;
        constexpr math::matrix<type, 3, 1> operator*(const math::matrix<type, 3, 1>& point) const;

    private:
        static math::matrix<type, 3, 3> left_jacobian_coupling(const math::matrix<type, 3, 1>& omega, const math::matrix<type, 3, 1>& upsilon);
    };

    template <typename type>
    class sim3 final {
    private:
        se3<type> transformation_se3;
        type scale_scalar;

    public:
        constexpr sim3();
        constexpr sim3(const se3<type>& initial_transformation_se3, type initial_scale_scalar);
        constexpr sim3(const type (&initial_rotation_quaternion_values)[4], const type (&initial_translation_vector)[3], const type initial_scale_scalar);
        static sim3 identity();
        constexpr const se3<type>& transformation() const;
        constexpr se3<type>& transformation();
        constexpr const type& scale() const;
        constexpr type& scale();
        constexpr sim3 inverse() const;
        static math::matrix<type, 4, 4> generator(size_t parameter_index);
        static math::matrix<type, 4, 1> generator_field(size_t parameter_index, const math::matrix<type, 4, 1>& point);
        static sim3 exp(const math::matrix<type, 7, 1>& omega_upsilon_sigma);
        constexpr math::matrix<type, 7, 1> log() const;
        static math::matrix<type, 7, 7> left_jacobian(const math::matrix<type, 7, 1>& omega_upsilon_sigma);
        static math::matrix<type, 7, 7> left_jacobian_inverse(const math::matrix<type, 7, 1>& omega_upsilon_sigma);
        static math::matrix<type, 7, 7> adjoint(const sim3& similarity);
        constexpr bool operator==(const sim3& rhs) const;
        constexpr bool operator!=(const sim3& rhs) const;
        constexpr sim3 operator*(const sim3& rhs) const;
        constexpr math::matrix<type, 3, 1> operator*(const math::matrix<type, 3, 1>& point) const;

    private:
        // The translation of exp is W * upsilon with W = c * I + a * omega_hat + b * omega_hat^2, the derivatives are by u = theta^2 and by sigma.
        struct translation_terms {
            type a;
            type b;
            type c;
            type a_u;
            type b_u;
            type a_sigma;
            type b_sigma;
            type c_sigma;
        };

        template <bool with_derivatives>
        static constexpr translation_terms translation_coefficients(type sigma, type theta);
        static constexpr void translation_inverse_coefficients(const translation_terms& terms, type theta_squared, type& inverse_a, type& inverse_b, type& inverse_c);
        static void left_jacobian_blocks(const math::matrix<type, 7, 1>& omega_upsilon_sigma, translation_terms& terms, math::matrix<type, 3, 3>& coupling, math::matrix<type, 3, 1>& sigma_column);
    };
}

namespace math {
    template <typename type>
    constexpr so3<type>::so3()
        : rotation_quaternion{ type(1), type(0), type(0), type(0) } {
    }

    template <typename type>
    constexpr so3<type>::so3(const so3& other) = default;
    template <typename type>
    constexpr so3<type>::so3(so3&& other) = default;
    template <typename type>
    constexpr so3<type>& so3<type>::operator=(const so3& other) = default;
    template <typename type>
    constexpr so3<type>& so3<type>::operator=(so3&& other) = default;

    template <typename type>
    constexpr so3<type>::so3(
        type initial_rotation_quaternion_w,
        type initial_rotation_quaternion_x,
        type initial_rotation_quaternion_y,
        type initial_rotation_quaternion_z
    )
        : rotation_quaternion{
            initial_rotation_quaternion_w,
            initial_rotation_quaternion_x,
            initial_rotation_quaternion_y,
            initial_rotation_quaternion_z
        } {
    }

    template <typename type>
    constexpr so3<type>::so3(
        const type (&initial_rotation_quaternion_values)[4]
    )
        : rotation_quaternion{
            initial_rotation_quaternion_values[0],
            initial_rotation_quaternion_values[1],
            initial_rotation_quaternion_values[2],
            initial_rotation_quaternion_values[3]
        } {
    }

    template <typename type>
    constexpr so3<type>::so3(const math::matrix<type, 3, 3>& initial_rotation_matrix) {
        ASSERT(([&initial_rotation_matrix]() {
                   for (size_t row = 0; row < 3; ++row) {
                       for (size_t column = 0; column < 3; ++column) {
                           type product = type(0);
                           for (size_t k = 0; k < 3; ++k) {
                               product += initial_rotation_matrix[k][row] * initial_rotation_matrix[k][column];
                           }
                           if (math::abs(product - ((row == column) ? type(1) : type(0))) > so3<type>::unit_tolerance) {
                               return false;
                           }
                       }
                   }
                   const type determinant =
                       initial_rotation_matrix[0][0] * ((initial_rotation_matrix[1][1] * initial_rotation_matrix[2][2]) - (initial_rotation_matrix[1][2] * initial_rotation_matrix[2][1])) -
                       initial_rotation_matrix[0][1] * ((initial_rotation_matrix[1][0] * initial_rotation_matrix[2][2]) - (initial_rotation_matrix[1][2] * initial_rotation_matrix[2][0])) +
                       initial_rotation_matrix[0][2] * ((initial_rotation_matrix[1][0] * initial_rotation_matrix[2][1]) - (initial_rotation_matrix[1][1] * initial_rotation_matrix[2][0]));
                   return determinant > type(0);
               }()),
               "A rotation must be constructed from a rotation matrix.");
        const type rotation_trace = initial_rotation_matrix[0][0] + initial_rotation_matrix[1][1] + initial_rotation_matrix[2][2];
        if (rotation_trace > 0) {
            const type square_root_trace_plus_one = math::sqrt(rotation_trace + 1);
            const type inverse_two_square_root_trace_plus_one = 0.5 / square_root_trace_plus_one;
            this->rotation_quaternion[0] = 0.5 * square_root_trace_plus_one;
            this->rotation_quaternion[1] = (initial_rotation_matrix[2][1] - initial_rotation_matrix[1][2]) * inverse_two_square_root_trace_plus_one;
            this->rotation_quaternion[2] = (initial_rotation_matrix[0][2] - initial_rotation_matrix[2][0]) * inverse_two_square_root_trace_plus_one;
            this->rotation_quaternion[3] = (initial_rotation_matrix[1][0] - initial_rotation_matrix[0][1]) * inverse_two_square_root_trace_plus_one;
            *this = this->normalised();
            return;
        }
        const size_t i = (initial_rotation_matrix[0][0] < initial_rotation_matrix[1][1]) ? (1 + (initial_rotation_matrix[1][1] < initial_rotation_matrix[2][2])) : (2 * (initial_rotation_matrix[0][0] < initial_rotation_matrix[2][2]));
        const size_t j = (i + 1) % 3;
        const size_t k = (j + 1) % 3;
        const type square_root_trace_plus_one = math::sqrt(initial_rotation_matrix[i][i] - initial_rotation_matrix[j][j] - initial_rotation_matrix[k][k] + 1.0);
        const type inverse_two_square_root_trace_plus_one = 0.5 / square_root_trace_plus_one;
        this->rotation_quaternion[0 + 0] = (initial_rotation_matrix[k][j] - initial_rotation_matrix[j][k]) * inverse_two_square_root_trace_plus_one;
        this->rotation_quaternion[1 + i] = 0.5 * square_root_trace_plus_one;
        this->rotation_quaternion[1 + j] = (initial_rotation_matrix[i][j] + initial_rotation_matrix[j][i]) * inverse_two_square_root_trace_plus_one;
        this->rotation_quaternion[1 + k] = (initial_rotation_matrix[i][k] + initial_rotation_matrix[k][i]) * inverse_two_square_root_trace_plus_one;
        *this = this->normalised();
    }

    template <typename type>
    constexpr math::matrix<type, 4, 1> so3<type>::get_quaternion() const {
        return math::matrix<type, 4, 1>(this->rotation_quaternion);
    }

    template <typename type>
    constexpr bool so3<type>::is_unit() const {
        const type length_squared = math::sqr(this->rotation_quaternion[0]) + math::sqr(this->rotation_quaternion[1]) + math::sqr(this->rotation_quaternion[2]) + math::sqr(this->rotation_quaternion[3]);
        return math::abs(length_squared - type(1)) <= so3<type>::unit_tolerance;
    }

    template <typename type>
    constexpr so3<type> so3<type>::normalised() const {
        const type length_squared = math::sqr(this->rotation_quaternion[0]) + math::sqr(this->rotation_quaternion[1]) + math::sqr(this->rotation_quaternion[2]) + math::sqr(this->rotation_quaternion[3]);
        if (length_squared < 0.000000000001) {
            return *this;
        }
        const type inverse_length = type(1) / math::sqrt(length_squared);
        return { inverse_length * this->rotation_quaternion[0], inverse_length * this->rotation_quaternion[1], inverse_length * this->rotation_quaternion[2], inverse_length * this->rotation_quaternion[3] };
    }

    template <typename type>
    constexpr math::matrix<type, 3, 3> so3<type>::get_matrix() const {
        ASSERT(this->is_unit(), "A rotation's quaternion must be a unit one.");
        const type length_squared = math::sqr(this->rotation_quaternion[0]) + math::sqr(this->rotation_quaternion[1]) + math::sqr(this->rotation_quaternion[2]) + math::sqr(this->rotation_quaternion[3]);
        const type inverse_length = (length_squared < 0.000000000001) ? type(0) : (type(1) / math::sqrt(length_squared));
        const type quaternion_w = inverse_length * this->rotation_quaternion[0];
        const type quaternion_x = inverse_length * this->rotation_quaternion[1];
        const type quaternion_y = inverse_length * this->rotation_quaternion[2];
        const type quaternion_z = inverse_length * this->rotation_quaternion[3];
        const type two_x = 2.0 * quaternion_x;
        const type two_y = 2.0 * quaternion_y;
        const type two_z = 2.0 * quaternion_z;
        const type two_x_x = two_x * quaternion_x;
        const type two_x_y = two_x * quaternion_y;
        const type two_x_z = two_x * quaternion_z;
        const type two_x_w = two_x * quaternion_w;
        const type two_y_y = two_y * quaternion_y;
        const type two_y_z = two_y * quaternion_z;
        const type two_y_w = two_y * quaternion_w;
        const type two_z_z = two_z * quaternion_z;
        const type two_z_w = two_z * quaternion_w;
        return { { { 1.0 - (two_y_y + two_z_z), two_x_y - two_z_w, two_x_z + two_y_w },
                   { two_x_y + two_z_w, 1.0 - (two_x_x + two_z_z), two_y_z - two_x_w },
                   { two_x_z - two_y_w, two_y_z + two_x_w, 1.0 - (two_x_x + two_y_y) } } };
    }

    template <typename type>
    constexpr so3<type> so3<type>::identity() {
        return { 1.0, 0.0, 0.0, 0.0 };
    }

    template <typename type>
    constexpr so3<type> so3<type>::rotation(const type& x, const type& y, const type& z) {
        return so3<type>::exp({ { x, y, z } });
    }

    template <typename type>
    constexpr so3<type> so3<type>::inverse() const {
        const type inverse_length_squared = 1.0 / (math::sqr(this->rotation_quaternion[0]) + math::sqr(this->rotation_quaternion[1]) + math::sqr(this->rotation_quaternion[2]) + math::sqr(this->rotation_quaternion[3]));
        return {
            +this->rotation_quaternion[0] * inverse_length_squared,
            -this->rotation_quaternion[1] * inverse_length_squared,
            -this->rotation_quaternion[2] * inverse_length_squared,
            -this->rotation_quaternion[3] * inverse_length_squared
        };
    }

    template <typename type>
    constexpr math::matrix<type, 3, 3> so3<type>::generator(size_t parameter_index) {
        ASSERT(parameter_index < 3, "The so3 algebra only has three parameters.");
        math::matrix<type, 3, 3> result = math::matrix<type, 3, 3>::zero();
        result[(parameter_index + 1) % 3][(parameter_index + 2) % 3] = -1;
        result[(parameter_index + 2) % 3][(parameter_index + 1) % 3] = +1;
        return result;
    }

    template <typename type>
    constexpr math::matrix<type, 3, 1> so3<type>::generator_field(size_t parameter_index, const math::matrix<type, 3, 1>& point) {
        ASSERT(parameter_index < 3, "The so3 algebra only has three parameters.");
        math::matrix<type, 3, 1> result;
        result[parameter_index] = 0;
        result[(parameter_index + 1) % 3] = -point[(parameter_index + 2) % 3];
        result[(parameter_index + 2) % 3] = +point[(parameter_index + 1) % 3];
        return result;
    }

    template <typename type>
    constexpr so3<type> so3<type>::exp(const math::matrix<type, 3, 1>& omega) {
        const type theta_squared = omega.get_length_squared();
        type real = 0;
        type imaginary_scale = 0;
        if (theta_squared < 1e-6 * 1e-6) {
            const type theta_quarted = theta_squared * theta_squared;
            real = 1.0 - (1.0 / 8.0) * theta_squared + (1.0 / 384.0) * theta_quarted;
            imaginary_scale = 0.5 - (1.0 / 48.0) * theta_squared + (1.0 / 3840.0) * theta_quarted;
        }
        else {
            const type theta = math::sqrt(theta_squared);
            const type theta_half = 0.5 * theta;
            real = math::cos(theta_half);
            imaginary_scale = math::sin(theta_half) / theta;
        }
        const math::matrix<type, 3, 1> imaginary = omega * imaginary_scale;
        return { real, imaginary[0], imaginary[1], imaginary[2] };
    }

    template <typename type>
    constexpr math::matrix<type, 3, 1> so3<type>::log() const {
        ASSERT(this->is_unit(), "A rotation's quaternion must be a unit one to take its logarithm.");
        const type real = this->rotation_quaternion[0];
        const math::matrix<type, 3, 1> imaginary = { { this->rotation_quaternion[1], this->rotation_quaternion[2], this->rotation_quaternion[3] } };
        const type imaginary_length_squared = imaginary.get_length_squared();
        type imaginary_multiplier;
        if (imaginary_length_squared < 1e-6 * 1e-6) {
            imaginary_multiplier = (2 / real) - ((2 * imaginary_length_squared) / (3 * real * real * real));
        }
        else {
            const type imaginary_length = math::sqrt(imaginary_length_squared);
            const type real_sign = static_cast<type>((((real >= 0)) << 1) - 1);
            imaginary_multiplier = 2 * math::atan2(real_sign * imaginary_length, real_sign * real) / imaginary_length;
        }
        return imaginary_multiplier * imaginary;
    }

    template <typename type>
    math::matrix<type, 3, 3> so3<type>::left_jacobian(const math::matrix<type, 3, 1>& omega) {
        const type theta_squared = omega.get_length_squared();
        const math::matrix<type, 3, 3> omega_hat = { { { 0, -omega[2], omega[1] },
                                                       { omega[2], 0, -omega[0] },
                                                       { -omega[1], omega[0], 0 } } };
        // Both coefficients are evaluated without cancellation, (1 - cos(theta)) / theta^2 in its half angle form, and
        // (theta - sin(theta)) / theta^3 by its series below one radian.
        type first = 0.5;
        type second = type(1) / type(6);
        if (theta_squared > 0) {
            const type theta = math::sqrt(theta_squared);
            const type half_sinc = math::sin(0.5 * theta) / (0.5 * theta);
            first = 0.5 * half_sinc * half_sinc;
            if (theta_squared < 1) {
                second = type(1.0 / 121645100408832000.0);
                second = type(1.0 / 355687428096000.0) - (theta_squared * second);
                second = type(1.0 / 1307674368000.0) - (theta_squared * second);
                second = type(1.0 / 6227020800.0) - (theta_squared * second);
                second = type(1.0 / 39916800.0) - (theta_squared * second);
                second = type(1.0 / 362880.0) - (theta_squared * second);
                second = type(1.0 / 5040.0) - (theta_squared * second);
                second = type(1.0 / 120.0) - (theta_squared * second);
                second = type(1.0 / 6.0) - (theta_squared * second);
            }
            else {
                second = (theta - math::sin(theta)) / (theta_squared * theta);
            }
        }
        return math::matrix<type, 3, 3>::identity() + (first * omega_hat) + (second * (omega_hat * omega_hat));
    }

    template <typename type>
    math::matrix<type, 3, 3> so3<type>::left_jacobian_inverse(const math::matrix<type, 3, 1>& omega) {
        const type theta_squared = omega.get_length_squared();
        const math::matrix<type, 3, 3> omega_hat = { { { 0, -omega[2], omega[1] },
                                                       { omega[2], 0, -omega[0] },
                                                       { -omega[1], omega[0], 0 } } };
        if (theta_squared < 1e-6 * 1e-6) {
            return math::matrix<type, 3, 3>::identity() - (0.5 * omega_hat) + ((omega_hat * omega_hat) * (1.0 / 12.0));
        }
        const type theta = math::sqrt(theta_squared);
        const type theta_half = 0.5 * theta;
        return math::matrix<type, 3, 3>::identity() - (0.5 * omega_hat) + (((1.0 - theta * math::cos(theta_half) / (2.0 * math::sin(theta_half))) / theta_squared) * (omega_hat * omega_hat));
    }

    template <typename type>
    constexpr bool so3<type>::operator==(const so3& rhs) const {
        return this->rotation_quaternion[0] == rhs.rotation_quaternion[0] &&
               this->rotation_quaternion[1] == rhs.rotation_quaternion[1] &&
               this->rotation_quaternion[2] == rhs.rotation_quaternion[2] &&
               this->rotation_quaternion[3] == rhs.rotation_quaternion[3];
    }

    template <typename type>
    constexpr bool so3<type>::operator!=(const so3& rhs) const {
        return this->rotation_quaternion[0] != rhs.rotation_quaternion[0] ||
               this->rotation_quaternion[1] != rhs.rotation_quaternion[1] ||
               this->rotation_quaternion[2] != rhs.rotation_quaternion[2] ||
               this->rotation_quaternion[3] != rhs.rotation_quaternion[3];
    }

    template <typename type>
    constexpr so3<type> so3<type>::operator-() const {
        return {
            -this->rotation_quaternion[0],
            -this->rotation_quaternion[1],
            -this->rotation_quaternion[2],
            -this->rotation_quaternion[3]
        };
    }

    template <typename type>
    constexpr so3<type> so3<type>::operator*(const so3& rhs) const {
        return {
            this->rotation_quaternion[0] * rhs.rotation_quaternion[0] - this->rotation_quaternion[1] * rhs.rotation_quaternion[1] - this->rotation_quaternion[2] * rhs.rotation_quaternion[2] - this->rotation_quaternion[3] * rhs.rotation_quaternion[3],
            this->rotation_quaternion[0] * rhs.rotation_quaternion[1] + this->rotation_quaternion[1] * rhs.rotation_quaternion[0] + this->rotation_quaternion[2] * rhs.rotation_quaternion[3] - this->rotation_quaternion[3] * rhs.rotation_quaternion[2],
            this->rotation_quaternion[0] * rhs.rotation_quaternion[2] - this->rotation_quaternion[1] * rhs.rotation_quaternion[3] + this->rotation_quaternion[2] * rhs.rotation_quaternion[0] + this->rotation_quaternion[3] * rhs.rotation_quaternion[1],
            this->rotation_quaternion[0] * rhs.rotation_quaternion[3] + this->rotation_quaternion[1] * rhs.rotation_quaternion[2] - this->rotation_quaternion[2] * rhs.rotation_quaternion[1] + this->rotation_quaternion[3] * rhs.rotation_quaternion[0]
        };
    }

    template <typename type>
    constexpr math::matrix<type, 3, 1> so3<type>::operator*(const math::matrix<type, 3, 1>& point) const {
        ASSERT(this->is_unit(), "A rotation's quaternion must be a unit one to rotate a point.");
        so3 point_quaternion = {
            0,
            point[0],
            point[1],
            point[2]
        };
        so3 rotation_quaternion_conjugate = {
            +this->rotation_quaternion[0],
            -this->rotation_quaternion[1],
            -this->rotation_quaternion[2],
            -this->rotation_quaternion[3]
        };
        const so3 result = *this * point_quaternion * rotation_quaternion_conjugate;
        return { { result.rotation_quaternion[1], result.rotation_quaternion[2], result.rotation_quaternion[3] } };
    }

    template <typename type>
    constexpr se3<type>::se3() = default;

    template <typename type>
    constexpr se3<type>::se3(const so3<type>& initial_rotation_so3, const math::matrix<type, 3, 1>& initial_translation_vector)
        : rotation_so3(initial_rotation_so3)
        , translation_vector(initial_translation_vector) {
    }

    template <typename type>
    constexpr se3<type>::se3(const type (&initial_rotation_quaternion_values)[4], const type (&initial_translation_vector)[3])
        : rotation_so3(initial_rotation_quaternion_values)
        , translation_vector(initial_translation_vector) {
    }

    template <typename type>
    constexpr const so3<type>& se3<type>::rotation() const {
        return this->rotation_so3;
    }

    template <typename type>
    constexpr so3<type>& se3<type>::rotation() {
        return this->rotation_so3;
    }

    template <typename type>
    constexpr const math::matrix<type, 3, 1>& se3<type>::translation() const {
        return this->translation_vector;
    }

    template <typename type>
    constexpr math::matrix<type, 3, 1>& se3<type>::translation() {
        return this->translation_vector;
    }

    template <typename type>
    se3<type> se3<type>::identity() {
        return { so3<type>::identity(), math::matrix<type, 3, 1>::zero() };
    }

    template <typename type>
    constexpr se3<type> se3<type>::inverse() const {
        const so3<type> rotation_so3_inverse = this->rotation_so3.inverse();
        return { rotation_so3_inverse, -(rotation_so3_inverse * this->translation_vector) };
    }

    template <typename type>
    math::matrix<type, 4, 4> se3<type>::generator(size_t parameter_index) {
        ASSERT(parameter_index < 6, "The se3 algebra only has six parameters.");
        math::matrix<type, 4, 4> result = math::matrix<type, 4, 4>::zero();
        if (parameter_index < 3) {
            const math::matrix<type, 3, 3> generated = so3<type>::generator(parameter_index);
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    result[i][j] = generated[i][j];
                }
            }
        }
        else {
            result[parameter_index % 3][3] = 1;
        }
        return result;
    }

    template <typename type>
    math::matrix<type, 4, 1> se3<type>::generator_field(size_t parameter_index, const math::matrix<type, 4, 1>& point) {
        ASSERT(parameter_index < 6, "The se3 algebra only has six parameters.");
        math::matrix<type, 4, 1> result = math::matrix<type, 4, 1>::zero();
        if (parameter_index < 3) {
            const math::matrix<type, 3, 1> point_3d = { { point[0], point[1], point[2] } };
            const math::matrix<type, 3, 1> generated_field = so3<type>::generator_field(parameter_index, point_3d);
            for (size_t i = 0; i < 3; ++i) {
                result[i] = generated_field[i];
            }
        }
        else {
            result[parameter_index % 3] = point[3];
        }
        return result;
    }

    template <typename type>
    se3<type> se3<type>::exp(const math::matrix<type, 6, 1>& omega_upsilon) {
        const math::matrix<type, 3, 1> omega = { { omega_upsilon[0], omega_upsilon[1], omega_upsilon[2] } };
        const math::matrix<type, 3, 1> upsilon = { { omega_upsilon[3], omega_upsilon[4], omega_upsilon[5] } };
        return { so3<type>::exp(omega), so3<type>::left_jacobian(omega) * upsilon };
    }

    template <typename type>
    constexpr math::matrix<type, 6, 1> se3<type>::log() const {
        const math::matrix<type, 3, 1> omega = this->rotation_so3.log();
        const math::matrix<type, 3, 1> upsilon = so3<type>::left_jacobian_inverse(omega) * this->translation_vector;
        return { { omega[0], omega[1], omega[2], upsilon[0], upsilon[1], upsilon[2] } };
    }

    template <typename type>
    math::matrix<type, 3, 3> se3<type>::left_jacobian_coupling(const math::matrix<type, 3, 1>& omega, const math::matrix<type, 3, 1>& upsilon) {
        // Barfoot's Q, the lower left block of the se3 left jacobian, with the coefficients (theta - sin(theta)) / theta^3,
        // (theta^2 + 2 cos(theta) - 2) / (2 theta^4) and (2 theta - 3 sin(theta) + theta cos(theta)) / (2 theta^5) evaluated by their series
        // below one radian, where the closed forms cancel.
        const type theta_squared = omega.get_length_squared();
        type first = 0;
        type second = 0;
        type third = 0;
        if (theta_squared < 1) {
            // The inverse factorials 1 / (k + 3)! for k = 0 to 18.
            const type inverse_factorials[19] = {
                type(1.0 / 6.0),
                type(1.0 / 24.0),
                type(1.0 / 120.0),
                type(1.0 / 720.0),
                type(1.0 / 5040.0),
                type(1.0 / 40320.0),
                type(1.0 / 362880.0),
                type(1.0 / 3628800.0),
                type(1.0 / 39916800.0),
                type(1.0 / 479001600.0),
                type(1.0 / 6227020800.0),
                type(1.0 / 87178291200.0),
                type(1.0 / 1307674368000.0),
                type(1.0 / 20922789888000.0),
                type(1.0 / 355687428096000.0),
                type(1.0 / 6402373705728000.0),
                type(1.0 / 121645100408832000.0),
                type(1.0 / 2432902008176640000.0),
                type(1.0 / 51090942171709440000.0)
            };
            for (size_t k = 9; k-- > 0;) {
                first = inverse_factorials[(2 * k)] - (theta_squared * first);
                second = inverse_factorials[(2 * k) + 1] - (theta_squared * second);
                third = (static_cast<type>(k + 1) * inverse_factorials[(2 * k) + 2]) - (theta_squared * third);
            }
        }
        else {
            const type theta = math::sqrt(theta_squared);
            const type sin_theta = math::sin(theta);
            const type cos_theta = math::cos(theta);
            first = (theta - sin_theta) / (theta_squared * theta);
            second = (theta_squared + (2 * cos_theta) - 2) / (2 * theta_squared * theta_squared);
            third = ((2 * theta) - (3 * sin_theta) + (theta * cos_theta)) / (2 * theta_squared * theta_squared * theta);
        }
        const math::matrix<type, 3, 3> omega_hat = { { { 0, -omega[2], omega[1] },
                                                       { omega[2], 0, -omega[0] },
                                                       { -omega[1], omega[0], 0 } } };
        const math::matrix<type, 3, 3> upsilon_hat = { { { 0, -upsilon[2], upsilon[1] },
                                                         { upsilon[2], 0, -upsilon[0] },
                                                         { -upsilon[1], upsilon[0], 0 } } };
        const math::matrix<type, 3, 3> omega_upsilon = omega_hat * upsilon_hat;
        const math::matrix<type, 3, 3> upsilon_omega = upsilon_hat * omega_hat;
        const math::matrix<type, 3, 3> omega_upsilon_omega = omega_upsilon * omega_hat;
        return (type(0.5) * upsilon_hat) +
               (first * (omega_upsilon + upsilon_omega + omega_upsilon_omega)) +
               (second * ((omega_hat * omega_upsilon) + (upsilon_omega * omega_hat) - (type(3) * omega_upsilon_omega))) +
               (third * ((omega_upsilon_omega * omega_hat) + (omega_hat * omega_upsilon_omega)));
    }

    template <typename type>
    math::matrix<type, 6, 6> se3<type>::left_jacobian(const math::matrix<type, 6, 1>& omega_upsilon) {
        const math::matrix<type, 3, 1> omega = { { omega_upsilon[0], omega_upsilon[1], omega_upsilon[2] } };
        const math::matrix<type, 3, 1> upsilon = { { omega_upsilon[3], omega_upsilon[4], omega_upsilon[5] } };
        const math::matrix<type, 3, 3> rotation_jacobian = so3<type>::left_jacobian(omega);
        const math::matrix<type, 3, 3> coupling = se3<type>::left_jacobian_coupling(omega, upsilon);
        math::matrix<type, 6, 6> result = math::matrix<type, 6, 6>::zero();
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                result[i][j] = rotation_jacobian[i][j];
                result[i + 3][j] = coupling[i][j];
                result[i + 3][j + 3] = rotation_jacobian[i][j];
            }
        }
        return result;
    }

    template <typename type>
    math::matrix<type, 6, 6> se3<type>::left_jacobian_inverse(const math::matrix<type, 6, 1>& omega_upsilon) {
        const math::matrix<type, 3, 1> omega = { { omega_upsilon[0], omega_upsilon[1], omega_upsilon[2] } };
        const math::matrix<type, 3, 1> upsilon = { { omega_upsilon[3], omega_upsilon[4], omega_upsilon[5] } };
        const math::matrix<type, 3, 3> rotation_jacobian_inverse = so3<type>::left_jacobian_inverse(omega);
        const math::matrix<type, 3, 3> coupling_block = se3<type>::left_jacobian_coupling(omega, upsilon);
        const math::matrix<type, 3, 3> inverse_coupling_block = -(rotation_jacobian_inverse * coupling_block * rotation_jacobian_inverse);
        math::matrix<type, 6, 6> result = math::matrix<type, 6, 6>::zero();
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                result[i][j] = rotation_jacobian_inverse[i][j];
                result[i + 3][j] = inverse_coupling_block[i][j];
                result[i + 3][j + 3] = rotation_jacobian_inverse[i][j];
            }
        }
        return result;
    }

    template <typename type>
    constexpr bool se3<type>::operator==(const se3& rhs) const {
        return (this->rotation_so3 == rhs.rotation_so3) && (this->translation_vector == rhs.translation_vector);
    }

    template <typename type>
    constexpr bool se3<type>::operator!=(const se3& rhs) const {
        return (this->rotation_so3 != rhs.rotation_so3) || (this->translation_vector != rhs.translation_vector);
    }

    template <typename type>
    constexpr se3<type> se3<type>::operator*(const se3& rhs) const {
        return {
            this->rotation_so3 * rhs.rotation_so3,
            (this->rotation_so3 * rhs.translation_vector) + this->translation_vector
        };
    }

    template <typename type>
    constexpr math::matrix<type, 3, 1> se3<type>::operator*(const math::matrix<type, 3, 1>& point) const {
        return (this->rotation_so3 * point) + this->translation_vector;
    }

    template <typename type>
    constexpr sim3<type>::sim3()
        : transformation_se3()
        , scale_scalar(type(1)) {
    }

    template <typename type>
    constexpr sim3<type>::sim3(const se3<type>& initial_transformation_se3, type initial_scale_scalar)
        : transformation_se3(initial_transformation_se3)
        , scale_scalar(initial_scale_scalar) {
    }

    template <typename type>
    constexpr sim3<type>::sim3(const type (&initial_rotation_quaternion_values)[4], const type (&initial_translation_vector)[3], const type initial_scale_scalar)
        : transformation_se3(initial_rotation_quaternion_values, initial_translation_vector)
        , scale_scalar(initial_scale_scalar) {
    }

    template <typename type>
    sim3<type> sim3<type>::identity() {
        return { se3<type>::identity(), 1.0 };
    }

    template <typename type>
    constexpr const se3<type>& sim3<type>::transformation() const {
        return this->transformation_se3;
    }

    template <typename type>
    constexpr se3<type>& sim3<type>::transformation() {
        return this->transformation_se3;
    }

    template <typename type>
    constexpr const type& sim3<type>::scale() const {
        return this->scale_scalar;
    }

    template <typename type>
    constexpr type& sim3<type>::scale() {
        return this->scale_scalar;
    }

    template <typename type>
    constexpr sim3<type> sim3<type>::inverse() const {
        const type scale_inverse = 1 / this->scale_scalar;
        se3<type> transformation_inverse = this->transformation_se3.inverse();
        transformation_inverse.translation() = transformation_inverse.translation() * scale_inverse;
        return { transformation_inverse, scale_inverse };
    }

    template <typename type>
    math::matrix<type, 4, 4> sim3<type>::generator(size_t parameter_index) {
        ASSERT(parameter_index < 7, "The sim3 algebra only has seven parameters.");
        math::matrix<type, 4, 4> result = math::matrix<type, 4, 4>::zero();
        if (parameter_index < 6) {
            result = se3<type>::generator(parameter_index);
        }
        else {
            result[0][0] = 1.0;
            result[1][1] = 1.0;
            result[2][2] = 1.0;
        }
        return result;
    }

    template <typename type>
    math::matrix<type, 4, 1> sim3<type>::generator_field(size_t parameter_index, const math::matrix<type, 4, 1>& point) {
        ASSERT(parameter_index < 7, "The sim3 algebra only has seven parameters.");
        math::matrix<type, 4, 1> result = math::matrix<type, 4, 1>::zero();
        if (parameter_index < 6) {
            result = se3<type>::generator_field(parameter_index, point);
        }
        else {
            result[0] = point[0];
            result[1] = point[1];
            result[2] = point[2];
        }
        return result;
    }

    template <typename type>
    template <bool with_derivatives>
    constexpr typename sim3<type>::translation_terms sim3<type>::translation_coefficients(const type sigma, const type theta) {
        // W = int_0^1 exp(sigma * t) * exp(t * omega_hat) dt = c * I + a * omega_hat + b * omega_hat^2. With z = sigma + i * theta and
        // phi(z) = (exp(z) - 1) / z these are c = phi(sigma), a = Im(phi(z)) / theta and b = (phi(sigma) - Re(phi(z))) / theta^2, all smooth
        // in sigma and u = theta^2, so each form below avoids cancelling terms. The derivatives are only computed when asked for.
        const type theta_squared = theta * theta;
        const type inverse_factorials[21] = {
            type(1.0),
            type(1.0 / 2.0),
            type(1.0 / 6.0),
            type(1.0 / 24.0),
            type(1.0 / 120.0),
            type(1.0 / 720.0),
            type(1.0 / 5040.0),
            type(1.0 / 40320.0),
            type(1.0 / 362880.0),
            type(1.0 / 3628800.0),
            type(1.0 / 39916800.0),
            type(1.0 / 479001600.0),
            type(1.0 / 6227020800.0),
            type(1.0 / 87178291200.0),
            type(1.0 / 1307674368000.0),
            type(1.0 / 20922789888000.0),
            type(1.0 / 355687428096000.0),
            type(1.0 / 6402373705728000.0),
            type(1.0 / 121645100408832000.0),
            type(1.0 / 2432902008176640000.0),
            type(1.0 / 51090942171709440000.0)
        };
        translation_terms terms = {};
        if ((sigma * sigma) + theta_squared <= 1) {
            // Horner's rule on phi(z) = sum_k z^k / (k + 1)!, tracking Re(phi(z)), Im(phi(z)) / theta and the difference to phi(sigma) over theta^2,
            // and their derivatives by u and by sigma.
            type series_c = inverse_factorials[20];
            type series_real = inverse_factorials[20];
            type series_a = 0;
            type series_b = 0;
            type real_u = 0;
            type real_sigma = 0;
            for (size_t k = 20; k-- > 0;) {
                const type next_b = (sigma * series_b) + series_a;
                const type next_a = series_real + (sigma * series_a);
                const type next_real = inverse_factorials[k] + (sigma * series_real) - (theta_squared * series_a);
                if constexpr (with_derivatives) {
                    const type next_b_u = (sigma * terms.b_u) + terms.a_u;
                    const type next_a_u = real_u + (sigma * terms.a_u);
                    const type next_real_u = (sigma * real_u) - series_a - (theta_squared * terms.a_u);
                    const type next_b_sigma = series_b + (sigma * terms.b_sigma) + terms.a_sigma;
                    const type next_a_sigma = real_sigma + series_a + (sigma * terms.a_sigma);
                    const type next_real_sigma = series_real + (sigma * real_sigma) - (theta_squared * terms.a_sigma);
                    terms.c_sigma = series_c + (sigma * terms.c_sigma);
                    terms.b_u = next_b_u;
                    terms.a_u = next_a_u;
                    real_u = next_real_u;
                    terms.b_sigma = next_b_sigma;
                    terms.a_sigma = next_a_sigma;
                    real_sigma = next_real_sigma;
                }
                series_c = inverse_factorials[k] + (sigma * series_c);
                series_b = next_b;
                series_a = next_a;
                series_real = next_real;
            }
            terms.a = series_a;
            terms.b = series_b;
            terms.c = series_c;
            return terms;
        }
        // Outside the unit circle the closed forms are rearranged so that none of their terms cancel, using the half angle form of
        // (1 - cos(theta)) / theta^2, and series for the derivatives of it and of sin(theta) / theta below one radian.
        const type scale = math::exp(sigma);
        if (math::abs(sigma) <= 1) {
            type series_c = inverse_factorials[20];
            for (size_t k = 20; k-- > 0;) {
                if constexpr (with_derivatives) {
                    terms.c_sigma = series_c + (sigma * terms.c_sigma);
                }
                series_c = inverse_factorials[k] + (sigma * series_c);
            }
            terms.c = series_c;
        }
        else {
            terms.c = (scale - 1) / sigma;
            if constexpr (with_derivatives) {
                terms.c_sigma = (scale - terms.c) / sigma;
            }
        }
        type sinc = 1;
        type half_angle = 0.5;
        if (theta > 0) {
            sinc = math::sin(theta) / theta;
            const type half_sinc = math::sin(0.5 * theta) / (0.5 * theta);
            half_angle = 0.5 * half_sinc * half_sinc;
        }
        const type modulus_squared = (sigma * sigma) + theta_squared;
        terms.a = ((sigma * ((scale * sinc) - terms.c)) + (scale * theta_squared * half_angle)) / modulus_squared;
        terms.b = (terms.c + (scale * ((sigma * half_angle) - sinc))) / modulus_squared;
        if constexpr (with_derivatives) {
            type sinc_u = 0;
            type half_angle_u = 0;
            if (theta_squared < 1) {
                for (size_t j = 9; j-- > 0;) {
                    sinc_u = -(static_cast<type>(j + 1) * inverse_factorials[(2 * j) + 2]) - (theta_squared * sinc_u);
                    half_angle_u = -(static_cast<type>(j + 1) * inverse_factorials[(2 * j) + 3]) - (theta_squared * half_angle_u);
                }
            }
            else {
                sinc_u = (math::cos(theta) - sinc) / (2 * theta_squared);
                half_angle_u = (sinc - (2 * half_angle)) / (2 * theta_squared);
            }
            terms.a_u = ((sigma * scale * sinc_u) + (scale * half_angle) + (scale * theta_squared * half_angle_u) - terms.a) / modulus_squared;
            terms.b_u = ((scale * ((sigma * half_angle_u) - sinc_u)) - terms.b) / modulus_squared;
            terms.a_sigma = (((scale * sinc) - terms.c) + (sigma * ((scale * sinc) - terms.c_sigma)) + (scale * theta_squared * half_angle) - (2 * sigma * terms.a)) / modulus_squared;
            terms.b_sigma = (terms.c_sigma + (scale * ((sigma * half_angle) - sinc)) + (scale * half_angle) - (2 * sigma * terms.b)) / modulus_squared;
        }
        return terms;
    }

    template <typename type>
    constexpr void sim3<type>::translation_inverse_coefficients(const translation_terms& terms, const type theta_squared, type& inverse_a, type& inverse_b, type& inverse_c) {
        // The inverse of W is in the same algebra, as omega_hat^3 = -theta^2 * omega_hat. The denominator is |phi(z)|^2, which only vanishes
        // for z = 2 * pi * k * i beyond the angles a logarithm returns.
        const type real = terms.c - (theta_squared * terms.b);
        const type modulus_squared = (real * real) + (theta_squared * terms.a * terms.a);
        inverse_c = type(1) / terms.c;
        inverse_a = -terms.a / modulus_squared;
        inverse_b = ((terms.a * terms.a) - (terms.b * real)) / (terms.c * modulus_squared);
    }

    template <typename type>
    sim3<type> sim3<type>::exp(const math::matrix<type, 7, 1>& omega_upsilon_sigma) {
        const math::matrix<type, 3, 1> omega = { { omega_upsilon_sigma[0], omega_upsilon_sigma[1], omega_upsilon_sigma[2] } };
        const math::matrix<type, 3, 1> upsilon = { { omega_upsilon_sigma[3], omega_upsilon_sigma[4], omega_upsilon_sigma[5] } };
        const type sigma = omega_upsilon_sigma[6];
        const type theta = math::sqrt(omega.get_length_squared());
        const math::matrix<type, 3, 3> omega_hat = { { { 0, -omega[2], omega[1] },
                                                       { omega[2], 0, -omega[0] },
                                                       { -omega[1], omega[0], 0 } } };
        const translation_terms terms = sim3<type>::translation_coefficients<false>(sigma, theta);
        const math::matrix<type, 3, 3> w = terms.a * omega_hat + terms.b * (omega_hat * omega_hat) + terms.c * math::matrix<type, 3, 3>::identity();
        return { { so3<type>::exp(omega), w * upsilon }, math::exp(sigma) };
    }

    template <typename type>
    constexpr math::matrix<type, 7, 1> sim3<type>::log() const {
        const math::matrix<type, 3, 1> omega = this->transformation_se3.rotation().log();
        const type theta = math::sqrt(omega.get_length_squared());
        const type sigma = math::log(this->scale_scalar);
        const math::matrix<type, 3, 3> omega_hat = { { { 0, -omega[2], omega[1] },
                                                       { omega[2], 0, -omega[0] },
                                                       { -omega[1], omega[0], 0 } } };
        // Inverting exp's own W makes log the exact inverse of the translation that exp computes.
        const translation_terms terms = sim3<type>::translation_coefficients<false>(sigma, theta);
        type inverse_a = 0;
        type inverse_b = 0;
        type inverse_c = 0;
        sim3<type>::translation_inverse_coefficients(terms, theta * theta, inverse_a, inverse_b, inverse_c);
        const math::matrix<type, 3, 3> w_inv = inverse_a * omega_hat + inverse_b * (omega_hat * omega_hat) + inverse_c * math::matrix<type, 3, 3>::identity();
        const math::matrix<type, 3, 1> upsilon = w_inv * this->transformation_se3.translation();
        return { { omega[0], omega[1], omega[2], upsilon[0], upsilon[1], upsilon[2], sigma } };
    }

    template <typename type>
    void sim3<type>::left_jacobian_blocks(const math::matrix<type, 7, 1>& omega_upsilon_sigma, translation_terms& terms, math::matrix<type, 3, 3>& coupling, math::matrix<type, 3, 1>& sigma_column) {
        // The left jacobian is [J, 0, 0; Q, W, s; 0, 0, 1], with J the so3 left jacobian and W the translation matrix of exp. As
        // exp(xi + delta) = exp(jacobian * delta) * exp(xi), its translation t = W * upsilon gives Q = dt / d omega + t^ * J and s = dt / d sigma - t.
        const math::matrix<type, 3, 1> omega = { { omega_upsilon_sigma[0], omega_upsilon_sigma[1], omega_upsilon_sigma[2] } };
        const math::matrix<type, 3, 1> upsilon = { { omega_upsilon_sigma[3], omega_upsilon_sigma[4], omega_upsilon_sigma[5] } };
        const type sigma = omega_upsilon_sigma[6];
        terms = sim3<type>::translation_coefficients<true>(sigma, math::sqrt(omega.get_length_squared()));
        const math::matrix<type, 3, 1> omega_upsilon = { { (omega[1] * upsilon[2]) - (omega[2] * upsilon[1]),
                                                           (omega[2] * upsilon[0]) - (omega[0] * upsilon[2]),
                                                           (omega[0] * upsilon[1]) - (omega[1] * upsilon[0]) } };
        const math::matrix<type, 3, 1> omega_omega_upsilon = { { (omega[1] * omega_upsilon[2]) - (omega[2] * omega_upsilon[1]),
                                                                 (omega[2] * omega_upsilon[0]) - (omega[0] * omega_upsilon[2]),
                                                                 (omega[0] * omega_upsilon[1]) - (omega[1] * omega_upsilon[0]) } };
        const math::matrix<type, 3, 1> translation = (terms.c * upsilon) + (terms.a * omega_upsilon) + (terms.b * omega_omega_upsilon);
        const math::matrix<type, 3, 3> translation_hat = { { { 0, -translation[2], translation[1] },
                                                             { translation[2], 0, -translation[0] },
                                                             { -translation[1], translation[0], 0 } } };
        const type omega_dot_upsilon = (omega[0] * upsilon[0]) + (omega[1] * upsilon[1]) + (omega[2] * upsilon[2]);
        // The derivative of a * (omega x upsilon) + b * (omega x (omega x upsilon)) by omega, where a and b depend on omega through u = |omega|^2.
        coupling = translation_hat * so3<type>::left_jacobian(omega);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                type derivative = terms.b * ((omega[i] * upsilon[j]) - (2 * upsilon[i] * omega[j]) + ((i == j) ? omega_dot_upsilon : type(0)));
                derivative += 2 * ((terms.a_u * omega_upsilon[i]) + (terms.b_u * omega_omega_upsilon[i])) * omega[j];
                coupling[i][j] += derivative;
            }
        }
        coupling[0][1] += terms.a * upsilon[2];
        coupling[0][2] -= terms.a * upsilon[1];
        coupling[1][0] -= terms.a * upsilon[2];
        coupling[1][2] += terms.a * upsilon[0];
        coupling[2][0] += terms.a * upsilon[1];
        coupling[2][1] -= terms.a * upsilon[0];
        sigma_column = ((terms.c_sigma - terms.c) * upsilon) + ((terms.a_sigma - terms.a) * omega_upsilon) + ((terms.b_sigma - terms.b) * omega_omega_upsilon);
    }

    template <typename type>
    math::matrix<type, 7, 7> sim3<type>::left_jacobian(const math::matrix<type, 7, 1>& omega_upsilon_sigma) {
        const math::matrix<type, 3, 1> omega = { { omega_upsilon_sigma[0], omega_upsilon_sigma[1], omega_upsilon_sigma[2] } };
        translation_terms terms = {};
        math::matrix<type, 3, 3> coupling;
        math::matrix<type, 3, 1> sigma_column;
        sim3<type>::left_jacobian_blocks(omega_upsilon_sigma, terms, coupling, sigma_column);
        const math::matrix<type, 3, 3> rotation_jacobian = so3<type>::left_jacobian(omega);
        const math::matrix<type, 3, 3> omega_hat = { { { 0, -omega[2], omega[1] },
                                                       { omega[2], 0, -omega[0] },
                                                       { -omega[1], omega[0], 0 } } };
        const math::matrix<type, 3, 3> w = terms.a * omega_hat + terms.b * (omega_hat * omega_hat) + terms.c * math::matrix<type, 3, 3>::identity();
        math::matrix<type, 7, 7> result = math::matrix<type, 7, 7>::zero();
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                result[i][j] = rotation_jacobian[i][j];
                result[i + 3][j] = coupling[i][j];
                result[i + 3][j + 3] = w[i][j];
            }
            result[i + 3][6] = sigma_column[i];
        }
        result[6][6] = 1.0;
        return result;
    }

    template <typename type>
    math::matrix<type, 7, 7> sim3<type>::left_jacobian_inverse(const math::matrix<type, 7, 1>& omega_upsilon_sigma) {
        // The inverse of [J, 0, 0; Q, W, s; 0, 0, 1] is [J^-1, 0, 0; -W^-1 * Q * J^-1, W^-1, -W^-1 * s; 0, 0, 1].
        const math::matrix<type, 3, 1> omega = { { omega_upsilon_sigma[0], omega_upsilon_sigma[1], omega_upsilon_sigma[2] } };
        translation_terms terms = {};
        math::matrix<type, 3, 3> coupling;
        math::matrix<type, 3, 1> sigma_column;
        sim3<type>::left_jacobian_blocks(omega_upsilon_sigma, terms, coupling, sigma_column);
        const math::matrix<type, 3, 3> rotation_jacobian_inverse = so3<type>::left_jacobian_inverse(omega);
        type inverse_a = 0;
        type inverse_b = 0;
        type inverse_c = 0;
        sim3<type>::translation_inverse_coefficients(terms, omega.get_length_squared(), inverse_a, inverse_b, inverse_c);
        const math::matrix<type, 3, 3> omega_hat = { { { 0, -omega[2], omega[1] },
                                                       { omega[2], 0, -omega[0] },
                                                       { -omega[1], omega[0], 0 } } };
        const math::matrix<type, 3, 3> w_inv = inverse_a * omega_hat + inverse_b * (omega_hat * omega_hat) + inverse_c * math::matrix<type, 3, 3>::identity();
        const math::matrix<type, 3, 3> inverse_coupling = -(w_inv * coupling * rotation_jacobian_inverse);
        const math::matrix<type, 3, 1> inverse_sigma_column = -(w_inv * sigma_column);
        math::matrix<type, 7, 7> result = math::matrix<type, 7, 7>::zero();
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                result[i][j] = rotation_jacobian_inverse[i][j];
                result[i + 3][j] = inverse_coupling[i][j];
                result[i + 3][j + 3] = w_inv[i][j];
            }
            result[i + 3][6] = inverse_sigma_column[i];
        }
        result[6][6] = 1.0;
        return result;
    }

    template <typename type>
    math::matrix<type, 7, 7> sim3<type>::adjoint(const sim3& similarity) {
        const math::matrix<type, 3, 3> rotation = similarity.transformation().rotation().get_matrix();
        const math::matrix<type, 3, 1>& translation = similarity.transformation().translation();
        const type scale = similarity.scale();
        const type translation_skew[3][3] = { { type(0), -translation[2], +translation[1] },
                                              { +translation[2], type(0), -translation[0] },
                                              { -translation[1], +translation[0], type(0) } };
        math::matrix<type, 7, 7> result = math::matrix<type, 7, 7>::zero();
        for (size_t row = 0; row < 3; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                result[row][column] = rotation[row][column];
                type translate_rotate = type(0);
                for (size_t k = 0; k < 3; ++k) {
                    translate_rotate += translation_skew[row][k] * rotation[k][column];
                }
                result[row + 3][column] = translate_rotate;
                result[row + 3][column + 3] = scale * rotation[row][column];
            }
            result[row + 3][6] = -translation[row];
        }
        result[6][6] = 1.0;
        return result;
    }

    template <typename type>
    constexpr bool sim3<type>::operator==(const sim3& rhs) const {
        return (this->transformation_se3 == rhs.transformation_se3) && (this->scale_scalar == rhs.scale_scalar);
    }

    template <typename type>
    constexpr bool sim3<type>::operator!=(const sim3& rhs) const {
        return (this->transformation_se3 != rhs.transformation_se3) || (this->scale_scalar != rhs.scale_scalar);
    }

    template <typename type>
    constexpr sim3<type> sim3<type>::operator*(const sim3& rhs) const {
        return { { this->transformation_se3.rotation() * rhs.transformation_se3.rotation(),
                   (this->transformation_se3.rotation() * rhs.transformation_se3.translation() * this->scale_scalar) + this->transformation_se3.translation() },
                 this->scale_scalar * rhs.scale_scalar };
    }

    template <typename type>
    constexpr math::matrix<type, 3, 1> sim3<type>::operator*(const math::matrix<type, 3, 1>& point) const {
        return this->transformation_se3.rotation() * point * this->scale_scalar + this->transformation_se3.translation();
    }
}

#endif // ZEROSLAM_MATH_LIE_HPP
