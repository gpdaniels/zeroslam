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
        constexpr bool operator==(const sim3& rhs) const;
        constexpr bool operator!=(const sim3& rhs) const;
        constexpr sim3 operator*(const sim3& rhs) const;
        constexpr math::matrix<type, 3, 1> operator*(const math::matrix<type, 3, 1>& point) const;
    };
}

namespace math {
    template <typename type>
    constexpr so3<type>::so3() = default;
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
        const type rotation_trace = initial_rotation_matrix[0][0] + initial_rotation_matrix[1][1] + initial_rotation_matrix[2][2];
        if (rotation_trace > 0) {
            const type square_root_trace_plus_one = math::sqrt(rotation_trace + 1);
            const type inverse_two_square_root_trace_plus_one = 0.5 / square_root_trace_plus_one;
            this->rotation_quaternion[0] = 0.5 * square_root_trace_plus_one;
            this->rotation_quaternion[1] = (initial_rotation_matrix[2][1] - initial_rotation_matrix[1][2]) * inverse_two_square_root_trace_plus_one;
            this->rotation_quaternion[2] = (initial_rotation_matrix[0][2] - initial_rotation_matrix[2][0]) * inverse_two_square_root_trace_plus_one;
            this->rotation_quaternion[3] = (initial_rotation_matrix[1][0] - initial_rotation_matrix[0][1]) * inverse_two_square_root_trace_plus_one;
            return;
        }
        const size_t i = (initial_rotation_matrix[0][0] < initial_rotation_matrix[1][1]) ? (1 + (initial_rotation_matrix[1][1] < initial_rotation_matrix[2][2])) : (2 * (initial_rotation_matrix[0][0] < initial_rotation_matrix[2][2]));
        const size_t j = (i + 1) % 3;
        const size_t k = (j + 1) % 3;
        const type square_root_trace_plus_one = math::sqrt(initial_rotation_matrix[i][i] - initial_rotation_matrix[j][j] - initial_rotation_matrix[k][k] + 1.0);
        const type inverse_two_square_root_trace_plus_one = 0.5 / square_root_trace_plus_one;
        this->rotation_quaternion[0 + 0] = (initial_rotation_matrix[j][k] - initial_rotation_matrix[k][j]) * inverse_two_square_root_trace_plus_one;
        this->rotation_quaternion[1 + i] = 0.5 * square_root_trace_plus_one;
        this->rotation_quaternion[1 + j] = (initial_rotation_matrix[i][j] + initial_rotation_matrix[j][i]) * inverse_two_square_root_trace_plus_one;
        this->rotation_quaternion[1 + k] = (initial_rotation_matrix[i][k] + initial_rotation_matrix[k][i]) * inverse_two_square_root_trace_plus_one;
    }

    template <typename type>
    constexpr math::matrix<type, 4, 1> so3<type>::get_quaternion() const {
        return math::matrix<type, 4, 1>(this->rotation_quaternion);
    }

    template <typename type>
    constexpr math::matrix<type, 3, 3> so3<type>::get_matrix() const {
        const type two_x = 2.0 * this->rotation_quaternion[1];
        const type two_y = 2.0 * this->rotation_quaternion[2];
        const type two_z = 2.0 * this->rotation_quaternion[3];
        const type two_x_x = two_x * this->rotation_quaternion[1];
        const type two_x_y = two_x * this->rotation_quaternion[2];
        const type two_x_z = two_x * this->rotation_quaternion[3];
        const type two_x_w = two_x * this->rotation_quaternion[0];
        const type two_y_y = two_y * this->rotation_quaternion[2];
        const type two_y_z = two_y * this->rotation_quaternion[3];
        const type two_y_w = two_y * this->rotation_quaternion[0];
        const type two_z_z = two_z * this->rotation_quaternion[3];
        const type two_z_w = two_z * this->rotation_quaternion[0];
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
        if (theta_squared < 1e-6 * 1e-6) {
            return math::matrix<type, 3, 3>::identity() + (0.5 * omega_hat) + ((1.0 / 6.0) * (omega_hat * omega_hat));
        }
        const type theta = math::sqrt(theta_squared);
        return math::matrix<type, 3, 3>::identity() + (((1 - math::cos(theta)) / (theta_squared)) * omega_hat) + (((theta - math::sin(theta)) / (theta_squared * theta)) * (omega_hat * omega_hat));
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
    math::matrix<type, 6, 6> se3<type>::left_jacobian(const math::matrix<type, 6, 1>& omega_upsilon) {
        math::matrix<type, 6, 6> result = math::matrix<type, 6, 6>::identity();
        math::matrix<type, 6, 6> adjoint = math::matrix<type, 6, 6>::zero();

        const math::matrix<type, 3, 3> omega_hat = { { { 0, -omega_upsilon[2], omega_upsilon[1] },
                                                       { omega_upsilon[2], 0, -omega_upsilon[0] },
                                                       { -omega_upsilon[1], omega_upsilon[0], 0 } } };
        const math::matrix<type, 3, 3> upsilon_hat = { { { 0, -omega_upsilon[5], omega_upsilon[4] },
                                                         { omega_upsilon[5], 0, -omega_upsilon[3] },
                                                         { -omega_upsilon[4], omega_upsilon[3], 0 } } };
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                adjoint[i][j] = omega_hat[i][j];
                adjoint[i + 3][j] = upsilon_hat[i][j];
                adjoint[i + 3][j + 3] = omega_hat[i][j];
            }
        }
        math::matrix<type, 6, 6> adjoint_power = adjoint;
        type factorial = 1.0;
        for (size_t n = 1; n < 20; ++n) {
            factorial *= static_cast<type>(n + 1);
            result = result + (adjoint_power * (1.0 / factorial));
            if (n < 19)
                adjoint_power = adjoint_power * adjoint;
        }
        return result;
    }

    template <typename type>
    math::matrix<type, 6, 6> se3<type>::left_jacobian_inverse(const math::matrix<type, 6, 1>& omega_upsilon) {
        const math::matrix<type, 3, 1> omega = { { omega_upsilon[0], omega_upsilon[1], omega_upsilon[2] } };
        const math::matrix<type, 3, 3> rotation_jacobian_inverse = so3<type>::left_jacobian_inverse(omega);
        const math::matrix<type, 6, 6> jacobian = left_jacobian(omega_upsilon);
        math::matrix<type, 3, 3> coupling_block;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                coupling_block[i][j] = jacobian[i + 3][j];
            }
        }
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
    constexpr sim3<type>::sim3() = default;

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
    sim3<type> sim3<type>::exp(const math::matrix<type, 7, 1>& omega_upsilon_sigma) {
        const math::matrix<type, 3, 1> omega = { { omega_upsilon_sigma[0], omega_upsilon_sigma[1], omega_upsilon_sigma[2] } };
        const math::matrix<type, 3, 1> upsilon = { { omega_upsilon_sigma[3], omega_upsilon_sigma[4], omega_upsilon_sigma[5] } };
        const type sigma = omega_upsilon_sigma[6];
        const type scale = math::exp(sigma);
        const type theta = math::sqrt(omega.get_length_squared());
        const math::matrix<type, 3, 3> omega_hat = { { { 0, -omega[2], omega[1] },
                                                       { omega[2], 0, -omega[0] },
                                                       { -omega[1], omega[0], 0 } } };
        const math::matrix<type, 3, 3> omega_hat_squared = omega_hat * omega_hat;
        type a, b, c;
        if (math::abs(sigma) < 1e-6) {
            c = 1 + (0.5 * sigma) + ((sigma * sigma) / 6);
            if (math::abs(theta) < 1e-6) {
                a = 0.5 + (sigma / 3) + ((sigma * sigma) / 8);
                b = (type(1) / type(6)) + (sigma / 8) + ((sigma * sigma) / 20);
            }
            else {
                const type theta_squared = theta * theta;
                const type sin_theta = math::sin(theta);
                const type cos_theta = math::cos(theta);
                a = ((1 - cos_theta) / theta_squared) + (sigma * ((sin_theta - (theta * cos_theta)) / (theta_squared * theta)));
                b = ((theta - sin_theta) / (theta_squared * theta)) + (sigma * (((0.5 * theta_squared) + 1 - cos_theta - (theta * sin_theta)) / (theta_squared * theta_squared)));
            }
        }
        else {
            c = (scale - 1) / sigma;
            if (math::abs(theta) < 1e-6) {
                const type sigma_squared = sigma * sigma;
                a = ((sigma - 1) * scale + 1) / sigma_squared;
                b = (scale * 0.5 * sigma_squared + scale - 1 - sigma * scale) / (sigma_squared * sigma);
            }
            else {
                const type theta_squared = theta * theta;
                const type scale_sin_theta = scale * math::sin(theta);
                const type scale_cos_theta = scale * math::cos(theta);
                const type theta_squared_plus_sigma_squared = theta_squared + sigma * sigma;
                a = (scale_sin_theta * sigma + (1 - scale_cos_theta) * theta) / (theta * theta_squared_plus_sigma_squared);
                b = (c - ((scale_cos_theta - 1) * sigma + scale_sin_theta * theta) / (theta_squared_plus_sigma_squared)) * 1 / (theta_squared);
            }
        }
        const math::matrix<type, 3, 3> w = a * omega_hat + b * omega_hat_squared + c * math::matrix<type, 3, 3>::identity();
        return { { so3<type>::exp(omega), w * upsilon }, scale };
    }

    template <typename type>
    constexpr math::matrix<type, 7, 1> sim3<type>::log() const {
        const math::matrix<type, 3, 1> omega = this->transformation_se3.rotation().log();
        const type theta = math::sqrt(omega.get_length_squared());
        const type sigma = math::log(this->scale_scalar);
        const math::matrix<type, 3, 3> omega_hat = { { { 0, -omega[2], omega[1] },
                                                       { omega[2], 0, -omega[0] },
                                                       { -omega[1], omega[0], 0 } } };
        const math::matrix<type, 3, 3> omega_hat_squared = omega_hat * omega_hat;
        const type scale_squared = this->scale_scalar * this->scale_scalar;
        const type theta_squared = theta * theta;
        const type sin_theta = math::sin(theta);
        const type cos_theta = math::cos(theta);
        type a, b, c;
        if (math::abs(sigma * sigma) < 1e-6) {
            c = 1 - (0.5 * sigma) + ((sigma * sigma) / 12);
            if (math::abs(theta_squared) < 1e-6) {
                a = -0.5 + (sigma / 6);
                b = type(1) / type(12);
            }
            else {
                a = -0.5 + (sigma * ((theta - sin_theta) / ((2 * theta) * (1 - cos_theta))));
                b = (theta * sin_theta + 2 * cos_theta - 2) / (2 * theta_squared * (cos_theta - 1));
            }
        }
        else {
            c = sigma / (this->scale_scalar - 1);
            if (math::abs(theta_squared) < 1e-6) {
                a = (-sigma * this->scale_scalar + this->scale_scalar - 1) / ((this->scale_scalar - 1) * (this->scale_scalar - 1));
                b = (scale_squared * sigma - 2 * scale_squared + this->scale_scalar * sigma + 2 * this->scale_scalar) / (2 * (scale_squared * this->scale_scalar) - 6 * scale_squared + 6 * this->scale_scalar - 2);
            }
            else {
                const type s_sin_theta = this->scale_scalar * sin_theta;
                const type s_cos_theta = this->scale_scalar * cos_theta;
                a = (theta * s_cos_theta - theta - sigma * s_sin_theta) / (theta * (scale_squared - 2 * s_cos_theta + 1));
                b = -this->scale_scalar *
                    (theta * s_sin_theta - theta * sin_theta + sigma * s_cos_theta - this->scale_scalar * sigma + sigma * cos_theta - sigma) /
                    (theta_squared * ((scale_squared * this->scale_scalar) - 2 * this->scale_scalar * s_cos_theta - scale_squared + 2 * s_cos_theta + this->scale_scalar - 1));
            }
        }
        const math::matrix<type, 3, 3> w_inv = a * omega_hat + b * omega_hat_squared + c * math::matrix<type, 3, 3>::identity();
        const math::matrix<type, 3, 1> upsilon = w_inv * this->transformation_se3.translation();
        return { { omega[0], omega[1], omega[2], upsilon[0], upsilon[1], upsilon[2], sigma } };
    }

    template <typename type>
    math::matrix<type, 7, 7> sim3<type>::left_jacobian(const math::matrix<type, 7, 1>& omega_upsilon_sigma) {
        math::matrix<type, 7, 7> result = math::matrix<type, 7, 7>::identity();
        math::matrix<type, 7, 7> adjoint = math::matrix<type, 7, 7>::zero();

        const type sigma = omega_upsilon_sigma[6];
        const math::matrix<type, 3, 3> omega_hat = { { { 0, -omega_upsilon_sigma[2], omega_upsilon_sigma[1] },
                                                       { omega_upsilon_sigma[2], 0, -omega_upsilon_sigma[0] },
                                                       { -omega_upsilon_sigma[1], omega_upsilon_sigma[0], 0 } } };
        const math::matrix<type, 3, 3> upsilon_hat = { { { 0, -omega_upsilon_sigma[5], omega_upsilon_sigma[4] },
                                                         { omega_upsilon_sigma[5], 0, -omega_upsilon_sigma[3] },
                                                         { -omega_upsilon_sigma[4], omega_upsilon_sigma[3], 0 } } };
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                adjoint[i][j] = omega_hat[i][j];
                adjoint[i + 3][j] = upsilon_hat[i][j];
                adjoint[i + 3][j + 3] = omega_hat[i][j];
            }
            adjoint[i + 3][i + 3] += sigma;
            adjoint[i + 3][6] = -omega_upsilon_sigma[3 + i];
        }
        math::matrix<type, 7, 7> adjoint_power = adjoint;
        type factorial = 1.0;
        for (size_t n = 1; n < 20; ++n) {
            factorial *= static_cast<type>(n + 1);
            result = result + (adjoint_power * (1.0 / factorial));
            if (n < 19)
                adjoint_power = adjoint_power * adjoint;
        }
        return result;
    }

    template <typename type>
    math::matrix<type, 7, 7> sim3<type>::left_jacobian_inverse(const math::matrix<type, 7, 1>& omega_upsilon_sigma) {
        const math::matrix<type, 3, 1> omega = { { omega_upsilon_sigma[0], omega_upsilon_sigma[1], omega_upsilon_sigma[2] } };
        const math::matrix<type, 3, 3> rotation_jacobian_inverse = so3<type>::left_jacobian_inverse(omega);
        const math::matrix<type, 7, 7> jacobian = left_jacobian(omega_upsilon_sigma);
        math::matrix<type, 3, 3> coupling_block;
        math::matrix<type, 3, 3> scaled_rotation_block;
        math::matrix<type, 3, 1> sigma_column;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                coupling_block[i][j] = jacobian[i + 3][j];
                scaled_rotation_block[i][j] = jacobian[i + 3][j + 3];
            }
            sigma_column[i] = jacobian[i + 3][6];
        }
        math::matrix<type, 3, 3> scaled_rotation_block_inverse;
        const bool invertible = math::invert(scaled_rotation_block, scaled_rotation_block_inverse);
        ASSERT(invertible, "The scaled rotation block of the sim3 left jacobian must be invertible.");
        static_cast<void>(invertible);
        const math::matrix<type, 3, 3> inverse_coupling_block = -(scaled_rotation_block_inverse * coupling_block * rotation_jacobian_inverse);
        const math::matrix<type, 3, 1> inverse_sigma_column = -(scaled_rotation_block_inverse * sigma_column);
        math::matrix<type, 7, 7> result = math::matrix<type, 7, 7>::zero();
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                result[i][j] = rotation_jacobian_inverse[i][j];
                result[i + 3][j] = inverse_coupling_block[i][j];
                result[i + 3][j + 3] = scaled_rotation_block_inverse[i][j];
            }
            result[i + 3][6] = inverse_sigma_column[i];
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
