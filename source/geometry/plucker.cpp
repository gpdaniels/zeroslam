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

#include "geometry/plucker.hpp"

#include "math/lie.hpp"
#include "math/math.hpp"

namespace geometry {
    plucker::plucker()
        : moment(math::matrix<double, 3, 1>::zero())
        , direction(math::matrix<double, 3, 1>({ 0.0, 0.0, 1.0 })) {
    }

    plucker::plucker(const math::matrix<double, 3, 1>& moment_value, const math::matrix<double, 3, 1>& direction_value)
        : moment(moment_value)
        , direction(direction_value) {
    }

    math::matrix<double, 3, 1> plucker::cross(const math::matrix<double, 3, 1>& lhs, const math::matrix<double, 3, 1>& rhs) {
        return math::matrix<double, 3, 1>({ (lhs[1] * rhs[2]) - (lhs[2] * rhs[1]),
                                            (lhs[2] * rhs[0]) - (lhs[0] * rhs[2]),
                                            (lhs[0] * rhs[1]) - (lhs[1] * rhs[0]) });
    }

    double plucker::dot(const math::matrix<double, 3, 1>& lhs, const math::matrix<double, 3, 1>& rhs) {
        return (lhs[0] * rhs[0]) + (lhs[1] * rhs[1]) + (lhs[2] * rhs[2]);
    }

    bool plucker::from_points(const math::matrix<double, 3, 1>& point_a, const math::matrix<double, 3, 1>& point_b, plucker& result) {
        const math::matrix<double, 3, 1> difference = point_b - point_a;
        const double length = math::sqrt(difference.get_length_squared());
        if (length < 1.0e-12) {
            return false;
        }
        result.direction = difference * (1.0 / length);
        result.moment = plucker::cross(point_a, result.direction);
        return true;
    }

    bool plucker::from_planes(
        const math::matrix<double, 3, 1>& normal_1,
        const double offset_1,
        const math::matrix<double, 3, 1>& normal_2,
        const double offset_2,
        const double minimum_sine,
        plucker& result
    ) {
        const math::matrix<double, 3, 1> direction_raw = plucker::cross(normal_1, normal_2);
        const double direction_length = math::sqrt(direction_raw.get_length_squared());
        const double normal_lengths = math::sqrt(normal_1.get_length_squared()) * math::sqrt(normal_2.get_length_squared());
        if ((normal_lengths < 1.0e-24) || (direction_length < minimum_sine * normal_lengths)) {
            return false;
        }
        const math::matrix<double, 3, 1> moment_raw = (normal_2 * offset_1) - (normal_1 * offset_2);
        result.direction = direction_raw * (1.0 / direction_length);
        result.moment = moment_raw * (1.0 / direction_length);
        return true;
    }

    plucker plucker::transformed(const math::matrix<double, 3, 3>& rotation, const math::matrix<double, 3, 1>& translation) const {
        const math::matrix<double, 3, 1> rotated_direction = rotation * this->direction;
        return plucker((rotation * this->moment) + plucker::cross(translation, rotated_direction), rotated_direction);
    }

    math::matrix<double, 3, 1> plucker::closest_point_to_origin() const {
        return plucker::cross(this->direction, this->moment);
    }

    double plucker::distance_to_origin() const {
        return math::sqrt(this->moment.get_length_squared());
    }

    math::matrix<double, 3, 1> plucker::project_point(const math::matrix<double, 3, 1>& point) const {
        const math::matrix<double, 3, 1> origin_point = this->closest_point_to_origin();
        return origin_point + (this->direction * plucker::dot(point - origin_point, this->direction));
    }

    bool plucker::closest_point_to_ray(
        const math::matrix<double, 3, 1>& ray_origin,
        const math::matrix<double, 3, 1>& ray_direction,
        math::matrix<double, 3, 1>& result
    ) const {
        const math::matrix<double, 3, 1> origin_point = this->closest_point_to_origin();
        const double vv = plucker::dot(this->direction, this->direction);
        const double vw = plucker::dot(this->direction, ray_direction);
        const double ww = plucker::dot(ray_direction, ray_direction);
        const math::matrix<double, 3, 1> difference = ray_origin - origin_point;
        const double determinant = (vv * -ww) - (vw * -vw);
        if (math::abs(determinant) < 1.0e-12 * math::max(vv * ww, 1.0e-24)) {
            return false;
        }
        const double dv = plucker::dot(difference, this->direction);
        const double dw = plucker::dot(difference, ray_direction);
        const double s = ((dv * -ww) - (dw * -vw)) / determinant;
        result = origin_point + (this->direction * s);
        return true;
    }

    void plucker::to_orthonormal(math::matrix<double, 3, 3>& u, double& w1, double& w2, double& scale) const {
        const double moment_length = math::sqrt(this->moment.get_length_squared());
        const double direction_length = math::sqrt(this->direction.get_length_squared());
        scale = math::sqrt((moment_length * moment_length) + (direction_length * direction_length));
        w1 = (scale > 0.0) ? (moment_length / scale) : 0.0;
        w2 = (scale > 0.0) ? (direction_length / scale) : 1.0;
        const math::matrix<double, 3, 1> u2 = (direction_length > 1.0e-18) ? (this->direction * (1.0 / direction_length)) : math::matrix<double, 3, 1>({ 0.0, 0.0, 1.0 });
        math::matrix<double, 3, 1> u1;
        if (moment_length > 1.0e-18) {
            u1 = this->moment * (1.0 / moment_length);
        }
        else {
            const double absolute[3] = { math::abs(u2[0]), math::abs(u2[1]), math::abs(u2[2]) };
            size_t axis = 0;
            if (absolute[1] < absolute[axis]) {
                axis = 1;
            }
            if (absolute[2] < absolute[axis]) {
                axis = 2;
            }
            math::matrix<double, 3, 1> seed = math::matrix<double, 3, 1>::zero();
            seed[axis] = 1.0;
            u1 = seed - (u2 * plucker::dot(seed, u2));
            u1 = u1 * (1.0 / math::sqrt(u1.get_length_squared()));
        }
        const math::matrix<double, 3, 1> u3 = plucker::cross(u1, u2);
        for (size_t row = 0; row < 3; ++row) {
            u[row][0] = u1[row];
            u[row][1] = u2[row];
            u[row][2] = u3[row];
        }
    }

    plucker plucker::from_orthonormal(const math::matrix<double, 3, 3>& u, const double w1, const double w2, const double scale) {
        const math::matrix<double, 3, 1> u1({ u[0][0], u[1][0], u[2][0] });
        const math::matrix<double, 3, 1> u2({ u[0][1], u[1][1], u[2][1] });
        return plucker(u1 * (scale * w1), u2 * (scale * w2));
    }

    bool plucker::oplus(const double delta_theta_1, const double delta_theta_2, const double delta_theta_3, const double delta_phi) {
        math::matrix<double, 3, 3> u;
        double w1 = 0.0;
        double w2 = 0.0;
        double scale = 0.0;
        this->to_orthonormal(u, w1, w2, scale);
        const math::matrix<double, 3, 3> updated_u = u * math::so3<double>::exp(math::matrix<double, 3, 1>({ delta_theta_1, delta_theta_2, delta_theta_3 })).get_matrix();
        const double cosine = math::cos(delta_phi);
        const double sine = math::sin(delta_phi);
        const double updated_w1 = (w1 * cosine) - (w2 * sine);
        const double updated_w2 = (w1 * sine) + (w2 * cosine);
        const plucker updated = plucker::from_orthonormal(updated_u, updated_w1, updated_w2, scale);
        const double direction_length = math::sqrt(updated.direction.get_length_squared());
        if (direction_length < 1.0e-12) {
            return false;
        }
        this->moment = updated.moment * (1.0 / direction_length);
        this->direction = updated.direction * (1.0 / direction_length);
        return true;
    }
}
