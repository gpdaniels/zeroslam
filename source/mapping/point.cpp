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

#include "mapping/point.hpp"

#include "math/math.hpp"

namespace mapping {
    point::point() {
        this->id = -1;
        this->location = math::matrix<double, 3, 1>::zero();
        this->colour = math::matrix<double, 3, 1>::zero();
        this->anchor_rotation = math::matrix<double, 3, 3>::identity();
        this->anchor_translation = math::matrix<double, 3, 1>::zero();
        this->inverse_parameters = math::matrix<double, 3, 1>::zero();
    }

    point::point(int input_id, const math::matrix<double, 3, 1>& input_location, const math::matrix<double, 3, 1>& input_colour) {
        this->id = input_id;
        this->location = input_location;
        this->colour = input_colour;
        this->anchor_rotation = math::matrix<double, 3, 3>::identity();
        this->anchor_translation = math::matrix<double, 3, 1>::zero();
        this->inverse_parameters = math::matrix<double, 3, 1>::zero();
    }

    void point::add_descriptor(const unsigned char* const bytes) {
        std::array<unsigned char, 32> added;
        for (size_t index = 0; index < 32; ++index) {
            added[index] = bytes[index];
        }
        if (this->descriptor_history.size() >= point::descriptor_history_maximum) {
            this->descriptor_history.erase(this->descriptor_history.begin());
        }
        this->descriptor_history.push_back(added);
        const auto hamming = [](const std::array<unsigned char, 32>& lhs, const std::array<unsigned char, 32>& rhs) {
            unsigned int distance = 0;
            for (size_t index = 0; index < 32; ++index) {
                unsigned char difference = static_cast<unsigned char>(lhs[index] ^ rhs[index]);
                while (difference != 0) {
                    distance += (difference & 1u);
                    difference = static_cast<unsigned char>(difference >> 1);
                }
            }
            return distance;
        };
        size_t medoid = this->descriptor_history.size() - 1;
        unsigned int medoid_sum = 0xFFFFFFFFu;
        for (size_t i = 0; i < this->descriptor_history.size(); ++i) {
            unsigned int sum = 0;
            for (size_t j = 0; j < this->descriptor_history.size(); ++j) {
                sum += hamming(this->descriptor_history[i], this->descriptor_history[j]);
            }
            if (sum <= medoid_sum) {
                medoid_sum = sum;
                medoid = i;
            }
        }
        for (size_t index = 0; index < 32; ++index) {
            this->descriptor[index] = this->descriptor_history[medoid][index];
        }
    }

    bool point::at_infinity() const {
        return this->inverse_depth && (this->inverse_parameters[2] <= point::minimum_inverse_depth);
    }

    void point::update_location_from_inverse_depth() {
        const double rho = this->inverse_parameters[2];
        const double effective_rho = (rho > point::minimum_inverse_depth) ? rho : point::minimum_inverse_depth;
        const math::matrix<double, 3, 1> bearing({ this->inverse_parameters[0], this->inverse_parameters[1], 1.0 });
        this->location = this->anchor_translation + ((this->anchor_rotation * bearing) * (1.0 / effective_rho));
    }

    bool point::update_inverse_depth_from_location() {
        const math::matrix<double, 3, 1> anchored = math::transpose(this->anchor_rotation) * (this->location - this->anchor_translation);
        if (!(anchored[2] > 0.0)) {
            return false;
        }
        this->inverse_parameters[0] = anchored[0] / anchored[2];
        this->inverse_parameters[1] = anchored[1] / anchored[2];
        this->inverse_parameters[2] = 1.0 / anchored[2];
        return true;
    }

    bool point::anchor(const math::matrix<double, 3, 3>& rotation_cw, const math::matrix<double, 3, 1>& translation_cw) {
        const math::matrix<double, 3, 3> rotation_wc = math::transpose(rotation_cw);
        const math::matrix<double, 3, 1> translation_wc = -(rotation_wc * translation_cw);
        const math::matrix<double, 3, 3> previous_rotation = this->anchor_rotation;
        const math::matrix<double, 3, 1> previous_translation = this->anchor_translation;
        this->anchor_rotation = rotation_wc;
        this->anchor_translation = translation_wc;
        if (!this->update_inverse_depth_from_location()) {
            this->anchor_rotation = previous_rotation;
            this->anchor_translation = previous_translation;
            return false;
        }
        this->inverse_depth = true;
        return true;
    }

    void point::anchor_at_infinity(const math::matrix<double, 3, 3>& rotation_cw, const math::matrix<double, 3, 1>& translation_cw, const math::matrix<double, 3, 1>& bearing_camera) {
        this->anchor_rotation = math::transpose(rotation_cw);
        this->anchor_translation = -(this->anchor_rotation * translation_cw);
        this->inverse_parameters = math::matrix<double, 3, 1>({ bearing_camera[0] / bearing_camera[2], bearing_camera[1] / bearing_camera[2], 0.0 });
        this->inverse_depth = true;
        this->update_location_from_inverse_depth();
        this->set_unbounded(math::matrix<double, 3, 1>({ 0.0, 0.0, 1.0 }));
    }

    void point::set_information(const math::matrix<double, 3, 3>& information, const math::matrix<double, 3, 1>& direction) {
        math::matrix<double, 3, 3> inverse;
        bool finite = math::invert(information, inverse);
        for (size_t row = 0; finite && (row < 3); ++row) {
            finite = math::isfinite(inverse[row][row]) && (inverse[row][row] > 0.0);
            for (size_t column = 0; finite && (column < 3); ++column) {
                finite = math::isfinite(inverse[row][column]);
            }
        }
        if (!finite) {
            if (this->inverse_depth) {
                this->set_unbounded(direction);
            }
            return;
        }
        if (this->inverse_depth && (this->uncertainty == uncertainty_kind::unbounded) && (this->inverse_parameters[2] < point::depth_significance_minimum * math::sqrt(inverse[2][2]))) {
            this->set_unbounded(direction);
            return;
        }
        this->uncertainty = uncertainty_kind::estimated;
        this->covariance = inverse;
        this->depth_direction = direction;
    }

    void point::set_unbounded(const math::matrix<double, 3, 1>& direction) {
        this->uncertainty = uncertainty_kind::unbounded;
        this->covariance = math::matrix<double, 3, 3>::zero();
        this->depth_direction = direction;
    }

    math::matrix<double, 2, 2> point::observation_information(const math::matrix<double, 0, 0>& jacobian, const double sigma) const {
        const double pixel_information = 1.0 / (sigma * sigma);
        math::matrix<double, 2, 2> pixel_only = math::matrix<double, 2, 2>::identity() * pixel_information;
        if ((this->uncertainty == uncertainty_kind::unknown) || (jacobian.rows() != 2) || (jacobian.cols() != 3)) {
            return pixel_only;
        }
        if (this->uncertainty == uncertainty_kind::unbounded) {
            double along[2] = { 0.0, 0.0 };
            for (size_t row = 0; row < 2; ++row) {
                for (size_t column = 0; column < 3; ++column) {
                    along[row] += jacobian[row][column] * this->depth_direction[column];
                }
            }
            const double length_squared = (along[0] * along[0]) + (along[1] * along[1]);
            if (length_squared < 1.0e-24) {
                return pixel_only;
            }
            math::matrix<double, 2, 2> information;
            for (size_t row = 0; row < 2; ++row) {
                for (size_t column = 0; column < 2; ++column) {
                    information[row][column] = pixel_information * (((row == column) ? 1.0 : 0.0) - (along[row] * along[column] / length_squared));
                }
            }
            return information;
        }
        if (this->uncertainty == uncertainty_kind::estimated) {
            return pixel_only;
        }
        math::matrix<double, 2, 2> total = math::matrix<double, 2, 2>::identity() * (sigma * sigma);
        for (size_t row = 0; row < 2; ++row) {
            for (size_t column = 0; column < 2; ++column) {
                double sum = 0.0;
                for (size_t a = 0; a < 3; ++a) {
                    double inner = 0.0;
                    for (size_t b = 0; b < 3; ++b) {
                        inner += this->covariance[a][b] * jacobian[column][b];
                    }
                    sum += jacobian[row][a] * inner;
                }
                total[row][column] += sum;
            }
        }
        math::matrix<double, 2, 2> information;
        if (!math::invert(total, information)) {
            return pixel_only;
        }
        return information;
    }
}
