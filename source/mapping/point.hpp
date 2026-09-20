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
#ifndef ZEROSLAM_MAPPING_POINT_HPP
#define ZEROSLAM_MAPPING_POINT_HPP

#include "math/matrix.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <array>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mapping {
    class point final {
    public:
        int id;
        math::matrix<double, 3, 1> location;
        math::matrix<double, 3, 1> colour;
        unsigned char descriptor[32] = {};
        constexpr static const size_t descriptor_history_maximum = 24;
        std::vector<std::array<unsigned char, 32>> descriptor_history;

        bool inverse_depth = false;
        math::matrix<double, 3, 3> anchor_rotation;
        math::matrix<double, 3, 1> anchor_translation;
        math::matrix<double, 3, 1> inverse_parameters;
        constexpr static const double minimum_inverse_depth = 1.0e-4;

        enum class uncertainty_kind {
            unknown,
            unbounded,
            estimated
        };
        uncertainty_kind uncertainty = uncertainty_kind::unknown;
        math::matrix<double, 3, 3> covariance;
        math::matrix<double, 3, 1> depth_direction;
        constexpr static const double depth_significance_minimum = 3.0;

    public:
        point();
        point(int input_id, const math::matrix<double, 3, 1>& input_location, const math::matrix<double, 3, 1>& input_colour);

    public:
        void add_descriptor(const unsigned char* const bytes);
        bool at_infinity() const;
        void update_location_from_inverse_depth();
        bool update_inverse_depth_from_location();
        bool anchor(const math::matrix<double, 3, 3>& rotation_cw, const math::matrix<double, 3, 1>& translation_cw);
        void anchor_at_infinity(const math::matrix<double, 3, 3>& rotation_cw, const math::matrix<double, 3, 1>& translation_cw, const math::matrix<double, 3, 1>& bearing_camera);

        void set_information(const math::matrix<double, 3, 3>& information, const math::matrix<double, 3, 1>& direction);

        void set_unbounded(const math::matrix<double, 3, 1>& direction);

        math::matrix<double, 2, 2> observation_information(const math::matrix<double, 0, 0>& jacobian, const double sigma) const;
    };
}

#endif // ZEROSLAM_MAPPING_POINT_HPP
