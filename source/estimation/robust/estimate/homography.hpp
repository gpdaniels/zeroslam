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
#ifndef ZEROSLAM_ESTIMATION_ROBUST_ESTIMATE_HOMOGRAPHY_HPP
#define ZEROSLAM_ESTIMATION_ROBUST_ESTIMATE_HOMOGRAPHY_HPP

#include "estimation/correspondence_2d_2d.hpp"
#include "estimation/robust/estimator.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace estimation::robust::estimate {
    template <typename type>
    class model_homography final {
    public:
        type homography[3][3];
    };

    // A homography mapping rhs to lhs from four normalised correspondences, scored by the symmetric transfer error; a singular homography marks every datum an outlier.
    template <typename type>
    class homography final
        : public estimator<correspondence_2d_2d<type>, 4, model_homography<type>, 1> {
    public:
        using model = model_homography<type>;

    public:
        virtual size_t generate_models(
            const correspondence_2d_2d<type>* const __restrict data,
            const size_t data_size,
            model* const __restrict models
        ) const override final;

        virtual void compute_residuals(
            const correspondence_2d_2d<type>* const __restrict data,
            const size_t data_size,
            const model& candidate,
            float* const __restrict residuals
        ) const override final;
    };

    extern template class homography<float>;
    extern template class homography<double>;
}

#endif // ZEROSLAM_ESTIMATION_ROBUST_ESTIMATE_HOMOGRAPHY_HPP
