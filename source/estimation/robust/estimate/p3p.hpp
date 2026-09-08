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
#ifndef ZEROSLAM_ESTIMATION_ROBUST_ESTIMATE_P3P_HPP
#define ZEROSLAM_ESTIMATION_ROBUST_ESTIMATE_P3P_HPP

#include "estimation/correspondence_2d_3d.hpp"
#include "estimation/robust/estimator.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace estimation::robust::estimate {
    template <typename type>
    class model_p3p final {
    public:
        type rotation[3][3];
        type translation[3];
    };

    // World to camera poses from three correspondences by the perspective three point solver, scored by one minus the cosine between the observed and reprojected bearings.
    template <typename type>
    class p3p final
        : public estimator<correspondence_2d_3d<type>, 3, model_p3p<type>, 4> {
    public:
        using model = model_p3p<type>;

    public:
        virtual size_t generate_models(
            const correspondence_2d_3d<type>* const __restrict data,
            const size_t data_size,
            model* const __restrict models
        ) const override final;

        virtual void compute_residuals(
            const correspondence_2d_3d<type>* const __restrict data,
            const size_t data_size,
            const model& candidate,
            float* const __restrict residuals
        ) const override final;
    };

    extern template class p3p<float>;
    extern template class p3p<double>;
}

#endif // ZEROSLAM_ESTIMATION_ROBUST_ESTIMATE_P3P_HPP
