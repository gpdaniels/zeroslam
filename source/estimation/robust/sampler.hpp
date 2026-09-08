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
#ifndef ZEROSLAM_ESTIMATION_ROBUST_SAMPLER_HPP
#define ZEROSLAM_ESTIMATION_ROBUST_SAMPLER_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace estimation::robust {
    // Interface of a data set sampler: prepare for a data size, then draw sample_size distinct indices per call.
    template <size_t template_sample_size>
    class sampler {
    private:
        static_assert(template_sample_size > 0, "Invalid number of samples, must be greater than zero.");

    public:
        constexpr static const size_t sample_size = template_sample_size;

    protected:
        virtual ~sampler() = default;
        sampler() = default;
        sampler(const sampler&) = default;
        sampler(sampler&&) = default;
        sampler& operator=(const sampler&) = default;
        sampler& operator=(sampler&&) = default;

    public:
        virtual void prepare(const size_t data_size) = 0;

        virtual void sample(size_t* const __restrict indices) = 0;
    };
}

#endif // ZEROSLAM_ESTIMATION_ROBUST_SAMPLER_HPP
