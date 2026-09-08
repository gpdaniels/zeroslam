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
#ifndef ZEROSLAM_ESTIMATION_ROBUST_SAMPLE_EXHAUSTIVE_HPP
#define ZEROSLAM_ESTIMATION_ROBUST_SAMPLE_EXHAUSTIVE_HPP

#include "core/assert.hpp"
#include "estimation/robust/sampler.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace estimation::robust::sample {
    // Enumerates every combination of sample_size indices in lexicographic order, wrapping around after the last.
    template <size_t sample_size>
    class exhaustive final
        : public sampler<sample_size> {
    private:
        size_t size = 0;
        size_t indices_next[sample_size] = {};

    public:
        virtual void prepare(
            const size_t data_size
        ) override final {
            ASSERT(data_size > 0, "Data size must be greater than zero.");
            ASSERT(data_size >= sample_size, "Data size must be greater or equal to the sample size.");
            this->size = data_size;
            for (size_t i = 0; i < sample_size; ++i) {
                this->indices_next[i] = i;
            }
        }

    public:
        virtual void sample(
            size_t* const __restrict indices
        ) override final {
            ASSERT(this->size > 0, "Data size must be greater than zero.");
            ASSERT(this->size >= sample_size, "Data size must be greater or equal to the sample size.");
            for (size_t i = 0; i < sample_size; ++i) {
                indices[i] = this->indices_next[i];
            }
            // Increment the last index; if it overflows find the rightmost index that can still grow, or wrap to the first combination.
            ++this->indices_next[sample_size - 1];
            if (this->indices_next[sample_size - 1] < this->size) {
                return;
            }
            size_t position = sample_size - 1;
            while (position > 0) {
                --position;
                if (this->indices_next[position] < (position + this->size - sample_size)) {
                    ++this->indices_next[position];
                    for (size_t i = position + 1; i < sample_size; ++i) {
                        this->indices_next[i] = this->indices_next[i - 1] + 1;
                    }
                    return;
                }
            }
            for (size_t i = 0; i < sample_size; ++i) {
                this->indices_next[i] = i;
            }
        }
    };
}

#endif // ZEROSLAM_ESTIMATION_ROBUST_SAMPLE_EXHAUSTIVE_HPP
