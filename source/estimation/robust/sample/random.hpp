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
#ifndef ZEROSLAM_ESTIMATION_ROBUST_SAMPLE_RANDOM_HPP
#define ZEROSLAM_ESTIMATION_ROBUST_SAMPLE_RANDOM_HPP

#include "core/assert.hpp"
#include "core/random_pcg.hpp"
#include "estimation/robust/sampler.hpp"

namespace {
    using size_t = decltype(sizeof(0));
}

namespace estimation::robust::sample {
    // Draws distinct indices uniformly with a fixed seed, so every run is reproducible.
    template <size_t sample_size>
    class random final
        : public sampler<sample_size> {
    private:
        core::random_pcg rng;
        size_t size = 0;

    public:
        virtual void prepare(
            const size_t data_size
        ) override final {
            ASSERT(data_size > 0, "Data size must be greater than zero.");
            ASSERT(data_size >= sample_size, "Data size must be greater or equal to the sample size.");
            this->size = data_size;
        }

    public:
        virtual void sample(
            size_t* const __restrict indices
        ) override final {
            ASSERT(this->size > 0, "Data size must be greater than zero.");
            ASSERT(this->size >= sample_size, "Data size must be greater or equal to the sample size.");
            ASSERT(this->size <= 0xFFFFFFFFull, "Data size must fit the 32-bit draw.");
            const unsigned int maximum_index = static_cast<unsigned int>(this->size - 1);
            for (size_t i = 0; i < sample_size; ++i) {
                bool unique = false;
                while (!unique) {
                    indices[i] = static_cast<size_t>(this->rng.get_random(0u, maximum_index));
                    unique = true;
                    for (size_t j = 0; j < i; ++j) {
                        if (indices[j] == indices[i]) {
                            unique = false;
                            break;
                        }
                    }
                }
            }
        }
    };
}

#endif // ZEROSLAM_ESTIMATION_ROBUST_SAMPLE_RANDOM_HPP
