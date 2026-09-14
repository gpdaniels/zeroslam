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
            size_t moved_positions[sample_size] = {};
            size_t moved_values[sample_size] = {};
            size_t moved_count = 0;
            const auto entry = [&moved_positions, &moved_values, &moved_count](const size_t position) -> size_t {
                for (size_t m = 0; m < moved_count; ++m) {
                    if (moved_positions[m] == position) {
                        return moved_values[m];
                    }
                }
                return position;
            };
            for (size_t i = 0; i < sample_size; ++i) {
                const size_t drawn = static_cast<size_t>(this->rng.get_random(static_cast<unsigned int>(i), static_cast<unsigned int>(this->size - 1)));
                indices[i] = entry(drawn);
                if ((drawn == i) || ((i + 1) == sample_size)) {
                    continue;
                }
                const size_t swapped = entry(i);
                size_t m = 0;
                while ((m < moved_count) && (moved_positions[m] != drawn)) {
                    ++m;
                }
                if (m == moved_count) {
                    ++moved_count;
                }
                moved_positions[m] = drawn;
                moved_values[m] = swapped;
            }
        }
    };
}

#endif // ZEROSLAM_ESTIMATION_ROBUST_SAMPLE_RANDOM_HPP
