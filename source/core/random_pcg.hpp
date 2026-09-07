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
#ifndef ZEROSLAM_CORE_RANDOM_PCG_HPP
#define ZEROSLAM_CORE_RANDOM_PCG_HPP

#include "core/assert.hpp"

namespace core {
    class random_pcg final {
    private:
        unsigned long long int state;
        unsigned long long int increment;

    public:
        ~random_pcg() = default;

        random_pcg()
            : state(0x853C49E6748FEA9Bull)
            , increment(0xDA3E39CB94B95BDBull) {
        }

        random_pcg(const random_pcg&) = default;
        random_pcg(random_pcg&&) = default;
        random_pcg& operator=(const random_pcg&) = default;
        random_pcg& operator=(random_pcg&&) = default;

        explicit random_pcg(unsigned long long int seed_value) {
            this->seed(seed_value);
        }

        void seed(unsigned long long int seed_value) {
            this->state = 0;
            this->increment = (seed_value << 1) | 1;
            this->get_random_raw();
            this->state += seed_value;
            this->get_random_raw();
        }

        unsigned int get_random_raw() {
            unsigned long long int state_previous = this->state;
            this->state = state_previous * 0x5851F42D4C957F2Dull + this->increment;
            unsigned int state_shift_xor_shift = static_cast<unsigned int>(((state_previous >> 18u) ^ state_previous) >> 27u);
            int rotation = static_cast<int>(state_previous >> 59u);
            return (state_shift_xor_shift >> rotation) | (state_shift_xor_shift << ((-rotation) & 31));
        }

    public:
        // A value in the open interval: 0 < value < 1.
        double get_random_exclusive() {
            return (static_cast<double>(this->get_random_raw()) + 0.5) * (1.0 / static_cast<double>(1ull << 32));
        }

        // A value in the half-open interval: 0 <= value < 1.
        double get_random_exclusive_top() {
            return static_cast<double>(this->get_random_raw()) * (1.0 / static_cast<double>(1ull << 32));
        }

        // A value in the closed interval: 0 <= value <= 1.
        double get_random_inclusive() {
            return static_cast<double>(this->get_random_raw()) * (1.0 / static_cast<double>((1ull << 32) - 1));
        }

        // An integer in the closed interval: inclusive_min <= value <= inclusive_max.
        unsigned int get_random(unsigned int inclusive_min, unsigned int inclusive_max) {
            ASSERT(inclusive_min <= inclusive_max, "Minimum bound must not exceed maximum bound.");
            const unsigned long long int range = static_cast<unsigned long long int>(inclusive_max) - static_cast<unsigned long long int>(inclusive_min) + 1ull;
            const unsigned long long int product = static_cast<unsigned long long int>(this->get_random_raw()) * range;
            return inclusive_min + static_cast<unsigned int>(product >> 32u);
        }

        // A value in the closed interval: inclusive_min <= value <= inclusive_max.
        double get_random(double inclusive_min, double inclusive_max) {
            ASSERT(inclusive_min < inclusive_max, "Minimum bound must be lower than maximum bound.");
            return (this->get_random_inclusive() * (inclusive_max - inclusive_min)) + inclusive_min;
        }
    };
}

#endif // ZEROSLAM_CORE_RANDOM_PCG_HPP
