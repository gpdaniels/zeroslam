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

#include "optimisation/vertices/calibration.hpp"

#include "core/assert.hpp"

namespace {
    // The vertex policy of a calibration, its local dimensions the count of free values, which the vertex takes when it is made.
    class calibration_values final {
    public:
        constexpr static const char* name = optimisation::vertices::calibration::name;
        constexpr static const int parameter_count = optimisation::vertices::calibration::parameter_count;
        constexpr static const int local_count = 1;

    private:
        int free_count;

    public:
        explicit calibration_values(const int free_count_value)
            : free_count(free_count_value) {
        }

        int get_parameter_count() const {
            return calibration_values::parameter_count;
        }

        int get_local_count() const {
            return (this->free_count > 0) ? this->free_count : 1;
        }

        void plus(double* const parameters, const double* const delta) const {
            const unsigned int mask = optimisation::vertices::calibration::mask(parameters);
            int local = 0;
            for (size_t value = 0; value < optimisation::vertices::calibration::maximum_values; ++value) {
                if ((mask & (1u << value)) != 0u) {
                    ASSERT(local < this->free_count, "A calibration vertex must have a local dimension for each free value.");
                    parameters[value] += delta[local];
                    ++local;
                }
            }
        }
    };
}

namespace optimisation::vertices {
    vertex calibration::make(const double* const values, const size_t count, const unsigned int free_mask) {
        double parameters[calibration::parameter_count] = {};
        unsigned int mask = 0;
        int free_count = 0;
        for (size_t value = 0; (value < count) && (value < calibration::maximum_values); ++value) {
            parameters[value] = (values != nullptr) ? values[value] : 0.0;
            if ((free_mask & (1u << value)) != 0u) {
                mask |= (1u << value);
                ++free_count;
            }
        }
        parameters[calibration::mask_index] = static_cast<double>(mask);

        vertex result{ calibration_values(free_count) };
        const bool set = result.set_parameters(&parameters[0], static_cast<size_t>(calibration::parameter_count));
        ASSERT(set, "A calibration vertex takes its values and its mask.");
        static_cast<void>(set);
        result.set_fixed(free_count == 0);
        return result;
    }

    unsigned int calibration::mask(const double* const parameters) {
        const double value = parameters[calibration::mask_index];
        if (!(value >= 0.0) || !(value < 256.0)) {
            return 0u;
        }
        return static_cast<unsigned int>(value);
    }

    int calibration::local_index(const double* const parameters, const size_t value) {
        const unsigned int bits = calibration::mask(parameters);
        if ((value >= calibration::maximum_values) || ((bits & (1u << value)) == 0u)) {
            return -1;
        }
        int local = 0;
        for (size_t lower = 0; lower < value; ++lower) {
            local += ((bits & (1u << lower)) != 0u) ? 1 : 0;
        }
        return local;
    }
}
