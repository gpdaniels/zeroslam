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

#include "optimisation/vertices/intrinsics.hpp"

#include "math/math.hpp"

namespace optimisation::vertices {
    intrinsics::intrinsics(const sensor::camera::model<double>& camera_model, const int* const parameter_locals, const size_t parameter_locals_length)
        : held(std::make_unique<state>())
        , count(0)
        , free_count(0) {
        this->held->camera = camera_model;
        for (size_t index = 0; index < vertex::maximum_parameters; ++index) {
            this->held->locals[index] = -1;
        }
        const size_t parameters = camera_model.get_parameter_count();
        double values[vertex::maximum_parameters] = {};
        if ((parameters == 0) || (parameters > vertex::maximum_parameters) || (parameter_locals == nullptr) || (parameter_locals_length != parameters) || !camera_model.get_parameters(&values[0], parameters)) {
            return;
        }
        int locals = 0;
        for (size_t index = 0; index < parameters; ++index) {
            if ((parameter_locals[index] < -1) || (parameter_locals[index] >= static_cast<int>(vertex::maximum_parameters))) {
                return;
            }
            locals = math::max(locals, parameter_locals[index] + 1);
        }
        // Each local dimension updates its first parameter by its step and the others of its group in proportion to them.
        for (int local = 0; local < locals; ++local) {
            int first = -1;
            for (size_t index = 0; index < parameters; ++index) {
                if (parameter_locals[index] != local) {
                    continue;
                }
                if (first < 0) {
                    first = static_cast<int>(index);
                }
                const double ratio = values[index] / values[first];
                this->held->weights[index] = ((static_cast<int>(index) == first) || !math::isfinite(ratio)) ? 1.0 : ratio;
                this->held->locals[index] = static_cast<signed char>(local);
            }
            if (first < 0) {
                for (size_t index = 0; index < vertex::maximum_parameters; ++index) {
                    this->held->locals[index] = -1;
                    this->held->weights[index] = 0.0;
                }
                return;
            }
        }
        this->count = static_cast<int>(parameters);
        this->free_count = locals;
    }

    intrinsics::intrinsics(const sensor::camera::model<double>& camera_model, const bool* const free_parameters, const size_t free_parameters_length)
        : intrinsics(camera_model, static_cast<const int*>(nullptr), 0) {
        const size_t parameters = camera_model.get_parameter_count();
        if ((free_parameters == nullptr) || (free_parameters_length != parameters) || (parameters > vertex::maximum_parameters)) {
            return;
        }
        int locals[vertex::maximum_parameters] = {};
        int next = 0;
        for (size_t index = 0; index < parameters; ++index) {
            locals[index] = free_parameters[index] ? next++ : -1;
        }
        *this = intrinsics(camera_model, &locals[0], parameters);
    }

    intrinsics::intrinsics(const intrinsics& other)
        : held(std::make_unique<state>(*other.held))
        , count(other.count)
        , free_count(other.free_count) {
    }

    intrinsics::intrinsics(intrinsics&& other)
        : intrinsics(static_cast<const intrinsics&>(other)) {
    }

    intrinsics& intrinsics::operator=(const intrinsics& other) {
        if (this != &other) {
            *this->held = *other.held;
            this->count = other.count;
            this->free_count = other.free_count;
        }
        return *this;
    }

    intrinsics& intrinsics::operator=(intrinsics&& other) {
        return *this = static_cast<const intrinsics&>(other);
    }

    intrinsics::~intrinsics() {
    }

    vertex intrinsics::make(const sensor::camera::model<double>& camera_model, const int* const parameter_locals, const size_t parameter_locals_length) {
        vertex made{ intrinsics(camera_model, parameter_locals, parameter_locals_length) };
        double parameters[vertex::maximum_parameters] = {};
        const size_t parameter_count_value = static_cast<size_t>(made.get_parameter_dimensions());
        if (!made.is_valid() || !camera_model.get_parameters(&parameters[0], parameter_count_value) || !made.set_parameters(&parameters[0], parameter_count_value)) {
            return vertex();
        }
        return made;
    }

    vertex intrinsics::make(const sensor::camera::model<double>& camera_model, const bool* const free_parameters, const size_t free_parameters_length) {
        vertex made{ intrinsics(camera_model, free_parameters, free_parameters_length) };
        double parameters[vertex::maximum_parameters] = {};
        const size_t parameter_count_value = static_cast<size_t>(made.get_parameter_dimensions());
        if (!made.is_valid() || !camera_model.get_parameters(&parameters[0], parameter_count_value) || !made.set_parameters(&parameters[0], parameter_count_value)) {
            return vertex();
        }
        return made;
    }

    int intrinsics::get_parameter_count() const {
        return this->count;
    }

    int intrinsics::get_local_count() const {
        return this->free_count;
    }

    void intrinsics::plus(double* const parameters, const double* const delta) const {
        double previous[vertex::maximum_parameters] = {};
        for (int index = 0; index < this->count; ++index) {
            previous[index] = parameters[index];
            const int local = this->held->locals[index];
            if (local >= 0) {
                parameters[index] += this->held->weights[index] * delta[local];
            }
        }
        // A step the model refuses is not taken, and the model is set back in case refusing it left it changed.
        if (!this->held->camera.set_parameters(parameters, static_cast<size_t>(this->count))) {
            for (int index = 0; index < this->count; ++index) {
                parameters[index] = previous[index];
            }
            static_cast<void>(this->held->camera.set_parameters(parameters, static_cast<size_t>(this->count)));
        }
    }

    void intrinsics::synchronise(const double* const parameters) const {
        static_cast<void>(this->held->camera.set_parameters(parameters, static_cast<size_t>(this->count)));
    }

    const sensor::camera::model<double>& intrinsics::get_camera() const {
        return this->held->camera;
    }

    int intrinsics::get_parameter_index(const int local_index) const {
        for (int index = 0; index < this->count; ++index) {
            if ((local_index >= 0) && (this->held->locals[index] == local_index)) {
                return index;
            }
        }
        return -1;
    }

    int intrinsics::get_local_index(const int parameter_index) const {
        return ((parameter_index >= 0) && (parameter_index < this->count)) ? static_cast<int>(this->held->locals[parameter_index]) : -1;
    }

    double intrinsics::get_weight(const int parameter_index) const {
        return (this->get_local_index(parameter_index) >= 0) ? this->held->weights[parameter_index] : 0.0;
    }

    void intrinsics::select_free_columns(const double* const jacobian_parameters, const size_t rows, const double scale, double* const jacobian_local) const {
        const size_t columns = static_cast<size_t>(this->count);
        const size_t locals = static_cast<size_t>(this->free_count);
        for (size_t row = 0; row < rows; ++row) {
            for (size_t local = 0; local < locals; ++local) {
                jacobian_local[(row * locals) + local] = 0.0;
            }
            for (size_t index = 0; index < columns; ++index) {
                const int local = this->held->locals[index];
                if (local >= 0) {
                    jacobian_local[(row * locals) + static_cast<size_t>(local)] += scale * (this->held->weights[index] * jacobian_parameters[(row * columns) + index]);
                }
            }
        }
    }
}
