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

#include "optimisation/vertex.hpp"

namespace optimisation {
    vertex::vertex()
        : storage{}
        , functions(nullptr)
        , ordering_id(0)
        , fixed(false)
        , marginalised(false)
        , parameters{}
        , parameters_backup{} {
    }

    vertex::vertex(const vertex& other)
        : storage{}
        , functions(other.functions)
        , ordering_id(other.ordering_id)
        , fixed(other.fixed)
        , marginalised(other.marginalised)
        , parameters{}
        , parameters_backup{} {
        for (size_t i = 0; i < vertex::maximum_parameters; ++i) {
            this->parameters[i] = other.parameters[i];
            this->parameters_backup[i] = other.parameters_backup[i];
        }
        if (this->functions != nullptr) {
            this->functions->copy(other.storage, this->storage);
        }
    }

    vertex::vertex(vertex&& other)
        : vertex(static_cast<const vertex&>(other)) {
    }

    vertex& vertex::operator=(const vertex& other) {
        if (this != &other) {
            this->clear();
            this->functions = other.functions;
            if (this->functions != nullptr) {
                this->functions->copy(other.storage, this->storage);
            }
            this->ordering_id = other.ordering_id;
            this->fixed = other.fixed;
            this->marginalised = other.marginalised;
            for (size_t i = 0; i < vertex::maximum_parameters; ++i) {
                this->parameters[i] = other.parameters[i];
                this->parameters_backup[i] = other.parameters_backup[i];
            }
        }
        return *this;
    }

    vertex& vertex::operator=(vertex&& other) {
        return *this = static_cast<const vertex&>(other);
    }

    vertex::~vertex() {
        this->clear();
    }

    void vertex::clear() {
        if (this->functions != nullptr) {
            this->functions->destroy(this->storage);
            this->functions = nullptr;
        }
    }

    bool vertex::is_valid() const {
        return this->functions != nullptr;
    }

    const char* vertex::name() const {
        return (this->functions != nullptr) ? this->functions->name : "";
    }

    void vertex::set_ordering_id(int id) {
        this->ordering_id = id;
    }

    int vertex::get_ordering_id() const {
        return this->ordering_id;
    }

    int vertex::get_parameter_dimensions() const {
        return (this->functions != nullptr) ? this->functions->parameter_count : 0;
    }

    int vertex::get_local_dimensions() const {
        return (this->functions != nullptr) ? this->functions->local_count : 0;
    }

    void vertex::set_fixed(bool fixed_value) {
        this->fixed = fixed_value;
    }

    bool vertex::is_fixed() const {
        return this->fixed;
    }

    void vertex::set_marginalised(bool marginalised_value) {
        this->marginalised = marginalised_value;
    }

    bool vertex::is_marginalised() const {
        return this->marginalised;
    }

    const double* vertex::get_parameters() const {
        return &this->parameters[0];
    }

    bool vertex::set_parameters(const double* const parameters_value, const size_t parameters_length) {
        if ((parameters_value == nullptr) || (parameters_length != static_cast<size_t>(this->get_parameter_dimensions()))) {
            return false;
        }
        for (size_t i = 0; i < parameters_length; ++i) {
            this->parameters[i] = parameters_value[i];
        }
        return true;
    }

    bool vertex::get_parameters(double* const parameters_value, const size_t parameters_length) const {
        if ((parameters_value == nullptr) || (parameters_length != static_cast<size_t>(this->get_parameter_dimensions()))) {
            return false;
        }
        for (size_t i = 0; i < parameters_length; ++i) {
            parameters_value[i] = this->parameters[i];
        }
        return true;
    }

    void vertex::backup() {
        for (size_t i = 0; i < vertex::maximum_parameters; ++i) {
            this->parameters_backup[i] = this->parameters[i];
        }
    }

    void vertex::restore() {
        for (size_t i = 0; i < vertex::maximum_parameters; ++i) {
            this->parameters[i] = this->parameters_backup[i];
        }
    }

    void vertex::plus(const double* const delta) {
        if (this->functions != nullptr) {
            this->functions->plus(this->storage, &this->parameters[0], delta);
        }
    }
}
