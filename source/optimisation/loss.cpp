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

#include "optimisation/loss.hpp"

namespace optimisation {
    loss::loss()
        : storage{}
        , functions(nullptr) {
    }

    loss::loss(const loss& other)
        : storage{}
        , functions(other.functions) {
        if (this->functions != nullptr) {
            this->functions->copy(other.storage, this->storage);
        }
    }

    loss::loss(loss&& other)
        : loss(static_cast<const loss&>(other)) {
    }

    loss& loss::operator=(const loss& other) {
        if (this != &other) {
            this->clear();
            this->functions = other.functions;
            if (this->functions != nullptr) {
                this->functions->copy(other.storage, this->storage);
            }
        }
        return *this;
    }

    loss& loss::operator=(loss&& other) {
        return *this = static_cast<const loss&>(other);
    }

    loss::~loss() {
        this->clear();
    }

    void loss::clear() {
        if (this->functions != nullptr) {
            this->functions->destroy(this->storage);
            this->functions = nullptr;
        }
    }

    bool loss::is_valid() const {
        return this->functions != nullptr;
    }

    const char* loss::name() const {
        return (this->functions != nullptr) ? this->functions->name : "";
    }

    void loss::compute(const double error_squared, math::matrix<double, 3, 1>& rho) const {
        if (this->functions == nullptr) {
            rho[0] = error_squared;
            rho[1] = 1.0;
            rho[2] = 0.0;
            return;
        }
        this->functions->compute(this->storage, error_squared, rho);
    }
}
