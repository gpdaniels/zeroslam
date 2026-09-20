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

#include "optimisation/edge.hpp"

#include "core/assert.hpp"

namespace optimisation {
    edge::edge()
        : storage{}
        , functions(nullptr)
        , ordering_id(0)
        , vertices()
        , residual()
        , jacobians()
        , information()
        , observation()
        , robust_loss() {
    }

    edge::edge(const edge& other)
        : storage{}
        , functions(other.functions)
        , ordering_id(other.ordering_id)
        , vertices(other.vertices)
        , residual(other.residual)
        , jacobians(other.jacobians)
        , information(other.information)
        , observation(other.observation)
        , robust_loss(other.robust_loss) {
        if (this->functions != nullptr) {
            this->functions->copy(other.storage, this->storage);
        }
    }

    edge::edge(edge&& other)
        : edge(static_cast<const edge&>(other)) {
    }

    edge& edge::operator=(const edge& other) {
        if (this != &other) {
            this->clear();
            this->functions = other.functions;
            if (this->functions != nullptr) {
                this->functions->copy(other.storage, this->storage);
            }
            this->ordering_id = other.ordering_id;
            this->vertices = other.vertices;
            this->residual = other.residual;
            this->jacobians = other.jacobians;
            this->information = other.information;
            this->observation = other.observation;
            this->robust_loss = other.robust_loss;
        }
        return *this;
    }

    edge& edge::operator=(edge&& other) {
        return *this = static_cast<const edge&>(other);
    }

    edge::~edge() {
        this->clear();
    }

    void edge::clear() {
        if (this->functions != nullptr) {
            this->functions->destroy(this->storage);
            this->functions = nullptr;
        }
    }

    bool edge::is_valid() const {
        return this->functions != nullptr;
    }

    const char* edge::name() const {
        return (this->functions != nullptr) ? this->functions->name : "";
    }

    int edge::get_ordering_id() const {
        return this->ordering_id;
    }

    void edge::set_ordering_id(int id) {
        this->ordering_id = id;
    }

    size_t edge::num_vertices() const {
        return this->vertices.size();
    }

    bool edge::add_vertex(vertex* node) {
        if (this->vertices.size() >= this->jacobians.size()) {
            return false;
        }
        this->vertices.push_back(node);
        return true;
    }

    const std::vector<vertex*>& edge::get_vertices() const {
        return this->vertices;
    }

    bool edge::set_vertices(const std::vector<vertex*>& vertices_value) {
        if (vertices_value.size() > this->jacobians.size()) {
            return false;
        }
        this->vertices = vertices_value;
        return true;
    }

    vertex* edge::get_vertex(int i) {
        return this->vertices[static_cast<size_t>(i)];
    }

    const vertex* edge::get_vertex(int i) const {
        return this->vertices[static_cast<size_t>(i)];
    }

    double edge::chi2() const {
        const size_t n = this->residual.rows();
        double result = 0;
        for (size_t x = 0; x < n; ++x) {
            double a_x = 0;
            for (size_t k = 0; k < n; ++k) {
                a_x += this->residual[k][0] * this->information[k][x];
            }
            result += a_x * this->residual[x][0];
        }
        return result;
    }

    double edge::robust_chi2() const {
        math::matrix<double, 3, 1> rho = math::matrix<double, 3, 1>::zero();
        this->robust_loss.compute(this->chi2(), rho);
        return rho[0];
    }

    const math::matrix<double, 0, 0>& edge::get_residual() const {
        return this->residual;
    }

    const std::vector<math::matrix<double, 0, 0>>& edge::get_jacobians() const {
        return this->jacobians;
    }

    const math::matrix<double, 0, 0>& edge::get_information() const {
        return this->information;
    }

    void edge::set_information(const math::matrix<double, 0, 0>& information_value) {
        this->information = information_value;
    }

    const loss& edge::get_loss() const {
        return this->robust_loss;
    }

    void edge::set_loss(const loss& robust_loss_value) {
        this->robust_loss = robust_loss_value;
    }

    void edge::robust_info(double& rho_delta, math::matrix<double, 0, 0>& robust_information, bool apply_triggs_correction) const {
        const double error_squared = this->chi2();
        math::matrix<double, 3, 1> rho;
        this->robust_loss.compute(error_squared, rho);
        rho_delta = rho[1];
        robust_information = rho[1] * this->information;
        if (apply_triggs_correction && ((rho[1] + 2.0 * rho[2] * error_squared) > 0.0)) {
            math::matrix<double, 0, 0> weight_error = this->information * this->residual;
            robust_information = robust_information + 2.0 * rho[2] * weight_error * math::transpose(weight_error);
        }
    }

    double edge::robust_weight() const {
        math::matrix<double, 3, 1> rho;
        this->robust_loss.compute(this->chi2(), rho);
        return rho[1];
    }

    const math::matrix<double, 0, 0>& edge::get_observation() const {
        return this->observation;
    }

    void edge::set_observation(const math::matrix<double, 0, 0>& observation_value) {
        this->observation = observation_value;
    }

    void edge::compute_residual() {
        if (this->functions != nullptr) {
            this->functions->compute_residual(this->storage, *this, this->residual);
        }
    }

    void edge::compute_jacobians() {
        if (this->functions == nullptr) {
            return;
        }
        if (this->functions->compute_jacobians != nullptr) {
            this->functions->compute_jacobians(this->storage, *this, this->jacobians);
            return;
        }
        const double delta = 1e-5;
        const double scalar = 1.0 / (2.0 * delta);

        const math::matrix<double, 0, 0> backup_residual = this->residual;

        for (size_t i = 0; i < this->vertices.size(); ++i) {
            const size_t vertex_dimensions = static_cast<size_t>(this->vertices[i]->get_local_dimensions());
            this->jacobians[i] = math::matrix<double, 0, 0>::zero(this->residual.rows(), vertex_dimensions);

            if (this->vertices[i]->is_fixed()) {
                continue;
            }

            const size_t parameter_dimensions = static_cast<size_t>(this->vertices[i]->get_parameter_dimensions());
            double backup[vertex::maximum_parameters] = {};
            const bool parameters_backed_up = this->vertices[i]->get_parameters(&backup[0], parameter_dimensions);
            ASSERT(parameters_backed_up, "A vertex must report the size of its own parameters.");
            static_cast<void>(parameters_backed_up);

            for (size_t d = 0; d < vertex_dimensions; ++d) {
                double add_delta[vertex::maximum_parameters] = {};

                add_delta[d] = delta;
                this->vertices[i]->plus(&add_delta[0]);
                this->functions->compute_residual(this->storage, *this, this->residual);
                const math::matrix<double, 0, 0> error_addition = this->residual;

                static_cast<void>(this->vertices[i]->set_parameters(&backup[0], parameter_dimensions));

                add_delta[d] = -delta;
                this->vertices[i]->plus(&add_delta[0]);
                this->functions->compute_residual(this->storage, *this, this->residual);
                const math::matrix<double, 0, 0> error_subtract = this->residual;

                static_cast<void>(this->vertices[i]->set_parameters(&backup[0], parameter_dimensions));

                for (size_t r = 0; r < this->residual.rows(); ++r) {
                    this->jacobians[i][r][d] = scalar * (error_addition[r][0] - error_subtract[r][0]);
                }
            }
        }

        this->residual = backup_residual;
    }
}
