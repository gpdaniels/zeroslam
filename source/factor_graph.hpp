/*
Copyright (C) 2025 Geoffrey Daniels. https://gpdaniels.com/

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
#ifndef FACTOR_GRAPH_HPP
#define FACTOR_GRAPH_HPP

#include "assert.hpp"
#include "lie.hpp"
#include "matrix.hpp"
#include "matrix_decomposition_cholesky.hpp"
#include "matrix_sparse_block.hpp"
#include "matrix_sparse_block_diagonal.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace factor_graph {

    // Base class for robust loss functions.
    class loss_function_base {
    public:
        virtual ~loss_function_base() = default;

    protected:
        loss_function_base() = default;

    public:
        // Compute rho vector for squared error.
        // rho[0] = robust function value.
        // rho[1] = first derivative.
        // rho[2] = second derivative.
        virtual void compute(double err2, matrix::matrix<double, 3, 1>& rho) const = 0;
    };

    // Base class for optimisation vertices.
    class vertex_base {
    protected:
        // Ordering id in the stacked variable vector.
        int ordering_id = 0;
        // Local dimension count for manifold optimisation.
        int local_dimension;
        // Flag indicating vertex is fixed.
        bool fixed = false;
        // Flag indicating vertex is marginalised.
        bool marginalised = false;
        // Parameters are stored as a column vector.
        matrix::matrix<double, 0, 0> parameters;
        // Backup used for rollback.
        matrix::matrix<double, 0, 0> parameters_backup;

    public:
        virtual ~vertex_base() = default;

    public:
        vertex_base(const vertex_base&) = default;
        vertex_base(vertex_base&&) = default;
        vertex_base& operator=(const vertex_base&) = default;
        vertex_base& operator=(vertex_base&&) = default;

    protected:
        explicit vertex_base(int parameter_dimension_value) {
            this->parameters = matrix::matrix<double, 0, 0>::zero(parameter_dimension_value, 1);
            this->local_dimension = parameter_dimension_value;
        }

        explicit vertex_base(int parameter_dimension_value, int local_dimension_value) {
            this->parameters = matrix::matrix<double, 0, 0>::zero(parameter_dimension_value, 1);
            this->local_dimension = local_dimension_value;
        }

    public:
        void set_ordering_id(int id) {
            this->ordering_id = id;
        };

        int get_ordering_id() const {
            return static_cast<int>(this->ordering_id);
        }

        int get_local_dimensions() const {
            return this->local_dimension;
        }

        void set_fixed(bool fixed_value = true) {
            this->fixed = fixed_value;
        }

        bool is_fixed() const {
            return this->fixed;
        }

        void set_marginalised(bool marginalised_value = true) {
            this->marginalised = marginalised_value;
        }

        bool is_marginalised() const {
            return this->marginalised;
        }

        const matrix::matrix<double, 0, 0>& get_parameters() const {
            return this->parameters;
        }

        void set_parameters(const matrix::matrix<double, 0, 0>& parameters_value) {
            this->parameters = parameters_value;
        }

        void backup() {
            this->parameters_backup = this->parameters;
        }

        void restore() {
            this->parameters = this->parameters_backup;
        }

    public:
        // Default additive update for dense parameter vectors.
        virtual void plus(const matrix::matrix<double, 0, 0>& delta) {
            for (size_t i = 0; i < this->parameters.rows(); ++i) {
                this->parameters[i][0] += delta[i][0];
            }
        }
    };

    // Base class for edges (factors).
    class edge_base {
    protected:
        // Ordering id for solver.
        int ordering_id;
        // Connected vertices for this edge.
        std::vector<vertex_base*> vertices;
        // Residual vector for the factor.
        matrix::matrix<double, 0, 0> residual;
        // Jacobians for each connected vertex.
        std::vector<matrix::matrix<double, 0, 0>> jacobians;
        // Information matrix of the measurement.
        matrix::matrix<double, 0, 0> information;
        // Observation vector.
        matrix::matrix<double, 0, 0> observation;
        // Optional robust loss function.
        loss_function_base* loss_function;

    public:
        virtual ~edge_base() = default;

    public:
        edge_base(const edge_base&) = default;
        edge_base(edge_base&&) = default;
        edge_base& operator=(const edge_base&) = default;
        edge_base& operator=(edge_base&&) = default;

    protected:
        // Construct an edge with residual dimension and number of vertices.
        explicit edge_base(int residual_dimension_value, int num_vertices) {
            this->vertices.reserve(num_vertices);
            this->residual = matrix::matrix<double, 0, 0>::zero(residual_dimension_value, 1);
            this->jacobians.resize(num_vertices);
            this->information = matrix::matrix<double, 0, 0>::identity(residual_dimension_value, residual_dimension_value);
            this->loss_function = nullptr;
        }

    public:
        int get_ordering_id() const {
            return this->ordering_id;
        }

        void set_ordering_id(int id) {
            this->ordering_id = id;
        }

        size_t num_vertices() const {
            return this->vertices.size();
        }

        bool add_vertex(vertex_base* vertex) {
            this->vertices.push_back(vertex);
            ASSERT(this->vertices.size() <= this->jacobians.size(), "Attempting to add too many vertices to an edge.");
            return true;
        }

        const std::vector<vertex_base*>& get_vertices() const {
            return this->vertices;
        }

        bool set_vertices(const std::vector<vertex_base*>& vertices_value) {
            this->vertices = vertices_value;
            return true;
        }

        vertex_base* get_vertex(int i) {
            return this->vertices[i];
        }

        double chi2() const {
            return (matrix::transpose(this->residual) * this->information * this->residual)[0][0];
        }

        double robust_chi2() const {
            if (!this->loss_function) {
                return this->chi2();
            }
            matrix::matrix<double, 3, 1> rho = matrix::matrix<double, 3, 1>::zero();
            this->loss_function->compute(this->chi2(), rho);
            return rho[0];
        }

        const matrix::matrix<double, 0, 0>& get_residual() const {
            return this->residual;
        }

        const std::vector<matrix::matrix<double, 0, 0>>& get_jacobians() const {
            return this->jacobians;
        }

        const matrix::matrix<double, 0, 0>& get_information() const {
            return this->information;
        }

        void set_information(const matrix::matrix<double, 0, 0>& information_value) {
            this->information = information_value;
        }

        loss_function_base* get_loss_function() {
            return this->loss_function;
        }

        void set_loss_function(loss_function_base* loss_function_value) {
            this->loss_function = loss_function_value;
        }

        void robust_info(double& rho_delta, matrix::matrix<double, 0, 0>& robust_information) const {
            if (!this->loss_function) {
                rho_delta = 1.0;
                robust_information = this->information;
                return;
            }
            const double error_squared = this->chi2();
            matrix::matrix<double, 3, 1> rho;
            this->loss_function->compute(error_squared, rho);
            rho_delta = rho[1];
            robust_information = rho[1] * this->information;
            if ((rho[1] + 2.0 * rho[2] * error_squared) > 0.0) {
                matrix::matrix<double, 0, 0> weight_error = this->information * this->residual;
                robust_information = robust_information + 2.0 * rho[2] * weight_error * matrix::transpose(weight_error);
            }
        }

        const matrix::matrix<double, 0, 0>& get_observation() const {
            return this->observation;
        }

        void set_observation(const matrix::matrix<double, 0, 0>& observation_value) {
            this->observation = observation_value;
        }

    public:
        virtual void compute_residual() = 0;

        virtual void compute_jacobians() {
            // Estimate the jacobian numerically by symmetric difference.
            const double delta = 1e-9;
            const double scalar = 1.0 / (2.0 * delta);

            // Save current residual to restore later.
            matrix::matrix<double, 0, 0> backup_residual = this->residual;

            for (size_t i = 0; i < this->vertices.size(); ++i) {
                const int vertex_dimensions = this->vertices[i]->get_local_dimensions();
                // Initialise jacobian block for this vertex to the correct size.
                // The residual dimension is known from the residual vector.
                this->jacobians[i] = matrix::matrix<double, 0, 0>::zero(this->residual.rows(), vertex_dimensions);

                if (this->vertices[i]->is_fixed()) {
                    continue;
                }

                // For each dimension of the vertex apply a small step and measure residual change.
                for (int d = 0; d < vertex_dimensions; ++d) {
                    // Save parameters for this vertex.
                    matrix::matrix<double, 0, 0> backup = this->vertices[i]->get_parameters();

                    // Prepare delta vector for additive perturbation.
                    matrix::matrix<double, 0, 0> add_delta = matrix::matrix<double, 0, 0>::zero(vertex_dimensions, 1);

                    // Forward perturbation.
                    add_delta[d][0] = delta;
                    this->vertices[i]->plus(add_delta);
                    this->compute_residual();
                    matrix::matrix<double, 0, 0> error_addition = this->residual;

                    // Restore parameters to original.
                    this->vertices[i]->set_parameters(backup);

                    // Backward perturbation.
                    add_delta[d][0] = -delta;
                    this->vertices[i]->plus(add_delta);
                    this->compute_residual();
                    matrix::matrix<double, 0, 0> error_subtract = this->residual;

                    // Restore parameters again.
                    this->vertices[i]->set_parameters(backup);

                    // Numerical derivative column.
                    const matrix::matrix<double, 0, 0> column = scalar * (error_addition - error_subtract);
                    for (size_t r = 0; r < this->residual.rows(); ++r) {
                        this->jacobians[i][static_cast<size_t>(r)][static_cast<size_t>(d)] = column[r][0];
                    }
                }
            }

            // Restore original residual.
            this->residual = backup_residual;
        }
    };

    // Factor graph solver.
    class factor_graph final {
    private:
        // Verbose logging flag.
        bool verbose;
        // Damping lambda used by Levenberg-Marquardt.
        double damping_lambda; // damping parameter.
        // Damping scale factor used when reducing/increasing lambda.
        double damping_factor; // damping parameter factor.
        // Current chi-squared cost.
        double chi_squared;

        // Vertex containers.
        std::vector<vertex_base*> vertices_general;
        std::vector<vertex_base*> vertices_marginalised;
        // Edge container.
        std::vector<edge_base*> edges;
        // Mapping from vertex to connected edges for quick lookup.
        std::unordered_multimap<vertex_base*, edge_base*> vertex_to_edge;

    private:
        // Matrix shape for Schur complement partitioning.
        //   [ h_pp h_pl ]
        //   [ h_lp h_ll ]
        matrix::sparse_block_diagonal<6> h_pp;         // pose-pose block diagonal.
        matrix::sparse_block_diagonal<3> h_ll;         // landmark-landmark block diagonal.
        matrix::sparse_block<6, 3> h_pl;               // pose-landmark off-diagonal blocks.
        matrix::sparse_block<3, 6> h_lp;               // landmark-pose off-diagonal blocks.
        matrix::sparse_block_diagonal<3> h_ll_inverse; // inverse of landmark-landmark block diagonal.
        matrix::matrix<double, 0, 0> vector_b;         // full right-hand side vector.
        matrix::matrix<double, 0, 0> delta_x;          // full solution vector.
        matrix::matrix<double, 0, 0> b_pp;             // pose part of vector_b.
        matrix::matrix<double, 0, 0> b_ll;             // landmark part of vector_b.

        int count_general_params = 0;
        int count_marginalised_params = 0;

    public:
        // Construct solver with optional verbose flag.
        factor_graph(bool verbose_in = false)
            : verbose(verbose_in) {
        }

    public:
        // Add a vertex to the graph.
        bool add_vertex(vertex_base* vertex) {
            for (vertex_base* v : this->vertices_general) {
                if (v == vertex)
                    return false;
            }
            for (vertex_base* v : this->vertices_marginalised) {
                if (v == vertex)
                    return false;
            }
            // Put marginalised vertices into separate container.
            if (vertex->is_marginalised()) {
                this->vertices_marginalised.push_back(vertex);
            }
            else {
                this->vertices_general.push_back(vertex);
            }
            return true;
        }

        // Remove a vertex and its connected edges from the graph.
        bool remove_vertex(vertex_base* vertex) {
            int found_index = -1;
            // Search general vertices first.
            for (int i = 0; i < static_cast<int>(this->vertices_general.size()); ++i) {
                if (this->vertices_general[i] == vertex) {
                    found_index = i;
                    break;
                }
            }
            // If not found, search marginalised vertices.
            if (found_index == -1) {
                for (int i = 0; i < static_cast<int>(this->vertices_marginalised.size()); ++i) {
                    if (this->vertices_marginalised[i] == vertex) {
                        found_index = i;
                        break;
                    }
                }
            }
            if (found_index == -1) {
                return false;
            }
            // Remove all connected edges first.
            std::vector<edge_base*> remove_edges = this->get_connected_edges(vertex);
            for (size_t i = 0; i < remove_edges.size(); i++) {
                this->remove_edge(remove_edges[i]);
            }
            // Erase the vertex from appropriate container.
            if (vertex->is_marginalised()) {
                this->vertices_marginalised.erase(this->vertices_marginalised.begin() + found_index);
            }
            else {
                this->vertices_general.erase(this->vertices_general.begin() + found_index);
            }
            vertex->set_ordering_id(-1);
            this->vertex_to_edge.erase(vertex);
            return true;
        }

        // Add an edge to the graph.
        bool add_edge(edge_base* edge) {
            for (edge_base* e : this->edges) {
                if (e == edge)
                    return false;
            }
            this->edges.push_back(edge);
            for (vertex_base* vertex : edge->get_vertices()) {
                this->vertex_to_edge.insert({ vertex, edge });
            }
            return true;
        }

        // Remove an edge from the graph.
        bool remove_edge(edge_base* edge) {
            int found_index = -1;
            for (int i = 0; i < static_cast<int>(this->edges.size()); ++i) {
                if (this->edges[i] == edge) {
                    found_index = i;
                    break;
                }
            }
            if (found_index == -1) {
                return false;
            }
            this->edges.erase(this->edges.begin() + found_index);
            return true;
        }

        // Get edges connected to a vertex using the vertex_to_edge map.
        std::vector<edge_base*> get_connected_edges(vertex_base* vertex) const {
            std::vector<edge_base*> edges_connected;
            auto range = this->vertex_to_edge.equal_range(vertex);
            for (auto iter = range.first; iter != range.second; ++iter) {
                for (edge_base* e : this->edges) {
                    if (e == iter->second) {
                        edges_connected.push_back(iter->second);
                        break;
                    }
                }
            }
            return edges_connected;
        }

    public:
        // Solve the factor graph using Levenberg-Marquardt with Schur complement.
        int solve(int iterations = 10) {
            if (this->verbose) {
                std::fflush(stdout);
            }

            if ((this->edges.size() == 0) || ((this->vertices_general.size() == 0) && (this->vertices_marginalised.size() == 0))) {
                if (this->verbose) {
                    std::fprintf(stderr, "Cannot solve problem without edges or vertices\n");
                }
                return 0;
            }

            // Compute initial residuals for all edges.
            for (edge_base* edge : this->edges) {
                edge->compute_residual();
            }

            this->set_ordering();

            this->make_hessian();

            // Initialise damping parameters.
            this->damping_factor = 2.0;
            this->chi_squared = this->get_current_chi();
            double max_diagonal = 0;
            for (int i = 0; i < this->count_general_params; ++i) {
                max_diagonal = std::max(math::abs(this->h_pp(i, i)), max_diagonal);
            }
            for (int i = 0; i < this->count_marginalised_params; ++i) {
                max_diagonal = std::max(math::abs(this->h_ll(i, i)), max_diagonal);
            }
            max_diagonal = std::min(5e10, max_diagonal);
            const double tau = 1e-5;
            this->damping_lambda = tau * max_diagonal;

            // Log initial state.
            if (this->verbose) {
                std::fprintf(stderr, "[INIT] iter: XXX, attempt XXX, chi = % 7.7f, rho = XXXX.XXXXXXX, base lambda = % 8.7f\n", this->chi_squared, this->damping_lambda);
            }

            // Levenberg-Marquardt optimisation loop.
            const int max_failures = 10;
            const double stop_threshold = 1e-10 * this->chi_squared;
            int success_count = 0;
            for (int iter = 0; iter < iterations; ++iter) {
                double last_chi_squared = this->chi_squared;
                int failure_count = 0;
                while (failure_count < max_failures) {
                    this->solve_linear_system();

                    this->update_states();

                    // Compute scale used in rho calculation.
                    double scale = 0;
                    for (int i = 0; i < (this->count_general_params + this->count_marginalised_params); ++i) {
                        scale += this->delta_x[i][0] * (this->damping_lambda * this->delta_x[i][0] + this->vector_b[i][0]);
                    }
                    scale += 1e-3;

                    // Recompute residuals and evaluate cost after the step.
                    for (edge_base* edge : this->edges) {
                        edge->compute_residual();
                    }
                    double tempChi = this->get_current_chi();
                    double rho = (this->chi_squared - tempChi) / scale;

                    // Determine if the step is successful.
                    bool good_step = false;
                    if ((rho > 0) && (math::isfinite(tempChi))) {
                        double alpha = 1.0 - math::pow((2.0 * rho - 1.0), 3.0);
                        alpha = math::min(alpha, 2.0 / 3.0);
                        double scaleFactor = math::max(1.0 / 3.0, alpha);
                        this->damping_lambda *= scaleFactor;
                        this->damping_factor = 2.0;
                        this->chi_squared = tempChi;
                        good_step = true;
                        ++success_count;
                        if (this->verbose) {
                            std::fprintf(stderr, "[GOOD] iter: % 3d, attempt % 3d, chi2 = % 7.7f, rho = % 7.7f, next lambda = % 8.7f\n", iter, failure_count, tempChi, rho, this->damping_lambda);
                        }
                    }
                    else {
                        this->damping_lambda *= this->damping_factor;
                        this->damping_factor *= 2.0;
                        good_step = false;
                        if (this->verbose) {
                            std::fprintf(stderr, "[ BAD] iter: % 3d, attempt % 3d, chi2 = % 7.7f, rho = % 7.7f, next lambda = % 8.7f\n", iter, failure_count, tempChi, rho, this->damping_lambda);
                        }
                    }
                    if (!math::isfinite(this->damping_lambda)) {
                        break;
                    }

                    if (good_step) {
                        // Recompute Hessian at new linearisation point.
                        this->make_hessian();
                        break;
                    }
                    else {
                        // Rollback parameters to previous state and try increasing lambda.
                        this->rollback_states();
                        ++failure_count;
                    }
                }
                // Early exit criteria for optimisation.
                if ((failure_count >= max_failures) || ((last_chi_squared - this->chi_squared) < stop_threshold) || (!math::isfinite(this->damping_lambda))) {
                    break;
                }
            }

            return success_count;
        }

        // Compute current chi-squared, optionally recomputing residuals first.
        double get_current_chi(bool recompute_residuals = true) {
            if (recompute_residuals) {
                for (edge_base* edge : this->edges) {
                    edge->compute_residual();
                }
            }
            double current_chi = 0.0;
            for (const edge_base* edge : this->edges) {
                current_chi += edge->robust_chi2();
            }
            return current_chi;
        }

    private:
        // Assign ordering ids for vertices and count total parameter sizes.
        void set_ordering() {
            this->count_general_params = 0;
            this->count_marginalised_params = 0;
            for (const auto& vertex : this->vertices_general) {
                vertex->set_ordering_id(this->count_general_params);
                this->count_general_params += vertex->get_local_dimensions();
            }
            for (const auto& vertex : this->vertices_marginalised) {
                // Order marginalised variables after the general variables.
                vertex->set_ordering_id(this->count_marginalised_params + this->count_general_params);
                this->count_marginalised_params += vertex->get_local_dimensions();
            }
        }

        // Build Hessian blocks and right-hand side vector.
        void make_hessian() {
            // Clear existing block structures and resize to match current variables.
            this->h_pp.diagonal().clear();
            this->h_pp.diagonal().resize(this->count_general_params / 6, matrix::matrix<double, 6, 6>::zero());
            this->h_ll.diagonal().clear();
            this->h_ll.diagonal().resize(this->count_marginalised_params / 3, matrix::matrix<double, 3, 3>::zero());
            this->h_pl.blocks().clear();
            this->h_pl.set_size(this->count_general_params, this->count_marginalised_params);
            this->h_lp.blocks().clear();
            this->h_lp.set_size(this->count_marginalised_params, this->count_general_params);
            this->h_ll_inverse.diagonal().clear();
            this->h_ll_inverse.diagonal().resize(this->count_marginalised_params / 3, matrix::matrix<double, 3, 3>::zero());
            this->vector_b = matrix::matrix<double, 0, 0>::zero(this->count_general_params + this->count_marginalised_params, 1);
            this->delta_x = matrix::matrix<double, 0, 0>::zero(this->count_general_params + this->count_marginalised_params, 1);

            // Accumulate Hessian and vector terms from all edges.
            for (const auto& edge : this->edges) {
                edge->compute_jacobians();
                const std::vector<vertex_base*>& edge_vertices = edge->get_vertices();
                const std::vector<matrix::matrix<double, 0, 0>>& jacobians = edge->get_jacobians();
                ASSERT(edge_vertices.size() == jacobians.size(), "Mismatching sizes between edge vertices and edge jacobians.");
                for (size_t i = 0; i < edge_vertices.size(); ++i) {
                    const vertex_base* v_i = edge_vertices[i];
                    if (v_i->is_fixed()) {
                        continue;
                    }
                    const matrix::matrix<double, 0, 0>& jacobian_i = jacobians[i];
                    const int index_i = v_i->get_ordering_id();
                    const int dim_i = v_i->get_local_dimensions();
                    double drho;
                    matrix::matrix<double, 0, 0> robust_information(edge->get_information().rows(), edge->get_information().cols());
                    edge->robust_info(drho, robust_information);
                    const matrix::matrix<double, 0, 0> jtw = matrix::transpose(jacobian_i) * robust_information;
                    for (size_t j = i; j < edge_vertices.size(); ++j) {
                        const vertex_base* v_j = edge_vertices[j];
                        if (v_j->is_fixed()) {
                            continue;
                        }
                        const matrix::matrix<double, 0, 0>& jacobian_j = jacobians[j];
                        const int index_j = v_j->get_ordering_id();
                        ASSERT(v_j->get_ordering_id() != -1, "Missing ordering id value of edge vertex.");
                        const matrix::matrix<double, 0, 0> hessian_block = jtw * jacobian_j;
                        // Four cases depending on marginalisation state of v_i and v_j.
                        if (v_i->is_marginalised() == false) {
                            if (v_j->is_marginalised() == false) {
                                // pose-pose block.
                                ASSERT(index_i == index_j, "Attempting to add a pose-pose block that is not on the block diagonal.");
                                ASSERT(hessian_block.cols() == 6 && hessian_block.rows() == 6, "Currently only 6 wide by 6 tall pose-pose blocks are supported.");
                                this->h_pp.diagonal()[index_i / 6] = this->h_pp.diagonal()[index_i / 6] + matrix::matrix<double, 6, 6>(hessian_block.data());
                            }
                            else {
                                // pose-landmark block.
                                ASSERT(index_i != index_j, "Attempting to add a pose-landmark block that is on the block diagonal.");
                                ASSERT(hessian_block.cols() == 3 && hessian_block.rows() == 6, "Currently only 3 wide by 6 tall pose-landmark blocks are supported.");
                                const size_t block_row = index_i / 6;
                                const size_t block_col = (index_j - this->count_general_params) / 3;
                                if (this->h_pl.blocks().find({ block_row, block_col }) == this->h_pl.blocks().end()) {
                                    this->h_pl.blocks()[{ block_row, block_col }] = matrix::matrix<double, 6, 3>(hessian_block.data());
                                }
                                else {
                                    this->h_pl.blocks()[{ block_row, block_col }] = this->h_pl.blocks()[{ block_row, block_col }] + matrix::matrix<double, 6, 3>(hessian_block.data());
                                }
                            }
                        }
                        else {
                            if (v_j->is_marginalised() == false) {
                                ASSERT(false, "This code path is unexpected for current edge vertex ordering conventions.");
                            }
                            else {
                                // landmark-landmark block.
                                ASSERT(index_i == index_j, "Attempting to add a landmark-landmark block that is not on the block diagonal.");
                                ASSERT(hessian_block.cols() == 3 && hessian_block.rows() == 3, "Currently only 3 wide by 6 tall landmark-landmark blocks are supported.");
                                this->h_ll.diagonal()[(index_j - this->count_general_params) / 3] = this->h_ll.diagonal()[(index_j - this->count_general_params) / 3] + matrix::matrix<double, 3, 3>(hessian_block.data());
                            }
                        }
                    }
                    // Update right-hand side vector B.
                    matrix::set_block(this->vector_b, index_i, 0, matrix::get_block(this->vector_b, index_i, 0, dim_i, 1) - drho * matrix::transpose(jacobian_i) * edge->get_information() * edge->get_residual());
                }
            }
            // Compute transposed off-diagonal blocks.
            this->h_lp = this->h_pl.get_transpose();
            this->b_pp = matrix::get_block(this->vector_b, 0, 0, this->count_general_params, 1);
            this->b_ll = matrix::get_block(this->vector_b, this->count_general_params, 0, this->count_marginalised_params, 1);
        }

        // Solve the linear system using Schur complement and a dense Cholesky for the reduced system.
        void solve_linear_system() {
            // Backup diagonal blocks in case we need to restore.
            const matrix::sparse_block_diagonal<6> h_pp_backup = this->h_pp;
            const matrix::sparse_block_diagonal<3> h_ll_backup = this->h_ll;

            // Apply current damping lambda to diagonal elements of Hpp and Hll.
            for (size_t d = 0; d < this->h_pp.diagonal().size(); ++d) {
                for (size_t i = 0; i < 6; ++i) {
                    this->h_pp.diagonal()[d][i][i] += this->damping_lambda;
                }
            }
            for (size_t d = 0; d < this->h_ll.diagonal().size(); ++d) {
                for (size_t i = 0; i < 3; ++i) {
                    this->h_ll.diagonal()[d][i][i] += this->damping_lambda;
                }
            }

            // Invert each marginalised block diagonal (Hll + lambda).
            for (const auto& landmark_vertex : this->vertices_marginalised) {
                if (landmark_vertex->is_fixed()) {
                    continue;
                }
                const int idx = landmark_vertex->get_ordering_id() - this->count_general_params;
                const int size = landmark_vertex->get_local_dimensions();
                ASSERT(size == 3, "Currently only landmarks of size 3 are supported.");
                ASSERT(idx % 3 == 0, "Landmark indexes must be aligned in sets of 3.");
                this->h_ll_inverse.diagonal()[idx / size] = matrix::invert(this->h_ll.diagonal()[idx / size]);
            }

            // Step 1: Schur complement setup.
            const matrix::sparse_block<6, 3> h_p_m_times_hll_inv = matrix::multiply(this->h_pl, this->h_ll_inverse);
            const matrix::matrix<double, 0, 0> b_pp_schur = this->b_pp - h_p_m_times_hll_inv.multiply(this->b_ll);
            const matrix::sparse_block<6, 6> h_pp_schur = matrix::subtract(this->h_pp, h_p_m_times_hll_inv.multiply(this->h_lp));

            // Step 2: solve reduced pose system Hpp_schur * delta_xpp = b_pp_schur.
            // Assemble dense matrix for Cholesky since a sparse block Cholesky is not provided here.
            matrix::matrix<double, 0, 0> hpp_schur_dense = matrix::matrix<double, 0, 0>::zero(h_pp_schur.rows(), h_pp_schur.cols());
            for (std::unordered_map<typename matrix::sparse_block<6, 6>::block_key<size_t, size_t>, typename matrix::sparse_block<6, 6>::block_type>::const_iterator iterator = h_pp_schur.blocks().begin(); iterator != h_pp_schur.blocks().end(); ++iterator) {
                for (int i = 0; i < 6; ++i) {
                    for (int j = 0; j < 6; ++j) {
                        hpp_schur_dense[iterator->first.first * 6 + i][iterator->first.second * 6 + j] = hpp_schur_dense[iterator->first.first * 6 + i][iterator->first.second * 6 + j] + iterator->second[i][j];
                    }
                }
            }
            matrix::matrix<double, 0, 0> matrix_lower(this->count_general_params, this->count_general_params);
            matrix::decompose_cholesky(hpp_schur_dense.data(), this->count_general_params, this->count_general_params, matrix_lower.data());
            matrix::matrix<double, 0, 0> delta_xpp = matrix::matrix<double, 0, 0>::zero(this->count_general_params, 1);
            if (!matrix::solve_cholesky(matrix_lower.data(), b_pp_schur.data(), this->count_general_params, this->count_general_params, delta_xpp.data())) {
                if (this->verbose) {
                    std::fprintf(stderr, "Cholesky solver failed!\n");
                }
                // Restore Hessian if solver fails.
                this->h_pp = h_pp_backup;
                this->h_ll = h_ll_backup;
                return;
            }

            // Place pose solution into full delta vector.
            matrix::set_block(this->delta_x, 0, 0, delta_xpp);

            // Step 3: back-substitute to compute marginalised variable updates.
            matrix::matrix<double, 0, 0> delta_xll = this->h_ll_inverse.multiply(this->b_ll - this->h_lp.multiply(delta_xpp));
            matrix::set_block(this->delta_x, this->count_general_params, 0, delta_xll);

            // Restore original Hessian blocks.
            this->h_pp = h_pp_backup;
            this->h_ll = h_ll_backup;
        }

        // Apply the computed delta_x to vertex states, backing up parameters first.
        void update_states() {
            for (auto vertex : this->vertices_general) {
                if (vertex->is_fixed()) {
                    continue;
                }
                vertex->backup();
                int idx = vertex->get_ordering_id();
                int dim = vertex->get_local_dimensions();
                matrix::matrix<double, 0, 0> delta = matrix::get_block(this->delta_x, idx, 0, dim, 1);
                vertex->plus(delta);
            }
            for (auto vertex : this->vertices_marginalised) {
                if (vertex->is_fixed()) {
                    continue;
                }
                vertex->backup();
                int idx = vertex->get_ordering_id();
                int dim = vertex->get_local_dimensions();
                matrix::matrix<double, 0, 0> delta = matrix::get_block(this->delta_x, idx, 0, dim, 1);
                vertex->plus(delta);
            }
        }

        // Roll back all vertex parameters to their backed up values.
        void rollback_states() {
            for (auto vertex : this->vertices_general) {
                if (vertex->is_fixed()) {
                    continue;
                }
                vertex->restore();
            }
            for (auto vertex : this->vertices_marginalised) {
                if (vertex->is_fixed()) {
                    continue;
                }
                vertex->restore();
            }
        }
    };
}

namespace factor_graph {
    class loss_trivial final
        : public loss_function_base {
    public:
        virtual void compute(double error_squared, matrix::matrix<double, 3, 1>& rho) const override final {
            rho[0] = error_squared;
            rho[1] = 1;
            rho[2] = 0;
        }
    };

    class loss_huber final
        : public loss_function_base {
    private:
        double delta;

    public:
        explicit loss_huber(double delta_value)
            : delta(delta_value) {
        }

        virtual void compute(double error_squared, matrix::matrix<double, 3, 1>& rho) const override final {
            const double delta_squared = this->delta * this->delta;
            if (error_squared <= delta_squared) {
                rho[0] = error_squared;
                rho[1] = 1.0;
                rho[2] = 0.0;
            }
            else {
                const double error = math::sqrt(error_squared);
                rho[0] = 2.0 * error * this->delta - delta_squared;
                rho[1] = this->delta / error;
                rho[2] = -0.5 * rho[1] / error_squared;
            }
        }
    };

    class loss_cauchy final
        : public loss_function_base {
    private:
        double delta;

    public:
        explicit loss_cauchy(double delta_value)
            : delta(delta_value) {
        }

        virtual void compute(double error_squared, matrix::matrix<double, 3, 1>& rho) const override final {
            const double delta_squared = this->delta * this->delta;
            const double delta_squared_reciprocal = 1.0 / delta_squared;
            const double aux = delta_squared_reciprocal * error_squared + 1.0;
            rho[0] = delta_squared * math::log(aux);
            rho[1] = 1.0 / aux;
            rho[2] = -delta_squared_reciprocal * math::sqr(rho[1]);
        }
    };

    class loss_tukey final
        : public loss_function_base {
    private:
        double delta;

    public:
        explicit loss_tukey(double delta_value)
            : delta(delta_value) {
        }

        virtual void compute(double error_squared, matrix::matrix<double, 3, 1>& rho) const override final {
            const double delta_squared = this->delta * this->delta;
            if (error_squared <= delta_squared) {
                const double aux = error_squared / delta_squared;
                const double one_minus_aux = 1.0 - aux;
                const double one_minus_aux_squared = one_minus_aux * one_minus_aux;
                const double one_minus_aux_cubed = one_minus_aux_squared * one_minus_aux;
                rho[0] = delta_squared * (1.0 - one_minus_aux_cubed) / 3.0;
                rho[1] = one_minus_aux_squared;
                rho[2] = -2.0 * one_minus_aux / delta_squared;
            }
            else {
                rho[0] = delta_squared / 3.0;
                rho[1] = 0;
                rho[2] = 0;
            }
        }
    };
}

namespace factor_graph {
    class vertex_pose final
        : public vertex_base {
    public:
        // Parameters: tx, ty, tz, qx, qy, qz, qw = 7 DoF (optimise on 6D manifold).
        vertex_pose()
            : vertex_base(7, 6) {
        }

        // Update on the manifold using left-multiplicative increment.
        virtual void plus(const matrix::matrix<double, 0, 0>& delta) override final {
            const lie::so3<double> current_rotation(this->parameters[6][0], this->parameters[3][0], this->parameters[4][0], this->parameters[5][0]);
            const matrix::matrix<double, 3, 1> current_translation = { { this->parameters[0][0], this->parameters[1][0], this->parameters[2][0] } };
            const lie::se3<double> current_pose(current_rotation, current_translation);
            const lie::se3<double> update = lie::se3<double>::exp({ { delta[0][0], delta[1][0], delta[2][0], delta[3][0], delta[4][0], delta[5][0] } });
            const lie::se3<double> next_pose = update * current_pose;
            this->parameters[0][0] = next_pose.translation()[0]; // tx.
            this->parameters[1][0] = next_pose.translation()[1]; // ty.
            this->parameters[2][0] = next_pose.translation()[2]; // tz.
            this->parameters[3][0] = next_pose.rotation().get_quaternion()[1]; // qx.
            this->parameters[4][0] = next_pose.rotation().get_quaternion()[2]; // qy.
            this->parameters[5][0] = next_pose.rotation().get_quaternion()[3]; // qz.
            this->parameters[6][0] = next_pose.rotation().get_quaternion()[0]; // qw.
        }
    };

    class vertex_point_xyz final
        : public vertex_base {
    public:
        vertex_point_xyz()
            : vertex_base(3) {
        }
    };
}

namespace factor_graph {
    template <typename camera_type>
    class edge_reprojection final
        : public edge_base {
    private:
        camera_type camera;

    public:
        edge_reprojection(const camera_type& camera_model)
            : edge_base(2, 2)
            , camera(camera_model) {
        }

        // Compute reprojection residual given a pose and a 3D landmark.
        virtual void compute_residual() override final {
            // Pose parameters from first vertex.
            const matrix::matrix<double, 0, 0>& pose_params = this->get_vertex(0)->get_parameters();
            const lie::se3 pose(lie::so3<double>(pose_params[6][0], pose_params[3][0], pose_params[4][0], pose_params[5][0]), { { pose_params[0][0], pose_params[1][0], pose_params[2][0] } });
            // Landmark parameters from second vertex.
            const matrix::matrix<double, 0, 0>& xyz_params = this->get_vertex(1)->get_parameters();
            const matrix::matrix<double, 3, 1> landmark_world({ xyz_params[0][0], xyz_params[1][0], xyz_params[2][0] });
            // Transform landmark into camera frame.
            const matrix::matrix<double, 3, 1> landmark_camera = pose * landmark_world;
            // Project into image plane using intrinsic matrix.
            matrix::matrix<double, 2, 1> projected = {{ 0.0, 0.0 }};
            if (!this->camera.project(landmark_camera.data(), projected.data())) {
                this->residual[0][0] = 0.0;
                this->residual[1][0] = 0.0;
                return;
            }
            this->residual[0][0] = this->get_observation()[0][0] - projected[0];
            this->residual[1][0] = this->get_observation()[1][0] - projected[1];
        }

        // Analytical jacobians for pose and landmark using camera model.
        virtual void compute_jacobians() override final {
            const matrix::matrix<double, 0, 0>& pose_params = this->get_vertex(0)->get_parameters();
            const lie::se3 pose(lie::so3<double>(pose_params[6][0], pose_params[3][0], pose_params[4][0], pose_params[5][0]), { { pose_params[0][0], pose_params[1][0], pose_params[2][0] } });
            const matrix::matrix<double, 0, 0>& xyz_params = this->get_vertex(1)->get_parameters();
            const matrix::matrix<double, 3, 1> landmark_world({ xyz_params[0][0], xyz_params[1][0], xyz_params[2][0] });
            const matrix::matrix<double, 3, 1> landmark_camera = pose * landmark_world;

            double projected[2];
            matrix::matrix<double, 2, 3> jacobian_camera;
            if (!this->camera.project(landmark_camera.data(), projected, jacobian_camera.data())) {
                this->jacobians[0] = matrix::matrix<double, 0, 0>::zero(2, 6);
                this->jacobians[1] = matrix::matrix<double, 0, 0>::zero(2, 3);
                return;
            }

            const double X = landmark_camera[0];
            const double Y = landmark_camera[1];
            const double Z = landmark_camera[2];

            // Point Jacobian wrt Pose increment (Left SE3 update: T' = exp(delta) * T)
            // d(exp(delta) * P) / d(omega, upsilon) = [-P^, I]
            matrix::matrix<double, 3, 6> jacobian_point_pose;
            jacobian_point_pose[0][0] = 0;   jacobian_point_pose[0][1] = Z;  jacobian_point_pose[0][2] = -Y;
            jacobian_point_pose[1][0] = -Z;  jacobian_point_pose[1][1] = 0;  jacobian_point_pose[1][2] = X;
            jacobian_point_pose[2][0] = Y;   jacobian_point_pose[2][1] = -X; jacobian_point_pose[2][2] = 0;
            jacobian_point_pose[0][3] = 1;   jacobian_point_pose[0][4] = 0;  jacobian_point_pose[0][5] = 0;
            jacobian_point_pose[1][3] = 0;   jacobian_point_pose[1][4] = 1;  jacobian_point_pose[1][5] = 0;
            jacobian_point_pose[2][3] = 0;   jacobian_point_pose[2][4] = 0;  jacobian_point_pose[2][5] = 1;

            // Residual Jacobian wrt Pose: d(obs - proj)/d_delta = -jacobian_camera * jacobian_point_pose
            const matrix::matrix<double, 2, 6> jacobian_pose_res = -(jacobian_camera * jacobian_point_pose);
            this->jacobians[0] = matrix::matrix<double, 0, 0>(2, 6, jacobian_pose_res.data());

            // Landmark Jacobian: d(obs - proj)/d_point_world = -jacobian_camera * d(P_camera)/d(P_world) = -jacobian_camera * R
            const matrix::matrix<double, 2, 3> jacobian_landmark_res = -(jacobian_camera * pose.rotation().get_matrix());
            this->jacobians[1] = matrix::matrix<double, 0, 0>(2, 3, jacobian_landmark_res.data());
        }
    };
}

#endif // FACTOR_GRAPH_HPP
