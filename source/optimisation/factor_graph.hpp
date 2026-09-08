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
#ifndef ZEROSLAM_OPTIMISATION_FACTOR_GRAPH_HPP
#define ZEROSLAM_OPTIMISATION_FACTOR_GRAPH_HPP

#include "math/matrix.hpp"
#include "math/matrix_sparse_block.hpp"
#include "math/matrix_sparse_block_diagonal.hpp"
#include "optimisation/edge.hpp"
#include "optimisation/vertex.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace optimisation {
    class factor_graph final {
    public:
        class diagnostics final {
        public:
            int rejected_attempts = 0;
        };

    private:
        double damping_lambda;
        double damping_factor;
        double chi_squared;

        std::deque<vertex> vertex_storage;
        std::deque<edge> edge_storage;

        std::vector<vertex*> vertices_general;
        std::vector<vertex*> vertices_marginalised;
        std::vector<edge*> edges;
        std::unordered_set<vertex*> vertex_set;
        std::unordered_set<edge*> edge_set;
        std::unordered_multimap<vertex*, edge*> vertex_to_edge;

    private:
        math::sparse_block_diagonal<6> h_pp;
        math::sparse_block_diagonal<3> h_ll;
        math::sparse_block<6, 3> h_pl;
        math::sparse_block<3, 6> h_lp;
        math::sparse_block_diagonal<3> h_ll_inverse;
        math::matrix<double, 0, 0> vector_b;
        math::matrix<double, 0, 0> delta_x;
        math::matrix<double, 0, 0> b_pp;
        math::matrix<double, 0, 0> b_ll;

        int count_general_params = 0;
        int count_marginalised_params = 0;

    private:
        std::vector<double> hessian_diagonal;
        std::vector<double> column_scale;
        std::vector<double> damping_diagonal;
        diagnostics last_diagnostics;

    public:
        factor_graph();

    public:
        const diagnostics& get_diagnostics() const;

    public:
        vertex* add_vertex(vertex&& node);

        bool remove_vertex(vertex* node);

        edge* add_edge(edge&& factor);

        bool remove_edge(edge* factor);

        std::vector<edge*> get_connected_edges(vertex* node) const;

    public:
        int solve(int iterations = 10, bool use_relative_convergence = false);

        double get_current_chi(bool recompute_residuals = true);

        bool compute_damped_step(double lambda, math::matrix<double, 0, 0>& step);

    private:
        double damping_weight(int index) const;

        void update_scaling();

        void linearise();

        bool compute_step();

        void set_ordering();

        void make_hessian();

        bool solve_linear_system();

        void update_states();

        void rollback_states();
    };
}

#endif // ZEROSLAM_OPTIMISATION_FACTOR_GRAPH_HPP
