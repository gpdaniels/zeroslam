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
#include "optimisation/edge.hpp"
#include "optimisation/landmark_block.hpp"
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
        enum class strategy {
            dense_schur,
            square_root,
            automatic
        };

        enum class precision {
            double_precision,
            single_precision
        };

        class diagnostics final {
        public:
            int rejected_attempts = 0;
            bool used_square_root = false;
            bool used_single_precision = false;
            int landmark_blocks = 0;
            int reduced_solves = 0;
            int reduced_iterations = 0;
            int reduced_failures = 0;
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

    public:
        constexpr static const int maximum_landmark_dimensions = 6;

    private:
        class coupling final {
        public:
            int general_index;
            int general_dimensions;
            size_t landmark_block;
            double block[vertex::maximum_parameters][maximum_landmark_dimensions];
        };

        class landmark_diagonal final {
        public:
            int offset;
            int dimensions;
            double block[maximum_landmark_dimensions][maximum_landmark_dimensions];
            double inverse[maximum_landmark_dimensions][maximum_landmark_dimensions];
        };

        constexpr static const size_t maximum_residuals = 16;

        template <typename scalar>
        class square_root_state final {
        public:
            std::vector<landmark_block<scalar>> blocks;
            std::vector<std::vector<edge*>> grouped_edges;
            std::unordered_map<const vertex*, size_t> landmark_indices;
            std::vector<size_t> block_vertices;
            std::vector<int> pose_block_begin;
            std::vector<int> pose_block_index;
            std::vector<int> pose_block_slot;
            std::vector<scalar> preconditioner;
            std::vector<scalar> right_hand_side;
            std::vector<scalar> general_right_hand_side;
            std::vector<scalar> increment;
            std::vector<scalar> scratch;
        };

    private:
        strategy solve_strategy = strategy::dense_schur;
        precision solve_precision = precision::double_precision;
        int square_root_parameter_threshold = 6 * 128;
        double conjugate_gradient_tolerance = 1e-2;
        int conjugate_gradient_iteration_limit = 0;
        bool square_root_active = false;
        square_root_state<double> square_root_double;
        square_root_state<float> square_root_single;

    private:
        math::matrix<double, 0, 0> h_pp;
        std::vector<landmark_diagonal> h_ll;
        std::vector<size_t> landmark_block_of_parameter;
        std::vector<coupling> couplings;
        std::vector<std::vector<size_t>> landmark_couplings;
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

        void set_strategy(const strategy solve_strategy_value);
        void set_precision(const precision solve_precision_value);
        void set_square_root_parameter_threshold(const int threshold);
        void set_conjugate_gradient_tolerance(const double tolerance);
        void set_conjugate_gradient_iteration_limit(const int limit);

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
        static void add_block(math::matrix<double, 0, 0>& target, size_t row, size_t col, const math::matrix<double, 0, 0>& block, double scale);

        void accumulate_general_edge(edge* factor);

        bool select_square_root_path() const;

        static bool invert_landmark_block(const double (&damped)[maximum_landmark_dimensions][maximum_landmark_dimensions], int dimensions, double (&inverse)[maximum_landmark_dimensions][maximum_landmark_dimensions]);

        template <typename scalar>
        void linearise_landmark_blocks(square_root_state<scalar>& state);

        template <typename scalar>
        bool solve_linear_system_qr(square_root_state<scalar>& state);

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
