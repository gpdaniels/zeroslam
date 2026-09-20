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
#ifndef ZEROSLAM_OPTIMISATION_LANDMARK_BLOCK_HPP
#define ZEROSLAM_OPTIMISATION_LANDMARK_BLOCK_HPP

#include "core/assert.hpp"
#include "math/math.hpp"
#include "math/matrix_decomposition_qr.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace optimisation {
    template <typename type>
    class landmark_block final {
    private:
        std::vector<type> storage;
        std::vector<type> undamped_head;
        std::vector<type> triangular_solution;
        std::vector<type> scratch;
        std::vector<int> pose_offsets;
        int landmark_offset = 0;
        int observation_rows = 0;
        int total_rows = 0;
        int total_cols = 0;
        int landmark_dimension = 0;
        int landmark_cols = 0;
        int landmark_col = 0;
        int residual_col = 0;
        int reduced_row = 0;
        bool damped = false;

    public:
        void configure(
            const int observation_count,
            const int dimension,
            const bool landmark_is_free,
            const int landmark_offset_value,
            const int* pose_offset_values,
            const int pose_count
        ) {
            ASSERT(observation_count > 0, "A landmark block needs at least one observation.");
            ASSERT(pose_count >= 0, "A landmark block cannot have a negative pose count.");
            ASSERT(dimension > 0, "A landmark block needs a positive landmark dimension.");
            this->observation_rows = 2 * observation_count;
            this->landmark_dimension = dimension;
            this->landmark_cols = landmark_is_free ? dimension : 0;
            this->reduced_row = landmark_is_free ? dimension : 0;
            this->total_rows = this->observation_rows + (landmark_is_free ? dimension : 0);
            this->landmark_col = 6 * pose_count;
            this->residual_col = this->landmark_col + this->landmark_cols;
            this->total_cols = this->residual_col + 1;
            this->landmark_offset = landmark_offset_value;
            this->pose_offsets.assign(pose_offset_values, pose_offset_values + pose_count);
            this->storage.assign(static_cast<size_t>(this->total_rows) * static_cast<size_t>(this->total_cols), static_cast<type>(0));
            this->undamped_head.assign(static_cast<size_t>(this->landmark_cols) * static_cast<size_t>(this->total_cols), static_cast<type>(0));
            this->triangular_solution.assign(static_cast<size_t>(this->landmark_cols), static_cast<type>(0));
            this->scratch.assign(static_cast<size_t>(this->total_rows + (2 * this->total_cols)), static_cast<type>(0));
            this->damped = false;
        }

        int rows() const {
            return this->total_rows;
        }

        int cols() const {
            return this->total_cols;
        }

        int reduced_rows() const {
            return this->total_rows - this->reduced_row;
        }

        int pose_count() const {
            return static_cast<int>(this->pose_offsets.size());
        }

        bool is_marginalised() const {
            return this->landmark_cols > 0;
        }

        int get_landmark_dimension() const {
            return this->landmark_cols;
        }

        int get_landmark_offset() const {
            return this->landmark_offset;
        }

        const type* row(const int index) const {
            ASSERT((index >= 0) && (index < this->total_rows), "Row index out of bounds.");
            return this->storage.data() + (static_cast<size_t>(index) * static_cast<size_t>(this->total_cols));
        }

    public:
        void set_observation(
            const int observation_slot,
            const int pose_slot,
            const type* jacobian_pose,
            const type* jacobian_landmark,
            const type* residual,
            const type* whitening
        ) {
            ASSERT((observation_slot >= 0) && ((2 * observation_slot) + 1 < this->observation_rows), "Observation slot out of bounds.");
            ASSERT(pose_slot < static_cast<int>(this->pose_offsets.size()), "Pose slot out of bounds.");
            for (int r = 0; r < 2; ++r) {
                type* const target = this->storage.data() + (static_cast<size_t>((2 * observation_slot) + r) * static_cast<size_t>(this->total_cols));
                if (pose_slot >= 0) {
                    for (int c = 0; c < 6; ++c) {
                        target[(6 * pose_slot) + c] = (whitening[(2 * r) + 0] * jacobian_pose[c]) + (whitening[(2 * r) + 1] * jacobian_pose[6 + c]);
                    }
                }
                if (this->landmark_cols > 0) {
                    for (int c = 0; c < this->landmark_cols; ++c) {
                        target[this->landmark_col + c] = (whitening[(2 * r) + 0] * jacobian_landmark[c]) + (whitening[(2 * r) + 1] * jacobian_landmark[this->landmark_cols + c]);
                    }
                }
                target[this->residual_col] = (whitening[(2 * r) + 0] * residual[0]) + (whitening[(2 * r) + 1] * residual[1]);
            }
        }

        template <typename output_type>
        void accumulate_gradient(output_type* vector_b) const {
            for (size_t slot = 0; slot < this->pose_offsets.size(); ++slot) {
                this->accumulate_gradient_slot(static_cast<int>(slot), vector_b + this->pose_offsets[slot]);
            }
            if (this->landmark_cols > 0) {
                this->accumulate_gradient_landmark(vector_b + this->landmark_offset);
            }
        }

        template <typename output_type>
        void accumulate_gradient_slot(const int slot, output_type* target) const {
            for (int r = 0; r < this->observation_rows; ++r) {
                const type* const source = this->row(r);
                const type residual = source[this->residual_col];
                const type* const jacobian = source + (6 * slot);
                for (int c = 0; c < 6; ++c) {
                    target[c] -= static_cast<output_type>(jacobian[c] * residual);
                }
            }
        }

        template <typename output_type>
        void accumulate_gradient_landmark(output_type* target) const {
            for (int r = 0; r < this->observation_rows; ++r) {
                const type* const source = this->row(r);
                const type residual = source[this->residual_col];
                const type* const jacobian = source + this->landmark_col;
                for (int c = 0; c < this->landmark_cols; ++c) {
                    target[c] -= static_cast<output_type>(jacobian[c] * residual);
                }
            }
        }

        template <typename output_type>
        void accumulate_hessian_diagonal(output_type* diagonal) const {
            for (size_t slot = 0; slot < this->pose_offsets.size(); ++slot) {
                this->accumulate_hessian_diagonal_slot(static_cast<int>(slot), diagonal + this->pose_offsets[slot]);
            }
            if (this->landmark_cols > 0) {
                this->accumulate_hessian_diagonal_landmark(diagonal + this->landmark_offset);
            }
        }

        template <typename output_type>
        void accumulate_hessian_diagonal_slot(const int slot, output_type* target) const {
            for (int r = 0; r < this->observation_rows; ++r) {
                const type* const jacobian = this->row(r) + (6 * slot);
                for (int c = 0; c < 6; ++c) {
                    target[c] += static_cast<output_type>(jacobian[c] * jacobian[c]);
                }
            }
        }

        template <typename output_type>
        void accumulate_hessian_diagonal_landmark(output_type* target) const {
            for (int r = 0; r < this->observation_rows; ++r) {
                const type* const jacobian = this->row(r) + this->landmark_col;
                for (int c = 0; c < this->landmark_cols; ++c) {
                    target[c] += static_cast<output_type>(jacobian[c] * jacobian[c]);
                }
            }
        }

        template <typename input_type>
        void scale_columns(const input_type* column_scale) {
            for (int r = 0; r < this->observation_rows; ++r) {
                type* const target = this->storage.data() + (static_cast<size_t>(r) * static_cast<size_t>(this->total_cols));
                for (size_t slot = 0; slot < this->pose_offsets.size(); ++slot) {
                    const input_type* const scale = column_scale + this->pose_offsets[slot];
                    type* const jacobian = target + (6 * static_cast<int>(slot));
                    for (int c = 0; c < 6; ++c) {
                        jacobian[c] *= static_cast<type>(scale[c]);
                    }
                }
                if (this->landmark_cols > 0) {
                    const input_type* const scale = column_scale + this->landmark_offset;
                    type* const jacobian = target + this->landmark_col;
                    for (int c = 0; c < this->landmark_cols; ++c) {
                        jacobian[c] *= static_cast<type>(scale[c]);
                    }
                }
            }
        }

        void perform_qr() {
            if (this->landmark_cols == 0) {
                return;
            }
            type* const data = this->storage.data();
            type* const workspace = this->scratch.data();
            const int reflections = math::min(this->landmark_cols, this->observation_rows - 1);
            for (int k = 0; k < reflections; ++k) {
                type beta = 0;
                type alpha = 0;
                const bool non_trivial = math::householder_vector(
                    data + (static_cast<size_t>(k) * static_cast<size_t>(this->total_cols)) + static_cast<size_t>(this->landmark_col + k),
                    this->observation_rows - k,
                    this->total_cols,
                    workspace,
                    beta,
                    alpha
                );
                if (!non_trivial) {
                    for (int i = k + 1; i < this->observation_rows; ++i) {
                        data[(static_cast<size_t>(i) * static_cast<size_t>(this->total_cols)) + static_cast<size_t>(this->landmark_col + k)] = 0;
                    }
                    continue;
                }
                if (this->landmark_col > 0) {
                    math::apply_householder_left(data, this->total_cols, k, 0, this->observation_rows - k, this->landmark_col, workspace, beta);
                }
                math::apply_householder_left(data, this->total_cols, k, this->landmark_col + k + 1, this->observation_rows - k, this->landmark_cols - k, workspace, beta);
                data[(static_cast<size_t>(k) * static_cast<size_t>(this->total_cols)) + static_cast<size_t>(this->landmark_col + k)] = alpha;
                for (int i = k + 1; i < this->observation_rows; ++i) {
                    data[(static_cast<size_t>(i) * static_cast<size_t>(this->total_cols)) + static_cast<size_t>(this->landmark_col + k)] = 0;
                }
            }
            for (int r = 0; r < this->landmark_cols; ++r) {
                for (int c = 0; c < this->total_cols; ++c) {
                    this->undamped_head[(static_cast<size_t>(r) * static_cast<size_t>(this->total_cols)) + static_cast<size_t>(c)] = (r < this->total_rows) ? this->storage[(static_cast<size_t>(r) * static_cast<size_t>(this->total_cols)) + static_cast<size_t>(c)] : static_cast<type>(0);
                }
            }
            this->damped = false;
        }

        void set_damping(const type lambda) {
            if (this->landmark_cols == 0) {
                return;
            }
            this->clear_damping();
            if (!(lambda > 0)) {
                this->damped = true;
                return;
            }
            type* const data = this->storage.data();
            const type root_lambda = math::sqrt(lambda);
            for (int d = 0; d < this->landmark_cols; ++d) {
                data[(static_cast<size_t>(this->observation_rows + d) * static_cast<size_t>(this->total_cols)) + static_cast<size_t>(this->landmark_col + d)] = root_lambda;
            }
            for (int c = 0; c < this->landmark_cols; ++c) {
                for (int d = this->landmark_cols - 1; d >= 0; --d) {
                    const int r = this->observation_rows + d;
                    if (r <= c) {
                        continue;
                    }
                    type* const pivot = data + (static_cast<size_t>(c) * static_cast<size_t>(this->total_cols));
                    type* const target = data + (static_cast<size_t>(r) * static_cast<size_t>(this->total_cols));
                    if (target[this->landmark_col + c] == 0) {
                        continue;
                    }
                    type cosine = 0;
                    type sine = 0;
                    math::givens(pivot[this->landmark_col + c], target[this->landmark_col + c], cosine, sine);
                    math::apply_givens_left(data, this->total_cols, c, r, 0, this->total_cols, cosine, sine);
                    target[this->landmark_col + c] = 0;
                }
            }
            this->damped = true;
        }

        void clear_damping() {
            if ((this->landmark_cols == 0) || !this->damped) {
                return;
            }
            for (int r = 0; (r < this->landmark_cols) && (r < this->total_rows); ++r) {
                for (int c = 0; c < this->total_cols; ++c) {
                    this->storage[(static_cast<size_t>(r) * static_cast<size_t>(this->total_cols)) + static_cast<size_t>(c)] = this->undamped_head[(static_cast<size_t>(r) * static_cast<size_t>(this->total_cols)) + static_cast<size_t>(c)];
                }
            }
            for (int d = 0; d < this->landmark_cols; ++d) {
                type* const target = this->storage.data() + (static_cast<size_t>(this->observation_rows + d) * static_cast<size_t>(this->total_cols));
                for (int c = 0; c < this->total_cols; ++c) {
                    target[c] = 0;
                }
            }
            this->damped = false;
        }

    public:
        void add_operator_product(const type* column_x, type* column_out) {
            if (!this->compute_operator_image(column_x)) {
                return;
            }
            for (size_t slot = 0; slot < this->pose_offsets.size(); ++slot) {
                this->add_operator_slot(static_cast<int>(slot), column_out + this->pose_offsets[slot]);
            }
        }

        bool compute_operator_image(const type* column_x) {
            const int reduced = this->reduced_rows();
            const int pose_columns = this->landmark_col;
            if ((reduced <= 0) || (pose_columns == 0)) {
                return false;
            }
            type* const gathered = this->scratch.data();
            type* const image = this->scratch.data() + pose_columns;
            for (size_t slot = 0; slot < this->pose_offsets.size(); ++slot) {
                for (int c = 0; c < 6; ++c) {
                    gathered[(6 * static_cast<int>(slot)) + c] = column_x[this->pose_offsets[slot] + c];
                }
            }
            for (int r = 0; r < reduced; ++r) {
                const type* const source = this->row(this->reduced_row + r);
                type sum = 0;
                for (int c = 0; c < pose_columns; ++c) {
                    sum += source[c] * gathered[c];
                }
                image[r] = sum;
            }
            return true;
        }

        void add_operator_slot(const int slot, type* target) const {
            const int reduced = this->reduced_rows();
            const type* const image = this->scratch.data() + this->landmark_col;
            type accumulated[6] = { 0, 0, 0, 0, 0, 0 };
            for (int r = 0; r < reduced; ++r) {
                const type* const source = this->row(this->reduced_row + r) + (6 * slot);
                const type value = image[r];
                for (int c = 0; c < 6; ++c) {
                    accumulated[c] += source[c] * value;
                }
            }
            for (int c = 0; c < 6; ++c) {
                target[c] += accumulated[c];
            }
        }

        void add_right_hand_side(type* column_b) const {
            for (size_t slot = 0; slot < this->pose_offsets.size(); ++slot) {
                this->add_right_hand_side_slot(static_cast<int>(slot), column_b + this->pose_offsets[slot]);
            }
        }

        void add_right_hand_side_slot(const int slot, type* target) const {
            const int reduced = this->reduced_rows();
            if ((reduced <= 0) || (this->landmark_col == 0)) {
                return;
            }
            for (int r = 0; r < reduced; ++r) {
                const type* const source = this->row(this->reduced_row + r);
                const type residual = source[this->residual_col];
                const type* const jacobian = source + (6 * slot);
                for (int c = 0; c < 6; ++c) {
                    target[c] -= jacobian[c] * residual;
                }
            }
        }

        void add_reduced_diagonal_blocks(type* blocks) const {
            for (size_t slot = 0; slot < this->pose_offsets.size(); ++slot) {
                this->add_reduced_diagonal_slot(static_cast<int>(slot), blocks + (static_cast<size_t>(this->pose_offsets[slot] / 6) * 36u));
            }
        }

        void add_reduced_diagonal_slot(const int slot, type* block) const {
            const int reduced = this->reduced_rows();
            if ((reduced <= 0) || (this->landmark_col == 0)) {
                return;
            }
            for (int r = 0; r < reduced; ++r) {
                const type* const jacobian = this->row(this->reduced_row + r) + (6 * slot);
                for (int i = 0; i < 6; ++i) {
                    for (int j = 0; j < 6; ++j) {
                        block[(6 * i) + j] += jacobian[i] * jacobian[j];
                    }
                }
            }
        }

        int get_pose_offset(const int slot) const {
            return this->pose_offsets[static_cast<size_t>(slot)];
        }

    public:
        bool back_substitute(const type* column_x, type* landmark_delta) {
            if (this->landmark_cols == 0) {
                for (int c = 0; c < this->landmark_dimension; ++c) {
                    landmark_delta[c] = 0;
                }
                return true;
            }
            type* const right_hand_side = this->scratch.data();
            for (int r = 0; r < this->landmark_cols; ++r) {
                const type* const source = this->row(r);
                type sum = source[this->residual_col];
                for (size_t slot = 0; slot < this->pose_offsets.size(); ++slot) {
                    const type* const jacobian = source + (6 * static_cast<int>(slot));
                    const type* const increment = column_x + this->pose_offsets[slot];
                    for (int c = 0; c < 6; ++c) {
                        sum += jacobian[c] * increment[c];
                    }
                }
                right_hand_side[r] = sum;
            }
            type* const solution = this->triangular_solution.data();
            if (!math::solve_upper_triangular(this->storage.data() + static_cast<size_t>(this->landmark_col), this->total_cols, this->landmark_cols, right_hand_side, solution)) {
                for (int c = 0; c < this->landmark_cols; ++c) {
                    landmark_delta[c] = 0;
                }
                return false;
            }
            for (int c = 0; c < this->landmark_cols; ++c) {
                landmark_delta[c] = -solution[c];
            }
            return true;
        }

        type minimum_triangular_diagonal() const {
            if (this->landmark_cols == 0) {
                return 0;
            }
            type minimum = math::abs(this->row(0)[this->landmark_col]);
            for (int i = 1; i < this->landmark_cols; ++i) {
                minimum = math::min(minimum, math::abs(this->row(i)[this->landmark_col + i]));
            }
            return minimum;
        }
    };
}

#endif // ZEROSLAM_OPTIMISATION_LANDMARK_BLOCK_HPP
