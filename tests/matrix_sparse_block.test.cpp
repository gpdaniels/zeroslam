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

#include "matrix_sparse_block.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-8) {
    if (std::isnan(lhs) && std::isnan(rhs))
        return true;
    if (std::isnan(lhs) != std::isnan(rhs))
        return false;
    if (std::isinf(lhs) != std::isinf(rhs))
        return false;
    if (std::signbit(lhs + epsilon) != std::signbit(rhs + epsilon))
        return false;
    if (std::isinf(lhs) && std::isinf(rhs))
        return true;
    return (std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon);
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    // Test basic construction.
    matrix::sparse_block<2, 3> sparse_block(4, 6);
    REQUIRE(sparse_block.rows() == 4);
    REQUIRE(sparse_block.cols() == 6);
    REQUIRE(sparse_block.validate());

    // Insert an out-of-range block and check validate() fails
    matrix::sparse_block<2, 2> sparse_block_bad;
    sparse_block_bad.blocks()[{ 2, 0 }] = {};
    REQUIRE(sparse_block_bad.validate() == false);

    // Test inserting a block at (0,0).
    matrix::sparse_block<2, 3>::block_type block_0;
    block_0(0, 0) = 1.0;
    block_0(0, 1) = 2.0;
    block_0(0, 2) = 3.0;
    block_0(1, 0) = 4.0;
    block_0(1, 1) = 5.0;
    block_0(1, 2) = 6.0;
    sparse_block.blocks()[{ 0, 0 }] = block_0;
    REQUIRE(sparse_block.validate());
    // Elements from the set blocks.
    REQUIRE(is_value_approx(sparse_block(0, 0), 1.0));
    REQUIRE(is_value_approx(sparse_block(0, 1), 2.0));
    REQUIRE(is_value_approx(sparse_block(0, 2), 3.0));
    REQUIRE(is_value_approx(sparse_block(1, 0), 4.0));
    REQUIRE(is_value_approx(sparse_block(1, 1), 5.0));
    REQUIRE(is_value_approx(sparse_block(1, 2), 6.0));
    // Missing elements should be zero.
    REQUIRE(is_value_approx(sparse_block(0, 3), 0.0));
    REQUIRE(is_value_approx(sparse_block(0, 4), 0.0));
    REQUIRE(is_value_approx(sparse_block(0, 5), 0.0));
    REQUIRE(is_value_approx(sparse_block(1, 3), 0.0));
    REQUIRE(is_value_approx(sparse_block(1, 4), 0.0));
    REQUIRE(is_value_approx(sparse_block(1, 5), 0.0));
    REQUIRE(is_value_approx(sparse_block(2, 3), 0.0));
    REQUIRE(is_value_approx(sparse_block(2, 4), 0.0));
    REQUIRE(is_value_approx(sparse_block(2, 5), 0.0));
    REQUIRE(is_value_approx(sparse_block(3, 3), 0.0));
    REQUIRE(is_value_approx(sparse_block(3, 4), 0.0));
    REQUIRE(is_value_approx(sparse_block(3, 5), 0.0));
    REQUIRE(is_value_approx(sparse_block(2, 0), 0.0));
    REQUIRE(is_value_approx(sparse_block(2, 1), 0.0));
    REQUIRE(is_value_approx(sparse_block(2, 2), 0.0));
    REQUIRE(is_value_approx(sparse_block(3, 0), 0.0));
    REQUIRE(is_value_approx(sparse_block(3, 1), 0.0));
    REQUIRE(is_value_approx(sparse_block(3, 2), 0.0));

    // Test inserting a block at (0,1) and (1,1).
    matrix::sparse_block<2, 3>::block_type block_1;
    block_1(0, 0) = 7.0;
    block_1(0, 1) = 8.0;
    block_1(0, 2) = 9.0;
    block_1(1, 0) = 10.0;
    block_1(1, 1) = 11.0;
    block_1(1, 2) = 12.0;
    sparse_block.blocks()[{ 0, 1 }] = block_1;
    sparse_block.blocks()[{ 1, 1 }] = block_1;
    REQUIRE(sparse_block.validate());
    // Elements from the set blocks.
    REQUIRE(is_value_approx(sparse_block(0, 0), 1.0));
    REQUIRE(is_value_approx(sparse_block(0, 1), 2.0));
    REQUIRE(is_value_approx(sparse_block(0, 2), 3.0));
    REQUIRE(is_value_approx(sparse_block(1, 0), 4.0));
    REQUIRE(is_value_approx(sparse_block(1, 1), 5.0));
    REQUIRE(is_value_approx(sparse_block(1, 2), 6.0));
    REQUIRE(is_value_approx(sparse_block(0, 3), 7.0));
    REQUIRE(is_value_approx(sparse_block(0, 4), 8.0));
    REQUIRE(is_value_approx(sparse_block(0, 5), 9.0));
    REQUIRE(is_value_approx(sparse_block(1, 3), 10.0));
    REQUIRE(is_value_approx(sparse_block(1, 4), 11.0));
    REQUIRE(is_value_approx(sparse_block(1, 5), 12.0));
    REQUIRE(is_value_approx(sparse_block(2, 3), 7.0));
    REQUIRE(is_value_approx(sparse_block(2, 4), 8.0));
    REQUIRE(is_value_approx(sparse_block(2, 5), 9.0));
    REQUIRE(is_value_approx(sparse_block(3, 3), 10.0));
    REQUIRE(is_value_approx(sparse_block(3, 4), 11.0));
    REQUIRE(is_value_approx(sparse_block(3, 5), 12.0));
    // Missing elements should be zero.
    REQUIRE(is_value_approx(sparse_block(2, 0), 0.0));
    REQUIRE(is_value_approx(sparse_block(2, 1), 0.0));
    REQUIRE(is_value_approx(sparse_block(2, 2), 0.0));
    REQUIRE(is_value_approx(sparse_block(3, 0), 0.0));
    REQUIRE(is_value_approx(sparse_block(3, 1), 0.0));
    REQUIRE(is_value_approx(sparse_block(3, 2), 0.0));

    // Test transpose.
    const matrix::sparse_block<3, 2> sparse_block_transpose = sparse_block.get_transpose();
    REQUIRE(sparse_block_transpose.rows() == 6);
    REQUIRE(sparse_block_transpose.cols() == 4);
    REQUIRE(sparse_block_transpose.validate());
    // Elements from the set blocks.
    REQUIRE(is_value_approx(sparse_block_transpose(0, 0), 1.0));
    REQUIRE(is_value_approx(sparse_block_transpose(1, 0), 2.0));
    REQUIRE(is_value_approx(sparse_block_transpose(2, 0), 3.0));
    REQUIRE(is_value_approx(sparse_block_transpose(0, 1), 4.0));
    REQUIRE(is_value_approx(sparse_block_transpose(1, 1), 5.0));
    REQUIRE(is_value_approx(sparse_block_transpose(2, 1), 6.0));
    REQUIRE(is_value_approx(sparse_block_transpose(3, 0), 7.0));
    REQUIRE(is_value_approx(sparse_block_transpose(4, 0), 8.0));
    REQUIRE(is_value_approx(sparse_block_transpose(5, 0), 9.0));
    REQUIRE(is_value_approx(sparse_block_transpose(3, 1), 10.0));
    REQUIRE(is_value_approx(sparse_block_transpose(4, 1), 11.0));
    REQUIRE(is_value_approx(sparse_block_transpose(5, 1), 12.0));
    REQUIRE(is_value_approx(sparse_block_transpose(3, 2), 7.0));
    REQUIRE(is_value_approx(sparse_block_transpose(4, 2), 8.0));
    REQUIRE(is_value_approx(sparse_block_transpose(5, 2), 9.0));
    REQUIRE(is_value_approx(sparse_block_transpose(3, 3), 10.0));
    REQUIRE(is_value_approx(sparse_block_transpose(4, 3), 11.0));
    REQUIRE(is_value_approx(sparse_block_transpose(5, 3), 12.0));
    // Missing elements should be zero.
    REQUIRE(is_value_approx(sparse_block_transpose(0, 2), 0.0));
    REQUIRE(is_value_approx(sparse_block_transpose(1, 2), 0.0));
    REQUIRE(is_value_approx(sparse_block_transpose(2, 2), 0.0));
    REQUIRE(is_value_approx(sparse_block_transpose(0, 3), 0.0));
    REQUIRE(is_value_approx(sparse_block_transpose(1, 3), 0.0));
    REQUIRE(is_value_approx(sparse_block_transpose(2, 3), 0.0));

    // Test addition.
    matrix::sparse_block<2, 3> sparse_block_sum = sparse_block.add(sparse_block);
    REQUIRE(sparse_block_sum.rows() == 4);
    REQUIRE(sparse_block_sum.cols() == 6);
    REQUIRE(sparse_block_sum.validate());
    // Elements from the set blocks.
    REQUIRE(is_value_approx(sparse_block_sum(0, 0), 2 * sparse_block(0, 0)));
    REQUIRE(is_value_approx(sparse_block_sum(0, 1), 2 * sparse_block(0, 1)));
    REQUIRE(is_value_approx(sparse_block_sum(0, 2), 2 * sparse_block(0, 2)));
    REQUIRE(is_value_approx(sparse_block_sum(1, 0), 2 * sparse_block(1, 0)));
    REQUIRE(is_value_approx(sparse_block_sum(1, 1), 2 * sparse_block(1, 1)));
    REQUIRE(is_value_approx(sparse_block_sum(1, 2), 2 * sparse_block(1, 2)));
    REQUIRE(is_value_approx(sparse_block_sum(0, 3), 2 * sparse_block(0, 3)));
    REQUIRE(is_value_approx(sparse_block_sum(0, 4), 2 * sparse_block(0, 4)));
    REQUIRE(is_value_approx(sparse_block_sum(0, 5), 2 * sparse_block(0, 5)));
    REQUIRE(is_value_approx(sparse_block_sum(1, 3), 2 * sparse_block(1, 3)));
    REQUIRE(is_value_approx(sparse_block_sum(1, 4), 2 * sparse_block(1, 4)));
    REQUIRE(is_value_approx(sparse_block_sum(1, 5), 2 * sparse_block(1, 5)));
    REQUIRE(is_value_approx(sparse_block_sum(2, 3), 2 * sparse_block(2, 3)));
    REQUIRE(is_value_approx(sparse_block_sum(2, 4), 2 * sparse_block(2, 4)));
    REQUIRE(is_value_approx(sparse_block_sum(2, 5), 2 * sparse_block(2, 5)));
    REQUIRE(is_value_approx(sparse_block_sum(3, 3), 2 * sparse_block(3, 3)));
    REQUIRE(is_value_approx(sparse_block_sum(3, 4), 2 * sparse_block(3, 4)));
    REQUIRE(is_value_approx(sparse_block_sum(3, 5), 2 * sparse_block(3, 5)));
    // Missing elements should be zero.
    REQUIRE(is_value_approx(sparse_block_transpose(0, 2), 0.0));
    REQUIRE(is_value_approx(sparse_block_transpose(1, 2), 0.0));
    REQUIRE(is_value_approx(sparse_block_transpose(2, 2), 0.0));
    REQUIRE(is_value_approx(sparse_block_transpose(0, 3), 0.0));
    REQUIRE(is_value_approx(sparse_block_transpose(1, 3), 0.0));
    REQUIRE(is_value_approx(sparse_block_transpose(2, 3), 0.0));

    // Test multiplication.
    matrix::sparse_block<2, 2> sparse_block_product = sparse_block.multiply(sparse_block_transpose);
    REQUIRE(sparse_block_product.rows() == 4);
    REQUIRE(sparse_block_product.cols() == 4);
    REQUIRE(sparse_block_product.validate());
    // All elements should now be set.
    REQUIRE(is_value_approx(sparse_block_product(0, 0), 208.0));
    REQUIRE(is_value_approx(sparse_block_product(0, 1), 298.0));
    REQUIRE(is_value_approx(sparse_block_product(0, 2), 194.0));
    REQUIRE(is_value_approx(sparse_block_product(0, 3), 266.0));
    REQUIRE(is_value_approx(sparse_block_product(1, 0), 298.0));
    REQUIRE(is_value_approx(sparse_block_product(1, 1), 442.0));
    REQUIRE(is_value_approx(sparse_block_product(1, 2), 266.0));
    REQUIRE(is_value_approx(sparse_block_product(1, 3), 365.0));
    REQUIRE(is_value_approx(sparse_block_product(2, 0), 194.0));
    REQUIRE(is_value_approx(sparse_block_product(2, 1), 266.0));
    REQUIRE(is_value_approx(sparse_block_product(2, 2), 194.0));
    REQUIRE(is_value_approx(sparse_block_product(2, 3), 266.0));
    REQUIRE(is_value_approx(sparse_block_product(3, 0), 266.0));
    REQUIRE(is_value_approx(sparse_block_product(3, 1), 365.0));
    REQUIRE(is_value_approx(sparse_block_product(3, 2), 266.0));
    REQUIRE(is_value_approx(sparse_block_product(3, 3), 365.0));

    // Multiply by a vector.
    matrix::matrix<double, 0, 0> vector_rhs = matrix::matrix<double, 0, 0>::zero(6, 1);
    vector_rhs(0, 0) = 1.0;
    vector_rhs(1, 0) = 2.0;
    vector_rhs(2, 0) = 3.0;
    vector_rhs(3, 0) = 4.0;
    vector_rhs(4, 0) = 5.0;
    vector_rhs(5, 0) = 6.0;
    matrix::matrix<double, 0, 0> sparse_block_vector_rhs = sparse_block.multiply(vector_rhs);
    REQUIRE(sparse_block_vector_rhs.rows() == 4);
    REQUIRE(sparse_block_vector_rhs.cols() == 1);
    REQUIRE(is_value_approx(sparse_block_vector_rhs(0, 0), 136.0));
    REQUIRE(is_value_approx(sparse_block_vector_rhs(1, 0), 199.0));
    REQUIRE(is_value_approx(sparse_block_vector_rhs(2, 0), 122.0));
    REQUIRE(is_value_approx(sparse_block_vector_rhs(3, 0), 167.0));

    // Right multiply by a vector.
    matrix::matrix<double, 0, 0> vector_lhs = matrix::matrix<double, 0, 0>::zero(1, 4);
    vector_lhs(0, 0) = 1.0;
    vector_lhs(0, 1) = 2.0;
    vector_lhs(0, 2) = 3.0;
    vector_lhs(0, 3) = 4.0;
    matrix::matrix<double, 0, 0> sparse_block_vector_lhs = sparse_block.multiply_right(vector_lhs);
    REQUIRE(sparse_block_vector_lhs.rows() == 1);
    REQUIRE(sparse_block_vector_lhs.cols() == 6);
    REQUIRE(is_value_approx(sparse_block_vector_lhs(0, 0), 9.0));
    REQUIRE(is_value_approx(sparse_block_vector_lhs(0, 1), 12.0));
    REQUIRE(is_value_approx(sparse_block_vector_lhs(0, 2), 15.0));
    REQUIRE(is_value_approx(sparse_block_vector_lhs(0, 3), 88.0));
    REQUIRE(is_value_approx(sparse_block_vector_lhs(0, 4), 98.0));
    REQUIRE(is_value_approx(sparse_block_vector_lhs(0, 5), 108.0));
}