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

#include "matrix_sparse_block_diagonal.hpp"

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

    // Basic construction and access tests for sparse_block_diagonal.
    {
        REQUIRE(matrix::sparse_block_diagonal<2>::block_size == 2);

        matrix::sparse_block_diagonal<2> diag;
        // Initially empty (0x0).
        REQUIRE(diag.rows() == 0);
        REQUIRE(diag.cols() == 0);
        REQUIRE(diag.diagonal().empty());

        // Create a diagonal of 2 blocks => 4x4 matrix.
        diag.diagonal().resize(2);
        REQUIRE(diag.rows() == 4);
        REQUIRE(diag.cols() == 4);

        // Fill block 0.
        matrix::sparse_block_diagonal<2>::block_type b0;
        b0(0, 0) = 1.0;
        b0(0, 1) = 2.0;
        b0(1, 0) = 3.0;
        b0(1, 1) = 4.0;
        diag.diagonal()[0] = b0;

        // Fill block 1.
        matrix::sparse_block_diagonal<2>::block_type b1;
        b1(0, 0) = 5.0;
        b1(0, 1) = 6.0;
        b1(1, 0) = 7.0;
        b1(1, 1) = 8.0;
        diag.diagonal()[1] = b1;

        // Ensure operator() returns the block entries on the diagonal.
        REQUIRE(is_value_approx(diag(0, 0), 1.0));
        REQUIRE(is_value_approx(diag(0, 1), 2.0));
        REQUIRE(is_value_approx(diag(1, 0), 3.0));
        REQUIRE(is_value_approx(diag(1, 1), 4.0));

        REQUIRE(is_value_approx(diag(2, 2), 5.0));
        REQUIRE(is_value_approx(diag(2, 3), 6.0));
        REQUIRE(is_value_approx(diag(3, 2), 7.0));
        REQUIRE(is_value_approx(diag(3, 3), 8.0));

        // Off-diagonal (different block indices) must be zero.
        REQUIRE(is_value_approx(diag(0, 2), 0.0));
        REQUIRE(is_value_approx(diag(1, 3), 0.0));
        REQUIRE(is_value_approx(diag(2, 0), 0.0));
        REQUIRE(is_value_approx(diag(3, 1), 0.0));
    }

    // Multiply (matrix * vector) and (vector * matrix) tests.
    {
        matrix::sparse_block_diagonal<2> diag;
        diag.diagonal().resize(2);

        matrix::sparse_block_diagonal<2>::block_type b0;
        b0(0, 0) = 1;
        b0(0, 1) = 2;
        b0(1, 0) = 3;
        b0(1, 1) = 4;
        matrix::sparse_block_diagonal<2>::block_type b1;
        b1(0, 0) = 5;
        b1(0, 1) = 6;
        b1(1, 0) = 7;
        b1(1, 1) = 8;
        diag.diagonal()[0] = b0;
        diag.diagonal()[1] = b1;

        // RHS column vector 4x1.
        matrix::matrix<double, 0, 0> rhs = matrix::matrix<double, 0, 0>::zero(4, 1);
        rhs(0, 0) = 1.0;
        rhs(1, 0) = 2.0;
        rhs(2, 0) = 3.0;
        rhs(3, 0) = 4.0;

        matrix::matrix<double, 0, 0> out = diag.multiply(rhs);
        REQUIRE(out.rows() == 4);
        REQUIRE(out.cols() == 1);
        // Test multiply left.
        // block0 * [1,2]^T = [1*1+2*2, 3*1+4*2] = [5,11]
        // block1 * [3,4]^T = [5*3+6*4, 7*3+8*4] = [39,53]
        REQUIRE(is_value_approx(out(0, 0), 5.0));
        REQUIRE(is_value_approx(out(1, 0), 11.0));
        REQUIRE(is_value_approx(out(2, 0), 39.0));
        REQUIRE(is_value_approx(out(3, 0), 53.0));

        // LHS row vector 1x4.
        matrix::matrix<double, 0, 0> lhs = matrix::matrix<double, 0, 0>::zero(1, 4);
        lhs(0, 0) = 1.0;
        lhs(0, 1) = 2.0;
        lhs(0, 2) = 3.0;
        lhs(0, 3) = 4.0;

        matrix::matrix<double, 0, 0> out_right = diag.multiply_right(lhs);
        REQUIRE(out_right.rows() == 1);
        REQUIRE(out_right.cols() == 4);
        // Test multiply right.
        // [1,2] * block0 = [1*1+2*3, 1*2+2*4] = [7,10]
        // [3,4] * block1 = [3*5+4*7, 3*6+4*8] = [43,50]
        REQUIRE(is_value_approx(out_right(0, 0), 7.0));
        REQUIRE(is_value_approx(out_right(0, 1), 10.0));
        REQUIRE(is_value_approx(out_right(0, 2), 43.0));
        REQUIRE(is_value_approx(out_right(0, 3), 50.0));
    }

    // Multiply (sparse_block * sparse_block_diagonal) tests.
    {
        // We'll use block size 2 and a 2-block diagonal -> 4x4 matrix.
        matrix::sparse_block_diagonal<2> rhs;
        rhs.diagonal().resize(2);
        matrix::sparse_block_diagonal<2>::block_type db0;
        db0(0, 0) = 2;
        db0(0, 1) = 0;
        db0(1, 0) = 0;
        db0(1, 1) = 2;
        matrix::sparse_block_diagonal<2>::block_type db1;
        db1(0, 0) = 3;
        db1(0, 1) = 1;
        db1(1, 0) = 4;
        db1(1, 1) = 5;
        rhs.diagonal()[0] = db0;
        rhs.diagonal()[1] = db1;

        // Create lhs sparse_block 4x4 (block dims 2x2)
        matrix::sparse_block<2, 2> lhs(4, 4);
        matrix::sparse_block<2, 2>::block_type lb00;
        lb00(0, 0) = 1;
        lb00(0, 1) = 2;
        lb00(1, 0) = 3;
        lb00(1, 1) = 4;
        matrix::sparse_block<2, 2>::block_type lb01;
        lb01(0, 0) = 5;
        lb01(0, 1) = 6;
        lb01(1, 0) = 7;
        lb01(1, 1) = 8;
        matrix::sparse_block<2, 2>::block_type lb10;
        lb10(0, 0) = 9;
        lb10(0, 1) = 10;
        lb10(1, 0) = 11;
        lb10(1, 1) = 12;
        lhs.blocks()[{ 0, 0 }] = lb00;
        lhs.blocks()[{ 0, 1 }] = lb01;
        lhs.blocks()[{ 1, 0 }] = lb10;

        // Multiply: result.blocks()[{i,j}] = lhs.blocks()[{i,j}] * rhs.diagonal()[j].
        matrix::sparse_block<2, 2> product = matrix::multiply(lhs, rhs);
        REQUIRE(product.rows() == 4);
        REQUIRE(product.cols() == 4);

        // Check block (0,0) = lb00 * db0  (db0 = 2*I) => each entry scaled by 2.
        matrix::sparse_block<2, 2>::block_type expected00 = lb00 * db0;
        matrix::sparse_block<2, 2>::block_type actual00 = product.blocks().at({ 0, 0 });
        for (size_t r = 0; r < 2; ++r) {
            for (size_t c = 0; c < 2; ++c) {
                REQUIRE(is_value_approx(expected00(r, c), actual00(r, c)));
            }
        }

        // Check block (0,1) = lb01 * db1.
        matrix::sparse_block<2, 2>::block_type expected01 = lb01 * db1;
        matrix::sparse_block<2, 2>::block_type actual01 = product.blocks().at({ 0, 1 });
        for (size_t r = 0; r < 2; ++r) {
            for (size_t c = 0; c < 2; ++c) {
                REQUIRE(is_value_approx(expected01(r, c), actual01(r, c)));
            }
        }

        // Check block (1,0) = lb10 * db0.
        matrix::sparse_block<2, 2>::block_type expected10 = lb10 * db0;
        matrix::sparse_block<2, 2>::block_type actual10 = product.blocks().at({ 1, 0 });
        for (size_t r = 0; r < 2; ++r) {
            for (size_t c = 0; c < 2; ++c) {
                REQUIRE(is_value_approx(expected10(r, c), actual10(r, c)));
            }
        }

        // Ensure there is no block (1,1) in result (lhs had no (1,1)).
        REQUIRE(product.blocks().find({ 1, 1 }) == product.blocks().end());
    }

    // Subtract (sparse_block_diagonal - sparse_block) tests.
    {
        // lhs diagonal: two blocks.
        matrix::sparse_block_diagonal<2> lhs;
        lhs.diagonal().resize(2);
        matrix::sparse_block_diagonal<2>::block_type da0;
        da0(0, 0) = 10;
        da0(0, 1) = 0;
        da0(1, 0) = 0;
        da0(1, 1) = 10;
        matrix::sparse_block_diagonal<2>::block_type da1;
        da1(0, 0) = 20;
        da1(0, 1) = 1;
        da1(1, 0) = 2;
        da1(1, 1) = 3;
        lhs.diagonal()[0] = da0;
        lhs.diagonal()[1] = da1;

        // rhs sparse_block with some blocks to subtract.
        matrix::sparse_block<2, 2> rhs(4, 4);
        matrix::sparse_block<2, 2>::block_type r00;
        r00(0, 0) = 1;
        r00(0, 1) = 1;
        r00(1, 0) = 1;
        r00(1, 1) = 1;
        matrix::sparse_block<2, 2>::block_type r11;
        r11(0, 0) = 2;
        r11(0, 1) = 2;
        r11(1, 0) = 2;
        r11(1, 1) = 2;
        // Also an off-diagonal block to ensure it is negated and included.
        matrix::sparse_block<2, 2>::block_type r01;
        r01(0, 0) = 3;
        r01(0, 1) = 3;
        r01(1, 0) = 3;
        r01(1, 1) = 3;
        rhs.blocks()[{ 0, 0 }] = r00;
        rhs.blocks()[{ 1, 1 }] = r11;
        rhs.blocks()[{ 0, 1 }] = r01;

        matrix::sparse_block<2, 2> result = matrix::subtract(lhs, rhs);
        // Result should contain diagonal blocks from lhs minus any rhs blocks on same position and negative of rhs blocks that had no lhs counterpart.
        matrix::sparse_block<2, 2>::block_type expect00;
        expect00(0, 0) = da0(0, 0) - r00(0, 0);
        expect00(0, 1) = da0(0, 1) - r00(0, 1);
        expect00(1, 0) = da0(1, 0) - r00(1, 0);
        expect00(1, 1) = da0(1, 1) - r00(1, 1);
        matrix::sparse_block<2, 2>::block_type actual00 = result.blocks().at({ 0, 0 });
        for (size_t r = 0; r < 2; ++r) {
            for (size_t c = 0; c < 2; ++c) {
                REQUIRE(is_value_approx(expect00(r, c), actual00(r, c)));
            }
        }

        // Check (1,1) = da1 - r11.
        matrix::sparse_block<2, 2>::block_type expect11;
        for (size_t r = 0; r < 2; ++r) {
            for (size_t c = 0; c < 2; ++c) {
                expect11(r, c) = da1(r, c) - r11(r, c);
            }
        }
        matrix::sparse_block<2, 2>::block_type actual11 = result.blocks().at({ 1, 1 });
        for (size_t r = 0; r < 2; ++r) {
            for (size_t c = 0; c < 2; ++c) {
                REQUIRE(is_value_approx(expect11(r, c), actual11(r, c)));
            }
        }

        // Check (0,1) existed only in rhs; result should contain -r01.
        matrix::sparse_block<2, 2>::block_type actual01 = result.blocks().at({ 0, 1 });
        for (size_t r = 0; r < 2; ++r) {
            for (size_t c = 0; c < 2; ++c) {
                REQUIRE(is_value_approx(-r01(r, c), actual01(r, c)));
            }
        }
    }

    return 0;
}
