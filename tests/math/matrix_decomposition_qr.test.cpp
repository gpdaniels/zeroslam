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

#include "math/matrix_decomposition_qr.hpp"

#include "core/random_pcg.hpp"
#include "math/matrix.hpp"
#include "math/matrix_decomposition_singular_value.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define REQUIRE(ASSERTION) static_cast<void>((ASSERTION) || (std::fprintf(stderr, "ERROR[%d]: Requirement '%s' failed.\n", __LINE__, #ASSERTION), __builtin_trap(), 0))

static inline double random_signed(core::random_pcg& rng) {
    return (static_cast<double>(rng.get_random_raw() % 2000000u) / 1000000.0) - 1.0;
}

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

    core::random_pcg rng;

    {
        const int shapes[][2] = { { 1, 1 }, { 2, 1 }, { 1, 2 }, { 3, 3 }, { 5, 3 }, { 3, 5 }, { 8, 4 }, { 12, 7 }, { 20, 20 }, { 23, 4 } };
        for (const auto& shape : shapes) {
            const int rows = shape[0];
            const int cols = shape[1];
            const int diagonal = (rows < cols) ? rows : cols;
            std::vector<double> a(static_cast<size_t>(rows * cols));
            for (double& value : a) {
                value = random_signed(rng);
            }
            for (int thin_pass = 0; thin_pass < 2; ++thin_pass) {
                const bool thin = (thin_pass == 1);
                const int q_cols = thin ? diagonal : rows;
                std::vector<double> q(static_cast<size_t>(rows * q_cols), 0.0);
                std::vector<double> r(static_cast<size_t>(rows * cols), 0.0);
                std::vector<double> workspace(static_cast<size_t>((rows * cols) + rows + diagonal), 0.0);
                math::decompose_qr_householder(a.data(), rows, cols, q.data(), r.data(), thin, workspace.data());

                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        if (j < i) {
                            REQUIRE(r[static_cast<size_t>((i * cols) + j)] == 0.0);
                        }
                    }
                }
                for (int i = 0; i < q_cols; ++i) {
                    for (int j = 0; j < q_cols; ++j) {
                        double dot = 0.0;
                        for (int k = 0; k < rows; ++k) {
                            dot += q[static_cast<size_t>((k * q_cols) + i)] * q[static_cast<size_t>((k * q_cols) + j)];
                        }
                        REQUIRE(is_value_approx(dot, (i == j) ? 1.0 : 0.0, 1e-12));
                    }
                }
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        double sum = 0.0;
                        for (int k = 0; k < q_cols; ++k) {
                            sum += q[static_cast<size_t>((i * q_cols) + k)] * r[static_cast<size_t>((k * cols) + j)];
                        }
                        REQUIRE(is_value_approx(sum, a[static_cast<size_t>((i * cols) + j)], 1e-12));
                    }
                }
            }
        }
    }

    {
        const int rows = 6;
        const int cols = 4;
        std::vector<double> a(static_cast<size_t>(rows * cols), 0.0);
        for (int i = 0; i < rows; ++i) {
            const double value = random_signed(rng);
            a[static_cast<size_t>((i * cols) + 0)] = value;
            a[static_cast<size_t>((i * cols) + 1)] = 2.0 * value;
            a[static_cast<size_t>((i * cols) + 2)] = 0.0;
            a[static_cast<size_t>((i * cols) + 3)] = random_signed(rng);
        }
        std::vector<double> q(static_cast<size_t>(rows * rows), 0.0);
        std::vector<double> r(static_cast<size_t>(rows * cols), 0.0);
        std::vector<double> workspace(static_cast<size_t>((rows * cols) + rows + cols), 0.0);
        math::decompose_qr_householder(a.data(), rows, cols, q.data(), r.data(), false, workspace.data());
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double sum = 0.0;
                for (int k = 0; k < rows; ++k) {
                    sum += q[static_cast<size_t>((i * rows) + k)] * r[static_cast<size_t>((k * cols) + j)];
                }
                REQUIRE(is_value_approx(sum, a[static_cast<size_t>((i * cols) + j)], 1e-12));
            }
        }
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < rows; ++j) {
                double dot = 0.0;
                for (int k = 0; k < rows; ++k) {
                    dot += q[static_cast<size_t>((k * rows) + i)] * q[static_cast<size_t>((k * rows) + j)];
                }
                REQUIRE(is_value_approx(dot, (i == j) ? 1.0 : 0.0, 1e-12));
            }
        }
        std::vector<double> rhs(static_cast<size_t>(cols), 1.0);
        std::vector<double> solution(static_cast<size_t>(cols), 12345.0);
        REQUIRE(!math::solve_upper_triangular(r.data(), cols, cols, rhs.data(), solution.data()));
        for (int i = 0; i < cols; ++i) {
            REQUIRE(solution[static_cast<size_t>(i)] == 0.0);
        }
    }

    {
        const int rows = 9;
        const int cols = 4;
        std::vector<double> a(static_cast<size_t>(rows * cols));
        for (double& value : a) {
            value = random_signed(rng);
        }
        std::vector<double> b(static_cast<size_t>(rows));
        for (double& value : b) {
            value = random_signed(rng);
        }

        std::vector<double> factored(a);
        std::vector<double> betas(static_cast<size_t>(cols), 0.0);
        std::vector<double> workspace(static_cast<size_t>(rows), 0.0);
        math::decompose_qr_householder_in_place(factored.data(), rows, cols, betas.data(), workspace.data());
        std::vector<double> qt_b(b);
        math::apply_householder_q_transpose_left(factored.data(), cols, betas.data(), cols, qt_b.data(), rows, 1, workspace.data());
        std::vector<double> solution_qr(static_cast<size_t>(cols), 0.0);
        REQUIRE(math::solve_upper_triangular(factored.data(), cols, cols, qt_b.data(), solution_qr.data()));

        std::vector<double> u(static_cast<size_t>(rows * rows), 0.0);
        std::vector<double> s(static_cast<size_t>(rows * cols), 0.0);
        std::vector<double> vt(static_cast<size_t>(cols * cols), 0.0);
        REQUIRE(math::decompose_singular_value(a.data(), static_cast<size_t>(cols), static_cast<size_t>(rows), u.data(), s.data(), vt.data()));
        std::vector<double> solution_svd(static_cast<size_t>(cols), 0.0);
        for (int j = 0; j < cols; ++j) {
            double ut_b = 0.0;
            for (int i = 0; i < rows; ++i) {
                ut_b += u[static_cast<size_t>((i * rows) + j)] * b[static_cast<size_t>(i)];
            }
            const double singular_value = s[static_cast<size_t>((j * cols) + j)];
            REQUIRE(std::abs(singular_value) > 1e-9);
            const double scaled = ut_b / singular_value;
            for (int k = 0; k < cols; ++k) {
                solution_svd[static_cast<size_t>(k)] += vt[static_cast<size_t>((j * cols) + k)] * scaled;
            }
        }
        for (int i = 0; i < cols; ++i) {
            REQUIRE(is_value_approx(solution_qr[static_cast<size_t>(i)], solution_svd[static_cast<size_t>(i)], 1e-9));
        }

        for (int j = 0; j < cols; ++j) {
            double gradient = 0.0;
            for (int i = 0; i < rows; ++i) {
                double residual = -b[static_cast<size_t>(i)];
                for (int k = 0; k < cols; ++k) {
                    residual += a[static_cast<size_t>((i * cols) + k)] * solution_qr[static_cast<size_t>(k)];
                }
                gradient += a[static_cast<size_t>((i * cols) + j)] * residual;
            }
            REQUIRE(is_value_approx(gradient, 0.0, 1e-10));
        }
    }

    {
        const double pairs[][2] = {
            { 3.0, 4.0 },
            { -3.0, 4.0 },
            { 3.0, -4.0 },
            { -3.0, -4.0 },
            { 1.0, 0.0 },
            { -1.0, 0.0 },
            { 0.0, 1.0 },
            { 0.0, -1.0 },
            { 0.0, 0.0 },
            { 1e-160, 1e-160 },
            { 1e160, 1.0 },
            { 1.0, 1e160 }
        };
        for (const auto& pair : pairs) {
            double c = 0.0;
            double s = 0.0;
            const double r = math::givens(pair[0], pair[1], c, s);
            REQUIRE(std::isfinite(c));
            REQUIRE(std::isfinite(s));
            REQUIRE(std::isfinite(r));
            REQUIRE(is_value_approx((c * c) + (s * s), 1.0, 1e-14));
            REQUIRE(is_value_approx((c * pair[0]) + (s * pair[1]), r, 1e-13));
            REQUIRE(is_value_approx((c * pair[1]) - (s * pair[0]), 0.0, 1e-13));
        }

        const int rows = 4;
        const int cols = 5;
        std::vector<double> original(static_cast<size_t>(rows * cols));
        for (double& value : original) {
            value = random_signed(rng);
        }
        std::vector<double> working(original);
        double c = 0.0;
        double s = 0.0;
        math::givens(working[static_cast<size_t>((1 * cols) + 2)], working[static_cast<size_t>((3 * cols) + 2)], c, s);
        math::apply_givens_left(working.data(), cols, 1, 3, 0, cols, c, s);
        REQUIRE(is_value_approx(working[static_cast<size_t>((3 * cols) + 2)], 0.0, 1e-14));
        math::apply_givens_left(working.data(), cols, 1, 3, 0, cols, c, -s);
        for (size_t i = 0; i < original.size(); ++i) {
            REQUIRE(is_value_approx(working[i], original[i], 1e-14));
        }
    }

    {
        const int rows = 10;
        const int cols = 3;
        const int extra_cols = 4;
        std::vector<double> a(static_cast<size_t>(rows * cols));
        for (double& value : a) {
            value = random_signed(rng);
        }
        std::vector<double> extra(static_cast<size_t>(rows * extra_cols));
        for (double& value : extra) {
            value = random_signed(rng);
        }

        std::vector<double> q(static_cast<size_t>(rows * rows), 0.0);
        std::vector<double> r(static_cast<size_t>(rows * cols), 0.0);
        std::vector<double> workspace(static_cast<size_t>((rows * cols) + rows + cols), 0.0);
        math::decompose_qr_householder(a.data(), rows, cols, q.data(), r.data(), false, workspace.data());

        std::vector<double> factored(a);
        std::vector<double> betas(static_cast<size_t>(cols), 0.0);
        std::vector<double> scratch(static_cast<size_t>(rows), 0.0);
        math::decompose_qr_householder_in_place(factored.data(), rows, cols, betas.data(), scratch.data());
        std::vector<double> applied(extra);
        math::apply_householder_q_transpose_left(factored.data(), cols, betas.data(), cols, applied.data(), rows, extra_cols, scratch.data());

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < extra_cols; ++j) {
                double expected = 0.0;
                for (int k = 0; k < rows; ++k) {
                    expected += q[static_cast<size_t>((k * rows) + i)] * extra[static_cast<size_t>((k * extra_cols) + j)];
                }
                REQUIRE(is_value_approx(applied[static_cast<size_t>((i * extra_cols) + j)], expected, 1e-12));
            }
        }
    }

    {
        const int size = 5;
        std::vector<double> r(static_cast<size_t>(size * size), 0.0);
        std::vector<double> expected(static_cast<size_t>(size));
        for (int i = 0; i < size; ++i) {
            expected[static_cast<size_t>(i)] = random_signed(rng);
            for (int j = i; j < size; ++j) {
                r[static_cast<size_t>((i * size) + j)] = random_signed(rng);
            }
            r[static_cast<size_t>((i * size) + i)] += (r[static_cast<size_t>((i * size) + i)] >= 0.0) ? 2.0 : -2.0;
        }
        std::vector<double> rhs(static_cast<size_t>(size), 0.0);
        for (int i = 0; i < size; ++i) {
            for (int j = i; j < size; ++j) {
                rhs[static_cast<size_t>(i)] += r[static_cast<size_t>((i * size) + j)] * expected[static_cast<size_t>(j)];
            }
        }
        std::vector<double> solution(static_cast<size_t>(size), 0.0);
        REQUIRE(math::solve_upper_triangular(r.data(), size, size, rhs.data(), solution.data()));
        for (int i = 0; i < size; ++i) {
            REQUIRE(is_value_approx(solution[static_cast<size_t>(i)], expected[static_cast<size_t>(i)], 1e-10));
        }

        std::vector<float> r_float(r.size());
        for (size_t i = 0; i < r.size(); ++i) {
            r_float[i] = static_cast<float>(r[i]);
        }
        std::vector<float> rhs_float(rhs.size());
        for (size_t i = 0; i < rhs.size(); ++i) {
            rhs_float[i] = static_cast<float>(rhs[i]);
        }
        std::vector<float> solution_float(static_cast<size_t>(size), 0.0f);
        REQUIRE(math::solve_upper_triangular(r_float.data(), size, size, rhs_float.data(), solution_float.data()));
        for (int i = 0; i < size; ++i) {
            REQUIRE(is_value_approx(static_cast<double>(solution_float[static_cast<size_t>(i)]), expected[static_cast<size_t>(i)], 1e-4));
        }
    }

    {
        const int rows = 14;
        const int cols = 6;
        std::vector<float> a(static_cast<size_t>(rows * cols));
        for (float& value : a) {
            value = static_cast<float>(random_signed(rng));
        }
        std::vector<float> q(static_cast<size_t>(rows * rows), 0.0f);
        std::vector<float> r(static_cast<size_t>(rows * cols), 0.0f);
        std::vector<float> workspace(static_cast<size_t>((rows * cols) + rows + cols), 0.0f);
        math::decompose_qr_householder(a.data(), rows, cols, q.data(), r.data(), false, workspace.data());
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < rows; ++k) {
                    sum += q[static_cast<size_t>((i * rows) + k)] * r[static_cast<size_t>((k * cols) + j)];
                }
                REQUIRE(is_value_approx(static_cast<double>(sum), static_cast<double>(a[static_cast<size_t>((i * cols) + j)]), 1e-5));
            }
        }
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < rows; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < rows; ++k) {
                    dot += q[static_cast<size_t>((k * rows) + i)] * q[static_cast<size_t>((k * rows) + j)];
                }
                REQUIRE(is_value_approx(static_cast<double>(dot), (i == j) ? 1.0 : 0.0, 1e-5));
            }
        }
    }

    {
        const int rows = 11;
        const int cols = 5;
        std::vector<double> a(static_cast<size_t>(rows * cols));
        for (double& value : a) {
            value = random_signed(rng);
        }
        std::vector<double> first(a);
        std::vector<double> second(a);
        std::vector<double> betas_first(static_cast<size_t>(cols), 0.0);
        std::vector<double> betas_second(static_cast<size_t>(cols), 0.0);
        std::vector<double> workspace(static_cast<size_t>(rows), 0.0);
        math::decompose_qr_householder_in_place(first.data(), rows, cols, betas_first.data(), workspace.data());
        math::decompose_qr_householder_in_place(second.data(), rows, cols, betas_second.data(), workspace.data());
        for (size_t i = 0; i < first.size(); ++i) {
            REQUIRE(first[i] == second[i]);
        }
        for (size_t i = 0; i < betas_first.size(); ++i) {
            REQUIRE(betas_first[i] == betas_second[i]);
        }
    }

    return EXIT_SUCCESS;
}
