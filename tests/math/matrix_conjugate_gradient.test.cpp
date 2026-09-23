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

#include "math/matrix_conjugate_gradient.hpp"

#include "core/random_pcg.hpp"
#include "math/matrix_decomposition_cholesky.hpp"

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

static std::vector<double> make_random_positive_definite(core::random_pcg& rng, const int size, const double shift) {
    std::vector<double> b(static_cast<size_t>(size * size));
    for (double& value : b) {
        value = random_signed(rng);
    }
    std::vector<double> a(static_cast<size_t>(size * size), 0.0);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double sum = 0.0;
            for (int k = 0; k < size; ++k) {
                sum += b[static_cast<size_t>((k * size) + i)] * b[static_cast<size_t>((k * size) + j)];
            }
            a[static_cast<size_t>((i * size) + j)] = sum;
        }
    }
    for (int i = 0; i < size; ++i) {
        a[static_cast<size_t>((i * size) + i)] += shift;
    }
    return a;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    core::random_pcg rng;

    {
        const int sizes[] = { 1, 2, 3, 6, 12, 25, 40 };
        for (const int size : sizes) {
            const std::vector<double> a = make_random_positive_definite(rng, size, 1.0);
            std::vector<double> expected(static_cast<size_t>(size));
            for (double& value : expected) {
                value = random_signed(rng);
            }
            std::vector<double> b(static_cast<size_t>(size), 0.0);
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    b[static_cast<size_t>(i)] += a[static_cast<size_t>((i * size) + j)] * expected[static_cast<size_t>(j)];
                }
            }

            const auto apply_operator = [&a, size](const double* input, double* output) {
                for (int i = 0; i < size; ++i) {
                    double sum = 0.0;
                    for (int j = 0; j < size; ++j) {
                        sum += a[static_cast<size_t>((i * size) + j)] * input[j];
                    }
                    output[i] = sum;
                }
            };

            std::vector<double> x(static_cast<size_t>(size), 0.0);
            std::vector<double> scratch(static_cast<size_t>(4 * size), 0.0);
            const math::conjugate_gradient_result result = math::conjugate_gradient(apply_operator, b.data(), x.data(), size, 10 * size, 1e-14, scratch.data());
            REQUIRE(result.converged);
            REQUIRE(result.iterations <= 10 * size);
            for (int i = 0; i < size; ++i) {
                REQUIRE(is_value_approx(x[static_cast<size_t>(i)], expected[static_cast<size_t>(i)], 1e-10));
            }
        }
    }

    {
        constexpr static const int size = 30;
        std::vector<double> a = make_random_positive_definite(rng, size, 1.0);
        std::vector<double> scale(static_cast<size_t>(size));
        for (int i = 0; i < size; ++i) {
            scale[static_cast<size_t>(i)] = ((i % 2) == 0) ? 1.0 : 1000.0;
        }
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                a[static_cast<size_t>((i * size) + j)] *= scale[static_cast<size_t>(i)] * scale[static_cast<size_t>(j)];
            }
        }
        std::vector<double> expected(static_cast<size_t>(size));
        for (double& value : expected) {
            value = random_signed(rng);
        }
        std::vector<double> b(static_cast<size_t>(size), 0.0);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                b[static_cast<size_t>(i)] += a[static_cast<size_t>((i * size) + j)] * expected[static_cast<size_t>(j)];
            }
        }
        const auto apply_operator = [&a](const double* input, double* output) {
            for (int i = 0; i < size; ++i) {
                double sum = 0.0;
                for (int j = 0; j < size; ++j) {
                    sum += a[static_cast<size_t>((i * size) + j)] * input[j];
                }
                output[i] = sum;
            }
        };
        const auto apply_preconditioner = [&a](const double* input, double* output) {
            for (int i = 0; i < size; ++i) {
                output[i] = input[i] / a[static_cast<size_t>((i * size) + i)];
            }
        };

        std::vector<double> scratch(static_cast<size_t>(4 * size), 0.0);
        std::vector<double> x_plain(static_cast<size_t>(size), 0.0);
        const math::conjugate_gradient_result plain = math::conjugate_gradient(apply_operator, b.data(), x_plain.data(), size, 50 * size, 1e-13, scratch.data());
        std::vector<double> x_preconditioned(static_cast<size_t>(size), 0.0);
        const math::conjugate_gradient_result preconditioned = math::conjugate_gradient(apply_operator, apply_preconditioner, b.data(), x_preconditioned.data(), size, 50 * size, 1e-13, scratch.data());

        REQUIRE(plain.converged);
        REQUIRE(preconditioned.converged);
        REQUIRE(preconditioned.iterations < plain.iterations);
        for (int i = 0; i < size; ++i) {
            REQUIRE(is_value_approx(x_plain[static_cast<size_t>(i)], expected[static_cast<size_t>(i)], 1e-7));
            REQUIRE(is_value_approx(x_preconditioned[static_cast<size_t>(i)], expected[static_cast<size_t>(i)], 1e-7));
        }
    }

    {
        constexpr static const int size = 20;
        const std::vector<double> a = make_random_positive_definite(rng, size, 5.0);
        std::vector<double> b(static_cast<size_t>(size));
        for (double& value : b) {
            value = random_signed(rng);
        }
        const auto apply_operator = [&a](const double* input, double* output) {
            for (int i = 0; i < size; ++i) {
                double sum = 0.0;
                for (int j = 0; j < size; ++j) {
                    sum += a[static_cast<size_t>((i * size) + j)] * input[j];
                }
                output[i] = sum;
            }
        };

        bool have_previous = false;
        double previous_objective = 0.0;
        double final_norm = 0.0;
        for (int cap = 1; cap <= size; ++cap) {
            std::vector<double> x(static_cast<size_t>(size), 0.0);
            std::vector<double> scratch(static_cast<size_t>(4 * size), 0.0);
            const math::conjugate_gradient_result result = math::conjugate_gradient(apply_operator, b.data(), x.data(), size, cap, 0.0, scratch.data());
            std::vector<double> image(static_cast<size_t>(size), 0.0);
            apply_operator(x.data(), image.data());
            double norm_squared = 0.0;
            double objective = 0.0;
            for (int i = 0; i < size; ++i) {
                const double residual = b[static_cast<size_t>(i)] - image[static_cast<size_t>(i)];
                norm_squared += residual * residual;
                objective += (0.5 * x[static_cast<size_t>(i)] * image[static_cast<size_t>(i)]) - (b[static_cast<size_t>(i)] * x[static_cast<size_t>(i)]);
            }
            final_norm = std::sqrt(norm_squared);
            REQUIRE(is_value_approx(final_norm, result.final_residual_norm, 1e-6));
            if (have_previous) {
                REQUIRE(objective <= previous_objective + (1e-12 * std::abs(previous_objective)));
            }
            previous_objective = objective;
            have_previous = true;
        }
        REQUIRE(final_norm < 1e-9);
    }

    {
        constexpr static const int size = 17;
        const std::vector<double> a = make_random_positive_definite(rng, size, 0.5);
        std::vector<double> b(static_cast<size_t>(size));
        for (double& value : b) {
            value = random_signed(rng);
        }
        const auto apply_operator = [&a](const double* input, double* output) {
            for (int i = 0; i < size; ++i) {
                double sum = 0.0;
                for (int j = 0; j < size; ++j) {
                    sum += a[static_cast<size_t>((i * size) + j)] * input[j];
                }
                output[i] = sum;
            }
        };
        std::vector<double> scratch(static_cast<size_t>(4 * size), 0.0);
        std::vector<double> first(static_cast<size_t>(size), 0.0);
        std::vector<double> second(static_cast<size_t>(size), 0.0);
        const math::conjugate_gradient_result result_first = math::conjugate_gradient(apply_operator, b.data(), first.data(), size, 100, 1e-12, scratch.data());
        const math::conjugate_gradient_result result_second = math::conjugate_gradient(apply_operator, b.data(), second.data(), size, 100, 1e-12, scratch.data());
        REQUIRE(result_first.iterations == result_second.iterations);
        for (int i = 0; i < size; ++i) {
            REQUIRE(first[static_cast<size_t>(i)] == second[static_cast<size_t>(i)]);
        }
    }

    {
        constexpr static const int size = 15;
        const std::vector<double> a = make_random_positive_definite(rng, size, 2.0);
        std::vector<double> expected(static_cast<size_t>(size));
        for (double& value : expected) {
            value = random_signed(rng);
        }
        std::vector<double> b(static_cast<size_t>(size), 0.0);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                b[static_cast<size_t>(i)] += a[static_cast<size_t>((i * size) + j)] * expected[static_cast<size_t>(j)];
            }
        }
        const auto apply_operator = [&a](const double* input, double* output) {
            for (int i = 0; i < size; ++i) {
                double sum = 0.0;
                for (int j = 0; j < size; ++j) {
                    sum += a[static_cast<size_t>((i * size) + j)] * input[j];
                }
                output[i] = sum;
            }
        };
        std::vector<double> scratch(static_cast<size_t>(4 * size), 0.0);
        std::vector<double> warm(expected);
        for (double& value : warm) {
            value += 0.01 * random_signed(rng);
        }
        const math::conjugate_gradient_result result = math::conjugate_gradient(apply_operator, b.data(), warm.data(), size, 10 * size, 1e-14, scratch.data());
        REQUIRE(result.converged);
        for (int i = 0; i < size; ++i) {
            REQUIRE(is_value_approx(warm[static_cast<size_t>(i)], expected[static_cast<size_t>(i)], 1e-9));
        }
    }

    {
        constexpr static const int size = 8;
        std::vector<double> a(static_cast<size_t>(size * size), 0.0);
        for (int r = 0; r < 3; ++r) {
            std::vector<double> vector(static_cast<size_t>(size));
            for (double& value : vector) {
                value = random_signed(rng);
            }
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    a[static_cast<size_t>((i * size) + j)] += vector[static_cast<size_t>(i)] * vector[static_cast<size_t>(j)];
                }
            }
        }
        std::vector<double> b(static_cast<size_t>(size));
        for (double& value : b) {
            value = random_signed(rng);
        }
        const auto apply_operator = [&a](const double* input, double* output) {
            for (int i = 0; i < size; ++i) {
                double sum = 0.0;
                for (int j = 0; j < size; ++j) {
                    sum += a[static_cast<size_t>((i * size) + j)] * input[j];
                }
                output[i] = sum;
            }
        };
        std::vector<double> x(static_cast<size_t>(size), 0.0);
        std::vector<double> scratch(static_cast<size_t>(4 * size), 0.0);
        const math::conjugate_gradient_result result = math::conjugate_gradient(apply_operator, b.data(), x.data(), size, 100, 1e-14, scratch.data());
        REQUIRE(!result.converged);
        REQUIRE(result.iterations <= 100);
        for (int i = 0; i < size; ++i) {
            REQUIRE(std::isfinite(x[static_cast<size_t>(i)]));
        }
        std::vector<double> image(static_cast<size_t>(size), 0.0);
        apply_operator(x.data(), image.data());
        double objective = 0.0;
        for (int i = 0; i < size; ++i) {
            objective += (0.5 * x[static_cast<size_t>(i)] * image[static_cast<size_t>(i)]) - (b[static_cast<size_t>(i)] * x[static_cast<size_t>(i)]);
        }
        REQUIRE(objective <= 0.0);
    }

    {
        const auto apply_operator = [](const double* input, double* output) {
            static_cast<void>(input);
            static_cast<void>(output);
        };
        const math::conjugate_gradient_result empty = math::conjugate_gradient(apply_operator, static_cast<const double*>(nullptr), static_cast<double*>(nullptr), 0, 10, 1e-12, static_cast<double*>(nullptr));
        REQUIRE(empty.iterations == 0);
        REQUIRE(empty.converged);

        constexpr static const int size = 4;
        const auto identity_operator = [](const double* input, double* output) {
            for (int i = 0; i < size; ++i) {
                output[i] = input[i];
            }
        };
        std::vector<double> b(static_cast<size_t>(size), 0.0);
        std::vector<double> x(static_cast<size_t>(size), 0.0);
        std::vector<double> scratch(static_cast<size_t>(4 * size), 0.0);
        const math::conjugate_gradient_result zero_rhs = math::conjugate_gradient(identity_operator, b.data(), x.data(), size, 10, 1e-12, scratch.data());
        REQUIRE(zero_rhs.iterations == 0);
        REQUIRE(zero_rhs.converged);
        for (int i = 0; i < size; ++i) {
            REQUIRE(x[static_cast<size_t>(i)] == 0.0);
        }
    }

    {
        constexpr static const int size = 10;
        const std::vector<double> a_double = make_random_positive_definite(rng, size, 3.0);
        std::vector<float> a(a_double.size());
        for (size_t i = 0; i < a_double.size(); ++i) {
            a[i] = static_cast<float>(a_double[i]);
        }
        std::vector<float> expected(static_cast<size_t>(size));
        for (float& value : expected) {
            value = static_cast<float>(random_signed(rng));
        }
        std::vector<float> b(static_cast<size_t>(size), 0.0f);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                b[static_cast<size_t>(i)] += a[static_cast<size_t>((i * size) + j)] * expected[static_cast<size_t>(j)];
            }
        }
        const auto apply_operator = [&a](const float* input, float* output) {
            for (int i = 0; i < size; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < size; ++j) {
                    sum += a[static_cast<size_t>((i * size) + j)] * input[j];
                }
                output[i] = sum;
            }
        };
        std::vector<float> x(static_cast<size_t>(size), 0.0f);
        std::vector<float> scratch(static_cast<size_t>(4 * size), 0.0f);
        const math::conjugate_gradient_result result = math::conjugate_gradient(apply_operator, b.data(), x.data(), size, 10 * size, 1e-6f, scratch.data());
        REQUIRE(result.converged);
        for (int i = 0; i < size; ++i) {
            REQUIRE(is_value_approx(static_cast<double>(x[static_cast<size_t>(i)]), static_cast<double>(expected[static_cast<size_t>(i)]), 1e-3));
        }
    }

    return EXIT_SUCCESS;
}
