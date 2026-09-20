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

#include "optimisation/landmark_block.hpp"

#include "core/random_pcg.hpp"
#include "math/matrix.hpp"

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

struct synthetic_observation final {
    int pose_slot;
    double jacobian_pose[12];
    double jacobian_landmark[6];
    double residual[2];
};

class reference_block final {
public:
    int pose_count;
    int pose_columns;
    std::vector<double> hessian_pose;
    std::vector<double> hessian_pose_landmark;
    std::vector<double> hessian_landmark;
    std::vector<double> gradient_pose;
    std::vector<double> gradient_landmark;

public:
    reference_block(const std::vector<synthetic_observation>& observations, const int poses)
        : pose_count(poses)
        , pose_columns(6 * poses)
        , hessian_pose(static_cast<size_t>(6 * poses * 6 * poses), 0.0)
        , hessian_pose_landmark(static_cast<size_t>(6 * poses * 3), 0.0)
        , hessian_landmark(9u, 0.0)
        , gradient_pose(static_cast<size_t>(6 * poses), 0.0)
        , gradient_landmark(3u, 0.0) {
        for (const synthetic_observation& observation : observations) {
            for (int r = 0; r < 2; ++r) {
                std::vector<double> row_pose(static_cast<size_t>(this->pose_columns), 0.0);
                if (observation.pose_slot >= 0) {
                    for (int c = 0; c < 6; ++c) {
                        row_pose[static_cast<size_t>((6 * observation.pose_slot) + c)] = observation.jacobian_pose[(6 * r) + c];
                    }
                }
                const double* const row_landmark = observation.jacobian_landmark + (3 * r);
                const double residual = observation.residual[r];
                for (int i = 0; i < this->pose_columns; ++i) {
                    for (int j = 0; j < this->pose_columns; ++j) {
                        this->hessian_pose[static_cast<size_t>((i * this->pose_columns) + j)] += row_pose[static_cast<size_t>(i)] * row_pose[static_cast<size_t>(j)];
                    }
                    for (int j = 0; j < 3; ++j) {
                        this->hessian_pose_landmark[static_cast<size_t>((i * 3) + j)] += row_pose[static_cast<size_t>(i)] * row_landmark[j];
                    }
                    this->gradient_pose[static_cast<size_t>(i)] -= row_pose[static_cast<size_t>(i)] * residual;
                }
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        this->hessian_landmark[static_cast<size_t>((i * 3) + j)] += row_landmark[i] * row_landmark[j];
                    }
                    this->gradient_landmark[static_cast<size_t>(i)] -= row_landmark[i] * residual;
                }
            }
        }
    }

    math::matrix<double, 3, 3> damped_landmark_inverse(const double lambda) const {
        math::matrix<double, 3, 3> damped;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                damped[static_cast<size_t>(i)][static_cast<size_t>(j)] = this->hessian_landmark[static_cast<size_t>((i * 3) + j)] + ((i == j) ? lambda : 0.0);
            }
        }
        math::matrix<double, 3, 3> inverse;
        REQUIRE(math::invert(damped, inverse));
        return inverse;
    }

    std::vector<double> schur_complement(const double lambda) const {
        const math::matrix<double, 3, 3> inverse = this->damped_landmark_inverse(lambda);
        std::vector<double> result(this->hessian_pose);
        for (int i = 0; i < this->pose_columns; ++i) {
            for (int j = 0; j < this->pose_columns; ++j) {
                double correction = 0.0;
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        correction += this->hessian_pose_landmark[static_cast<size_t>((i * 3) + a)] * inverse[static_cast<size_t>(a)][static_cast<size_t>(b)] * this->hessian_pose_landmark[static_cast<size_t>((j * 3) + b)];
                    }
                }
                result[static_cast<size_t>((i * this->pose_columns) + j)] -= correction;
            }
        }
        return result;
    }

    std::vector<double> schur_gradient(const double lambda) const {
        const math::matrix<double, 3, 3> inverse = this->damped_landmark_inverse(lambda);
        std::vector<double> result(this->gradient_pose);
        for (int i = 0; i < this->pose_columns; ++i) {
            double correction = 0.0;
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    correction += this->hessian_pose_landmark[static_cast<size_t>((i * 3) + a)] * inverse[static_cast<size_t>(a)][static_cast<size_t>(b)] * this->gradient_landmark[static_cast<size_t>(b)];
                }
            }
            result[static_cast<size_t>(i)] -= correction;
        }
        return result;
    }

    std::vector<double> back_substitution(const double lambda, const std::vector<double>& increment_pose) const {
        const math::matrix<double, 3, 3> inverse = this->damped_landmark_inverse(lambda);
        double right_hand_side[3];
        for (int a = 0; a < 3; ++a) {
            double value = this->gradient_landmark[static_cast<size_t>(a)];
            for (int i = 0; i < this->pose_columns; ++i) {
                value -= this->hessian_pose_landmark[static_cast<size_t>((i * 3) + a)] * increment_pose[static_cast<size_t>(i)];
            }
            right_hand_side[a] = value;
        }
        std::vector<double> result(3u, 0.0);
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                result[static_cast<size_t>(a)] += inverse[static_cast<size_t>(a)][static_cast<size_t>(b)] * right_hand_side[b];
            }
        }
        return result;
    }
};

static optimisation::landmark_block<double> build_block(const std::vector<synthetic_observation>& observations, const int poses, const bool landmark_is_free) {
    std::vector<int> pose_offsets(static_cast<size_t>(poses));
    for (int i = 0; i < poses; ++i) {
        pose_offsets[static_cast<size_t>(i)] = 6 * i;
    }
    optimisation::landmark_block<double> block;
    block.configure(static_cast<int>(observations.size()), 3, landmark_is_free, 6 * poses, pose_offsets.data(), poses);
    const double identity[4] = { 1.0, 0.0, 0.0, 1.0 };
    for (size_t i = 0; i < observations.size(); ++i) {
        block.set_observation(static_cast<int>(i), observations[i].pose_slot, observations[i].jacobian_pose, observations[i].jacobian_landmark, observations[i].residual, identity);
    }
    return block;
}

static std::vector<synthetic_observation> make_observations(core::random_pcg& rng, const int count, const int poses, const int fixed_pose_observations) {
    std::vector<synthetic_observation> observations(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        synthetic_observation& observation = observations[static_cast<size_t>(i)];
        observation.pose_slot = (i < fixed_pose_observations) ? -1 : ((poses > 0) ? ((i - fixed_pose_observations) % poses) : -1);
        for (double& value : observation.jacobian_pose) {
            value = random_signed(rng);
        }
        for (double& value : observation.jacobian_landmark) {
            value = random_signed(rng);
        }
        for (double& value : observation.residual) {
            value = random_signed(rng);
        }
    }
    return observations;
}

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    core::random_pcg rng;

    {
        const int shapes[][3] = {
            { 2, 2, 0 },
            { 3, 3, 0 },
            { 5, 4, 0 },
            { 10, 5, 0 },
            { 20, 8, 0 },
            { 4, 2, 2 },
            { 6, 1, 3 },
            { 3, 0, 3 },
            { 1, 1, 0 },
            { 2, 1, 1 }
        };
        const double lambdas[] = { 1e-6, 1e-2, 1.0, 1e2 };

        for (const auto& shape : shapes) {
            const int observation_count = shape[0];
            const int poses = shape[1];
            const std::vector<synthetic_observation> observations = make_observations(rng, observation_count, poses, shape[2]);
            const reference_block reference(observations, poses);
            optimisation::landmark_block<double> block = build_block(observations, poses, true);

            {
                std::vector<double> gradient(static_cast<size_t>((6 * poses) + 3), 0.0);
                block.accumulate_gradient(gradient.data());
                for (int i = 0; i < 6 * poses; ++i) {
                    REQUIRE(is_value_approx(gradient[static_cast<size_t>(i)], reference.gradient_pose[static_cast<size_t>(i)], 1e-12));
                }
                for (int i = 0; i < 3; ++i) {
                    REQUIRE(is_value_approx(gradient[static_cast<size_t>((6 * poses) + i)], reference.gradient_landmark[static_cast<size_t>(i)], 1e-12));
                }
            }

            block.perform_qr();

            if (observation_count >= 2) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        double sum = 0.0;
                        for (int k = 0; k < 3; ++k) {
                            sum += block.row(k)[(6 * poses) + i] * block.row(k)[(6 * poses) + j];
                        }
                        REQUIRE(is_value_approx(sum, reference.hessian_landmark[static_cast<size_t>((i * 3) + j)], 1e-11));
                    }
                }
            }

            for (const double lambda : lambdas) {
                block.set_damping(lambda);

                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        double sum = 0.0;
                        for (int k = 0; k < 3; ++k) {
                            sum += block.row(k)[(6 * poses) + i] * block.row(k)[(6 * poses) + j];
                        }
                        const double expected = reference.hessian_landmark[static_cast<size_t>((i * 3) + j)] + ((i == j) ? lambda : 0.0);
                        REQUIRE(is_value_approx(sum, expected, 1e-10));
                    }
                }
                for (int r = 3; r < block.rows(); ++r) {
                    for (int c = 0; c < 3; ++c) {
                        REQUIRE(block.row(r)[(6 * poses) + c] == 0.0);
                    }
                }

                const std::vector<double> expected_schur = reference.schur_complement(lambda);
                for (int column = 0; column < 6 * poses; ++column) {
                    std::vector<double> basis(static_cast<size_t>(6 * poses), 0.0);
                    basis[static_cast<size_t>(column)] = 1.0;
                    std::vector<double> image(static_cast<size_t>(6 * poses), 0.0);
                    block.add_operator_product(basis.data(), image.data());
                    for (int row = 0; row < 6 * poses; ++row) {
                        REQUIRE(is_value_approx(image[static_cast<size_t>(row)], expected_schur[static_cast<size_t>((row * 6 * poses) + column)], 1e-9));
                    }
                }

                {
                    const std::vector<double> expected_gradient = reference.schur_gradient(lambda);
                    std::vector<double> gradient(static_cast<size_t>(6 * poses), 0.0);
                    block.add_right_hand_side(gradient.data());
                    for (int i = 0; i < 6 * poses; ++i) {
                        REQUIRE(is_value_approx(gradient[static_cast<size_t>(i)], expected_gradient[static_cast<size_t>(i)], 1e-9));
                    }
                }

                {
                    std::vector<double> blocks(static_cast<size_t>(poses * 36), 0.0);
                    block.add_reduced_diagonal_blocks(blocks.data());
                    for (int p = 0; p < poses; ++p) {
                        for (int i = 0; i < 6; ++i) {
                            for (int j = 0; j < 6; ++j) {
                                const int row = (6 * p) + i;
                                const int column = (6 * p) + j;
                                REQUIRE(is_value_approx(blocks[static_cast<size_t>((p * 36) + (6 * i) + j)], expected_schur[static_cast<size_t>((row * 6 * poses) + column)], 1e-9));
                            }
                        }
                    }
                }

                {
                    std::vector<double> increment(static_cast<size_t>(6 * poses));
                    for (double& value : increment) {
                        value = random_signed(rng);
                    }
                    double delta[3] = { 0.0, 0.0, 0.0 };
                    REQUIRE(block.back_substitute(increment.data(), delta));
                    const std::vector<double> expected_delta = reference.back_substitution(lambda, increment);
                    for (int i = 0; i < 3; ++i) {
                        REQUIRE(is_value_approx(delta[static_cast<size_t>(i)], expected_delta[static_cast<size_t>(i)], 1e-9));
                    }
                }
            }
        }
    }

    {
        const std::vector<synthetic_observation> observations = make_observations(rng, 7, 4, 1);
        optimisation::landmark_block<double> block = build_block(observations, 4, true);
        block.perform_qr();
        std::vector<double> snapshot(static_cast<size_t>(block.rows() * block.cols()));
        for (int r = 0; r < block.rows(); ++r) {
            for (int c = 0; c < block.cols(); ++c) {
                snapshot[static_cast<size_t>((r * block.cols()) + c)] = block.row(r)[c];
            }
        }
        for (const double lambda : { 1e-8, 1.0, 1e6, 1e-3 }) {
            block.set_damping(lambda);
            block.clear_damping();
            for (int r = 0; r < block.rows(); ++r) {
                for (int c = 0; c < block.cols(); ++c) {
                    REQUIRE(block.row(r)[c] == snapshot[static_cast<size_t>((r * block.cols()) + c)]);
                }
            }
        }
        block.set_damping(0.25);
        std::vector<double> damped(static_cast<size_t>(block.rows() * block.cols()));
        for (int r = 0; r < block.rows(); ++r) {
            for (int c = 0; c < block.cols(); ++c) {
                damped[static_cast<size_t>((r * block.cols()) + c)] = block.row(r)[c];
            }
        }
        block.set_damping(0.25);
        for (int r = 0; r < block.rows(); ++r) {
            for (int c = 0; c < block.cols(); ++c) {
                REQUIRE(block.row(r)[c] == damped[static_cast<size_t>((r * block.cols()) + c)]);
            }
        }
    }

    {
        const int poses = 3;
        const std::vector<synthetic_observation> observations = make_observations(rng, 6, poses, 0);
        const reference_block reference(observations, poses);
        optimisation::landmark_block<double> block = build_block(observations, poses, false);
        REQUIRE(!block.is_marginalised());
        block.perform_qr();
        block.set_damping(1.0);
        REQUIRE(block.rows() == 12);
        REQUIRE(block.reduced_rows() == 12);
        for (int column = 0; column < 6 * poses; ++column) {
            std::vector<double> basis(static_cast<size_t>(6 * poses), 0.0);
            basis[static_cast<size_t>(column)] = 1.0;
            std::vector<double> image(static_cast<size_t>(6 * poses), 0.0);
            block.add_operator_product(basis.data(), image.data());
            for (int row = 0; row < 6 * poses; ++row) {
                REQUIRE(is_value_approx(image[static_cast<size_t>(row)], reference.hessian_pose[static_cast<size_t>((row * 6 * poses) + column)], 1e-10));
            }
        }
        std::vector<double> gradient(static_cast<size_t>(6 * poses), 0.0);
        block.add_right_hand_side(gradient.data());
        for (int i = 0; i < 6 * poses; ++i) {
            REQUIRE(is_value_approx(gradient[static_cast<size_t>(i)], reference.gradient_pose[static_cast<size_t>(i)], 1e-10));
        }
        double delta[3] = { 1.0, 2.0, 3.0 };
        std::vector<double> increment(static_cast<size_t>(6 * poses), 0.5);
        REQUIRE(block.back_substitute(increment.data(), delta));
        REQUIRE(delta[0] == 0.0);
        REQUIRE(delta[1] == 0.0);
        REQUIRE(delta[2] == 0.0);
    }

    {
        const int poses = 1;
        std::vector<synthetic_observation> observations = make_observations(rng, 3, poses, 0);
        const double whitening[4] = { 0.75, 0.25, 0.25, 1.5 };
        std::vector<synthetic_observation> whitened = observations;
        for (synthetic_observation& observation : whitened) {
            synthetic_observation source = observation;
            for (int r = 0; r < 2; ++r) {
                for (int c = 0; c < 6; ++c) {
                    observation.jacobian_pose[(6 * r) + c] = (whitening[(2 * r) + 0] * source.jacobian_pose[c]) + (whitening[(2 * r) + 1] * source.jacobian_pose[6 + c]);
                }
                for (int c = 0; c < 3; ++c) {
                    observation.jacobian_landmark[(3 * r) + c] = (whitening[(2 * r) + 0] * source.jacobian_landmark[c]) + (whitening[(2 * r) + 1] * source.jacobian_landmark[3 + c]);
                }
                observation.residual[r] = (whitening[(2 * r) + 0] * source.residual[0]) + (whitening[(2 * r) + 1] * source.residual[1]);
            }
        }
        const reference_block reference(whitened, poses);

        std::vector<int> pose_offsets{ 0 };
        optimisation::landmark_block<double> block;
        block.configure(static_cast<int>(observations.size()), 3, true, 6, pose_offsets.data(), poses);
        for (size_t i = 0; i < observations.size(); ++i) {
            block.set_observation(static_cast<int>(i), observations[i].pose_slot, observations[i].jacobian_pose, observations[i].jacobian_landmark, observations[i].residual, whitening);
        }
        std::vector<double> gradient(9u, 0.0);
        block.accumulate_gradient(gradient.data());
        for (int i = 0; i < 6; ++i) {
            REQUIRE(is_value_approx(gradient[static_cast<size_t>(i)], reference.gradient_pose[static_cast<size_t>(i)], 1e-12));
        }
        for (int i = 0; i < 3; ++i) {
            REQUIRE(is_value_approx(gradient[static_cast<size_t>(6 + i)], reference.gradient_landmark[static_cast<size_t>(i)], 1e-12));
        }
    }

    {
        const int poses = 3;
        const std::vector<synthetic_observation> observations = make_observations(rng, 6, poses, 0);
        const reference_block reference(observations, poses);
        const int offsets[3] = { 18, 6, 42 };
        const int total = 48;
        std::vector<int> sorted_offsets{ offsets[0], offsets[1], offsets[2] };
        int order[3] = { 1, 0, 2 };
        sorted_offsets[0] = offsets[order[0]];
        sorted_offsets[1] = offsets[order[1]];
        sorted_offsets[2] = offsets[order[2]];
        optimisation::landmark_block<double> block;
        block.configure(static_cast<int>(observations.size()), 3, true, total, sorted_offsets.data(), poses);
        const double identity[4] = { 1.0, 0.0, 0.0, 1.0 };
        for (size_t i = 0; i < observations.size(); ++i) {
            int slot = -1;
            for (int s = 0; s < 3; ++s) {
                if (order[s] == observations[i].pose_slot) {
                    slot = s;
                }
            }
            block.set_observation(static_cast<int>(i), slot, observations[i].jacobian_pose, observations[i].jacobian_landmark, observations[i].residual, identity);
        }
        block.perform_qr();
        block.set_damping(0.5);
        const std::vector<double> expected_schur = reference.schur_complement(0.5);
        for (int reference_column = 0; reference_column < 6 * poses; ++reference_column) {
            const int reference_pose = reference_column / 6;
            int slot = -1;
            for (int s = 0; s < 3; ++s) {
                if (order[s] == reference_pose) {
                    slot = s;
                }
            }
            std::vector<double> basis(static_cast<size_t>(total + 3), 0.0);
            basis[static_cast<size_t>(sorted_offsets[static_cast<size_t>(slot)] + (reference_column % 6))] = 1.0;
            std::vector<double> image(static_cast<size_t>(total + 3), 0.0);
            block.add_operator_product(basis.data(), image.data());
            for (int reference_row = 0; reference_row < 6 * poses; ++reference_row) {
                const int row_pose = reference_row / 6;
                int row_slot = -1;
                for (int s = 0; s < 3; ++s) {
                    if (order[s] == row_pose) {
                        row_slot = s;
                    }
                }
                const double actual = image[static_cast<size_t>(sorted_offsets[static_cast<size_t>(row_slot)] + (reference_row % 6))];
                REQUIRE(is_value_approx(actual, expected_schur[static_cast<size_t>((reference_row * 6 * poses) + reference_column)], 1e-9));
            }
        }
    }

    {
        const int poses = 4;
        const std::vector<synthetic_observation> observations = make_observations(rng, 9, poses, 0);
        const reference_block reference(observations, poses);
        std::vector<int> pose_offsets(static_cast<size_t>(poses));
        for (int i = 0; i < poses; ++i) {
            pose_offsets[static_cast<size_t>(i)] = 6 * i;
        }
        optimisation::landmark_block<float> block;
        block.configure(static_cast<int>(observations.size()), 3, true, 6 * poses, pose_offsets.data(), poses);
        const float identity[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
        for (size_t i = 0; i < observations.size(); ++i) {
            float jacobian_pose[12];
            float jacobian_landmark[6];
            float residual[2];
            for (int c = 0; c < 12; ++c) {
                jacobian_pose[c] = static_cast<float>(observations[i].jacobian_pose[c]);
            }
            for (int c = 0; c < 6; ++c) {
                jacobian_landmark[c] = static_cast<float>(observations[i].jacobian_landmark[c]);
            }
            residual[0] = static_cast<float>(observations[i].residual[0]);
            residual[1] = static_cast<float>(observations[i].residual[1]);
            block.set_observation(static_cast<int>(i), observations[i].pose_slot, jacobian_pose, jacobian_landmark, residual, identity);
        }
        block.perform_qr();
        block.set_damping(1.0f);
        const std::vector<double> expected_schur = reference.schur_complement(1.0);
        for (int column = 0; column < 6 * poses; ++column) {
            std::vector<float> basis(static_cast<size_t>(6 * poses), 0.0f);
            basis[static_cast<size_t>(column)] = 1.0f;
            std::vector<float> image(static_cast<size_t>(6 * poses), 0.0f);
            block.add_operator_product(basis.data(), image.data());
            for (int row = 0; row < 6 * poses; ++row) {
                REQUIRE(is_value_approx(static_cast<double>(image[static_cast<size_t>(row)]), expected_schur[static_cast<size_t>((row * 6 * poses) + column)], 1e-4));
            }
        }
    }

    {
        const std::vector<synthetic_observation> observations = make_observations(rng, 8, 3, 2);
        optimisation::landmark_block<double> first = build_block(observations, 3, true);
        optimisation::landmark_block<double> second = build_block(observations, 3, true);
        first.perform_qr();
        second.perform_qr();
        second.set_damping(1.0);
        second.clear_damping();
        first.set_damping(2.0);
        second.set_damping(2.0);
        for (int r = 0; r < first.rows(); ++r) {
            for (int c = 0; c < first.cols(); ++c) {
                REQUIRE(first.row(r)[c] == second.row(r)[c]);
            }
        }
    }

    return EXIT_SUCCESS;
}
