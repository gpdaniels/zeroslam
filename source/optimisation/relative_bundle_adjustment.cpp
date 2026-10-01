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

#include "optimisation/relative_bundle_adjustment.hpp"

#include "math/math.hpp"
#include "math/matrix_decomposition_cholesky.hpp"
#include "optimisation/edges/reprojection.hpp"
#include "optimisation/loss.hpp"
#include "optimisation/losses/huber.hpp"

namespace optimisation {
    namespace {
        // A transform as the matrix and translation of (x, w) -> (M * x + t * w, w) on a homogeneous point.
        class factor final {
        public:
            double matrix[3][3];
            double translation[3];
        };

        factor factor_of(const math::sim3<double>& similarity) {
            factor result;
            const math::matrix<double, 3, 3> rotation = similarity.transformation().rotation().get_matrix();
            const math::matrix<double, 3, 1>& translation = similarity.transformation().translation();
            for (size_t row = 0; row < 3; ++row) {
                for (size_t column = 0; column < 3; ++column) {
                    result.matrix[row][column] = similarity.scale() * rotation[row][column];
                }
                result.translation[row] = translation[row];
            }
            return result;
        }

        void apply(const factor& transform, const double (&point)[4], double (&result)[4]) {
            for (size_t row = 0; row < 3; ++row) {
                result[row] = (transform.matrix[row][0] * point[0]) + (transform.matrix[row][1] * point[1]) + (transform.matrix[row][2] * point[2]) + (transform.translation[row] * point[3]);
            }
            result[3] = point[3];
        }

        // One active transform's share of an observation's jacobian.
        class transform_jacobian final {
        public:
            int offset;
            int dimensions;
            double values[2][7];
        };

        // A landmark's coupling to one active transform, W = J_t^T * w * J_l.
        class coupling final {
        public:
            int offset;
            int dimensions;
            double values[7][3];
        };

        class landmark_system final {
        public:
            double information[3][3];
            double gradient[3];
            std::vector<coupling> couplings;
        };

        class workspace final {
        public:
            std::vector<factor> forward;
            std::vector<factor> inverse;
            // F_a * ... * F_(m-1) * l for each step a of the path being linearised, then l itself.
            std::vector<double> suffix;
            // The matrices of F_0 * ... * F_(a-1) for each step a, then of the whole chain.
            std::vector<double> prefix;
            // The jacobian of the prediction by each step's transform, two rows of seven per step.
            std::vector<double> steps;
        };

        void prepare(const relative_bundle_adjustment& problem, workspace& scratch) {
            scratch.forward.resize(problem.transforms.size());
            scratch.inverse.resize(problem.transforms.size());
            for (size_t index = 0; index < problem.transforms.size(); ++index) {
                scratch.forward[index] = factor_of(problem.transforms[index].estimate);
                scratch.inverse[index] = factor_of(problem.transforms[index].estimate.inverse());
            }
        }

        const factor& factor_at(const workspace& scratch, const relative_bundle_adjustment::step& link) {
            return link.inverse ? scratch.inverse[static_cast<size_t>(link.transform)] : scratch.forward[static_cast<size_t>(link.transform)];
        }

        // The landmark's point in the observing frame scaled by its inverse depth, l = (x, y, 1, rho) taken through the chain.
        void chain(const relative_bundle_adjustment& problem, const relative_bundle_adjustment::observation& measured, const workspace& scratch, double (&point)[4]) {
            const double* const parameters = problem.landmarks[static_cast<size_t>(measured.landmark)].parameters;
            double current[4] = { parameters[0], parameters[1], 1.0, parameters[2] };
            for (size_t a = measured.path.size(); a-- > 0;) {
                double next[4];
                apply(factor_at(scratch, measured.path[a]), current, next);
                for (size_t index = 0; index < 4; ++index) {
                    current[index] = next[index];
                }
            }
            for (size_t index = 0; index < 4; ++index) {
                point[index] = current[index];
            }
        }

        // The residual observed - predicted, and the jacobian of the prediction by the point, with the behind-camera penalty
        // of the reprojection edges when the point does not project. The point is homogeneous, so the penalty is of its
        // direction, q_z / |q|, which a similarity cannot lower by shrinking the point.
        void project(const sensor::model& camera, const relative_bundle_adjustment::observation& measured, const double (&point)[4], double (&residual)[2], double (&jacobian)[2][3]) {
            double projected[2] = { 0.0, 0.0 };
            double projection_jacobian[6] = {};
            const double q[3] = { point[0], point[1], point[2] };
            if (!camera.project(&q[0], &projected[0], &projection_jacobian[0])) {
                const double length = math::sqrt((q[0] * q[0]) + (q[1] * q[1]) + (q[2] * q[2]));
                if (!(length > 0.0)) {
                    residual[0] = edges::reprojection::behind_camera_residual(-1.0);
                    residual[1] = residual[0];
                    for (size_t row = 0; row < 2; ++row) {
                        for (size_t column = 0; column < 3; ++column) {
                            jacobian[row][column] = 0.0;
                        }
                    }
                    return;
                }
                const double cosine = q[2] / length;
                residual[0] = edges::reprojection::behind_camera_residual(cosine);
                residual[1] = residual[0];
                for (size_t column = 0; column < 3; ++column) {
                    const double derivative = (((column == 2) ? 1.0 : 0.0) - (cosine * q[column] / length)) / length;
                    jacobian[0][column] = edges::reprojection::behind_camera_slope * derivative;
                    jacobian[1][column] = jacobian[0][column];
                }
                return;
            }
            residual[0] = measured.pixel[0] - projected[0];
            residual[1] = measured.pixel[1] - projected[1];
            for (size_t row = 0; row < 2; ++row) {
                for (size_t column = 0; column < 3; ++column) {
                    jacobian[row][column] = projection_jacobian[(3 * row) + column];
                }
            }
        }

        // The residual of one observation, the jacobian of its prediction by each step's transform into scratch.steps, and
        // by the landmark. A transform is updated as T * exp(delta), and exp(delta) moves a homogeneous point (Y, w) by
        // G(Y, w) * delta with G = [-[Y]x, w * I, Y]. So a forward step moves the point by P_(a+1) * G(v_(a+1)) * delta, where
        // v_(a+1) is the point entering the step and P_(a+1) the matrix of the chain up to and including it, and an inverse
        // step, (T * exp(delta))^-1 = exp(-delta) * T^-1, by -P_a * G(v_a) * delta with v_a the point leaving it.
        void linearise_observation(const relative_bundle_adjustment& problem, const relative_bundle_adjustment::observation& measured, workspace& scratch, double (&residual)[2], double (&landmark_jacobian)[2][3]) {
            const size_t steps = measured.path.size();
            scratch.suffix.resize(4 * (steps + 1));
            scratch.prefix.resize(9 * (steps + 1));
            scratch.steps.resize(14 * steps);
            const double* const parameters = problem.landmarks[static_cast<size_t>(measured.landmark)].parameters;
            double current[4] = { parameters[0], parameters[1], 1.0, parameters[2] };
            double origin[4] = { 0.0, 0.0, 0.0, 1.0 };
            for (size_t index = 0; index < 4; ++index) {
                scratch.suffix[(4 * steps) + index] = current[index];
            }
            for (size_t a = steps; a-- > 0;) {
                const factor& transform = factor_at(scratch, measured.path[a]);
                double next[4];
                apply(transform, current, next);
                double next_origin[4];
                apply(transform, origin, next_origin);
                for (size_t index = 0; index < 4; ++index) {
                    current[index] = next[index];
                    origin[index] = next_origin[index];
                    scratch.suffix[(4 * a) + index] = next[index];
                }
            }
            double running[3][3] = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } };
            for (size_t a = 0;; ++a) {
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t column = 0; column < 3; ++column) {
                        scratch.prefix[(9 * a) + (3 * row) + column] = running[row][column];
                    }
                }
                if (a == steps) {
                    break;
                }
                const factor& transform = factor_at(scratch, measured.path[a]);
                double product[3][3];
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t column = 0; column < 3; ++column) {
                        product[row][column] = (running[row][0] * transform.matrix[0][column]) + (running[row][1] * transform.matrix[1][column]) + (running[row][2] * transform.matrix[2][column]);
                    }
                }
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t column = 0; column < 3; ++column) {
                        running[row][column] = product[row][column];
                    }
                }
            }

            double projection[2][3];
            project(problem.cameras[static_cast<size_t>(measured.camera)], measured, current, residual, projection);

            for (size_t a = 0; a < steps; ++a) {
                const bool inverse = measured.path[a].inverse;
                const double* const point = &scratch.suffix[4 * (inverse ? a : (a + 1))];
                const double* const left = &scratch.prefix[9 * (inverse ? a : (a + 1))];
                const double sign = inverse ? -1.0 : 1.0;
                const double generators[3][7] = {
                    { 0.0, point[2], -point[1], point[3], 0.0, 0.0, point[0] },
                    { -point[2], 0.0, point[0], 0.0, point[3], 0.0, point[1] },
                    { point[1], -point[0], 0.0, 0.0, 0.0, point[3], point[2] },
                };
                double moved[3][7];
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t column = 0; column < 7; ++column) {
                        moved[row][column] = sign * ((left[(3 * row) + 0] * generators[0][column]) + (left[(3 * row) + 1] * generators[1][column]) + (left[(3 * row) + 2] * generators[2][column]));
                    }
                }
                double* const values = &scratch.steps[14 * a];
                for (size_t row = 0; row < 2; ++row) {
                    for (size_t column = 0; column < 7; ++column) {
                        values[(7 * row) + column] = (projection[row][0] * moved[0][column]) + (projection[row][1] * moved[1][column]) + (projection[row][2] * moved[2][column]);
                    }
                }
            }

            // The point is M * (x, y, 1) + t * rho for the matrix M and translation t of the whole chain.
            const double* const composed = &scratch.prefix[9 * steps];
            for (size_t row = 0; row < 2; ++row) {
                for (size_t column = 0; column < 2; ++column) {
                    landmark_jacobian[row][column] = (projection[row][0] * composed[column]) + (projection[row][1] * composed[3 + column]) + (projection[row][2] * composed[6 + column]);
                }
                landmark_jacobian[row][2] = (projection[row][0] * origin[0]) + (projection[row][1] * origin[1]) + (projection[row][2] * origin[2]);
            }
        }

        double observation_cost(const relative_bundle_adjustment& problem, const relative_bundle_adjustment::observation& measured, const loss& robust, const workspace& scratch) {
            double point[4];
            chain(problem, measured, scratch, point);
            double residual[2];
            double jacobian[2][3];
            project(problem.cameras[static_cast<size_t>(measured.camera)], measured, point, residual, jacobian);
            const double chi2 = ((residual[0] * residual[0]) + (residual[1] * residual[1])) / (measured.sigma * measured.sigma);
            math::matrix<double, 3, 1> rho = math::matrix<double, 3, 1>::zero();
            // An observation without a finite residual costs as one a thousand sigma out, which no step is drawn by.
            robust.compute(math::isfinite(chi2) ? chi2 : 1.0e6, rho);
            return rho[0];
        }

        double total_cost(const relative_bundle_adjustment& problem, const loss& robust, workspace& scratch) {
            prepare(problem, scratch);
            double total = 0.0;
            for (const relative_bundle_adjustment::length_prior& prior : problem.priors) {
                const math::matrix<double, 3, 1>& translation = problem.transforms[static_cast<size_t>(prior.transform)].estimate.transformation().translation();
                const double error = prior.length - math::sqrt(translation.get_length_squared());
                total += prior.information * error * error;
            }
            for (const relative_bundle_adjustment::observation& measured : problem.observations) {
                total += observation_cost(problem, measured, robust, scratch);
            }
            return total;
        }
    }

    math::matrix<double, 3, 1> relative_bundle_adjustment::point_in_frame(const observation& measured) const {
        workspace scratch;
        prepare(*this, scratch);
        double point[4];
        chain(*this, measured, scratch, point);
        return { { point[0], point[1], point[2] } };
    }

    bool relative_bundle_adjustment::predict(const observation& measured, double (&pixel)[2]) const {
        const math::matrix<double, 3, 1> point = this->point_in_frame(measured);
        return this->cameras[static_cast<size_t>(measured.camera)].project(point.data(), &pixel[0]);
    }

    void relative_bundle_adjustment::linearise(const observation& measured, double (&residual)[2], std::vector<math::matrix<double, 2, 7>>& step_jacobians, math::matrix<double, 2, 3>& landmark_jacobian) const {
        workspace scratch;
        prepare(*this, scratch);
        double landmark_values[2][3];
        linearise_observation(*this, measured, scratch, residual, landmark_values);
        step_jacobians.resize(measured.path.size());
        for (size_t a = 0; a < measured.path.size(); ++a) {
            for (size_t row = 0; row < 2; ++row) {
                for (size_t column = 0; column < 7; ++column) {
                    step_jacobians[a][row][column] = scratch.steps[(14 * a) + (7 * row) + column];
                }
            }
        }
        for (size_t row = 0; row < 2; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                landmark_jacobian[row][column] = landmark_values[row][column];
            }
        }
    }

    double relative_bundle_adjustment::cost(const double huber_delta) const {
        const loss robust{ losses::huber(huber_delta) };
        workspace scratch;
        return total_cost(*this, robust, scratch);
    }

    relative_bundle_adjustment::summary relative_bundle_adjustment::solve(const int rounds, const double huber_delta) {
        summary result;
        const loss robust{ losses::huber(huber_delta) };
        workspace scratch;

        std::vector<int> transform_offset(this->transforms.size(), -1);
        int dimensions = 0;
        for (size_t index = 0; index < this->transforms.size(); ++index) {
            if (this->transforms[index].active) {
                transform_offset[index] = dimensions;
                dimensions += this->transforms[index].similarity ? 7 : 6;
                ++result.active_transforms;
            }
        }
        std::vector<int> landmark_slot(this->landmarks.size(), -1);
        int landmark_count = 0;
        for (size_t index = 0; index < this->landmarks.size(); ++index) {
            if (this->landmarks[index].active) {
                landmark_slot[index] = landmark_count++;
            }
        }
        result.active_landmarks = landmark_count;

        double current_cost = total_cost(*this, robust, scratch);
        result.initial_cost = current_cost;
        result.final_cost = current_cost;
        if ((dimensions == 0) && (landmark_count == 0)) {
            return result;
        }

        const size_t system_size = static_cast<size_t>(dimensions);
        std::vector<double> hessian(system_size * system_size);
        std::vector<double> gradient(system_size);
        std::vector<landmark_system> landmark_systems(static_cast<size_t>(landmark_count));
        std::vector<transform_jacobian> jacobians;
        std::vector<double> reduced(system_size * system_size);
        std::vector<double> reduced_rhs(system_size);
        std::vector<double> factorised(system_size * system_size);
        std::vector<double> transform_step(system_size);
        std::vector<double> scaling(system_size);
        std::vector<double> landmark_step(3 * static_cast<size_t>(landmark_count));
        std::vector<math::matrix<double, 3, 3>> inverses(static_cast<size_t>(landmark_count));
        std::vector<char> invertible(static_cast<size_t>(landmark_count), 0);
        std::vector<math::sim3<double>> saved_transforms(this->transforms.size());
        std::vector<double> saved_landmarks(3 * this->landmarks.size());

        double lambda = 1e-3;
        double nu = 2.0;
        int rejections = 0;
        for (int round = 0; round < rounds; ++round) {
            ++result.iterations;
            for (double& value : hessian) {
                value = 0.0;
            }
            for (double& value : gradient) {
                value = 0.0;
            }
            for (landmark_system& system : landmark_systems) {
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t column = 0; column < 3; ++column) {
                        system.information[row][column] = 0.0;
                    }
                    system.gradient[row] = 0.0;
                }
                system.couplings.clear();
            }

            prepare(*this, scratch);
            for (const observation& measured : this->observations) {
                double residual[2];
                double landmark_jacobian[2][3];
                linearise_observation(*this, measured, scratch, residual, landmark_jacobian);
                // A point at the camera's plane, or past where its distortion is defined, gives no usable linearisation.
                bool finite = math::isfinite(residual[0]) && math::isfinite(residual[1]);
                for (size_t index = 0; finite && (index < 14 * measured.path.size()); ++index) {
                    finite = math::isfinite(scratch.steps[index]);
                }
                for (size_t index = 0; finite && (index < 6); ++index) {
                    finite = math::isfinite(landmark_jacobian[index / 3][index % 3]);
                }
                if (!finite) {
                    continue;
                }
                const double sigma_squared = measured.sigma * measured.sigma;
                const double chi2 = ((residual[0] * residual[0]) + (residual[1] * residual[1])) / sigma_squared;
                math::matrix<double, 3, 1> rho = math::matrix<double, 3, 1>::zero();
                robust.compute(chi2, rho);
                const double weight = rho[1] / sigma_squared;

                // Each active transform's jacobian, summed over the steps it appears in.
                jacobians.clear();
                for (size_t a = 0; a < measured.path.size(); ++a) {
                    const size_t transform_index = static_cast<size_t>(measured.path[a].transform);
                    const int offset = transform_offset[transform_index];
                    if (offset < 0) {
                        continue;
                    }
                    transform_jacobian* target = nullptr;
                    for (transform_jacobian& existing : jacobians) {
                        if (existing.offset == offset) {
                            target = &existing;
                        }
                    }
                    if (target == nullptr) {
                        jacobians.push_back(transform_jacobian{ offset, this->transforms[transform_index].similarity ? 7 : 6, {} });
                        target = &jacobians.back();
                    }
                    for (size_t row = 0; row < 2; ++row) {
                        for (size_t column = 0; column < 7; ++column) {
                            target->values[row][column] += scratch.steps[(14 * a) + (7 * row) + column];
                        }
                    }
                }

                for (const transform_jacobian& first : jacobians) {
                    for (int row = 0; row < first.dimensions; ++row) {
                        const size_t first_index = static_cast<size_t>(first.offset + row);
                        gradient[first_index] += weight * ((first.values[0][row] * residual[0]) + (first.values[1][row] * residual[1]));
                        for (const transform_jacobian& second : jacobians) {
                            for (int column = 0; column < second.dimensions; ++column) {
                                hessian[(first_index * system_size) + static_cast<size_t>(second.offset + column)] += weight * ((first.values[0][row] * second.values[0][column]) + (first.values[1][row] * second.values[1][column]));
                            }
                        }
                    }
                }
                const int slot = landmark_slot[static_cast<size_t>(measured.landmark)];
                if (slot < 0) {
                    continue;
                }
                landmark_system& system = landmark_systems[static_cast<size_t>(slot)];
                for (size_t row = 0; row < 3; ++row) {
                    system.gradient[row] += weight * ((landmark_jacobian[0][row] * residual[0]) + (landmark_jacobian[1][row] * residual[1]));
                    for (size_t column = 0; column < 3; ++column) {
                        system.information[row][column] += weight * ((landmark_jacobian[0][row] * landmark_jacobian[0][column]) + (landmark_jacobian[1][row] * landmark_jacobian[1][column]));
                    }
                }
                for (const transform_jacobian& first : jacobians) {
                    coupling* target = nullptr;
                    for (coupling& existing : system.couplings) {
                        if (existing.offset == first.offset) {
                            target = &existing;
                        }
                    }
                    if (target == nullptr) {
                        system.couplings.push_back(coupling{ first.offset, first.dimensions, {} });
                        target = &system.couplings.back();
                    }
                    for (int row = 0; row < first.dimensions; ++row) {
                        for (size_t column = 0; column < 3; ++column) {
                            target->values[row][column] += weight * ((first.values[0][row] * landmark_jacobian[0][column]) + (first.values[1][row] * landmark_jacobian[1][column]));
                        }
                    }
                }
            }

            // A length prior acts on the translation, which T * exp(delta) moves by scale * R * upsilon to first order.
            for (const length_prior& prior : this->priors) {
                const int offset = transform_offset[static_cast<size_t>(prior.transform)];
                if (offset < 0) {
                    continue;
                }
                const math::sim3<double>& estimate = this->transforms[static_cast<size_t>(prior.transform)].estimate;
                const math::matrix<double, 3, 1>& translation = estimate.transformation().translation();
                const double length = math::sqrt(translation.get_length_squared());
                if (!(length > 1e-12)) {
                    continue;
                }
                const math::matrix<double, 3, 3> rotation = estimate.transformation().rotation().get_matrix();
                double jacobian[3];
                for (size_t column = 0; column < 3; ++column) {
                    jacobian[column] = estimate.scale() * ((translation[0] * rotation[0][column]) + (translation[1] * rotation[1][column]) + (translation[2] * rotation[2][column])) / length;
                }
                const double error = prior.length - length;
                const size_t first_index = static_cast<size_t>(offset + 3);
                for (size_t row = 0; row < 3; ++row) {
                    gradient[first_index + row] += prior.information * jacobian[row] * error;
                    for (size_t column = 0; column < 3; ++column) {
                        hessian[((first_index + row) * system_size) + first_index + column] += prior.information * jacobian[row] * jacobian[column];
                    }
                }
            }

            // The damping is Marquardt's, the diagonal of the information, floored at a fraction of its largest so that a
            // parameter the observations hold nothing on takes no step, rather than one its rounding errors make for it.
            double information_largest = 0.0;
            for (size_t index = 0; index < system_size; ++index) {
                information_largest = math::max(information_largest, hessian[(index * system_size) + index]);
            }
            for (const landmark_system& system : landmark_systems) {
                for (size_t row = 0; row < 3; ++row) {
                    information_largest = math::max(information_largest, system.information[row][row]);
                }
            }
            const double damping_floor = math::max(information_largest * 1.0e-9, 1.0e-300);

            // The damped system, with the landmarks eliminated: S = (U + lambda * D) - sum_k W_k * (V_k + lambda * D)^-1 * W_k^T.
            // A landmark whose block does not invert is held for the round.
            bool factorised_ok = false;
            while (!factorised_ok) {
                for (size_t index = 0; index < system_size * system_size; ++index) {
                    reduced[index] = hessian[index];
                }
                for (size_t index = 0; index < system_size; ++index) {
                    reduced[(index * system_size) + index] += lambda * math::max(hessian[(index * system_size) + index], damping_floor);
                    reduced_rhs[index] = gradient[index];
                }
                for (size_t k = 0; k < landmark_systems.size(); ++k) {
                    const landmark_system& system = landmark_systems[k];
                    math::matrix<double, 3, 3> damped;
                    for (size_t row = 0; row < 3; ++row) {
                        for (size_t column = 0; column < 3; ++column) {
                            damped[row][column] = system.information[row][column];
                        }
                        damped[row][row] += lambda * math::max(system.information[row][row], damping_floor);
                    }
                    invertible[k] = math::invert(damped, inverses[k]) ? 1 : 0;
                    if (invertible[k] == 0) {
                        continue;
                    }
                    const math::matrix<double, 3, 3>& inverse = inverses[k];
                    double eliminated[3];
                    for (size_t row = 0; row < 3; ++row) {
                        eliminated[row] = (inverse[row][0] * system.gradient[0]) + (inverse[row][1] * system.gradient[1]) + (inverse[row][2] * system.gradient[2]);
                    }
                    for (const coupling& first : system.couplings) {
                        double scaled[7][3];
                        for (int row = 0; row < first.dimensions; ++row) {
                            for (size_t column = 0; column < 3; ++column) {
                                scaled[row][column] = (first.values[row][0] * inverse[0][column]) + (first.values[row][1] * inverse[1][column]) + (first.values[row][2] * inverse[2][column]);
                            }
                            reduced_rhs[static_cast<size_t>(first.offset + row)] -= (first.values[row][0] * eliminated[0]) + (first.values[row][1] * eliminated[1]) + (first.values[row][2] * eliminated[2]);
                        }
                        for (const coupling& second : system.couplings) {
                            for (int row = 0; row < first.dimensions; ++row) {
                                const size_t first_index = static_cast<size_t>(first.offset + row);
                                for (int column = 0; column < second.dimensions; ++column) {
                                    reduced[(first_index * system_size) + static_cast<size_t>(second.offset + column)] -= (scaled[row][0] * second.values[column][0]) + (scaled[row][1] * second.values[column][1]) + (scaled[row][2] * second.values[column][2]);
                                }
                            }
                        }
                    }
                }
                // The parameters' information spans many orders of magnitude, from a translation seen only through points at
                // infinity to one seen through points by the camera, so the system is scaled to a unit diagonal for the
                // factorisation's pivots to be judged against each parameter's own information.
                double largest = 0.0;
                for (size_t index = 0; index < system_size; ++index) {
                    largest = math::max(largest, reduced[(index * system_size) + index]);
                }
                const double smallest = math::max(largest * 1.0e-15, 1.0e-300);
                for (size_t index = 0; index < system_size; ++index) {
                    scaling[index] = 1.0 / math::sqrt(math::max(reduced[(index * system_size) + index], smallest));
                }
                for (size_t row = 0; row < system_size; ++row) {
                    for (size_t column = 0; column < system_size; ++column) {
                        reduced[(row * system_size) + column] *= scaling[row] * scaling[column];
                    }
                    reduced_rhs[row] *= scaling[row];
                }
                factorised_ok = (system_size == 0) || (math::decompose_cholesky(reduced.data(), dimensions, dimensions, factorised.data()) && math::solve_cholesky(factorised.data(), reduced_rhs.data(), dimensions, dimensions, transform_step.data()));
                for (size_t index = 0; factorised_ok && (index < system_size); ++index) {
                    transform_step[index] *= scaling[index];
                    factorised_ok = math::isfinite(transform_step[index]);
                }
                if (!factorised_ok) {
                    lambda *= nu;
                    nu *= 2.0;
                    if (lambda > 1e12) {
                        return result;
                    }
                }
            }
            for (size_t k = 0; k < landmark_systems.size(); ++k) {
                const landmark_system& system = landmark_systems[k];
                double rhs[3] = { system.gradient[0], system.gradient[1], system.gradient[2] };
                for (const coupling& link : system.couplings) {
                    for (size_t column = 0; column < 3; ++column) {
                        for (int row = 0; row < link.dimensions; ++row) {
                            rhs[column] -= link.values[row][column] * transform_step[static_cast<size_t>(link.offset + row)];
                        }
                    }
                }
                for (size_t row = 0; row < 3; ++row) {
                    landmark_step[(3 * k) + row] = (invertible[k] != 0) ? ((inverses[k][row][0] * rhs[0]) + (inverses[k][row][1] * rhs[1]) + (inverses[k][row][2] * rhs[2])) : 0.0;
                }
            }

            // The reduction the damped model predicts, delta^T * (g + lambda * D * delta).
            double predicted = 0.0;
            for (size_t index = 0; index < system_size; ++index) {
                predicted += transform_step[index] * (gradient[index] + (lambda * math::max(hessian[(index * system_size) + index], damping_floor) * transform_step[index]));
            }
            for (size_t k = 0; k < landmark_systems.size(); ++k) {
                for (size_t row = 0; row < 3; ++row) {
                    const double value = landmark_step[(3 * k) + row];
                    predicted += value * (landmark_systems[k].gradient[row] + (lambda * math::max(landmark_systems[k].information[row][row], damping_floor) * value));
                }
            }

            for (size_t index = 0; index < this->transforms.size(); ++index) {
                saved_transforms[index] = this->transforms[index].estimate;
                const int offset = transform_offset[index];
                if (offset < 0) {
                    continue;
                }
                math::matrix<double, 7, 1> delta = math::matrix<double, 7, 1>::zero();
                const size_t transform_dimensions = this->transforms[index].similarity ? 7 : 6;
                for (size_t row = 0; row < transform_dimensions; ++row) {
                    delta[row] = transform_step[static_cast<size_t>(offset) + row];
                }
                math::sim3<double> updated = this->transforms[index].estimate * math::sim3<double>::exp(delta);
                updated.transformation().rotation() = updated.transformation().rotation().normalised();
                this->transforms[index].estimate = updated;
            }
            for (size_t index = 0; index < this->landmarks.size(); ++index) {
                for (size_t row = 0; row < 3; ++row) {
                    saved_landmarks[(3 * index) + row] = this->landmarks[index].parameters[row];
                }
                const int slot = landmark_slot[index];
                if (slot < 0) {
                    continue;
                }
                for (size_t row = 0; row < 3; ++row) {
                    this->landmarks[index].parameters[row] += landmark_step[(3 * static_cast<size_t>(slot)) + row];
                }
            }

            const double next_cost = total_cost(*this, robust, scratch);
            if ((predicted > 0.0) && (next_cost < current_cost)) {
                const double gain = (current_cost - next_cost) / predicted;
                const double improvement = current_cost - next_cost;
                current_cost = next_cost;
                result.final_cost = current_cost;
                ++result.accepted;
                rejections = 0;
                const double ratio = (2.0 * gain) - 1.0;
                lambda *= math::max(1.0 / 3.0, 1.0 - (ratio * ratio * ratio));
                nu = 2.0;
                if (improvement <= 1e-9 * current_cost) {
                    break;
                }
            }
            else {
                for (size_t index = 0; index < this->transforms.size(); ++index) {
                    this->transforms[index].estimate = saved_transforms[index];
                }
                for (size_t index = 0; index < this->landmarks.size(); ++index) {
                    for (size_t row = 0; row < 3; ++row) {
                        this->landmarks[index].parameters[row] = saved_landmarks[(3 * index) + row];
                    }
                }
                lambda *= nu;
                nu *= 2.0;
                if ((++rejections >= 10) || (lambda > 1e12)) {
                    break;
                }
            }
        }
        return result;
    }
}
