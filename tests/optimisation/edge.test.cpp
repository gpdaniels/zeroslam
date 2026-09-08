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

#include "optimisation/edge.hpp"

#include "optimisation/loss.hpp"
#include "optimisation/losses/huber.hpp"
#include "optimisation/vertex.hpp"
#include "optimisation/vertices/point.hpp"
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

static inline bool is_value_approx(double lhs, double rhs, double epsilon = 1e-9) {
    return std::abs(lhs - rhs) <= (epsilon * (std::abs(lhs) + std::abs(rhs))) + epsilon;
}

class distance_edge final {
public:
    constexpr static const char* name = "distance";
    constexpr static const int residual_count = 1;
    constexpr static const int vertex_count = 2;

    void compute_residual(const optimisation::edge& context, math::matrix<double, 0, 0>& residual) const {
        const double* const lhs = context.get_vertex(0)->get_parameters();
        const double* const rhs = context.get_vertex(1)->get_parameters();
        double sum = 0.0;
        for (size_t i = 0; i < 3; ++i) {
            sum += (lhs[i] - rhs[i]) * (lhs[i] - rhs[i]);
        }
        residual[0][0] = context.get_observation()[0][0] - std::sqrt(sum);
    }
};

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    optimisation::vertex lhs{ optimisation::vertices::point() };
    optimisation::vertex rhs{ optimisation::vertices::point() };
    const double lhs_location[3] = { 0.0, 0.0, 0.0 };
    const double rhs_location[3] = { 3.0, 4.0, 0.0 };
    REQUIRE(lhs.set_parameters(&lhs_location[0], 3));
    REQUIRE(rhs.set_parameters(&rhs_location[0], 3));

    optimisation::edge edge{ distance_edge() };
    REQUIRE(edge.is_valid());
    REQUIRE(edge.add_vertex(&lhs));
    REQUIRE(edge.add_vertex(&rhs));
    REQUIRE(!edge.add_vertex(&rhs));
    REQUIRE(edge.num_vertices() == 2);
    REQUIRE(edge.get_vertex(1) == &rhs);
    edge.set_observation(math::matrix<double, 0, 0>(1, 1, math::matrix<double, 1, 1>{ { 6.0 } }.data()));

    edge.compute_residual();
    REQUIRE(is_value_approx(edge.get_residual()[0][0], 1.0));
    REQUIRE(is_value_approx(edge.chi2(), 1.0));
    REQUIRE(is_value_approx(edge.robust_chi2(), 1.0));
    edge.set_information(4.0 * math::matrix<double, 0, 0>::identity(1, 1));
    REQUIRE(is_value_approx(edge.chi2(), 4.0));

    edge.compute_jacobians();
    REQUIRE(edge.get_jacobians().size() == 2);
    REQUIRE(is_value_approx(edge.get_jacobians()[0][0][0], 0.6, 1e-6));
    REQUIRE(is_value_approx(edge.get_jacobians()[0][0][1], 0.8, 1e-6));
    REQUIRE(is_value_approx(edge.get_jacobians()[1][0][0], -0.6, 1e-6));
    REQUIRE(is_value_approx(edge.get_jacobians()[1][0][1], -0.8, 1e-6));
    REQUIRE(is_value_approx(edge.get_residual()[0][0], 1.0));

    lhs.set_fixed();
    edge.compute_jacobians();
    REQUIRE(edge.get_jacobians()[0][0][0] == 0.0);
    REQUIRE(is_value_approx(edge.get_jacobians()[1][0][1], -0.8, 1e-6));

    edge.set_loss(optimisation::loss(optimisation::losses::huber(1.0)));
    REQUIRE(edge.get_loss().is_valid());
    REQUIRE(edge.robust_chi2() < edge.chi2());
    double rho_delta = 0.0;
    math::matrix<double, 0, 0> robust_information;
    edge.robust_info(rho_delta, robust_information);
    REQUIRE(rho_delta < 1.0);
    REQUIRE(is_value_approx(robust_information[0][0], rho_delta * 4.0));

    return EXIT_SUCCESS;
}
