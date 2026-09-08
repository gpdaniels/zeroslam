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

#include "estimation/robust/estimator.hpp"

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

class mean_estimator final
    : public estimation::robust::estimator<float, 2, float, 1> {
public:
    virtual size_t generate_models(const float* const __restrict data, const size_t data_size, float* const __restrict models) const override final {
        if (data_size < 2) {
            return 0;
        }
        models[0] = 0.5f * (data[0] + data[1]);
        return 1;
    }

    virtual void compute_residuals(const float* const __restrict data, const size_t data_size, const float& candidate, float* const __restrict residuals) const override final {
        for (size_t i = 0; i < data_size; ++i) {
            residuals[i] = std::abs(data[i] - candidate);
        }
    }
};

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(mean_estimator::sample_size == 2);
    static_assert(mean_estimator::models_size == 1);

    {
        const mean_estimator estimator;
        const float data[4] = { 1.0f, 3.0f, 2.5f, 10.0f };
        float model = 0.0f;
        REQUIRE(estimator.generate_models(data, 2, &model) == 1);
        REQUIRE(model == 2.0f);
        REQUIRE(estimator.generate_models(data, 1, &model) == 0);
        float residuals[4];
        estimator.compute_residuals(data, 4, model, residuals);
        REQUIRE(residuals[0] == 1.0f);
        REQUIRE(residuals[1] == 1.0f);
        REQUIRE(residuals[2] == 0.5f);
        REQUIRE(residuals[3] == 8.0f);
    }

    return EXIT_SUCCESS;
}
