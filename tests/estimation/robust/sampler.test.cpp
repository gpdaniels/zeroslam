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

#include "estimation/robust/sampler.hpp"

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

class counting_sampler final
    : public estimation::robust::sampler<3> {
private:
    size_t next = 0;

public:
    virtual void prepare(const size_t data_size) override final {
        this->next = data_size;
    }

    virtual void sample(size_t* const __restrict indices) override final {
        for (size_t i = 0; i < 3; ++i) {
            indices[i] = this->next + i;
        }
    }
};

int main(int argc, char* argv[]) {
    static_cast<void>(argc);
    static_cast<void>(argv);

    static_assert(counting_sampler::sample_size == 3);
    static_assert(estimation::robust::sampler<7>::sample_size == 7);

    {
        counting_sampler sampler;
        sampler.prepare(10);
        size_t indices[3] = {};
        sampler.sample(indices);
        REQUIRE(indices[0] == 10);
        REQUIRE(indices[1] == 11);
        REQUIRE(indices[2] == 12);
    }

    return EXIT_SUCCESS;
}
