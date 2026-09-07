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
#ifndef ZEROSLAM_CORE_FILTER_HPP
#define ZEROSLAM_CORE_FILTER_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace core {
    class filter final {
    public:
        // Remove every element the test accepts, order is not preserved.
        template <typename type, typename test_function_type>
        static void remove_if(
            type* __restrict const features,
            size_t& features_count,
            const test_function_type test_function
        ) {
            if (features_count == 0)
                return;
            type* front = features;
            type* back = features + features_count - 1;
            while (front <= back) {
                if (test_function(*front)) {
                    while ((back != front) && (test_function(*back))) {
                        --back;
                    }
                    if (back != front) {
                        *front = static_cast<type&&>(*back);
                    }
                    --back;
                }
                ++front;
            }
            features_count = static_cast<size_t>(back + 1 - features);
        }
    };
}

#endif // ZEROSLAM_CORE_FILTER_HPP
