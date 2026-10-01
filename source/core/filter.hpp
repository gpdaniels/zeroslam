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
            // The kept elements are those below end, so the last candidate is end - 1 and no index goes before the first element.
            size_t front = 0;
            size_t end = features_count;
            while (front < end) {
                if (test_function(features[front])) {
                    while (((end - 1) != front) && (test_function(features[end - 1]))) {
                        --end;
                    }
                    if ((end - 1) != front) {
                        features[front] = static_cast<type&&>(features[end - 1]);
                    }
                    --end;
                }
                ++front;
            }
            features_count = end;
        }
    };
}

#endif // ZEROSLAM_CORE_FILTER_HPP
