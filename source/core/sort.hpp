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
#ifndef ZEROSLAM_CORE_SORT_HPP
#define ZEROSLAM_CORE_SORT_HPP

namespace {
    using size_t = decltype(sizeof(0));
}

namespace core {
    class sort final {
    public:
        template <typename type, typename comparitor_function_type>
        static void insertion(
            type* __restrict const features,
            const size_t features_count,
            const comparitor_function_type& comparitor_function
        ) {
            for (size_t i = 1; i < features_count; ++i) {
                const type value = features[i];
                size_t j = i;
                while ((j > 0) && comparitor_function(value, features[j - 1])) {
                    features[j] = features[j - 1];
                    --j;
                }
                features[j] = value;
            }
        }

        template <typename type, typename comparitor_function_type>
        static void heap(
            type* __restrict const features,
            const size_t features_count,
            const comparitor_function_type& comparitor_function
        ) {
            constexpr static const auto sift_down = [](
                                                        type* __restrict const heap_features,
                                                        size_t index,
                                                        const size_t heap_size,
                                                        const comparitor_function_type& heap_comparitor_function
                                                    ) {
                while (true) {
                    size_t largest = index;
                    const size_t left = 2 * index + 1;
                    const size_t right = 2 * index + 2;
                    if ((left < heap_size) && heap_comparitor_function(heap_features[largest], heap_features[left])) {
                        largest = left;
                    }
                    if ((right < heap_size) && heap_comparitor_function(heap_features[largest], heap_features[right])) {
                        largest = right;
                    }
                    if (largest == index) {
                        break;
                    }
                    const type temp = heap_features[index];
                    heap_features[index] = heap_features[largest];
                    heap_features[largest] = temp;
                    index = largest;
                }
            };
            if (features_count < 2) {
                return;
            }
            for (size_t i = features_count / 2; i-- > 0;) {
                sift_down(features, i, features_count, comparitor_function);
            }
            for (size_t i = features_count - 1; i > 0; --i) {
                const type temp = features[0];
                features[0] = features[i];
                features[i] = temp;
                sift_down(features, 0, i, comparitor_function);
            }
        }

        template <typename type, typename comparitor_function_type>
        static void quick(
            type* __restrict features,
            size_t features_count,
            const comparitor_function_type& comparitor_function,
            int depth_limit = -1
        ) {
            constexpr static const size_t insertion_sort_threshold = 16;
            constexpr static const auto swap = [](type& lhs, type& rhs) {
                type lhs_copy = lhs;
                lhs = rhs;
                rhs = lhs_copy;
            };
            if (depth_limit < 0) {
                depth_limit = 0;
                for (size_t n = features_count; n > 1; n >>= 1) {
                    ++depth_limit;
                }
                depth_limit *= 2;
            }
            while (features_count > 1) {
                if (features_count <= insertion_sort_threshold) {
                    sort::insertion(features, features_count, comparitor_function);
                    return;
                }
                if (depth_limit <= 0) {
                    sort::heap(features, features_count, comparitor_function);
                    return;
                }
                --depth_limit;
                const size_t mid = features_count / 2;
                const size_t last = features_count - 1;
                if (comparitor_function(features[mid], features[0])) {
                    swap(features[0], features[mid]);
                }
                if (comparitor_function(features[last], features[mid])) {
                    swap(features[mid], features[last]);
                }
                if (comparitor_function(features[mid], features[0])) {
                    swap(features[0], features[mid]);
                }
                swap(features[0], features[mid]);

                size_t index_left = 1;
                size_t index_right = features_count;
                while (index_left < index_right) {
                    if (comparitor_function(features[index_left], features[0])) {
                        ++index_left;
                    }
                    else {
                        --index_right;
                        swap(features[index_left], features[index_right]);
                    }
                }
                --index_left;
                swap(features[index_left], features[0]);
                const size_t left_count = index_left + 1;
                const size_t right_count = features_count - index_right;
                if (left_count < right_count) {
                    sort::quick(features, left_count, comparitor_function, depth_limit);
                    features += index_right;
                    features_count = right_count;
                }
                else {
                    sort::quick(&features[index_right], right_count, comparitor_function, depth_limit);
                    features_count = left_count;
                }
            }
        }
    };
}

#endif // ZEROSLAM_CORE_SORT_HPP
