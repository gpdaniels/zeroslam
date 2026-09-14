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

#include "feature/detector/mser.hpp"

#include "core/arena.hpp"
#include "core/arena_allocator.hpp"
#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::detector {
    namespace {
        size_t detect_polarity(const unsigned char* __restrict const data, const int width, const int height, const int stride, const bool invert, const mser::options& settings, const size_t buffer_size, point* __restrict buffer, size_t written) {
            const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
            std::vector<unsigned char, core::arena_allocator<unsigned char>> level(pixels);
            size_t histogram[257] = { 0 };
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const unsigned char value = data[(static_cast<long>(y) * stride) + x];
                    const unsigned char g = invert ? static_cast<unsigned char>(255 - value) : value;
                    level[(static_cast<size_t>(y) * static_cast<size_t>(width)) + static_cast<size_t>(x)] = g;
                    ++histogram[static_cast<size_t>(g) + 1];
                }
            }
            for (int g = 1; g <= 256; ++g) {
                histogram[g] += histogram[g - 1];
            }
            std::vector<int, core::arena_allocator<int>> order(pixels);
            {
                size_t fill[256];
                for (int g = 0; g < 256; ++g) {
                    fill[g] = histogram[g];
                }
                for (size_t p = 0; p < pixels; ++p) {
                    order[fill[level[p]]++] = static_cast<int>(p);
                }
            }
            std::vector<int, core::arena_allocator<int>> parent(pixels, -1);
            std::vector<int, core::arena_allocator<int>> node_of(pixels, -1);

            struct node final {
                int level;
                int area;
                double sum_x;
                double sum_y;
                int parent;
                int main_child;
                float variation;
                bool stable;
            };

            std::vector<node, core::arena_allocator<node>> nodes;
            nodes.reserve(pixels / 4 + 16);
            const auto find = [&](int p) {
                int root = p;
                while (parent[static_cast<size_t>(root)] >= 0) {
                    root = parent[static_cast<size_t>(root)];
                }
                while (parent[static_cast<size_t>(p)] >= 0) {
                    const int next = parent[static_cast<size_t>(p)];
                    parent[static_cast<size_t>(p)] = root;
                    p = next;
                }
                return root;
            };
            for (size_t index = 0; index < pixels; ++index) {
                const int p = order[index];
                const int g = level[static_cast<size_t>(p)];
                const int px = p % width;
                const int py = p / width;
                int current = static_cast<int>(nodes.size());
                nodes.push_back(node{ g, 1, static_cast<double>(px), static_cast<double>(py), -1, -1, 0.0f, false });
                node_of[static_cast<size_t>(p)] = current;
                parent[static_cast<size_t>(p)] = -1;
                int root = p;
                const int neighbours[4][2] = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
                for (int n = 0; n < 4; ++n) {
                    const int nx = px + neighbours[n][0];
                    const int ny = py + neighbours[n][1];
                    if ((nx < 0) || (nx >= width) || (ny < 0) || (ny >= height)) {
                        continue;
                    }
                    const int q = (ny * width) + nx;
                    if (node_of[static_cast<size_t>(q)] < 0) {
                        continue;
                    }
                    const int other_root = find(q);
                    if (other_root == root) {
                        continue;
                    }
                    const int other = node_of[static_cast<size_t>(other_root)];
                    node& current_node = nodes[static_cast<size_t>(current)];
                    node& other_node = nodes[static_cast<size_t>(other)];
                    if (other_node.level == g) {
                        node& survivor = (other_node.area >= current_node.area) ? other_node : current_node;
                        node& absorbed = (other_node.area >= current_node.area) ? current_node : other_node;
                        survivor.area += absorbed.area;
                        survivor.sum_x += absorbed.sum_x;
                        survivor.sum_y += absorbed.sum_y;
                        if ((absorbed.main_child >= 0) && ((survivor.main_child < 0) || (nodes[static_cast<size_t>(absorbed.main_child)].area > nodes[static_cast<size_t>(survivor.main_child)].area))) {
                            survivor.main_child = absorbed.main_child;
                        }
                        const int survivor_index = (&survivor == &other_node) ? other : current;
                        current = survivor_index;
                    }
                    else {
                        other_node.parent = current;
                        current_node.area += other_node.area;
                        current_node.sum_x += other_node.sum_x;
                        current_node.sum_y += other_node.sum_y;
                        if ((current_node.main_child < 0) || (other_node.area > nodes[static_cast<size_t>(current_node.main_child)].area)) {
                            current_node.main_child = other;
                        }
                    }
                    parent[static_cast<size_t>(other_root)] = root;
                    node_of[static_cast<size_t>(root)] = current;
                }
            }
            for (size_t i = 0; i < nodes.size(); ++i) {
                node& n = nodes[i];
                if ((n.area < settings.minimum_area) || (n.area > settings.maximum_area)) {
                    n.variation = 1.0e9f;
                    continue;
                }
                int up = static_cast<int>(i);
                while ((nodes[static_cast<size_t>(up)].parent >= 0) && (nodes[static_cast<size_t>(nodes[static_cast<size_t>(up)].parent)].level <= n.level + settings.delta)) {
                    up = nodes[static_cast<size_t>(up)].parent;
                }
                int down = static_cast<int>(i);
                while ((nodes[static_cast<size_t>(down)].main_child >= 0) && (nodes[static_cast<size_t>(nodes[static_cast<size_t>(down)].main_child)].level >= n.level - settings.delta)) {
                    down = nodes[static_cast<size_t>(down)].main_child;
                }
                n.variation = static_cast<float>(nodes[static_cast<size_t>(up)].area - nodes[static_cast<size_t>(down)].area) / static_cast<float>(n.area);
            }
            for (size_t i = 0; i < nodes.size(); ++i) {
                node& n = nodes[i];
                if (n.variation > settings.maximum_variation) {
                    continue;
                }
                const bool better_parent = (n.parent >= 0) && (nodes[static_cast<size_t>(n.parent)].variation < n.variation);
                const bool better_child = (n.main_child >= 0) && (nodes[static_cast<size_t>(n.main_child)].variation < n.variation);
                n.stable = !better_parent && !better_child;
            }
            for (size_t i = 0; i < nodes.size(); ++i) {
                node& n = nodes[i];
                if (!n.stable || (n.parent < 0)) {
                    continue;
                }
                int ancestor = n.parent;
                while ((ancestor >= 0) && !nodes[static_cast<size_t>(ancestor)].stable) {
                    ancestor = nodes[static_cast<size_t>(ancestor)].parent;
                }
                if ((ancestor >= 0) && (static_cast<float>(nodes[static_cast<size_t>(ancestor)].area - n.area) < settings.minimum_diversity * static_cast<float>(nodes[static_cast<size_t>(ancestor)].area))) {
                    if (nodes[static_cast<size_t>(ancestor)].variation <= n.variation) {
                        n.stable = false;
                    }
                    else {
                        nodes[static_cast<size_t>(ancestor)].stable = false;
                    }
                }
            }
            for (size_t i = 0; (i < nodes.size()) && (written < buffer_size); ++i) {
                const node& n = nodes[i];
                if (!n.stable) {
                    continue;
                }
                point feature;
                feature.x = static_cast<float>(n.sum_x / static_cast<double>(n.area));
                feature.y = static_cast<float>(n.sum_y / static_cast<double>(n.area));
                feature.response = 1.0f - n.variation;
                feature.angle = math::sqrt(static_cast<float>(n.area) / 3.14159265358979323846f);
                feature.octave = 0;
                buffer[written++] = feature;
            }
            return written;
        }
    }

    size_t mser::detect(const unsigned char* __restrict const data, const int width, const int height, const int stride, const options& settings, const size_t feature_point_buffer_size, point* __restrict feature_point_buffer) {
        if ((width <= 0) || (height <= 0) || (feature_point_buffer_size == 0)) {
            return 0;
        }
        core::arena::scope scratch;
        size_t written = detect_polarity(data, width, height, stride, false, settings, feature_point_buffer_size, feature_point_buffer, 0);
        written = detect_polarity(data, width, height, stride, true, settings, feature_point_buffer_size, feature_point_buffer, written);
        return written;
    }
}
