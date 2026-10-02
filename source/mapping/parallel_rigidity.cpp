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

#include "mapping/parallel_rigidity.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <unordered_map>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace mapping {
    namespace {
        class disjoint_sets final {
        public:
            std::vector<size_t> parents;

        public:
            explicit disjoint_sets(const size_t count)
                : parents(count) {
                for (size_t index = 0; index < count; ++index) {
                    this->parents[index] = index;
                }
            }

            size_t find(size_t index) {
                while (this->parents[index] != index) {
                    this->parents[index] = this->parents[this->parents[index]];
                    index = this->parents[index];
                }
                return index;
            }

            // The smaller root is kept, so the sets do not depend on the order of the joins.
            bool join(const size_t lhs, const size_t rhs) {
                const size_t lhs_root = this->find(lhs);
                const size_t rhs_root = this->find(rhs);
                if (lhs_root == rhs_root) {
                    return false;
                }
                if (lhs_root < rhs_root) {
                    this->parents[rhs_root] = lhs_root;
                }
                else {
                    this->parents[lhs_root] = rhs_root;
                }
                return true;
            }
        };

        constexpr static const size_t none = static_cast<size_t>(-1);
    }

    parallel_rigidity::result parallel_rigidity::analyse(const std::vector<track>& tracks) {
        result outcome;

        std::vector<int> frame_ids;
        for (const track& seen : tracks) {
            frame_ids.insert(frame_ids.end(), seen.frames.begin(), seen.frames.end());
        }
        std::sort(frame_ids.begin(), frame_ids.end());
        frame_ids.erase(std::unique(frame_ids.begin(), frame_ids.end()), frame_ids.end());
        const unsigned long long frame_count = static_cast<unsigned long long>(frame_ids.size());

        // Each track's observers as frame indices, and whether each observation is still in a four-loop.
        std::vector<std::vector<size_t>> observers(tracks.size());
        std::vector<std::vector<char>> looped(tracks.size());
        for (size_t index = 0; index < tracks.size(); ++index) {
            for (const int frame_id : tracks[index].frames) {
                observers[index].push_back(static_cast<size_t>(std::lower_bound(frame_ids.begin(), frame_ids.end(), frame_id) - frame_ids.begin()));
            }
            std::sort(observers[index].begin(), observers[index].end());
            observers[index].erase(std::unique(observers[index].begin(), observers[index].end()), observers[index].end());
            looped[index].assign(observers[index].size(), 1);
        }

        // An edge for every pair of frames that see a point together.
        std::unordered_map<unsigned long long, size_t> edge_of;
        std::vector<std::pair<size_t, size_t>> edges;
        for (const std::vector<size_t>& seen : observers) {
            for (size_t first = 0; first < seen.size(); ++first) {
                for (size_t second = first + 1; second < seen.size(); ++second) {
                    const unsigned long long key = (static_cast<unsigned long long>(seen[first]) * frame_count) + static_cast<unsigned long long>(seen[second]);
                    if (edge_of.insert({ key, edges.size() }).second) {
                        edges.push_back({ seen[first], seen[second] });
                    }
                }
            }
        }
        const auto edge_between = [&edge_of, frame_count](const size_t lhs, const size_t rhs) {
            const size_t low = (lhs < rhs) ? lhs : rhs;
            const size_t high = (lhs < rhs) ? rhs : lhs;
            return edge_of.at((static_cast<unsigned long long>(low) * frame_count) + static_cast<unsigned long long>(high));
        };

        // Step 1: an edge needs two points that both its frames still see, and an observation an edge to another frame
        // seeing the point, which leaves every remaining observation in a four-loop. Dropping either can break the other.
        std::vector<char> edge_alive(edges.size(), 1);
        std::vector<size_t> shared(edges.size(), 0);
        for (bool changed = true; changed;) {
            changed = false;
            std::fill(shared.begin(), shared.end(), 0);
            for (size_t index = 0; index < tracks.size(); ++index) {
                const std::vector<size_t>& seen = observers[index];
                for (size_t first = 0; first < seen.size(); ++first) {
                    for (size_t second = first + 1; (looped[index][first] != 0) && (second < seen.size()); ++second) {
                        shared[edge_between(seen[first], seen[second])] += static_cast<size_t>(looped[index][second]);
                    }
                }
            }
            for (size_t edge = 0; edge < edges.size(); ++edge) {
                if ((edge_alive[edge] != 0) && (shared[edge] < 2)) {
                    edge_alive[edge] = 0;
                    changed = true;
                }
            }
            for (size_t index = 0; index < tracks.size(); ++index) {
                const std::vector<size_t>& seen = observers[index];
                size_t remaining = 0;
                for (size_t first = 0; first < seen.size(); ++first) {
                    if (looped[index][first] == 0) {
                        continue;
                    }
                    bool supported = false;
                    for (size_t second = 0; !supported && (second < seen.size()); ++second) {
                        supported = (second != first) && (looped[index][second] != 0) && (edge_alive[edge_between(seen[first], seen[second])] != 0);
                    }
                    if (supported) {
                        ++remaining;
                    }
                    else {
                        looped[index][first] = 0;
                        changed = true;
                    }
                }
                for (size_t first = 0; (remaining < 2) && (first < seen.size()); ++first) {
                    changed = changed || (looped[index][first] != 0);
                    looped[index][first] = 0;
                }
            }
        }

        // Step 2: edges sharing a frame and a point merge, the four-loops of each edge being merged already.
        disjoint_sets edge_sets(edges.size());
        for (size_t index = 0; index < tracks.size(); ++index) {
            const std::vector<size_t>& seen = observers[index];
            for (size_t first = 0; first < seen.size(); ++first) {
                if (looped[index][first] == 0) {
                    continue;
                }
                size_t joined = none;
                for (size_t second = 0; second < seen.size(); ++second) {
                    if ((second == first) || (looped[index][second] == 0)) {
                        continue;
                    }
                    const size_t edge = edge_between(seen[first], seen[second]);
                    if (edge_alive[edge] == 0) {
                        continue;
                    }
                    if (joined == none) {
                        joined = edge;
                    }
                    else {
                        edge_sets.join(joined, edge);
                    }
                }
            }
        }
        std::vector<size_t> set_of_root(edges.size(), none);
        size_t set_count = 0;
        for (size_t edge = 0; edge < edges.size(); ++edge) {
            if ((edge_alive[edge] != 0) && (set_of_root[edge_sets.find(edge)] == none)) {
                set_of_root[edge_sets.find(edge)] = set_count++;
            }
        }
        const auto set_of = [&edge_sets, &set_of_root](const size_t edge) {
            return set_of_root[edge_sets.find(edge)];
        };
        std::vector<std::vector<size_t>> sets_of_track(tracks.size());
        for (size_t index = 0; index < tracks.size(); ++index) {
            const std::vector<size_t>& seen = observers[index];
            for (size_t first = 0; first < seen.size(); ++first) {
                for (size_t second = first + 1; (looped[index][first] != 0) && (second < seen.size()); ++second) {
                    const size_t edge = edge_between(seen[first], seen[second]);
                    if ((looped[index][second] != 0) && (edge_alive[edge] != 0)) {
                        sets_of_track[index].push_back(set_of(edge));
                    }
                }
            }
            std::sort(sets_of_track[index].begin(), sets_of_track[index].end());
            sets_of_track[index].erase(std::unique(sets_of_track[index].begin(), sets_of_track[index].end()), sets_of_track[index].end());
        }

        // Step 3: subgraphs sharing two points merge, until no two do.
        disjoint_sets merged(set_count);
        std::vector<size_t> roots;
        for (bool changed = true; changed;) {
            changed = false;
            std::unordered_map<unsigned long long, size_t> common;
            std::vector<std::pair<size_t, size_t>> pairs;
            for (const std::vector<size_t>& sets : sets_of_track) {
                roots.clear();
                for (const size_t set : sets) {
                    roots.push_back(merged.find(set));
                }
                std::sort(roots.begin(), roots.end());
                roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
                for (size_t first = 0; first < roots.size(); ++first) {
                    for (size_t second = first + 1; second < roots.size(); ++second) {
                        if (++common[(static_cast<unsigned long long>(roots[first]) * static_cast<unsigned long long>(set_count)) + static_cast<unsigned long long>(roots[second])] == 2) {
                            pairs.push_back({ roots[first], roots[second] });
                        }
                    }
                }
            }
            std::sort(pairs.begin(), pairs.end());
            for (const std::pair<size_t, size_t>& pair : pairs) {
                changed = merged.join(pair.first, pair.second) || changed;
            }
        }

        std::vector<component> by_root(set_count);
        for (size_t edge = 0; edge < edges.size(); ++edge) {
            if (edge_alive[edge] == 0) {
                continue;
            }
            component& part = by_root[merged.find(set_of(edge))];
            part.frames.push_back(frame_ids[edges[edge].first]);
            part.frames.push_back(frame_ids[edges[edge].second]);
        }
        for (size_t index = 0; index < tracks.size(); ++index) {
            for (const size_t set : sets_of_track[index]) {
                by_root[merged.find(set)].points.push_back(tracks[index].point);
            }
        }
        for (component& part : by_root) {
            if (part.frames.empty()) {
                continue;
            }
            std::sort(part.frames.begin(), part.frames.end());
            part.frames.erase(std::unique(part.frames.begin(), part.frames.end()), part.frames.end());
            std::sort(part.points.begin(), part.points.end());
            part.points.erase(std::unique(part.points.begin(), part.points.end()), part.points.end());
            outcome.components.push_back(static_cast<component&&>(part));
        }
        std::sort(outcome.components.begin(), outcome.components.end(), [](const component& lhs, const component& rhs) {
            if (lhs.frames.size() != rhs.frames.size()) {
                return lhs.frames.size() > rhs.frames.size();
            }
            if (lhs.points.size() != rhs.points.size()) {
                return lhs.points.size() > rhs.points.size();
            }
            return lhs.frames.front() < rhs.frames.front();
        });

        for (size_t index = 0; index < tracks.size(); ++index) {
            for (size_t first = 0; first < observers[index].size(); ++first) {
                if (looped[index][first] == 0) {
                    outcome.unlooped.push_back({ tracks[index].point, frame_ids[observers[index][first]] });
                }
            }
        }
        std::sort(outcome.unlooped.begin(), outcome.unlooped.end());
        return outcome;
    }
}
